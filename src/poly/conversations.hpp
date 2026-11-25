// src/poly/conversations.hpp - the conversations a server has read, kept so that a prompt which starts like one of
// them is not read again from its first token.
//
// The engine holds one sequence at a time.  Coming back to an earlier point of a sequence takes two things: the
// recurrent state at that point, which is a checkpoint, and the rows its tokens left in the attention caches.  The
// rows of the sequence in the engine are there already.  A conversation that makes room for another is parked:
// its rows are copied to host memory, up to its last checkpoint.  What is saved is written once and never changed,
// so conversations that start alike share it, and a parked conversation comes back bit for bit.
//
// Nothing here knows the model.  E is the engine:
//     void reset();
//     int64_t pos() const;
//     size_t state_bytes() const;  void save_state(uint8_t*);  void load_state(const uint8_t*, int64_t pos);
//     size_t row_bytes() const;    void save_rows(int64_t first, int64_t n, uint8_t*);
//     void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::poly {

struct CacheOptions {
    int checkpoints = 6;       ///< kept per conversation; 0: a prompt is reread unless it goes on from the last token
    int64_t every = 0;         ///< a checkpoint every so many prompt tokens, besides the ones at the turns' starts
                               ///< (checkpoints_at()) and the one before the prompt's last token
    int64_t park_mib = 0;      ///< host memory for parked conversations; 0: none is parked
    int park_slots = 4;        ///< parked conversations at most
    int64_t park_min = 1024;   ///< a conversation with fewer tokens before its last checkpoint is not parked
    std::function<bool(size_t)> admit;   ///< whether the host can spare this many bytes; unset: it can
    FILE* log = nullptr;       ///< where parking is reported; the server reads these lines
    uint64_t fingerprint = 0;  ///< what a session file must match to be loaded: a file holds raw state and rows, which
                               ///< mean nothing to another model or build (main.cpp fills it from the engine)
};

template<class E>
class Conversations {
public:
    /// Why a session file could not be written or read; main.cpp names it in the answer to the server
    /// (SERR <invalid|memory|storage|io>).
    enum class FileError { invalid, memory, storage, io };
    Conversations(E& engine, CacheOptions o)
        : e_(engine), o_(std::move(o)), slots_((size_t) std::max(o_.park_slots, 0) + 1) {}

    /// The tokens the engine has read.
    const std::vector<int32_t>& seq() const { return slots_[cur_].seq; }

    /// Leaves the engine at the furthest point of `ids` that can be come back to, and returns that position.
    int64_t resume(const std::vector<int32_t>& ids) {
        const int64_t n = (int64_t) ids.size();
        anchors_.clear();
        size_t best = cur_;
        int64_t at = 0, same = 0;
        for (size_t i = 0; i < slots_.size(); ++i) {
            const int64_t c = common(slots_[i].seq, ids), r = reach(i, c, n);
            if (r > at || (r == at && i == cur_)) best = i, at = r, same = c;
        }
        ++clock_;
        if (at == 0) {
            // nothing kept starts like this
            if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
            slots_[cur_] = Slot{};
            e_.reset();
        } else if (best != cur_ || at < len(cur())) {
            if (best != cur_) enter(best, at);
            // The conversation is left at `at`.  If the prompt the client sent for it went on from there, the
            // client may send that again, so the conversation is parked and this one goes on in a copy.
            if (same >= cur().prompt || !branch(at)) cut(cur(), at);
            e_.load_state(state_at(cur(), at), at);
        }
        cur().stamp = clock_;
        return at;
    }

    /// One more checkpoint of the prompt resume() took up: where its chat's last turn begins, which the server
    /// knows from the template.  The chat's next request renders the conversation up to there as this one does,
    /// but this turn's header and reply not always in the same tokens (a template drops the reply's thinking), so
    /// the checkpoint before the prompt's last token is seldom where it goes on from.  A place the read does not
    /// pass adds none.
    void turn_at(int64_t at) { checkpoints_at(at > 0 ? std::vector<int64_t>{at} : std::vector<int64_t>{}); }

    /// Every place the prompt's turns begin, as token-prefix lengths, which the server reads from the template:
    /// a checkpoint goes at each, so a chat is come back to exactly where a client can take it up again - an
    /// edited or regenerated turn starts at one of them, whatever its token interval.  The last `checkpoints` of
    /// them are kept, beside the one before the prompt's last token.
    void checkpoints_at(std::vector<int64_t> at) {
        anchors_.clear();
        for (const int64_t p : at)
            if (p > 0) anchors_.push_back(p);
        std::sort(anchors_.begin(), anchors_.end());
        anchors_.erase(std::unique(anchors_.begin(), anchors_.end()), anchors_.end());
    }

    /// Of `want` tokens from position `at`, how many to read before the next checkpoint is due.
    int64_t until_checkpoint(int64_t at, int64_t want) const {
        if (o_.checkpoints < 1) return want;
        if (o_.every > 0) want = std::min(want, (at / o_.every + 1) * o_.every - at);
        const auto next = std::upper_bound(anchors_.begin(), anchors_.end(), at);
        if (next != anchors_.end()) want = std::min(want, *next - at);
        return want;
    }

    /// Before the engine reads the token at `at`; `last` says it is the prompt's last.  Takes a checkpoint if one is
    /// due.  One sits before the last token of every prompt, so coming back to it leaves a token to read and the
    /// logits that come with it.
    void before(int64_t at, bool last) {
        if (o_.checkpoints < 1 || at < 1) return;
        const bool anchor = std::binary_search(anchors_.begin(), anchors_.end(), at);
        if (!last && !anchor && (o_.every < 1 || at % o_.every != 0)) return;
        if (const BlobPtr b = mark(at)) e_.save_state(b->p.get());
    }

    /// For an engine that reads past checkpoints in one go, of n tokens from `at`: the checkpoints due after the
    /// read's first token.  `out` gets, for each, how many of the read's tokens come before it and where its
    /// state_bytes() go; null where no state is wanted, and the engine still breaks its work there as if one
    /// were.  The engine writes a state when it has it.  The checkpoints are the conversation's once read() has
    /// the read's tokens, and not at all when the read is given up (rewind()).
    ///
    /// A state is wanted for the checkpoints the conversation would still have, had each been taken in turn as
    /// before() takes one.  A long prompt has more due than are kept, and a state can be hundreds of MiB.
    void inside(int64_t at, int64_t n, std::vector<std::pair<int64_t, uint8_t*>>& out) {
        out.clear();
        due_.clear();
        after_.clear();
        if (o_.checkpoints < 1) return;
        std::vector<int64_t> due;
        if (o_.every > 0)
            for (int64_t p = (at / o_.every + 1) * o_.every; p < at + n; p += o_.every) due.push_back(p);
        for (const int64_t p : anchors_)
            if (p > at && p < at + n && (o_.every < 1 || p % o_.every != 0)) due.push_back(p);
        std::sort(due.begin(), due.end());
        due.erase(std::unique(due.begin(), due.end()), due.end());
        if (due.empty()) return;
        for (const Checkpoint& m : cur().marks) after_.push_back(m.pos);
        for (const int64_t p : due) {
            after_.push_back(p);
            while (after_.size() > (size_t) o_.checkpoints) after_.erase(after_.begin() + (std::ptrdiff_t) drop_of(after_));
        }
        for (const int64_t p : due) {
            const bool stays = std::find(after_.begin(), after_.end(), p) != after_.end();
            due_.push_back({p, stays ? blob(e_.state_bytes()) : nullptr});
            out.emplace_back(p - at, due_.back().state ? due_.back().state->p.get() : nullptr);
        }
    }

    /// The engine has read these tokens.  A read that inside() was asked about is done with them, and its
    /// checkpoints take their places.
    void read(const int32_t* t, int64_t n) {
        Slot& s = cur();
        s.seq.insert(s.seq.end(), t, t + n);
        if (due_.empty()) return;
        const auto stays = [&](const Checkpoint& m) {
            return m.state && std::find(after_.begin(), after_.end(), m.pos) != after_.end();
        };
        std::vector<Checkpoint> marks;
        for (const Checkpoint& m : s.marks)
            if (stays(m)) marks.push_back(m);
        for (const Checkpoint& m : due_)
            if (stays(m)) marks.push_back(m);
        s.marks = std::move(marks);
        due_.clear();
    }

    /// A read was given up part way: what it left in the engine is not the conversation's, and the checkpoints
    /// inside it are not taken.  The conversation goes back to its last checkpoint, which the engine is left
    /// at, or is forgotten when it has none.  Returns where the conversation is then.
    int64_t rewind() {
        due_.clear();
        Slot& s = cur();
        if (s.marks.empty()) {
            slots_[cur_] = Slot{};
            e_.reset();
            return 0;
        }
        const int64_t at = s.marks.back().pos;
        cut(s, at);
        e_.load_state(s.marks.back().state->p.get(), at);
        return at;
    }
    /// The first n tokens of the conversation are the prompt the client sent.
    void prompt_was(int64_t n) { cur().prompt = std::min(n, len(cur())); }
    /// After an engine error: the conversation being read is lost.  The caller resets the engine.
    void drop() {
        slots_[cur_] = Slot{};
        due_.clear();
    }

    /// The engine's sequence is about to hold another conversation, read elsewhere (a batch slot's copy): the one
    /// it holds is parked, as resume() parks it, or forgotten.  adopt() then says what it holds.
    void leave() {
        if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
        slots_[cur_] = Slot{};
        due_.clear();
    }
    /// The engine holds `seq` now, with the state at its end: a prompt that goes on from there reads none of it.
    void adopt(std::vector<int32_t> seq) {
        Slot& s = cur();
        s = Slot{};
        s.seq = std::move(seq);
        s.prompt = len(s);
        s.stamp = ++clock_;
    }

    /// The parked conversations, the bytes they hold, and how many were pushed out so far.
    size_t parked() const { return held(slots_.size()).count; }
    size_t parked_bytes() const { return held(slots_.size()).bytes; }
    size_t evictions() const { return evictions_; }

private:
    using Clock = std::chrono::steady_clock;

    struct Blob {
        std::unique_ptr<uint8_t[]> p;
        size_t bytes = 0;
    };
    using BlobPtr = std::shared_ptr<const Blob>;
    struct Checkpoint {
        int64_t pos;
        BlobPtr state;
    };
    // The blob holds `count` rows from row `first`; the first `keep` of them are this conversation's.
    struct Rows {
        BlobPtr blob;
        int64_t first, count, keep;
    };
    struct Slot {
        std::vector<int32_t> seq;
        int64_t prompt = 0;             // seq's first tokens that the client sent
        std::vector<Checkpoint> marks;  // by position
        std::vector<Rows> rows;         // back to back from row 0; a parked conversation has all of its rows
        uint64_t stamp = 0;
    };
    struct Held {
        size_t count = 0, bytes = 0;
    };
    // A row chunk's place in a session file: `rows` positions from `first`, its packed payload at `at`.
    struct ChunkAt {
        int64_t first, rows;
        uint64_t at;
    };

    // A session file's payload is mostly zeros: the rows hold a block's compressed row once per block, so most of
    // their bytes are the positions a block does not end at.  A run of zeros is written as its length - record
    // u32 n, bit 31 set: n zero bytes; clear: n literal bytes follow.  A zero run shorter than 16 bytes is left in
    // the literal: its record would cost more than the bytes it saves.
    static constexpr uint32_t kZeroRun = 0x80000000u;

    static void pack_append(std::vector<uint8_t>& out, const uint8_t* p, size_t n) {
        const auto put32 = [&out](uint32_t v) {
            const uint8_t* b = (const uint8_t*) &v;
            out.insert(out.end(), b, b + sizeof v);
        };
        size_t lit = 0, i = 0;
        while (i < n) {
            if (p[i] != 0) {
                ++i;
                continue;
            }
            size_t z = i;
            while (z < n && p[z] == 0) ++z;
            if (z - i < 16) {
                i = z;
                continue;
            }
            if (i > lit) {
                put32((uint32_t) (i - lit));
                out.insert(out.end(), p + lit, p + i);
            }
            for (size_t run = z - i; run > 0;) {
                const uint32_t m = (uint32_t) std::min<size_t>(run, 0x7fffffffu);
                put32(m | kZeroRun);
                run -= m;
            }
            i = lit = z;
        }
        if (n > lit) {
            put32((uint32_t) (n - lit));
            out.insert(out.end(), p + lit, p + n);
        }
    }

    // One packed payload, `n` bytes of it, through `get`.
    template <class Get>
    static bool unpack_into(Get&& get, uint8_t* dst, size_t n) {
        for (size_t done = 0; done < n;) {
            uint32_t h = 0;
            if (!get(&h, sizeof h)) return false;
            const size_t m = h & ~kZeroRun;
            if (m == 0 || m > n - done) return false;
            if (h & kZeroRun) std::memset(dst + done, 0, m);
            else if (!get(dst + done, m)) return false;
            done += m;
        }
        return true;
    }

    // The same payload's structure without reading its literals: true when its records make exactly `n` bytes.
    static bool walk_packed(FILE* f, size_t n) {
        for (size_t done = 0; done < n;) {
            uint32_t h = 0;
            if (std::fread(&h, sizeof h, 1, f) != 1) return false;
            const size_t m = h & ~kZeroRun;
            if (m == 0 || m > n - done) return false;
            if (!(h & kZeroRun) && std::fseek(f, (long) m, SEEK_CUR) != 0) return false;
            done += m;
        }
        return true;
    }

    Slot& cur() { return slots_[cur_]; }
    const Slot& cur() const { return slots_[cur_]; }
    static int64_t len(const Slot& s) { return (int64_t) s.seq.size(); }
    static int64_t saved(const Slot& s) { return s.rows.empty() ? 0 : s.rows.back().first + s.rows.back().keep; }

    static int64_t common(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return (int64_t) n;
    }

    // How far into a prompt of n tokens slot i gets it, given that they start with `same` tokens alike.
    int64_t reach(size_t i, int64_t same, int64_t n) const {
        const Slot& s = slots_[i];
        if (i == cur_ && same > 0 && same == len(s) && n > same) return same;   // the engine is right there
        int64_t at = 0;
        for (const Checkpoint& m : s.marks)
            if (m.pos <= std::min(same, n - 1)) at = m.pos;
        return at;
    }

    static const uint8_t* state_at(const Slot& s, int64_t at) {
        for (const Checkpoint& m : s.marks)
            if (m.pos == at) return m.state->p.get();
        return nullptr;   // resume() only goes to positions reach() found
    }

    static BlobPtr blob(size_t bytes) {
        auto b = std::make_shared<Blob>();
        b->p.reset(new (std::nothrow) uint8_t[bytes]);
        b->bytes = bytes;
        return b->p ? b : nullptr;
    }

    // A checkpoint of the current conversation at `at`, its state still to be written.  Null: there is one there,
    // or no memory for it.
    BlobPtr mark(int64_t at) {
        Slot& s = cur();
        if (!s.marks.empty() && s.marks.back().pos == at) return nullptr;
        const BlobPtr b = blob(e_.state_bytes());
        if (!b) return nullptr;
        s.marks.push_back({at, b});
        while (s.marks.size() > (size_t) o_.checkpoints) {
            std::vector<int64_t> pos;
            for (const Checkpoint& m : s.marks) pos.push_back(m.pos);
            s.marks.erase(s.marks.begin() + (std::ptrdiff_t) drop_of(pos));
        }
        return b;
    }

    // Of more checkpoints than are kept, the one that goes: the one whose neighbours are closest together.  The
    // last two stay, the end of this prompt and of the one before being where a client most often comes back to.
    static size_t drop_of(const std::vector<int64_t>& pos) {
        size_t drop = 0;
        int64_t gap = INT64_MAX;
        for (size_t k = 0; k + 2 < pos.size(); ++k) {
            const int64_t g = pos[k + 1] - (k ? pos[k - 1] : 0);
            if (g < gap) gap = g, drop = k;
        }
        return drop;
    }

    static void cut(Slot& s, int64_t at) {
        if (at >= len(s)) return;
        s.seq.resize((size_t) at);
        s.prompt = std::min(s.prompt, at);
        while (!s.marks.empty() && s.marks.back().pos > at) s.marks.pop_back();
        while (!s.rows.empty() && s.rows.back().first >= at) s.rows.pop_back();
        if (!s.rows.empty()) s.rows.back().keep = std::min(s.rows.back().keep, at - s.rows.back().first);
    }

    // The parked conversations and what they hold; a blob several of them share counts once.  The current one
    // counts when `with` is its slot.
    Held held(size_t with) const {
        Held h;
        std::vector<const Blob*> seen;
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot& s = slots_[i];
            if (s.seq.empty() || (i == cur_ && i != with)) continue;
            ++h.count;
            for (const Checkpoint& m : s.marks) seen.push_back(m.state.get());
            for (const Rows& r : s.rows) seen.push_back(r.blob.get());
        }
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (const Blob* b : seen) h.bytes += b->bytes;
        return h;
    }

    // Makes the parked conversations, the current one among them and `extra` bytes more, fit the budget and the
    // slots.  The least recently used goes first, whatever its length: a long conversation can lose its place to
    // several short ones.  Slot `keep` stays; it is the one about to take the engine, if any, so it frees a slot.
    // False: it does not fit, or the host cannot spare the bytes.
    bool fit(size_t extra, size_t keep) {
        const size_t budget = (size_t) o_.park_mib << 20;
        for (;;) {
            const Held h = held(cur_);
            const size_t leaving = keep < slots_.size() && keep != cur_ && !slots_[keep].seq.empty() ? 1 : 0;
            if (h.bytes + extra <= budget && h.count - leaving <= (size_t) o_.park_slots) break;
            size_t out = slots_.size();
            for (size_t i = 0; i < slots_.size(); ++i)
                if (i != cur_ && i != keep && !slots_[i].seq.empty() &&
                    (out == slots_.size() || slots_[i].stamp < slots_[out].stamp))
                    out = i;
            if (out == slots_.size()) return false;
            slots_[out] = Slot{};
            ++evictions_;
        }
        return extra == 0 || !o_.admit || o_.admit(extra);
    }

    // Copies the rows the current conversation still has only in the engine, up to its last checkpoint; what it
    // read after that cannot be come back to and is dropped.  False: it cannot be kept.
    bool park(size_t keep) {
        Slot& s = cur();
        if (o_.park_mib < 1 || o_.park_slots < 1 || s.marks.empty() || s.marks.back().pos < o_.park_min) return false;
        const Clock::time_point t0 = Clock::now();
        const int64_t end = s.marks.back().pos, from = saved(s);
        cut(s, end);
        const size_t bytes = (size_t) (end - from) * e_.row_bytes();
        const BlobPtr b = from < end && fit(bytes, keep) ? blob(bytes) : nullptr;
        if (from < end && !b) {
            say("skipped", end, t0, slots_.size());
            return false;
        }
        if (b) {
            e_.save_rows(from, end - from, b->p.get());
            s.rows.push_back({b, from, end - from, end - from});
        }
        if (!fit(0, keep)) return false;
        say("parked", end, t0, cur_);
        return true;
    }

    // An empty slot other than `keep`.
    size_t take(size_t keep) {
        size_t out = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (i == keep) continue;
            if (slots_[i].seq.empty()) return i;
            if (out == slots_.size() || slots_[i].stamp < slots_[out].stamp) out = i;
        }
        slots_[out] = Slot{};   // park() left one empty, so this is not reached
        ++evictions_;
        return out;
    }

    // Slot i takes the engine, with its rows before `need` back in the caches.  The conversation that leaves is
    // parked if it can be.  Rows that both have from the same blob are in the caches already.
    void enter(size_t i, int64_t need) {
        const Clock::time_point t0 = Clock::now();
        const Slot& in = slots_[i];
        int64_t have = 0;
        for (size_t k = 0; k < in.rows.size() && k < cur().rows.size() && in.rows[k].blob == cur().rows[k].blob; ++k) {
            have = in.rows[k].first + std::min(in.rows[k].keep, cur().rows[k].keep);
            if (in.rows[k].keep != cur().rows[k].keep) break;
        }
        if (cur().seq.empty() || !park(i)) slots_[cur_] = Slot{};
        for (const Rows& r : in.rows) {
            const int64_t a = std::max(r.first, have), b = std::min(r.first + r.keep, need);
            if (a < b) e_.load_rows(r.blob->p.get(), r.first, r.count, a, b - a);
        }
        cur_ = i;
        say("restored", need, t0, slots_.size());
    }

    // Parks the current conversation whole and goes on at `at` in a copy of its start, which shares what is
    // saved.  False: it cannot be parked.
    bool branch(int64_t at) {
        const size_t from = cur_;
        if (!park(slots_.size())) return false;
        const size_t to = take(from);
        const Slot& a = slots_[from];
        Slot b;
        b.seq.assign(a.seq.begin(), a.seq.begin() + (std::ptrdiff_t) at);
        b.prompt = std::min(a.prompt, at);
        for (const Checkpoint& m : a.marks)
            if (m.pos <= at) b.marks.push_back(m);
        for (const Rows& r : a.rows)
            if (r.first < at) b.rows.push_back({r.blob, r.first, r.count, std::min(r.keep, at - r.first)});
        slots_[to] = std::move(b);
        cur_ = to;
        return true;
    }

public:
    // ---- the session file: one conversation whole, states and rows, so that it comes back after a restart, or
    // stays on the disk while the RAM keeps others.  The marks are what park() keeps, plus one more at the end of
    // what the engine has - its live state - so a restored conversation reads nothing of itself back at all.
    //
    // The payloads are packed: a conversation's rows hold a block's compressed row once per block, so about 85% of
    // their bytes are the positions a block does not end at, and a state's ring has its own empty stretches.  A run
    // of zeros is written as its length (see pack_append()); a real DeepSeek4 file comes out about a third of the
    // bytes.  The rows are written and read a chunk of positions at a time through one fixed stage buffer, so a save
    // never holds gigabytes of host memory and a restore never reads the file into one blob.
    //
    // Layout (little endian, version 2):
    //     "SPCSV2\0\0"  u64 fingerprint  i64 tokens  i64 prompt  tokens x i32
    //     u32 marks, each { i64 pos  u64 raw  u64 packed  packed }
    //     u32 row entries, each { i64 first  i64 keep  u32 chunks }
    //         a chunk: { u32 rows  u64 packed  packed }
    // The fingerprint is what main.cpp makes of the engine's geometry: raw state of another model or build means
    // nothing here, so such a file is refused rather than loaded.
    static constexpr int64_t kChunkRows = 512;   // positions a row chunk stages: about 29 MiB at DeepSeek4's rows

    bool save_file(const std::string& path, std::string& why) {
        last_file_bytes_ = 0;
        last_file_error_ = FileError::io;
        Slot& s = cur();
        const int64_t end = std::min(len(s), e_.pos());
        if (end < 1 || s.marks.empty()) {
            last_file_error_ = FileError::invalid;
            why = "no conversation to save";
            return false;
        }
        const Clock::time_point t0 = Clock::now();
        std::vector<Checkpoint> marks;
        for (const Checkpoint& m : s.marks)
            if (m.pos <= end) marks.push_back(m);
        const bool live = marks.empty() || marks.back().pos < end;
        const size_t row = e_.row_bytes();
        std::vector<uint8_t> stage((size_t) std::min(kChunkRows, end) * row), packed;
        // one chunk is staged and one packed copy of it is held, and the state at the end is taken whole
        const size_t need = stage.size() * 2 + (live ? e_.state_bytes() : 0);
        if (o_.admit && !o_.admit(need)) {
            last_file_error_ = FileError::memory;
            why = "the host cannot spare " + std::to_string(need >> 20) + " MiB for the session file";
            return false;
        }
        BlobPtr tip;    // the state at the end: what the engine holds now
        if (live) {
            tip = blob(e_.state_bytes());
            if (!tip) {
                last_file_error_ = FileError::memory;
                why = "not enough RAM for the state to be written";
                return false;
            }
            e_.save_state(tip->p.get());
        }
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) {
            last_file_error_ = errno == ENOSPC ? FileError::storage : FileError::io;
            why = "cannot write " + path + ": " + std::strerror(errno);
            return false;
        }
        const auto put = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
        const auto put64 = [&](uint64_t v) { return put(&v, sizeof v); };
        const auto put32 = [&](uint32_t v) { return put(&v, sizeof v); };
        const auto puti64 = [&](int64_t v) { return put(&v, sizeof v); };
        const auto pack_of = [&](const uint8_t* p, size_t n) {
            packed.clear();
            pack_append(packed, p, n);
            return packed.size();
        };
        const auto put_packed = [&](size_t n_packed) {
            return put64(n_packed) && (n_packed == 0 || put(packed.data(), n_packed));
        };
        bool ok = put("SPCSV2\0\0", 8) && put64(o_.fingerprint) && puti64((int64_t) s.seq.size()) &&
                  puti64(s.prompt) && put(s.seq.data(), s.seq.size() * sizeof(int32_t)) &&
                  put32((uint32_t) (marks.size() + (live ? 1 : 0)));
        for (size_t k = 0; ok && k < marks.size(); ++k)
            ok = puti64(marks[k].pos) && put64(marks[k].state->bytes) &&
                 put_packed(pack_of(marks[k].state->p.get(), marks[k].state->bytes));
        if (ok && live) ok = puti64(end) && put64(tip->bytes) && put_packed(pack_of(tip->p.get(), tip->bytes));
        // Row entries: the parked blobs first, then what only the engine holds, a chunk of positions at a time.
        int64_t covered = 0;
        uint32_t entries = 0;
        for (const Rows& r : s.rows) {
            if (r.first >= end) break;
            ++entries;
            covered = r.first + std::min(r.keep, end - r.first);
        }
        if (covered < end) ++entries;
        ok = ok && put32(entries);
        for (const Rows& r : s.rows) {
            if (!ok || r.first >= end) break;
            const int64_t keep = std::min(r.keep, end - r.first);
            ok = puti64(r.first) && puti64(keep) && put32((uint32_t) ((keep + kChunkRows - 1) / kChunkRows));
            for (int64_t at = 0; ok && at < keep; at += kChunkRows) {
                const int64_t m = std::min(kChunkRows, keep - at);
                ok = put32((uint32_t) m) &&
                     put_packed(pack_of(r.blob->p.get() + (size_t) at * row, (size_t) m * row));
            }
        }
        if (ok && covered < end) {
            const int64_t keep = end - covered;
            ok = puti64(covered) && puti64(keep) && put32((uint32_t) ((keep + kChunkRows - 1) / kChunkRows));
            for (int64_t at = covered; ok && at < end; at += kChunkRows) {
                const int64_t m = std::min(kChunkRows, end - at);
                e_.save_rows(at, m, stage.data());
                ok = put32((uint32_t) m) && put_packed(pack_of(stage.data(), (size_t) m * row));
            }
        }
        const long at = std::ftell(f);
        if (std::fclose(f) != 0 || !ok) {
            if (errno == ENOSPC) last_file_error_ = FileError::storage;
            why = "cannot write " + path;
            return false;
        }
        last_file_bytes_ = (size_t) at;
        say("saved", end, t0, cur_);
        return true;
    }

    // The conversation of `path` takes the engine; the one the engine held is parked if it can be, and dropped
    // when it cannot.  The end state the file holds puts the engine at the conversation's end, so its next turn
    // reads none of it again.  The file is walked whole before the engine is touched: a damaged one is refused
    // with the conversation as it was, never half mounted.
    bool load_file(const std::string& path, std::string& why) {
        last_file_bytes_ = 0;
        last_file_error_ = FileError::io;
        const Clock::time_point t0 = Clock::now();
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            why = "cannot read " + path + ": " + std::strerror(errno);
            return false;
        }
        const auto get = [&](void* p, size_t n) { return n == 0 || std::fread(p, 1, n, f) == n; };
        char magic[8] = {};
        uint64_t fp = 0;
        int64_t n = 0, prompt = 0;
        uint32_t n_marks = 0, n_entries = 0;
        bool ok = get(magic, 8) && std::memcmp(magic, "SPCSV2", 6) == 0 && get(&fp, sizeof fp) && get(&n, sizeof n) &&
                  get(&prompt, sizeof prompt);
        if (!ok) {
            std::fclose(f);
            last_file_error_ = FileError::invalid;
            why = "not a session file: " + path;
            return false;
        }
        if (fp != o_.fingerprint) {
            std::fclose(f);
            last_file_error_ = FileError::invalid;
            why = "the session file is of another model or build";
            return false;
        }
        if (n < 1 || n > e_.max_context()) {
            std::fclose(f);
            last_file_error_ = FileError::invalid;
            why = "the session file holds " + std::to_string((long long) n) + " tokens, which the context cannot keep";
            return false;
        }
        Slot in;
        in.seq.resize((size_t) n);
        ok = get(in.seq.data(), (size_t) n * sizeof(int32_t)) && get(&n_marks, sizeof n_marks);
        // The marks are read whole (a state is small beside the rows): they are what the conversation comes back at.
        for (uint32_t k = 0; ok && k < n_marks; ++k) {
            int64_t pos = 0;
            uint64_t raw = 0, packed = 0;
            ok = get(&pos, sizeof pos) && get(&raw, sizeof raw) && get(&packed, sizeof packed);
            if (!ok) break;
            if (raw != e_.state_bytes() || pos < 1 || pos > n || (!in.marks.empty() && pos <= in.marks.back().pos)) {
                ok = false;
                last_file_error_ = FileError::invalid;
                why = "the session file's checkpoints do not fit this engine";
                break;
            }
            BlobPtr b = blob((size_t) raw);
            if (!b) {
                ok = false;
                last_file_error_ = FileError::memory;
                why = "not enough RAM for the states of the session";
                break;
            }
            const long at = std::ftell(f);
            if (!(ok = unpack_into(get, b->p.get(), (size_t) raw)) || (uint64_t) (std::ftell(f) - at) != packed) {
                last_file_error_ = FileError::invalid;
                why = "the session file is damaged (a checkpoint)";
                ok = false;
                break;
            }
            in.marks.push_back({pos, b});
        }
        if (ok && in.marks.empty()) {
            ok = false;
            last_file_error_ = FileError::invalid;
            why = "the session file holds no checkpoint";
        }
        // The rows: the chunks are located now and read one at a time once the engine takes the conversation, so
        // the walk refuses a damaged file before a device byte is written.
        const size_t row = e_.row_bytes();
        std::vector<ChunkAt> chunks;
        int64_t rows_end = 0;
        ok = ok && get(&n_entries, sizeof n_entries);
        for (uint32_t k = 0; ok && k < n_entries; ++k) {
            int64_t first = 0, keep = 0;
            uint32_t n_chunks = 0;
            ok = get(&first, sizeof first) && get(&keep, sizeof keep) && get(&n_chunks, sizeof n_chunks);
            if (!ok) break;
            if (first != rows_end || keep < 1 || first + keep > n || n_chunks == 0 ||
                n_chunks != (uint32_t) ((keep + kChunkRows - 1) / kChunkRows)) {
                ok = false;
                last_file_error_ = FileError::invalid;
                why = "the session file's rows do not fit this engine";
                break;
            }
            int64_t got = 0;
            for (uint32_t c = 0; ok && c < n_chunks; ++c) {
                uint32_t rows = 0;
                uint64_t packed = 0;
                ok = get(&rows, sizeof rows) && get(&packed, sizeof packed);
                if (!ok) break;
                if (rows == 0 || rows > (uint32_t) kChunkRows || got + rows > keep) {
                    ok = false;
                    why = "the session file's rows do not fit this engine";
                    last_file_error_ = FileError::invalid;
                    break;
                }
                const long at = std::ftell(f);
                if (!(ok = walk_packed(f, (size_t) rows * row)) || (uint64_t) (std::ftell(f) - at) != packed) {
                    ok = false;
                    why = "the session file is damaged (its rows)";
                    last_file_error_ = FileError::invalid;
                    break;
                }
                chunks.push_back({first + got, (int64_t) rows, (uint64_t) at});
                got += rows;
            }
            if (ok && got != keep) {
                ok = false;
                why = "the session file's rows do not fit this engine";
                last_file_error_ = FileError::invalid;
            }
            rows_end = first + keep;
        }
        const long at = std::ftell(f);
        if (!ok) {
            std::fclose(f);
            if (last_file_error_ == FileError::io) last_file_error_ = FileError::invalid;
            if (why.empty()) why = "the session file is damaged";
            return false;
        }
        // The host stage the mount reads through: one chunk, checked before a byte is taken
        std::vector<uint8_t> stage((size_t) std::min<int64_t>(kChunkRows, std::max<int64_t>(rows_end, 1)) * row);
        if (o_.admit && !o_.admit(stage.size())) {
            std::fclose(f);
            last_file_error_ = FileError::memory;
            why = "the host cannot spare " + std::to_string(stage.size() >> 20) + " MiB to mount the session file";
            return false;
        }
        in.prompt = std::max<int64_t>(0, std::min(prompt, n));
        // The engine takes this conversation: the one it held is parked if it can be, and dropped when it cannot -
        // the operator asked for this one.  A file that holds no state at its end leaves the conversation at its
        // last checkpoint.
        if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
        slots_[cur_] = std::move(in);
        const int64_t mark = cur().marks.back().pos;
        cut(cur(), mark);
        e_.reset();
        e_.load_state(state_at(cur(), mark), mark);
        for (const ChunkAt& c : chunks) {
            const int64_t to = std::min(c.first + c.rows, mark);
            if (to <= c.first) continue;
            std::fseek(f, (long) c.at, SEEK_SET);
            if (!unpack_into(get, stage.data(), (size_t) c.rows * row))
                throw std::runtime_error("the session file changed while it was being mounted");
            e_.load_rows(stage.data(), c.first, c.rows, c.first, to - c.first);
        }
        std::fclose(f);
        cur().stamp = ++clock_;
        last_file_bytes_ = at > 0 ? (size_t) at : 0;
        say("loaded", mark, t0, cur_);
        return true;
    }

    size_t last_file_bytes() const { return last_file_bytes_; }
    /// The protocol's word for why save_file()/load_file() last said no.
    const char* last_file_error() const {
        switch (last_file_error_) {
            case FileError::memory: return "memory";
            case FileError::storage: return "storage";
            case FileError::io: return "io";
            default: return "invalid";
        }
    }

private:

    // `with` is the current slot when its conversation is the one just parked.
    void say(const char* what, int64_t tokens, Clock::time_point t0, size_t with) const {
        if (!o_.log) return;
        const Held h = held(with);
        std::fprintf(o_.log,
                     "strata-poly: conversation cache: %s %lld tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu\n",
                     what, (long long) tokens, std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                     h.count, h.bytes, evictions_);
    }

    E& e_;
    CacheOptions o_;
    std::vector<Slot> slots_;   // the current conversation and the parked ones
    size_t cur_ = 0;
    uint64_t clock_ = 0;
    size_t evictions_ = 0;
    size_t last_file_bytes_ = 0;    // the bytes the last save_file() wrote, or the file's size after load_file()
    FileError last_file_error_ = FileError::invalid;
    std::vector<Checkpoint> due_;   // inside()'s checkpoints, until read() makes them the conversation's
    std::vector<int64_t> after_;    // where the conversation's checkpoints are once that read is done
    std::vector<int64_t> anchors_;  // where the prompt's turns begin (checkpoints_at()); empty: none
};

}  // namespace strata::poly
