// src/poly/ram_pool.hpp - a RAM pool that keeps the most used experts outside VRAM in the process's own
// memory, the fork's native take on upstream's --ram-experts-mib idea (VecSzn's slot pool) without its
// memory-layer refactor: the page cache holding a 100 GB model file competes with everything else in RAM and
// gives no residency promise, while this pool is anonymous memory the kernel cannot reclaim under pressure
// short of swapping - the thing --read-heat's page-cache warming could not promise.
//
// A slot holds one expert whole (gate|up, then down).  copy_out() serves a hit and fill() steals the least
// recently used slot; the metadata moves under one mutex and the bytes move outside it, and a slot somebody
// is busy copying is never stolen.  Two calls filling the same expert at once both write the same bytes,
// which is benign; the last one finished holds the tick.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

namespace strata::poly {

class RamPool {
public:
    /// A pool of `bytes` bytes holding `slot_bytes`-byte experts of `layers` x `experts` keys.
    RamPool(size_t bytes, size_t slot_bytes, int layers, int experts)
        : slot_bytes_(slot_bytes), n_layers_(layers), n_experts_(experts),
          keys_((size_t) layers * (size_t) experts, -1) {
        n_slots_ = (int) (bytes / slot_bytes);
        if (n_slots_ > 0) {
            buf_ = new (std::nothrow) uint8_t[(size_t) n_slots_ * slot_bytes];
            if (!buf_) n_slots_ = 0;
        }
        busy_.assign((size_t) n_slots_, 0);
        tick_.assign((size_t) n_slots_, 0);
        who_.assign((size_t) n_slots_, -1);
    }

    ~RamPool() { delete[] buf_; }

    RamPool(const RamPool&) = delete;
    RamPool& operator=(const RamPool&) = delete;

    int slots() const { return n_slots_; }
    size_t slot_bytes() const { return slot_bytes_; }

    /// 1 when the expert's bytes sit in the pool now.
    bool holds(int il, int e) {
        std::lock_guard<std::mutex> lk(m_);
        return keys_[(size_t) il * n_experts_ + (size_t) e] >= 0;
    }

    /// On a hit copies the expert's gate|up into dst_gu and its down into dst_down, and makes it the most
    /// recently used.  On a miss leaves the destinations alone and returns 0.  The slot's busy flag keeps
    /// its bytes stable between the two locks, so the slot number rides in a local.
    bool copy_out(int il, int e, void* dst_gu, void* dst_down, size_t gu2, size_t d_bytes) {
        if (!n_slots_) return false;
        int s = -1;
        uint64_t now;
        {
            std::lock_guard<std::mutex> lk(m_);
            s = keys_[(size_t) il * n_experts_ + (size_t) e];
            if (s < 0 || busy_[(size_t) s]) return false;
            busy_[(size_t) s] = 1;
            now = ++clock_;
        }
        std::memcpy(dst_gu, buf_ + (size_t) s * slot_bytes_, gu2);
        std::memcpy(dst_down, buf_ + (size_t) s * slot_bytes_ + gu2, d_bytes);
        std::lock_guard<std::mutex> lk(m_);
        busy_[(size_t) s] = 0;
        tick_[(size_t) s] = now;
        return true;
    }

    /// Copies the expert (gate|up, then down) into a free or least-recently-used slot - never one a copy_out
    /// is reading.  With every slot busy the fill is dropped quietly: the read itself already happened.
    void fill(int il, int e, const void* src_gu, const void* src_down, size_t gu2, size_t d_bytes) {
        if (!n_slots_ || gu2 + d_bytes > slot_bytes_) return;
        const size_t x = (size_t) il * n_experts_ + (size_t) e;
        int s = -1;
        uint64_t now = ++clock_;
        {
            std::lock_guard<std::mutex> lk(m_);
            uint64_t oldest = UINT64_MAX;
            for (int q = 0; q < n_slots_; ++q)
                if (!busy_[(size_t) q] && tick_[(size_t) q] < oldest) oldest = tick_[(size_t) q], s = q;
            if (s < 0) return;
            // the key may have been re-pointed at another slot by a concurrent fill of the same expert
            if (who_[(size_t) s] >= 0 && keys_[(size_t) who_[(size_t) s]] == s) keys_[(size_t) who_[(size_t) s]] = -1;
            busy_[(size_t) s] = 1;
            who_[(size_t) s] = (int32_t) x;
            keys_[x] = (int32_t) s;
        }
        std::memcpy(buf_ + (size_t) s * slot_bytes_, src_gu, gu2);
        std::memcpy(buf_ + (size_t) s * slot_bytes_ + gu2, src_down, d_bytes);
        std::lock_guard<std::mutex> lk(m_);
        busy_[(size_t) s] = 0;
        tick_[(size_t) s] = now;
    }

private:
    size_t slot_bytes_ = 0;
    int n_layers_ = 1, n_experts_ = 1, n_slots_ = 0;
    uint8_t* buf_ = nullptr;
    std::vector<int32_t> keys_;      // (layer, expert) -> slot, -1 = not kept
    std::vector<int32_t> who_;       // slot -> (layer, expert) key, -1 = empty
    std::vector<uint8_t> busy_;      // the slot's bytes are being copied in or out
    std::vector<uint64_t> tick_;     // the last completed use, the steal order
    std::mutex m_;
    std::atomic<uint64_t> clock_{0};   // the ticket dispenser
};

}  // namespace strata::poly
