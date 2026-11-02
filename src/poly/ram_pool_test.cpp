// src/poly/ram_pool_test.cpp - RamPool against the host's expectations: a hit returns the bytes of the last
// fill, a miss leaves the destinations alone, eviction goes by the least recently used, and copies running
// on many threads never tear an expert (every slot's bytes carry its (layer, expert) key in every word).
//
// usage: poly_ram_pool_test   (exit 0: everything as expected)
#include "ram_pool.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using namespace strata::poly;

constexpr size_t kGu2 = 3 << 20, kDown = 1 << 20;   // a slot: 4 MiB
constexpr uint32_t kPat(uint64_t il, uint64_t e, uint64_t w) {
    return (uint32_t) (il * 1000003u + e * 97u + w * 7919u);
}

int failures = 0;

void want(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

// paints the expert's pattern over the word views, or checks it against them
void paint(uint32_t* gu, uint32_t* down, int il, int e, bool check) {
    for (size_t w = 0; w < kGu2 / 4; ++w) {
        if (check) {
            if (gu[w] != kPat(il, e, w)) { want(false, "a hit's gate|up bytes are another expert's"); return; }
        } else gu[w] = kPat(il, e, w);
    }
    for (size_t w = 0; w < kDown / 4; ++w) {
        if (check) {
            if (down[w] != kPat(il, e, w + (1u << 28))) { want(false, "a hit's down bytes are another expert's"); return; }
        } else down[w] = kPat(il, e, w + (1u << 28));
    }
}

}  // namespace

int main() {
    std::vector<uint8_t> gu(kGu2), down(kDown), junk(kGu2, 0xAB), junkd(kDown, 0xAB);
    auto* gw = (uint32_t*) gu.data();
    auto* dw = (uint32_t*) down.data();
    auto* jw = (uint32_t*) junk.data();
    auto* jdw = (uint32_t*) junkd.data();

    // a miss leaves the destinations alone
    {
        RamPool p(4 << 20, kGu2 + kDown, 2, 256);
        want(p.slots() == 1, "4 MiB of 4 MiB slots = one slot");
        want(!p.copy_out(0, 5, junk.data(), junkd.data(), kGu2, kDown), "an empty pool misses");
        want(junk[0] == 0xAB && junkd[0] == 0xAB, "a miss leaves the destinations alone");
    }
    // a hit returns the last fill; with one slot a second expert's fill evicts it
    {
        RamPool p(4 << 20, kGu2 + kDown, 2, 256);
        paint(gw, dw, 0, 5, false);
        p.fill(0, 5, gw, dw, kGu2, kDown);
        p.fill(0, 5, gw, dw, kGu2, kDown);   // refilling its own key is benign
        std::memset(gu.data(), 0xEE, kGu2);
        want(p.copy_out(0, 5, gu.data(), down.data(), kGu2, kDown), "a filled expert hits");
        want(gw[0] == kPat(0, 5, 0) && dw[0] == kPat(0, 5, 1u << 28), "a hit's bytes are the fill's");
        paint(gw, dw, 1, 7, false);
        p.fill(1, 7, gw, dw, kGu2, kDown);   // steals the one slot
        std::memset(gu.data(), 0xEE, kGu2);
        want(!p.copy_out(0, 5, gu.data(), down.data(), kGu2, kDown), "the evicted expert misses");
        want(p.copy_out(1, 7, gu.data(), down.data(), kGu2, kDown), "the stealer hits");
        p.fill(0, 5, gw, dw, kGu2, kDown);   // 0,5 steals back (1,7 held the newer tick)
        std::memset(gu.data(), 0xEE, kGu2);
        want(!p.copy_out(1, 7, gu.data(), down.data(), kGu2, kDown), "LRU evicted the older tick");
        want(p.copy_out(0, 5, gu.data(), down.data(), kGu2, kDown), "the newest fill hits");
    }
    // the down half rides behind gate|up in the slot, per call's sizes
    {
        RamPool p(4 << 20, kGu2 + kDown, 1, 4);
        paint(gw, dw, 0, 3, false);
        p.fill(0, 3, gw, dw, kGu2, kDown);
        std::memset(gu.data(), 0xEE, kGu2);
        std::memset(down.data(), 0xEE, kDown);
        want(p.copy_out(0, 3, gu.data(), down.data(), kGu2, kDown), "hit after fill");
        paint(gw, dw, 0, 3, true);
    }
    // many threads copy and fill different experts; every hit's bytes are their expert's
    {
        RamPool p(64 << 20, kGu2 + kDown, 4, 256);   // 16 slots, 8 threads, 1024 keys: heavy stealing
        std::atomic<int> bad{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < 8; ++t)
            ts.emplace_back([&, t] {
                std::vector<uint8_t> g(kGu2), d(kDown);
                auto* g32 = (uint32_t*) g.data();
                auto* d32 = (uint32_t*) d.data();
                for (int q = 0; q < 40; ++q) {
                    const int il = (t * 40 + q) % 4, e = (t * 17 + q * 31) % 256;
                    if (p.copy_out(il, e, g.data(), d.data(), kGu2, kDown)) {
                        for (size_t w = 0; w < kGu2 / 4; w += 997)
                            if (g32[w] != kPat(il, e, w)) { ++bad; break; }
                        for (size_t w = 0; w < kDown / 4; w += 997)
                            if (d32[w] != kPat(il, e, w + (1u << 28))) { ++bad; break; }
                    }
                    paint(g32, d32, il, e, false);
                    p.fill(il, e, g.data(), d.data(), kGu2, kDown);
                }
            });
        for (std::thread& t : ts) t.join();
        want(bad.load() == 0, "no thread read another expert's bytes");
    }
    if (failures) std::printf("%d failed\n", failures);
    else std::printf("all passed\n");
    return failures ? 1 : 0;
}
