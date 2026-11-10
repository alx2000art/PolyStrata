#!/usr/bin/env python3
# pool_tune.py - the ladder's tuner in the spirit of llama.cpp's moe-cache-tune.py: instead of running the
# server for every pool size, it replays a recorded routing stream through a copy of the engine's own cache
# logic (recent_'s half-life of 128, follow()'s trades, --ram-experts-mib's LRU, the demotion of evictions)
# and predicts, for each pool size, the VRAM share of routed experts - the number whose rise is the whole
# point of a bigger pool: a routed expert out of VRAM is a CPU expert (the done line's hit rate).
#
#   1) run the server once with STRATA_POLY_ROUTE_DUMP=/tmp/route.bin (the prod config, the pool it has);
#   2) stop the server (the dump closes cleanly; SIGKILL truncates it);
#   3) python3 pool_tune.py /tmp/route.bin --vram-experts 615 --pool-mib 6144 12288 --fact-vram 62.4
#
# The replay of the run's own settings has to land near the facts (--fact-vram; the engine's DONE line also
# carries pool_hits= and pool_misses=) before other sizes are believed.  An upload's arrival time depends on
# its lane: a pool hit arrives in --hit-windows windows, a miss (the file: page cache or SSD) in
# --miss-windows - that is the way a bigger pool buys VRAM share: it turns slow misses into fast hits.
# The recommendation is the smallest pool within 0.5 points of the best VRAM share.  Needs numpy.
import argparse
import array
import struct
import sys

import numpy as np


def read_dump(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8 or head[:4] != b"PRTD":
            sys.exit(f"{path}: not a PRTD route dump (run the engine with STRATA_POLY_ROUTE_DUMP)")
        n_layers, n_expert = struct.unpack("<HH", head[4:8])
        size = f.seek(0, 2)
        f.seek(8)
        codes = array.array("H")
        codes.fromfile(f, (size - 8) // 2)
    return n_layers, n_expert, codes


def windows(codes, ne):
    """A window's events end where il does not grow (moe() visits the layers in order)."""
    out, cur, prev = [], [], -1
    for code in codes:
        il = code // ne
        if il < prev and cur:
            out.append(cur)
            cur = []
        cur.append(code)
        prev = il
    if cur:
        out.append(cur)
    return out


class Ladder:
    def __init__(self, n_experts, vram_experts, pool_slots, n_swap, window_tokens, hit_windows,
                 miss_windows, demote, warmup):
        self.n = n_experts
        self.cached = np.zeros(n_experts, dtype=bool)
        self.coming = np.zeros(n_experts, dtype=bool)
        self.recent = np.zeros(n_experts, dtype=np.float32)
        self.vram, self.pool = vram_experts, pool_slots
        self.pool_lru = []
        self.pool_set = set()
        self.swap_busy = []              # (arrive_window, code)
        self.n_swap = n_swap
        self.keep = np.float32(2.0 ** (-window_tokens / 128.0))   # kSwapHalf
        self.hit_windows, self.miss_windows = hit_windows, miss_windows
        self.demote, self.warmup = demote, warmup
        self.window = 0
        self.vram_hits = self.misses = self.pool_hits = self.pool_misses = 0
        self.swaps = self.demotes = 0
        self.events = 0

    def _pool_touch(self, code):
        """The pool's LRU; returns 1 when the code was already in it."""
        if code in self.pool_set:
            self.pool_lru.remove(code)
            self.pool_lru.append(code)
            return True
        if self.pool <= 0:
            return False
        if len(self.pool_lru) >= self.pool:
            self.pool_set.discard(self.pool_lru.pop(0))
        self.pool_set.add(code)
        self.pool_lru.append(code)
        return False

    def _trade(self):
        """follow(): the free uploads take the hottest uncached against the coolest cached (>= 2r + 2)."""
        while len(self.swap_busy) < self.n_swap:
            r_in = np.where(~self.cached & ~self.coming, self.recent, np.float32(-1.0))
            best_in = int(r_in.argmax())
            if r_in[best_in] < 2.0:
                return
            r_out = np.where(self.cached, self.recent, np.float32(1e30))
            best_out = int(r_out.argmin())
            if r_in[best_in] < 2.0 * r_out[best_out] + 2.0:
                return
            if self.demote:              # the evicted expert settles into the pool
                self._pool_touch(best_out)
                self.demotes += 1
            self.cached[best_out] = False
            if self._pool_touch(best_in):
                self.pool_hits += 1
                arrive = self.hit_windows
            else:
                self.pool_misses += 1
                arrive = self.miss_windows
            self.coming[best_in] = True
            self.swap_busy.append((self.window + arrive, best_in))
            self.swaps += 1

    def step_window(self, codes):
        still = []
        for a, code in self.swap_busy:
            if a <= self.window:
                self.coming[code] = False
                if self.cached.sum() < self.vram:
                    self.cached[code] = True
            else:
                still.append((a, code))
        self.swap_busy = still
        self.events += len(codes)
        late = self.events > self.warmup
        cs = np.fromiter(codes, dtype=np.int64, count=len(codes))
        self.recent[cs] += 1.0
        if late:
            hits_window = int(self.cached[cs].sum())
            self.vram_hits += hits_window
            self.misses += len(codes) - hits_window
        self.recent *= self.keep
        if self.cached.sum() < self.vram:    # load(): the cache starts filling, no trade asked
            fill = np.where(~self.cached & ~self.coming)[0]
            if len(fill):
                top = fill[np.argsort(self.recent[fill])[-(self.vram - int(self.cached.sum())):]]
                self.cached[top] = True
        self._trade()
        self.window += 1


def replay(nl, ne, codes, args, pool_slots, demote):
    sim = Ladder(nl * ne, args.vram_experts, pool_slots, args.expert_swaps, args.window_tokens,
                 args.hit_windows, args.miss_windows, demote, args.warmup)
    wins = windows(codes, ne)
    misses = 0
    for w in wins:
        sim.step_window(w)
    total = sim.vram_hits + sim.misses
    share = 100.0 * sim.vram_hits / total if total else 0.0
    ph = 100.0 * sim.pool_hits / max(1, sim.pool_hits + sim.pool_misses)
    return share, ph, sim.swaps, sim.demotes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="PRTD route dump(s), replayed in order as one stream")
    ap.add_argument("--vram-experts", type=int, required=True, help="the cache's experts (the start line's)")
    ap.add_argument("--pool-mib", type=int, nargs="+",
                    default=[2048, 4096, 6144, 8192, 12288, 16384, 24576], help="the sizes to replay")
    ap.add_argument("--slot-mib", type=float, default=10.3, help="one expert slot's MiB")
    ap.add_argument("--expert-swaps", type=int, default=64, help="simultaneous uploads (follow())")
    ap.add_argument("--window-tokens", type=int, default=1, help="tokens a decode window writes (recent decay)")
    ap.add_argument("--hit-windows", type=int, default=1, help="windows an upload takes from a pool hit")
    ap.add_argument("--miss-windows", type=int, default=3, help="windows an upload takes from the file")
    ap.add_argument("--demote", action="store_true", help="also replay the demotion of evictions")
    ap.add_argument("--warmup", type=int, default=1_000_000, help="events before the metrics start counting")
    ap.add_argument("--limit-events", type=int, default=0, help="only the first N events (0: all)")
    ap.add_argument("--fact-vram", type=float, default=None, help="the done line's expert-cache hit %%")
    ap.add_argument("--own-mib", type=int, default=None, help="this run's own pool, for the facts' check")
    args = ap.parse_args()

    nl = ne = 0
    streams = []
    for p in args.dumps:
        a, b, codes = read_dump(p)
        nl, ne = a, b
        streams.append(codes)
        print(f"{p}: {len(codes):,} routed experts, {nl} layers x {ne} experts")
    all_codes = array.array("H")
    for s in streams:
        all_codes.extend(s)
    if args.limit_events:
        all_codes = all_codes[: args.limit_events]

    rows = []
    for mib in args.pool_mib:
        slots = int(mib * 1024 * 1024 / (args.slot_mib * 1024 * 1024))
        for demote in ([False, True] if args.demote else [False]):
            share, ph, swaps, demotes = replay(nl, ne, all_codes, args, slots, demote)
            rows.append((mib, demote, share, ph, swaps, demotes))

    print(f"\n{'pool MiB':>9}  {'demote':>6}  {'VRAM %':>7}  {'pool hit %':>10}  {'swaps':>8}  {'demotes':>8}")
    for mib, demote, share, ph, swaps, demotes in rows:
        print(f"{mib:>9}  {str(demote):>6}  {share:>7.2f}  {ph:>10.2f}  {swaps:>8}  {demotes:>8}")

    if args.own_mib is not None and args.fact_vram is not None:
        own = [r for r in rows if r[0] == args.own_mib and not r[1]]
        if own:
            print(f"\ncheck: the run's own pool {own[0][0]} MiB replays to {own[0][2]:.2f}% VRAM share "
                  f"vs the done line's {args.fact_vram:.1f}% (difference {own[0][2] - args.fact_vram:+.2f})")

    best = max(r[2] for r in rows)
    pick = next(r for r in rows if best - r[2] <= 0.5)
    print(f"recommendation: pool {pick[0]} MiB (demote={pick[1]}), VRAM share {pick[2]:.2f}% "
          f"vs the best {best:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
