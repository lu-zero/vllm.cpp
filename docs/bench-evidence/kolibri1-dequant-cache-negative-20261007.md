# Kolibri-1 CPU: dequant-cache lever — NEGATIVE result (MODEL-TEXT-kolibri-1)

Date: 2026-10-07. Host: aarch64, 128 cores, 255 GB RAM, Linux. Worktree
`/tmp/vllm-kolibri-neon2`, branch `row/kolibri-neon2`, base 323f81ca6.
All runs `VLLM_CPP_CPU_THREADS=8`; both binaries built from this tree
(Release, `-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TENSTORRENT=OFF`), GPU work inside
the session per policy.

VERDICT: the lever was REVERTED. The cache is mechanically sound and roughly
2x on gate-forward time, but with an active cache the row's token gate FAILS
(7 hard flips; worst topk logit diff 7.15 against a 2.5 band), so it cannot
land under the bit-exactness contract. Follow-up owned by
ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2. Raw logs: `/tmp/base-w3.log`,
`/tmp/after-w3b.log`, `/tmp/after-w3c.log`, `/tmp/w3-verify2.log`,
`/tmp/w3-trip.log`, `/tmp/inc-base*.log`, `/tmp/inc-after8*.log`,
`/tmp/noevict.log` (session /tmp; not archived).

## Lever chosen, and why

The two recorded levers after the NEON decode were (a) a dequant cache and
(b) linear_gemm NEON. The profile ranks dequant_fp8_block (248.28 s of 513.43
s at 150 forwards, 48%) and linear_gemm (222.6 s, co-dominant) nearly equal,
but they are not equally reachable: a GEMM re-vectorization reorders the
bf16 accumulation chain and cannot meet the "identical chain + fingerprints"
bar by construction, while a cache stores exactly what the kernel emits and
is bit-faithful by construction. The cache was implemented.

## Implementation (as reverted)

`kolibri1_dequant_cache.h`: process-wide LRU keyed on raw weight identity
(packed-bytes pointer, scale-grid pointer, n, k, block_n, block_k); entries
hold the decoded bf16 block via `DBuf::ReleaseShared` (the pool-block lease)
plus the tensor view; byte budget `VT_KOLIBRI1_DEQUANT_CACHE_MB` (default
8192, LRU eviction, absolute bound — an entry larger than the budget
degenerates to the uncached path); hit/miss/decode-call/eviction counters;
`VT_KOLIBRI1_PROFILE` Report extended with the counters. `LinearBT`'s fp8 arm
consulted the cache; the miss callback was the exact uncached body.

## Red-first evidence (unit gate)

- RED (base binary, no cache seam): the new
  `test_kolibri1_dequant` case does not exist; compile-error red on the base
  tree, then a behavioral mutation red — hits disabled in a scratch copy →
  `CHECK_EQ(decode_calls, dec0+1)` and `CHECK_EQ(hits, 1)` FAIL (2 failed
  assertions), restored byte-for-byte after.
- GREEN: 4/4 cases, 20/20 assertions — a second identical `GetOrDequant`
  never re-invokes the decode callback and returns bit-identical raw-uint16
  bytes (the very block the kernel filled); a sub-entry budget forces
  eviction and the next call re-decodes bit-identically.

## Baseline (before), gate shape

`test_kolibri1_w3`, 8 threads, profiled, `/usr/bin/time -v`, 150 forwards:

| axis | baseline |
|---|---|
| dequant_fp8_block @150f | 254.40 s |
| linear_gemm @150f | 224.61 s |
| total_forward @150f | 524.08 s |
| whole-gate wall | 9:18.71 (max RSS 76.5 GB) |
| gate | 900/900, chain 141/145 (4 near-tie, 0 hard), worst topk 2.18646 |

## Production-shape bench (before / after)

The shipped W3 harness re-prefills the whole chain per step, which is not the
production decode shape; a scratch bench (NOT committed) prefills the
128-token prompt once and then runs 63 incremental t=1 decode steps through
one persistent KV topology, greedy, deterministic 128-token prompt
(`(i*7919+13) % 128000`). Same host, 8 threads, both binaries from this tree.
Cache OFF (uncached) vs cache ON (8 GiB default):

| axis | cache OFF | cache ON | multiple |
|---|---|---|---|
| decode wall (63 tok), best of 2 | 49.39 s | 23.52 s | **2.10x** |
| decode tok/s | 1.276 | 2.679 | |
| prefill wall | 12.67 s | 12.74 s | 1.00x |
| greedy chain | 109726 | 113260 | **DIFFERENT — see below** |

Budget sweep (same bench, cache ON): 4 GiB → hits=0 (per-decode-step working
set ≈ 4.3 GB: attention projections ≈ 3.4 GB + ~18 experts/layer); 8 GiB →
52% hits, decode 23.5 s; 16 GiB → decode 18.2 s; 32 GiB → 17.1 s; 200 GiB
(no eviction) → 16.6 s. The standing set grows with the hit rate (an expert
kept resident is reused by later steps), so the win and the budget both grow
together; the working set is the bound that matters, not the decode cost.

## Correctness: the lever FAILS the row gate with an active cache

| configuration | W3 result |
|---|---|
| cache OFF (hits=0, any budget) | 900/900, chain 141/145, worst topk 2.18646 — identical to landed |
| cache ON, hits=0 (4 GiB) | identical to landed |
| cache ON, 8 GiB (W3 working set ≈ 10 GB → hits=0) | identical to landed |
| cache ON, 16 GiB (hits>0: 10026 hits / 5729 misses at 10 forwards) | **FAIL: 7 hard flips, worst topk 7.15, worst sum diff 478753** |

With hits active the forward gets FASTER as designed — dequant_fp8_block
1.65 s vs 16.5 s per 10 forwards, total_forward 18.26 s vs ~35 s (~1.9x) —
but the tokens change materially (first decode-step top logit 5.86 vs 7.08,
different token; the full re-prefill bench shows the same at its first
incremental step).

## Diagnosis (as far as this session reached)

1. The cache is byte-faithful at the seam: a per-hit `memcmp` of the cached
   block against a fresh decode matched on every hit in the bench build, and
   the unit gate pins it exhaustively at kernel level.
2. Despite byte-identical weights, hits change outputs. A diagnostics W3 run
   aborted at the FIRST hit whose cached block no longer matched a fresh
   decode: `CACHE MISMATCH n=512 k=2560 first=0 ndiff=1099739` — the cached
   block (still owned by a live entry) contained another tensor's data. The
   decode itself is deterministic (two fresh decodes compared equal).
3. A decode-target tripwire fired `POOL DOUBLE-HAND-OUT`: `DequantFp8Block`'s
   fresh DBuf was handed a pointer still owned by a live cache entry — i.e.
   blocks leased via `DBuf::ReleaseShared` while cached are re-handed to new
   allocations. A raw `DevicePool::Put/Get` free-list tripwire was
   inconclusive (its static set spans multiple pool instances; 2.45 M prints
   even with the cache disabled), so the exact double-ownership path —
   a missed lease inside the pool, a second `Put`, or a borrow/lend
   interaction specific to the cache's lease pattern — is NOT localized.
4. Process-state sensitivity is real and pre-existing in the scratch
   incremental harness: an uncached pass that runs AFTER a cached pass in the
   same process produced a different chain (17385) than the uncached binary
   (109726), while the landed W3 harness (fresh topology per forward) is
   stable. Whether that sensitivity is only the scratch harness or reaches
   the incremental-decode path is open and belongs to the follow-up issue.

Repro recipe (any tree containing the reverted header — see the issue for the
diff): wire `LinearBT`'s fp8 arm through `kolibri1_dequant_cache::GetOrDequant`,
build `test_kolibri1_w3`, run `VT_KOLIBRI1_DEQUANT_CACHE_MB=16384
VLLM_CPP_CPU_THREADS=8 test_kolibri1_w3` → gate red as above; add the per-hit
memcmp to catch the first aliased block.

## Gates (post-revert, recorded negative)

- Tree reverted to base 323f81ca6 for all product, test and cmake paths
  (`git diff` empty); nothing of the lever lands.
- `scripts/check-agent-record.py`: run before push (see commit).
- The negative result itself is the deliverable of this session; the next
  attempt needs the pool-lease defect localized first.
