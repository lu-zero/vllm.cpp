ID: ISSUE-LOCAL-01M3HB51EC31P82SERM924N9WR
Title: The dispatch-cost row: decompose the ~30 ms/op, then the levers
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

Both paths pay ~30 ms per-op dispatch (eager 1,037 ops at ~32 ms; trace-side ~30 ms/command); kernels are 1.27 s/forward -- 25x headroom. The W1 GEMV phase pattern extended to op granularity: per-op phase attribution (host construction, launch submission, completion wait), the per-class breakdown (572 matmul-class ops; the keepquant repair web's 360x420MiB transients), the amortization probe (steady-state collapse = launch latency, amortizable by fusion; flat = per-op host work), and the lever verdict that binds the next rows: fusion ours, descriptor work ours, or launch cost tt-metal's. Spec: tenstorrent-dispatch-cost.md.

## Resolution

-
