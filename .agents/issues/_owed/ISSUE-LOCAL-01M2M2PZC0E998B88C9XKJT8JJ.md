ID: ISSUE-LOCAL-01M2M2PZC0E998B88C9XKJT8JJ
Title: Split tenstorrent_ops.cpp into per-module translation units
Row: -
State: CLOSED
Kind: refactor
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-24
Closed: 2026-09-24

## Problem

src/vt/tenstorrent/tenstorrent_ops.cpp is 10,445 lines / 501 KB and grows with every keep-quant and GDN wave; it carries at least ten distinct modules (residency/slots, weight staging, embedding shadows, PagedKv shadows, keep-quant decode+grouped+int8-dot ~3.2k lines, PA/RAC metadata, GDN prefill/decode ~2.1k lines, trace capture, host-free decode, W4d alloc trace) separated by banner comments. The size makes every fix require reading past unrelated sections and concentrates merge pressure in one file. Mechanical split along the existing banner seams into per-module TUs plus a shared internal header; staged one module per PR, residency first. Gates: full backend suite unchanged (77/77), 0.8B vehicle unchanged, record anchors repaired per PR. No behavior changes, no checker changes.

## Resolution

All six stages landed on main 2026-09-16..2026-09-24: stage 1 internal header + residency (#3146-era), stage 2 keepquant, stage 3 gdn (#3281, 8ee16c33e lineage), stage 4 paged (#3293, 8ee16c33e), stage 5 capture (#3296, 1e53767c4), stage 6 residue (#3297, e2a6d0ae8). tenstorrent_ops.cpp 10,445 lines / 501 KB -> 1,483 lines of registration + core op kernels + glue; the modules live in residency (1,711) / keepquant (2,089) / gdn (2,272) / paged (2,398) / capture (472). The device suite held 92/92 cases / 525,723 assertions byte-for-byte at every stage gate (evidence: docs/bench-evidence/tt-split{4,5,6}-gate-20260924.md); every stage's mutations red and restored. No behavior edit landed in any stage.
