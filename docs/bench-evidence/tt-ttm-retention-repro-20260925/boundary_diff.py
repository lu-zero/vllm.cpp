#!/usr/bin/env python3
"""TT-LEDGER boundary analysis: diff the live DRAM set across request boundaries.

Segments the log at [TT-FWD] route T>1 markers (prefill events) and reports,
for each boundary, the live DRAM set aggregated by (size, tt-metal alloc frame,
ttnn frame, bench frame) plus the NEW classes vs the previous boundary.
"""
import re
import sys
from collections import defaultdict

LEDGER_A = re.compile(r"\[TTLEDGER\] A id=(\d+) type=(\S+) addr=(\d+) size=(\d+)(?: ctx='([^']*)')?(?: bt=(\S+))?")
LEDGER_F = re.compile(r"\[TTLEDGER\] F id=(\d+) type=(\S+)")
FWD = re.compile(r"\[TT-FWD\] route T=(\d+)")

path = sys.argv[1]

def frames_sig(bt):
    fr = bt.split(';')
    tt = next((x for x in fr if x.startswith('libtt_metal.so')), '?')
    tn = next((x for x in fr if x.startswith('_ttnn')), '?')
    bench = next((x for x in fr if x.startswith('vllm-bench')), '?')
    return tt, tn, bench

boundary_live = []   # (label, live_dict)
live = {}
seg_idx = 0
seg_names = ['load']
with open(path, errors='replace') as f:
    for line in f:
        m = FWD.search(line)
        if m and int(m.group(1)) > 1:
            boundary_live.append((f"before-prefill-{len(boundary_live)}", dict(live)))
        m = LEDGER_A.match(line)
        if m:
            bid = int(m.group(1))
            typ = m.group(2)
            size = int(m.group(4))
            bt = m.group(6) or 'NOBT'
            live[bid] = (typ, size, frames_sig(bt), m.group(5) or '')
            continue
        m = LEDGER_F.match(line)
        if m:
            live.pop(int(m.group(1)), None)

boundary_live.append(('end', dict(live)))

def agg_of(live_dict):
    agg = defaultdict(lambda: [0, 0])
    for bid, (typ, size, sig, ctx) in live_dict.items():
        if typ != 'DRAM':
            continue
        agg[(size, sig, ctx)][0] += 1
        agg[(size, sig, ctx)][1] += size
    return agg

prev = None
for name, lv in boundary_live:
    agg = agg_of(lv)
    total = sum(v[1] for v in agg.values())
    print(f"\n=== {name}: liveDRAM={total/2**20:.1f}MiB in {len(lv)} buffers, {len(agg)} classes")
    if prev is not None:
        deltas = []
        for k, v in agg.items():
            pv = prev.get(k)
            d = v[1] - (pv[1] if pv else 0)
            if d != 0:
                deltas.append((d, v[0] - (pv[0] if pv else 0), k))
        deltas.sort(key=lambda x: -abs(x[0]))
        print(f"  net={(sum(d[0] for d in deltas))/2**20:+.1f}MiB; top deltas:")
        for d, cnt, (size, sig, ctx) in deltas[:10]:
            print(f"    {d/2**20:+10.2f}MiB x{cnt:+4d} size={size:>11d} ctx={ctx[:30]!r}")
            print(f"        tt={sig[0]} tn={sig[1]} bench={sig[2]}")
    prev = agg
