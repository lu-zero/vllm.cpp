#!/usr/bin/env python3
"""W31 launch-address trace analyzer.

Input: the w31-trace-1.stderr.log (interleaved [W28-CANARY] and [TT-TA] lines).

For every ARRIVED-ZEROED canary check, take the victim buffer base address
printed by the adjacent addr= canary line, then scan the whole [TT-TA] stream
for launches whose patched (or retained-CRTA) address equals the victim base.
Print, per victim: every matching TA line with its stream position, so the
EARLIER programs holding the same address (the stale writer candidates) are
visible against the flip position.
"""
import re
import sys

path = sys.argv[1]
ta_addr = re.compile(r"\[TT-TA\] kernel=(\S+) param=(\S+) addr=(\d+)")
ta_empty = re.compile(r"\[TT-TA\] kernel=(\S+) binding_handles=EMPTY.*")
canary_addr = re.compile(r"\[W28-CANARY\] #(\d+) addr=0x([0-9a-f]+) size=(\d+)")
canary_verdict = re.compile(r"\[W28-CANARY\] #(\d+) pre=(\d+) post=\2 .*verdict=(\S+)")

lines = open(path, errors="replace").read().splitlines()
victims = {}  # call# -> base addr
cur_addr = None
for i, ln in enumerate(lines):
    m = canary_addr.search(ln)
    if m:
        cur_addr = (int(m.group(1)), int(m.group(2), 16), int(m.group(3)))
        continue
    v = canary_verdict.search(ln)
    if v and v.group(3) == "ARRIVED-ZEROED":
        n = int(v.group(1))
        if cur_addr and cur_addr[0] == n:
            victims[n] = cur_addr[1]
        cur_addr = None

print(f"victims: " + ", ".join(f"#{n} base=0x{a:x}" for n, a in victims.items()))
for n, base in victims.items():
    print(f"\n=== victim #{n} base=0x{base:x} — TA lines with addr==base ===")
    for i, ln in enumerate(lines):
        m = ta_addr.search(ln)
        if m and int(m.group(3)) == base:
            print(f"  line {i}: kernel={m.group(1)} param={m.group(2)}")

# Unbound kernels: retained CRTA words may hold a stale victim address.
print("\n=== EMPTY-bindings kernels whose CRTA dump contains a victim base ===")
for i, ln in enumerate(lines):
    m = ta_empty.search(ln)
    if not m or "crta_words=" not in ln:
        continue
    words = [w for w in ln.split() if re.fullmatch(r"[0-9a-f]{8}", w)]
    vals = [int(w, 16) for w in words]
    for n, base in victims.items():
        if base in vals:
            print(f"  victim #{n} (0x{base:x}) in kernel={m.group(1)} at line {i}: {ln}")

# Legacy per-core runtime args ([TT-RTA]): scan every printed arg for a value
# inside any victim window.
rta = re.compile(r"\[TT-RTA\] kernel=(\S+) core=(\S+) args:(.*)")
print("\n=== TT-RTA lines with an arg inside a victim window ===")
for i, ln in enumerate(lines):
    m = rta.search(ln)
    if not m:
        continue
    vals = [int(w, 16) for w in m.group(3).split() if re.fullmatch(r"[0-9a-f]+", w)]
    for n, base in victims.items():
        # Per-bank extent = canary size / 8 (pages round-robin, bank = idx % 8).
        if any(base <= v < base + 33423360 // 8 for v in vals):
            print(f"  victim #{n} line {i}: {ln.strip()}")
            break
