import re, sys
# W33 analyzer: scan [TT-DEVECHO] device-arg readbacks against each victim's
# per-bank window [base, base+0x3FC000). A device-resident stale base shows a
# hit at a 256B-aligned (or exact-base) address; unaligned in-window words are
# kernel-text instruction encodings (the same false-positive class W31 named:
# an arbitrary 32-bit RISC-V word can numerically land in a DRAM window).
log = sys.argv[1]
bases = [0x231d6f00, 0x2a146f00, 0x2d7cef00]
EXT = 0x3FC000
lines = 0
raw = {b: [] for b in bases}
aligned = {b: [] for b in bases}
for line in open(log, errors='replace'):
    if not line.startswith('[TT-DEVECHO]'):
        continue
    m = re.match(r'\[TT-DEVECHO\] lseq=(\d+) rid=(\d+) chip=(\d+) core=(\d+)-(\d+) base=0x([0-9a-f]+) size=(\d+) words=(\d+)(.*)', line)
    if not m:
        continue
    lines += 1
    lseq, rid, slot = int(m[1]), int(m[2]), int(m[6], 16)
    core = f'{m[4]}-{m[5]}'
    for tok in m.group(9).split():
        i, w = tok.split(':')
        if not w:
            continue
        v = int(w, 16)
        for b in bases:
            if b <= v < b + EXT:
                raw[b].append((lseq, core, hex(slot), i, hex(v)))
                if (v & 0xFF) == 0:
                    aligned[b].append((lseq, core, hex(slot), i, hex(v)))
print(f"devecho lines parsed: {lines}")
for b in bases:
    print(f"\n=== victim window [0x{b:x}, 0x{b + EXT:x}) ===")
    print(f"raw in-window words: {len(raw[b])}")
    for h in raw[b][:10]:
        print(f"  raw lseq={h[0]} core={h[1]} slot={h[2]} word[{h[3]}]={h[4]}")
    exact = [h for h in raw[b] if int(h[4], 16) == b]
    print(f"256B-aligned in-window words: {len(aligned[b])}; exact-base words: {len(exact)}")
    for h in aligned[b][:10]:
        print(f"  ALIGNED lseq={h[0]} core={h[1]} slot={h[2]} word[{h[3]}]={h[4]}")
    for h in exact:
        print(f"  EXACT-BASE lseq={h[0]} core={h[1]} slot={h[2]} word[{h[3]}]={h[4]}")
