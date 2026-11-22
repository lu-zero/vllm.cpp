import re, sys
log = sys.argv[1]
bases = [0x231d6f00, 0x2a146f00, 0x2d7cef00]
EXT = 0x3FC000
allocs = []  # (seq, uid, addr, size) first-registration only (extra=0x1... actually extra=1 first)
seen = set()
events = []
for line in open(log, errors='replace'):
    if line.startswith('[TT-CMD] seq='):
        m = re.match(r'\[TT-CMD\] seq=(\d+) kind=(\S+) uid=(\d+) addr=0x([0-9a-f]+) size=0x([0-9a-f]+) extra=0x([0-9a-f]+)', line)
        if m:
            s,k,u,a,z,e = int(m[1]),m[2],int(m[3]),int(m[4],16),int(m[5],16),int(m[6],16)
            events.append((s,k,u,a,z,e))
            if k=='alloc' and u not in seen:
                seen.add(u); allocs.append((s,u,a,z))
# dedupe: the buffer impl double-registers (uid then uid+1, same addr); treat pairs
addr_map = {}
for s,u,a,z in allocs:
    addr_map.setdefault(a, []).append((s,u,z))
print("=== neighbors immediately below each victim (extent end == or near victim base) ===")
for b in bases:
    cands = [(s,u,a,z) for a,lst in addr_map.items() for (s,u,z) in lst if a < b and a + z*8 > b - 2*EXT]  # z is per-bank extent
    cands.sort(key=lambda t:-t[0])
    print(f"victim 0x{b:x}: below-neighbors (seq,uid,addr,perbank): {[(s,u,hex(a),hex(z)) for s,u,a,z in cands[:6]]}")
# write overflow: per uid, max (addr+size) seen in bufwrite vs alloc perbank extent
maxw = {}
for s,k,u,a,z,e in events:
    if k.startswith('bufwrite'):
        if u not in maxw or a+z > maxw[u][0]: maxw[u]=(a+z,s)
ext = {u:(a,z) for s,u,a,z in allocs}
print("\n=== bufwrite overflow: txn end beyond declared per-bank extent (excluding base-addr mismatch) ===")
ovf=0
for u,(end,s) in sorted(maxw.items()):
    if u in ext:
        a,z = ext[u]
        if end > a + z:
            ovf+=1
            if ovf<15: print(f"uid={u} alloc addr=0x{a:x} perbank=0x{z:x} max_txn_end=0x{end:x} (+0x{end-a-z:x}) at seq={s}")
print(f"total uids with txn-end beyond alloc extent: {ovf}")
# occupant timeline of each window
print("\n=== occupants of each victim window over time ===")
free_seq = {}
for s,k,u,a,z,e in events:
    if k=='free' and u not in free_seq: free_seq[u]=s
for b in bases:
    occ = [(s,u,a,z) for s,u,a,z in allocs if a <= b < a+z]
    occ.sort()
    print(f"victim 0x{b:x}:")
    for s,u,a,z in occ:
        print(f"  seq={s} uid={u} addr=0x{a:x} perbank=0x{z:x} freed_seq={free_seq.get(u,'LIVE')}")
