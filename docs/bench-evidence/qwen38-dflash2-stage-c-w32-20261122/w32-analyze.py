import re, sys
log = sys.argv[1]
bases = [0x231d6f00, 0x2a146f00, 0x2d7cef00]
EXT = 0x3FC000
win = {b: (b, b+EXT) for b in bases}
uid2alloc = {}   # uid -> (seq, addr, size)
events = []      # (seq, kind, uid, addr, size, extra)
cur = {}         # seq -> event idx, to attach victim alloc
victim_alloc = {}  # base -> (uid, seq)
for line in open(log, errors='replace'):
    if line.startswith('[TT-CMD] seq='):
        m = re.match(r'\[TT-CMD\] seq=(\d+) kind=(\S+) uid=(\d+) addr=0x([0-9a-f]+) size=0x([0-9a-f]+) extra=0x([0-9a-f]+)', line)
        if m:
            s,k,u,a,z,e = int(m[1]),m[2],int(m[3]),int(m[4],16),int(m[5],16),int(m[6],16)
            events.append((s,k,u,a,z,e))
            if k=='alloc': uid2alloc[u]=(s,a,z)
    elif 'W28-STAGE' in line:
        m = re.search(r'addr=0x([0-9a-f]+)', line)
        if m:
            a=int(m[1],16)
            if a in win:
                victim_alloc[a] = ('stage', len(events))
print(f"events: {len(events)}")
uid_alloc_seq = {}
for s,k,u,a,z,e in events:
    if k=='alloc' and u not in uid_alloc_seq: uid_alloc_seq[u]=s
occu = []  # (start_seq, end_seq_or_None, uid, addr, size)
for s,k,u,a,z,e in events:
    if k=='alloc':
        occu.append([s, None, u, a, z])
    elif k=='free':
        for o in reversed(occu):
            if o[2]==u and o[1] is None:
                o[1]=s; break
def owner_at(seq, addr):
    for o in occu:
        if o[1] is not None and o[0] <= seq < o[1] and o[3] <= addr < o[3]+max(o[4],1):
            return o
    return None
hits = {b: [] for b in bases}
for s,k,u,a,z,e in events:
    if not k.startswith('bufwrite'): continue
    for b,(lo,hi) in win.items():
        # a is the txn noc addr (bank-interleaved linear addr); window check on addr and on extra(base)
        if lo <= a < hi or lo <= e < hi:
            hits[b].append((s,k,u,a,z,e))
for b in bases:
    print(f"\n=== victim window 0x{b:x}..0x{b+EXT:x} : {len(hits[b])} bufwrite hits ===")
    vic_alloc = next((o for o in occu if o[1] is None and o[3] <= b < o[3]+max(o[4],1)), None)
    # victim alloc: the alloc owning addr b around the strike; find nearest alloc with addr==b
    vic = [o for o in occu if o[3]==b]
    if vic: print(f"victim-ish allocs at addr: {[(o[0],o[1],o[2],hex(o[4])) for o in vic][:10]}")
    for s,k,u,a,z,e in hits[b][:40]:
        own = owner_at(s, a if lo0win(a,b) else e) if False else owner_at(s, b)
        print(f"seq={s} {k} uid={u} addr=0x{a:x} size=0x{z:x} base=0x{e:x} owner_now={None if not own else (own[2], hex(own[3]), hex(own[4]), own[1])}")
def lo0win(*args): return False
