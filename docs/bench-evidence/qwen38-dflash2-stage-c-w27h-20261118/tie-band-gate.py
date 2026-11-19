#!/usr/bin/env python3
# The W27 tie-band gate (ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H): the
# drafter-ON token-identity gate redefined per the ratified near-tie precedent
# (the B2b kolibri near-tie-gap gate; kNearTieMnats = 500 in
# tests/parity/test_commandr_paged_engine.cpp — the same threshold here).
#
# The W26 acceptance trace proved the accept walk and commit FAITHFUL and the
# 3-token shift an M=9-vs-M=1 BFP8 near-tie flip (top-2 gaps 0.03-0.5 nats,
# inside the 0.75% max-rel band), so a byte-identity gate cannot pass over the
# lever. This gate asserts, per run:
#
#   (a) per-step acceptance telemetry within the recorded band — the number of
#       committed tokens per verify block parsed from the [ACCEPT] trace lines
#       (the acceptance-trace instrument feeds the gate);
#   (b) the emitted stream equals the EAGER (drafter-OFF) stream EXCEPT at
#       positions the eager logits' top-2 gap marks near-tie: a mismatch at
#       position p is permissible only when the eager gap at p, at p-1, or at
#       p+1 (a one-position phase slip of a periodic cycle) is below
#       --near-tie-mnats (default 500 = the ratified band);
#   (c) zero NaN/Inf anywhere — no NaN/Inf marker in the run's stderr, every
#       emitted token id inside [0, 2**31) and non-zero mid-stream (the
#       degenerate [220, 0, 0, ...] corruption signature), hard fail.
#
# Exit 0 = PASS, 1 = FAIL, 2 = usage. The corrupted-weights red proof runs
# this same gate on the planted-corruption leg and MUST exit 1.
import argparse
import json
import re
import sys


def load_ids(path):
    with open(path) as f:
        ids = json.load(f)
    assert isinstance(ids, list) and len(ids) == 1, f"{path}: expect one row"
    return ids[0]


def parse_gaps(path):
    # [EAGER-TOP2] step=N tok=T gap_mnats=G lines -> list indexed by step.
    gaps, toks = [], []
    pat = re.compile(r"\[EAGER-TOP2\] step=(\d+) tok=(-?\d+) gap_mnats=([0-9.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                gaps.append(float(m.group(3)))
                toks.append(int(m.group(2)))
    return gaps, toks


def parse_accept(stderr_path):
    # [ACCEPT] step=S ... emitted=[ t0 t1 ... ] -> committed tokens per verify
    # block; acceptance per step = emitted_count - 1 (k drafts, one bonus row).
    per_step = {}
    pat = re.compile(
        r"\[ACCEPT\] step=(\d+) req=(\d+).*emitted=\[([^\]]*)\]")
    with open(stderr_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                step, req = int(m.group(1)), int(m.group(2))
                emitted = [int(t) for t in m.group(3).split()]
                per_step.setdefault(step, []).append(len(emitted) - 1)
    return [min(v) for _, v in sorted(per_step.items())]  # per step, first req


def scan_fatal(stderr_path):
    pat = re.compile(r"\b(NaN|nan|Inf|inf)\b")
    hits = []
    with open(stderr_path) as f:
        for i, line in enumerate(f, 1):
            if pat.search(line):
                hits.append((i, line.strip()[:160]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager-ids", required=True)
    ap.add_argument("--eager-gaps", required=True,
                    help="stderr log of the VT_EAGER_TOP2_TRACE OFF leg")
    ap.add_argument("--on-ids", required=True)
    ap.add_argument("--on-stderr", required=True)
    ap.add_argument("--accept-trace", default=None,
                    help="stderr carrying the [ACCEPT] lines; default: "
                         "--on-stderr")
    ap.add_argument("--near-tie-mnats", type=float, default=500.0,
                    help="the ratified near-tie band (kNearTieMnats)")
    ap.add_argument("--accept-band", default=None,
                    help="min,max of the mean accepted-drafts per verify "
                         "step, recorded from the healthy legs")
    ap.add_argument("--corrupted", action="store_true",
                    help="the planted-corruption red leg: expect FAIL")
    ap.add_argument("--verdict", default=None, help="write the verdict JSON")
    args = ap.parse_args()

    eager = load_ids(args.eager_ids)
    on = load_ids(args.on_ids)
    gaps, gap_toks = parse_gaps(args.eager_gaps)
    findings = []
    ok = True

    # (c) NaN/Inf — the corruption signature. Checked FIRST and hard.
    fatal = scan_fatal(args.on_stderr)
    bad_ids = [(i, t) for i, t in enumerate(on)
               if t <= 0 or t >= (1 << 31)]
    if fatal:
        ok = False
        findings.append(f"(c) FAIL: {len(fatal)} NaN/Inf markers in stderr, "
                        f"first: {fatal[0][1]}")
    else:
        findings.append("(c) zero NaN/Inf markers: OK")
    if bad_ids:
        ok = False
        findings.append(f"(c) FAIL: {len(bad_ids)} degenerate token ids, "
                        f"first {bad_ids[0]}")

    # (b) the tie-band stream comparison.
    if len(gaps) < len(eager):
        ok = False
        findings.append(f"(b) FAIL: {len(gaps)} eager gap records for "
                        f"{len(eager)} eager positions — the reference leg "
                        f"is short (raise VT_EAGER_TOP2_TRACE)")
    near_tie = [g < args.near_tie_mnats for g in gaps]
    mism = [i for i in range(min(len(eager), len(on)))
            if eager[i] != on[i]]
    illegal = []
    for i in mism:
        cover = [near_tie[j] for j in (i - 1, i, i + 1)
                 if 0 <= j < len(near_tie)]
        if not any(cover):
            illegal.append(i)
    if illegal:
        ok = False
        findings.append(
            f"(b) FAIL: {len(illegal)} mismatches OUTSIDE the tie band "
            f"(threshold {args.near_tie_mnats} mnats), first at {illegal[0]}: "
            f"eager={eager[illegal[0]]} on={on[illegal[0]]} "
            f"gap={gaps[illegal[0]]:.1f}")
    else:
        findings.append(f"(b) {len(mism)} mismatches, all inside the tie "
                        f"band (threshold {args.near_tie_mnats} mnats): OK")

    # (a) the acceptance band.
    if args.accept_band:
        lo, hi = (float(x) for x in args.accept_band.split(","))
        acc = parse_accept(args.accept_trace or args.on_stderr)
        if not acc:
            ok = False
            findings.append("(a) FAIL: no [ACCEPT] telemetry found")
        else:
            mean = sum(acc) / len(acc)
            if not (lo <= mean <= hi):
                ok = False
                findings.append(f"(a) FAIL: mean accepted drafts/step "
                                f"{mean:.2f} outside the recorded band "
                                f"[{lo},{hi}] ({len(acc)} steps)")
            else:
                findings.append(f"(a) mean accepted drafts/step {mean:.2f} "
                                f"in [{lo},{hi}] over {len(acc)} steps: OK")

    for f in findings:
        print(f"  {f}")
    verdict = "PASS" if ok else "FAIL"
    if args.corrupted and ok:
        # The red leg must fail: a green gate on a corrupted run means the
        # gate cannot separate corrupted from healthy — NEEDS_DECISION.
        print("RED-PROOF ERROR: the corrupted leg PASSED — the gate does not "
              "distinguish corrupted from healthy at these thresholds")
        verdict = "RED-PROOF-INVALID"
        ok = False
    print(f"TIE-BAND GATE: {verdict} "
          f"(near_tie_mnats={args.near_tie_mnats}, "
          f"mismatches={len(mism)}, in_band="
          f"{len(mism) - len(illegal)})")
    if args.verdict:
        with open(args.verdict, "w") as f:
            json.dump({"verdict": verdict,
                       "near_tie_mnats": args.near_tie_mnats,
                       "mismatches": mism,
                       "findings": findings}, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
