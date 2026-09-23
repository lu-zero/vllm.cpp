ID: ISSUE-LOCAL-01M37W0HP6JTNJP55S74159T02
Title: Flip VT_TT_KEEPQUANT_INT8DOT to default-on: the default-off denominator is invalidated and both arms are in-band
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

The W4b lever stayed default-off (#3031) because its e2e lane failed the 500-mnat band against llama.cpp b10451 INCREMENTAL greedy. The 2026-09-22 GSQ investigation proved that denominator invalid (the pin's own batch decode disagrees with its incremental greedy at identical causal positions). Re-adjudicated against the batch decode, both arms are in-band (default 26.6, INT8DOT 245.3 mnats, 16/16 each) and the lever cuts TPOT 1253->642 ms (2x) and trace demand -33.8%. The default flip is the row's pending decision; this spec is its gate plan.

## Resolution

-
