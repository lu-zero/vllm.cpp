# INT8DOT flip gate: anchor-triple hashes (TT build, P150, 2026-09-24)

Anchor prompt: gsq_p0.i32 (65 tokens), 16 greedy tokens, concurrency 4,
seed 0, ignore-eos, one fresh device hold per leg, identical flags.
Oracle: pinned llama.cpp b10451 batch decode teacher-forced on each leg's own
stream (/tmp/lanegate-oracle/oracle-dump, threads=8, ctx=128).

leg            token_stream                       oracle_tf_sha256
default(unset) 220,17,220,16,220,16,...           37db59084984bbbf1881a3e33fb54e3ecaa1a716ebfd7e85f25f1175030534e9
optout(=0)     220,17,220,16,220,16,...           37db59084984bbbf1881a3e33fb54e3ecaa1a716ebfd7e85f25f1175030534e9
lever(=1)      220,17,220,17,220,16,...           38c808444c3ea1b4029d10b60371ea81be47b83aec0b68bf42e19e406afaa456

default == optout: tokens byte-identical AND full-logit dump byte-identical.
lever differs (the flip's declared byte-diff).
