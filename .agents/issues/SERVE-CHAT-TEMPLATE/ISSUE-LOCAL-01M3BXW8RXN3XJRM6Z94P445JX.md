ID: ISSUE-LOCAL-01M3BXW8RXN3XJRM6Z94P445JX
Title: test_openai_api_server fails on main: chat-template kwargs rejection breaks served requests
Row: SERVE-CHAT-TEMPLATE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-25
Updated: 2026-09-25
Closed: 2026-09-25

## Problem

At main 636926736, test_openai_api_server fails: the server logs 'chat template kwargs from request may not set messages/tools: the renderer supplies it' and rejects the request bodies the test sends (CI job 107975227373, 2026-09-25). A kwargs-tightening change (chat-template or ABI v29 era, #3301-adjacent) tightened request validation without updating the server's own test requests — or vice versa. Blocks every open PR's build-test-cpu.

## Resolution

The kwargs attribution was wrong, and the run that falsified it also found
the real failure. Reproduced at main 53f5af17d on the CPU lane
(build-test-cpu recipe, `./build/tests/test_openai_api_server`): every
chat_template_kwargs case is GREEN — the two 'may not set' log lines the
issue quoted are the EXPECTED output of the passing #1681 case
'chat_template_kwargs cannot forge the conversation'
(tests/vllm/entrypoints/openai/test_api_server.cpp:1108), which sends those
bodies precisely to assert the 400-by-name contract of
src/vllm/entrypoints/chat_template.cpp:256. The filter and its tests landed
together in ea9b7e30e and are mutually consistent; nothing about kwargs was
stale.

The actual red: two /v1/systemone cases — 'systemone dispatch — jev-
compatible answers shape' (:6035) and 'systemone separate — N passes, summed
tokens' (:6130) — noul 0.0 vs 0.95/0.88, entities 0 vs 1. MODEL-KEV (4756d264c)
aligned ParseSystemOneBody's noul labels with kev/api.py to_record
(option_text("no", criteria["false"]) / option_text("yes", criteria["true"])),
which the decision path needs byte-exactly, but q.labels is ALSO the GLiNER
NER path's label set, and a noul question's NER label is its INSTRUCTION.
Fixed 2026-09-25 by splitting the two meanings: NerLabels(q)
(include/vllm/entrypoints/openai/systemone.h) returns the instruction for
noul and q.labels otherwise; ParseSystemOneBody builds all_labels,
BuildSystemOneAnswer scores/entities, and handle_systemone_separate's
per-question NER call all use it, while RenderDecisionOptions keeps kev's
option texts. Red-first: both cases failed pre-fix (0.0 vs 0.95, 0 vs 1);
after: full test_openai_api_server 102/102, plus test_kev 25/25,
test_laya 7/7, test_laya_sequence 11/11, test_capi 73/73,
test_model_registry 24/24 green. Closed by the same commit that carries
this file update.
