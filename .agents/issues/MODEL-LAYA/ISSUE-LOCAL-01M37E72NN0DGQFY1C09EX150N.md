ID: ISSUE-LOCAL-01M37E72NN0DGQFY1C09EX150N
Title: MODEL-LAYA registration breaks the model-registry test build: 48 initializers in a 47-slot array
Row: MODEL-LAYA
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: -

## Problem

#3263 added the Laya architecture to the example-config architecture list in tests/vllm/models/test_model_registry.cpp but left the std::array extent at 47. Every Linux compile job fails with 'too many initializers' at test_model_registry.cpp:780, including main's own CI (run 35871777376 family).

## Resolution

-
