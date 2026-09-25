ID: ISSUE-LOCAL-01M3BXW82K17KT9TYN9H6H99VP
Title: test_qwen4_exp_layer_loop fails on main: by-name second-prompt cache movement is 0
Row: ROAD-V1-D2
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-25
Updated: 2026-09-25
Closed: -

## Problem

At main 636926736, test_qwen4_exp_layer_loop fails: 'by-name second-prompt logit movement: 0' — CHECK(moved > 0) with 0 of 128 indexer-cache words and 0 of 192 paged K/V words moved across two prompts (CI job 107975227373, 2026-09-25). The by-name cache path stopped moving words between prompts — a regression in the KV by-name/multi-cache work landing between 2026-09-12 and now. Blocks every open PR's build-test-cpu.

## Resolution

-
