ID: ISSUE-LOCAL-01M3BXW8RXN3XJRM6Z94P445JX
Title: test_openai_api_server fails on main: chat-template kwargs rejection breaks served requests
Row: SERVE-CHAT-TEMPLATE
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-25
Updated: 2026-09-25
Closed: -

## Problem

At main 636926736, test_openai_api_server fails: the server logs 'chat template kwargs from request may not set messages/tools: the renderer supplies it' and rejects the request bodies the test sends (CI job 107975227373, 2026-09-25). A kwargs-tightening change (chat-template or ABI v29 era, #3301-adjacent) tightened request validation without updating the server's own test requests — or vice versa. Blocks every open PR's build-test-cpu.

## Resolution

-
