---
title: Dual-Model Arbitration Chat
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
---

# Dual-Model Arbitration Chat

Two free LLMs (Nemotron + GPT-OSS on NVIDIA NIM) answer every prompt. The
answers are formalised in propositional logic and checked exactly
(self-consistency, compatibility, entailment, equivalence). Unless they are
logically equivalent, the models critique each other and choose DEFEND or
REVISE; a judge scores both candidates twice with the order swapped, and the
payoff matrix is resolved by maximin (near-tie -> merged answer).

## Secrets (Settings -> Variables and secrets)

| Name | Value |
|---|---|
| `NVIDIA_API_KEY` | your NVIDIA key (required) |
| `JUDGE_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `JUDGE_API_KEY` | your Gemini key |
| `JUDGE_NAME` | `gemini-2.5-flash` |

Optional overrides: `MODEL_A_NAME`, `MODEL_B_NAME`, `MODEL_B_BASE_URL`,
`MODEL_B_API_KEY`.
