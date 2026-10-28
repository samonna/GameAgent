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

Two free LLMs answer every prompt; a game-theoretic protocol plus exact
propositional-logic checks decide what the user sees.

```
arbiter/
  providers.py   OpenAI-compatible clients for players A, B and judge J
  prompts.py     prompt templates for every round
  logic.py       propositional parser + truth-table tautology/SAT/entailment checks
  game.py        the game: rounds, payoff matrix, maximin selection, game log
app.py           Gradio chat UI (shows the full game log under each answer)
eval.py          offline accuracy: A vs B vs arbitrated final, on a JSONL set
data/eval_set.jsonl   starter evaluation questions with reference answers
tests/test_logic.py   unit tests for the logic engine (python -m pytest)
```

## The protocol

| Round | What happens | Calls |
|---|---|---|
| 1 Proposal | A and B answer in parallel | 2 |
| 2 Logic | J formalises both answers over shared atoms; `logic.py` checks exactly: each answer satisfiable (no self-contradiction), A∧B satisfiable (compatible), A→B / B→A tautologies (entailment), A↔B tautology (equivalence → consensus, stop) | 1 |
| 3 Critique | *(full)* each player attacks the other's answer | 2 |
| 4 Revision | *(full)* each player sees the critique of its own answer and plays DEFEND or REVISE | 2 |
| 5 Scoring | J scores both candidates on correctness/completeness/consistency twice, with order swapped (removes position bias) | 2 |
| 6 Selection | payoff `U[c][j] = 0.6·correct + 0.25·complete + 0.15·consistent − penalties`; choose by **maximin**; margin inside tie band → judge's merged answer | 0 |

Penalties: −3 for a self-contradictory answer (from round 2), −0.5 for DEFEND
when the judge still finds errors.

## Secrets (Space → Settings → Variables and secrets)

| Name | Required | Default |
|---|---|---|
| `NVIDIA_API_KEY` | yes | — used for A, B and J unless overridden |
| `MODEL_A_NAME` | no | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| `MODEL_B_NAME` | no | `openai/gpt-oss-120b` |
| `MODEL_B_BASE_URL`, `MODEL_B_API_KEY` | no | second provider for B (e.g. Groq) |
| `JUDGE_NAME`, `JUDGE_BASE_URL`, `JUDGE_API_KEY` | no | third model as judge |

Every run is appended to `runs.jsonl` (ephemeral on a free Space).

## Run locally
```bash
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-...
python -m pytest -q tests          # logic engine
python app.py                      # http://127.0.0.1:7860
python eval.py data/eval_set.jsonl # accuracy: A vs B vs final
```

## Honest limits
- The truth-table checks are exact, but the *formalisation* is done by a model;
  a bad formalisation gives a wrong relation. The log shows the atoms and
  formulas so you can audit them.
- Maximin is robust to one biased judge round, not to a judge that is wrong in
  both orders. For math/code, add an executor (run the code, check the number):
  that is the only step that gives a true payoff signal.
- Free-tier rate limits: ~9 calls per turn in full mode.
