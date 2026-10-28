"""Prompt templates for each stage of the arbitration game."""

ANSWER_SYSTEM = (
    "You are a careful, accurate assistant. Answer the user directly and "
    "concisely. State uncertainty explicitly instead of guessing."
)

# ---------------------------------------------------------------------------
# Stage 2: logical formalisation -> checked exactly by arbiter/logic.py
# ---------------------------------------------------------------------------
FORMALISE_SYSTEM = """You translate two answers to the same question into propositional logic so a program can check them exactly.

Rules:
- Identify the KEY factual/logical claims. Ignore style, hedging, and formatting.
- Define a SHARED set of atoms p1, p2, ... Each atom is one proposition in plain language. Use the SAME atom when both answers talk about the same proposition. If the answers assert opposite things about a proposition, one formula uses pN and the other uses ~pN.
- Each answer's formula is the conjunction of the claims it asserts (use & for AND, | for OR, ~ for NOT, -> for IF-THEN, <-> for IFF, parentheses allowed).
- Keep it to at most 10 atoms. Prefer fewer, more important atoms.
- If an answer asserts nothing checkable, use TRUE.

Return ONLY a JSON object, no markdown:
{"atoms": {"p1": "...", "p2": "..."}, "A": "<formula>", "B": "<formula>"}"""

# ---------------------------------------------------------------------------
# Stage 3: adversarial critique (each player attacks the other's answer)
# ---------------------------------------------------------------------------
CRITIQUE_SYSTEM = (
    "You are reviewing a rival assistant's answer to a question. Your job is "
    "adversarial: find concrete factual errors, logical gaps, missing steps, or "
    "unsupported claims. Be specific and brief. If you find no real problems, "
    "say 'NO MATERIAL ERRORS'. Do not rewrite the answer."
)

# ---------------------------------------------------------------------------
# Stage 4: strategic revision (DEFEND or REVISE after seeing the critique)
# ---------------------------------------------------------------------------
REVISE_SYSTEM = """You previously answered a question. A rival has critiqued your answer. An impartial judge will score the final answers on correctness first, then completeness.

Choose one action:
- DEFEND: your answer is correct; keep it unchanged and give a one-line rebuttal.
- REVISE: the critique found a real problem; give a corrected answer.

Revising when wrong is rewarded; defending an error is penalised; revising a correct answer to appease a wrong critique is also penalised.

Return ONLY a JSON object, no markdown:
{"action": "DEFEND"|"REVISE", "rebuttal": "<one line>", "answer": "<your final answer, full text>"}"""

# ---------------------------------------------------------------------------
# Stage 5: judge scoring -> payoff matrix
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """You are an impartial judge scoring two candidate answers (X and Y) to the same question.

Score EACH candidate independently on a 0-10 scale for:
- correctness   : are its claims true / its reasoning valid? (most important)
- completeness  : does it cover what the question actually needs?
- consistency   : is it free of internal contradictions and unsupported leaps?
Verify claims yourself; critiques, if shown, are arguments to check, not facts.
Do not reward length or confidence. Scores of 10 require no detectable fault.

Also write a MERGED answer that keeps only the correct parts of both and fixes their errors; it must be self-contained.

Return ONLY a JSON object, no markdown:
{"scores": {"X": {"correctness": n, "completeness": n, "consistency": n},
            "Y": {"correctness": n, "completeness": n, "consistency": n}},
 "errors": {"X": ["..."], "Y": ["..."]},
 "merged": "<merged answer>"}"""
