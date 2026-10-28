"""
The arbitration game.

Players: A and B (answering models). Referee: J (judge model).
Nature: the propositional-logic engine (exact, no model involved).

Rounds
  1. Proposal      A and B answer independently, in parallel.
  2. Logic         J formalises both answers over shared atoms; the logic
                   engine checks self-consistency, compatibility, entailment,
                   equivalence (tautology checks by truth table).
                   Equivalent + both consistent  -> consensus, game ends.
  3. Critique      (full mode) each player attacks the other's answer.
  4. Revision      (full mode) each player sees the critique of its own answer
                   and chooses DEFEND or REVISE. This is the strategic move.
  5. Scoring       J scores both final candidates twice, with the presentation
                   order swapped, to remove position bias. Scores + logic
                   penalties form the payoff matrix U[candidate][judge-round].
  6. Selection     maximin: choose the candidate whose WORST judge score is
                   highest (robust against a single biased judge round).
                   If the maximin margin is inside the tie band, neither pure
                   strategy dominates -> return the judge's MERGED answer.

Everything is recorded in GameResult.log for display and in runs.jsonl for
offline evaluation (see eval.py).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from . import logic, prompts
from .providers import CFG, ProviderError, chat, chat_json

# Payoff weights (sum to 1) and penalties
W_CORRECT, W_COMPLETE, W_CONSIST = 0.60, 0.25, 0.15
PENALTY_SELF_CONTRADICTION = 3.0   # subtracted from a self-inconsistent answer
PENALTY_DEFEND_AFTER_ERRORS = 0.5  # defended, yet judge still lists errors
TIE_MARGIN = 0.75                  # maximin margin below this -> merge


@dataclass
class GameResult:
    question: str
    mode: str
    answers: dict[str, str] = field(default_factory=dict)      # round 1
    logic: dict[str, Any] = field(default_factory=dict)        # round 2
    critiques: dict[str, str] = field(default_factory=dict)    # round 3 (key = author)
    revisions: dict[str, dict] = field(default_factory=dict)   # round 4
    candidates: dict[str, str] = field(default_factory=dict)   # final text per player
    judge_rounds: list[dict] = field(default_factory=list)     # round 5 raw (mapped to A/B)
    payoff: dict[str, list[float]] = field(default_factory=dict)  # U[c] = [r1, r2]
    maximin: dict[str, float] = field(default_factory=dict)
    selection: str = ""            # "A" | "B" | "merge" | "consensus" | "fallback-A" ...
    margin: float = 0.0
    confidence: float = 0.0
    final: str = ""
    merged: str | None = None
    log: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latency_s: float = 0.0

    def note(self, line: str) -> None:
        self.log.append(line)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _history_messages(history: list[dict], question: str) -> list[dict]:
    msgs = [{"role": "system", "content": prompts.ANSWER_SYSTEM}]
    for m in history or []:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            msgs.append({"role": m["role"],
                         "content": m["content"].split("\n<details>")[0].strip()})
    msgs.append({"role": "user", "content": question})
    return msgs


def _parallel(fa, fb):
    with ThreadPoolExecutor(max_workers=2) as ex:
        a, b = ex.submit(fa), ex.submit(fb)
        return a.result(), b.result()


def _safe(fn, errors: list[str], label: str):
    try:
        return fn()
    except ProviderError as e:
        errors.append(f"{label}: {e}")
        return None


def _score(s: dict) -> float:
    try:
        return (W_CORRECT * float(s["correctness"]) + W_COMPLETE * float(s["completeness"])
                + W_CONSIST * float(s["consistency"]))
    except (KeyError, TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------
def play(question: str, history: list[dict] | None = None, mode: str = "full",
         temperature: float = 0.2) -> GameResult:
    t0 = time.time()
    g = GameResult(question=question, mode=mode)
    la, lb = CFG["A"]["label"], CFG["B"]["label"]
    msgs = _history_messages(history or [], question)

    # ---- Round 1: proposal -------------------------------------------------
    ans_a, ans_b = _parallel(
        lambda: _safe(lambda: chat("A", msgs, temperature), g.errors, la),
        lambda: _safe(lambda: chat("B", msgs, temperature), g.errors, lb),
    )
    if ans_a is None and ans_b is None:
        g.selection, g.final = "error", "Both models failed:\n\n" + "\n".join(g.errors)
        return _finish(g, t0)
    if ans_a is None or ans_b is None:
        who = "B" if ans_a is None else "A"
        g.answers = {who: ans_a or ans_b}
        g.selection, g.final, g.confidence = f"fallback-{who}", (ans_a or ans_b), 0.4
        g.note(f"Round 1: only {CFG[who]['label']} answered; returning it unjudged.")
        return _finish(g, t0)
    g.answers = {"A": ans_a, "B": ans_b}
    g.candidates = dict(g.answers)
    g.note(f"Round 1 (proposal): {la} and {lb} answered in parallel.")

    # ---- Round 2: logic / tautology checks ---------------------------------
    pa = None
    form = _safe(lambda: chat_json("J", [
        {"role": "system", "content": prompts.FORMALISE_SYSTEM},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nANSWER A:\n{ans_a}\n\nANSWER B:\n{ans_b}"},
    ]), g.errors, "formalise")
    if form and "A" in form and "B" in form:
        pa = logic.analyse_pair(str(form["A"]), str(form["B"]), form.get("atoms"))
        g.logic = asdict(pa) | {"relation": pa.relation}
        if pa.ok:
            g.note(f"Round 2 (logic): A = `{pa.formula_a}`, B = `{pa.formula_b}` -> relation **{pa.relation}**"
                   f" (A self-consistent: {pa.a_consistent}, B self-consistent: {pa.b_consistent}, "
                   f"A&B satisfiable: {pa.compatible}, A->B tautology: {pa.a_entails_b}, "
                   f"B->A tautology: {pa.b_entails_a}).")
        else:
            g.note(f"Round 2 (logic): formalisation could not be parsed ({pa.error}); skipping logic checks.")
    else:
        g.note("Round 2 (logic): formaliser failed; skipping logic checks.")

    if pa and pa.ok and pa.equivalent and pa.a_consistent and pa.b_consistent:
        # Both players already chose the same strategy: a trivial equilibrium.
        shorter = "A" if len(ans_a) <= len(ans_b) else "B"
        g.selection, g.final, g.confidence = "consensus", g.answers[shorter], 0.9
        g.note(f"A <-> B is a tautology: the answers are logically equivalent. "
               f"Consensus reached; returning the shorter one ({CFG[shorter]['label']}).")
        return _finish(g, t0)

    # ---- Rounds 3 + 4: critique and strategic revision (full mode) ---------
    if mode == "full":
        def critique(author: str, target_answer: str) -> str | None:
            return _safe(lambda: chat(author, [
                {"role": "system", "content": prompts.CRITIQUE_SYSTEM},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nRIVAL'S ANSWER:\n{target_answer}"},
            ], 0.1, 600), g.errors, f"critique by {CFG[author]['label']}")

        crit_by_a, crit_by_b = _parallel(lambda: critique("A", ans_b), lambda: critique("B", ans_a))
        g.critiques = {k: v for k, v in (("A", crit_by_a), ("B", crit_by_b)) if v}
        g.note(f"Round 3 (critique): {la} attacked {lb}'s answer and vice versa.")

        def revise(player: str, own_answer: str, critique_of_it: str | None) -> dict | None:
            if not critique_of_it:
                return None
            r = _safe(lambda: chat_json(player, [
                {"role": "system", "content": prompts.REVISE_SYSTEM},
                {"role": "user", "content": (f"QUESTION:\n{question}\n\nYOUR ANSWER:\n{own_answer}\n\n"
                                             f"CRITIQUE OF YOUR ANSWER:\n{critique_of_it}")},
            ], 0.1, 1800), g.errors, f"revision by {CFG[player]['label']}")
            if r and str(r.get("action", "")).upper() in ("DEFEND", "REVISE") and r.get("answer"):
                r["action"] = str(r["action"]).upper()
                return r
            return None

        rev_a, rev_b = _parallel(lambda: revise("A", ans_a, crit_by_b),
                                 lambda: revise("B", ans_b, crit_by_a))
        for p, r in (("A", rev_a), ("B", rev_b)):
            if r:
                g.revisions[p] = r
                if r["action"] == "REVISE":
                    g.candidates[p] = str(r["answer"])
        moves = ", ".join(f"{CFG[p]['label']}: {g.revisions[p]['action']}" if p in g.revisions
                          else f"{CFG[p]['label']}: (no move)" for p in ("A", "B"))
        g.note(f"Round 4 (revision): {moves}.")

    # ---- Round 5: scoring (two judge rounds, order swapped) ----------------
    def judge(order: tuple[str, str]) -> dict | None:
        x, y = order
        body = (f"QUESTION:\n{question}\n\n=== CANDIDATE X ===\n{g.candidates[x]}\n\n"
                f"=== CANDIDATE Y ===\n{g.candidates[y]}\n")
        if g.critiques:
            cx = g.critiques.get(y)   # critique OF x was written BY y
            cy = g.critiques.get(x)
            if cx:
                body += f"\n=== CRITIQUE OF X (by rival) ===\n{cx}\n"
            if cy:
                body += f"\n=== CRITIQUE OF Y (by rival) ===\n{cy}\n"
        r = _safe(lambda: chat_json("J", [
            {"role": "system", "content": prompts.JUDGE_SYSTEM},
            {"role": "user", "content": body},
        ], 0.0, 2200), g.errors, f"judge order {x}{y}")
        if not r or "scores" not in r:
            return None
        # map X/Y back to A/B
        sc, er = r.get("scores", {}), r.get("errors", {})
        return {"order": f"{x}{y}",
                "scores": {x: sc.get("X", {}), y: sc.get("Y", {})},
                "errors": {x: er.get("X", []), y: er.get("Y", [])},
                "merged": r.get("merged")}

    j1, j2 = _parallel(lambda: judge(("A", "B")), lambda: judge(("B", "A")))
    g.judge_rounds = [j for j in (j1, j2) if j]
    if not g.judge_rounds:
        g.selection, g.final, g.confidence = "fallback-A", g.candidates["A"], 0.3
        g.note("Round 5 (scoring): judge failed in both orders; returning A unscored.")
        return _finish(g, t0)
    g.merged = next((j["merged"] for j in g.judge_rounds if j.get("merged")), None)

    # ---- Payoff matrix U[candidate][judge-round] ----------------------------
    for c in ("A", "B"):
        row = []
        for j in g.judge_rounds:
            u = _score(j["scores"].get(c, {}))
            if pa and pa.ok and not getattr(pa, f"{c.lower()}_consistent"):
                u -= PENALTY_SELF_CONTRADICTION
            if g.revisions.get(c, {}).get("action") == "DEFEND" and j["errors"].get(c):
                u -= PENALTY_DEFEND_AFTER_ERRORS
            row.append(round(u, 2))
        g.payoff[c] = row
    g.note("Round 5 (scoring): payoff matrix (rows = candidates, cols = judge rounds "
           + ", ".join(j["order"] for j in g.judge_rounds) + "): "
           + "; ".join(f"{CFG[c]['label']} = {g.payoff[c]}" for c in ("A", "B")) + ".")

    # ---- Selection: maximin, tie band -> merge --------------------------------
    g.maximin = {c: min(v) for c, v in g.payoff.items()}
    ranked = sorted(g.maximin.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_v), (_, second_v) = ranked[0], ranked[1]
    g.margin = round(best_v - second_v, 2)
    if g.margin < TIE_MARGIN and g.merged:
        g.selection, g.final = "merge", g.merged
        g.note(f"Selection: maximin margin {g.margin} < {TIE_MARGIN} -> no pure strategy dominates; "
               f"returning the judge's merged answer.")
    else:
        g.selection, g.final = best, g.candidates[best]
        g.note(f"Selection: maximin picks {CFG[best]['label']} "
               f"(worst-case score {best_v} vs {second_v}, margin {g.margin}).")
    # confidence: margin, judge-round agreement, and whether logic found contradictions
    agree = len({max(("A", "B"), key=lambda c: g.payoff[c][i]) for i in range(len(g.judge_rounds))}) == 1
    g.confidence = round(min(1.0, 0.5 + 0.1 * min(g.margin, 3) + (0.15 if agree else 0)
                             + (0.05 if pa and pa.ok and pa.compatible else 0)), 2)
    return _finish(g, t0)


def _finish(g: GameResult, t0: float) -> GameResult:
    g.latency_s = round(time.time() - t0, 2)
    try:
        with open("runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")
    except OSError:
        pass
    return g
