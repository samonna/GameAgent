"""
Dual-model arbitration chatbot - SINGLE-FILE build.
Everything (logic engine, providers, prompts, game, Gradio UI) is in this
one file so it can be pasted straight into a Hugging Face Space.
Needs requirements.txt with:  gradio>=5.0  and  openai>=1.40
"""
from __future__ import annotations

# ============================================================================
# logic.py
# ============================================================================
"""
Propositional logic engine (no external dependencies).

The LLM formalises each answer as a propositional formula over a shared set of
atoms (p1, p2, ...). This module then does *exact* checks by truth table:

  is_tautology(f)      -> f is true under every assignment
  is_satisfiable(f)    -> some assignment makes f true
  self_consistent(f)   -> alias of is_satisfiable (an answer that is
                          unsatisfiable contradicts itself)
  entails(f, g)        -> (f -> g) is a tautology
  equivalent(f, g)     -> (f <-> g) is a tautology
  consistent(f, g)     -> (f & g) is satisfiable (the two answers can both
                          be true at once)

Formula syntax (what the formaliser prompt asks the model to emit):
  atoms        p1, p2, ... (any identifier)
  negation     ~p   or  !p
  conjunction  p & q   or  p && q
  disjunction  p | q   or  p || q
  implication  p -> q
  biconditional p <-> q
  parentheses  ( ... )
  constants    TRUE, FALSE
Precedence (tightest first): ~, &, |, ->, <->.  '->' is right-associative.
"""


import itertools
import re
from dataclasses import dataclass
from typing import Any

MAX_ATOMS = 16  # 2**16 = 65,536 assignments: fast enough, caps runaway formulas

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<iff><->)|(?P<imp>->)|(?P<and>&&|&)|(?P<or>\|\||\|)"
    r"|(?P<not>~|!)|(?P<lp>\()|(?P<rp>\))|(?P<atom>[A-Za-z_][A-Za-z0-9_]*))"
)


class LogicError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Parsing -> AST (nested tuples)
# ---------------------------------------------------------------------------
def tokenize(src: str) -> list[tuple[str, str]]:
    pos, out = 0, []
    src = src.strip()
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            raise LogicError(f"Unexpected character at {pos}: {src[pos:pos+10]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def take(self, kind: str) -> str:
        if self.peek() != kind:
            raise LogicError(f"Expected {kind}, got {self.peek()}")
        tok = self.toks[self.i][1]
        self.i += 1
        return tok

    def parse(self) -> Any:
        node = self.iff()
        if self.peek() is not None:
            raise LogicError(f"Trailing tokens from position {self.i}")
        return node

    def iff(self) -> Any:
        left = self.imp()
        while self.peek() == "iff":
            self.take("iff")
            left = ("iff", left, self.imp())
        return left

    def imp(self) -> Any:  # right-associative
        left = self.or_()
        if self.peek() == "imp":
            self.take("imp")
            return ("imp", left, self.imp())
        return left

    def or_(self) -> Any:
        left = self.and_()
        while self.peek() == "or":
            self.take("or")
            left = ("or", left, self.and_())
        return left

    def and_(self) -> Any:
        left = self.not_()
        while self.peek() == "and":
            self.take("and")
            left = ("and", left, self.not_())
        return left

    def not_(self) -> Any:
        if self.peek() == "not":
            self.take("not")
            return ("not", self.not_())
        if self.peek() == "lp":
            self.take("lp")
            node = self.iff()
            self.take("rp")
            return node
        name = self.take("atom")
        if name.upper() == "TRUE":
            return ("const", True)
        if name.upper() == "FALSE":
            return ("const", False)
        return ("atom", name)


def parse(src: str) -> Any:
    if not src or not src.strip():
        raise LogicError("Empty formula")
    return _Parser(tokenize(src)).parse()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def atoms_of(node: Any, acc: set[str] | None = None) -> set[str]:
    acc = set() if acc is None else acc
    kind = node[0]
    if kind == "atom":
        acc.add(node[1])
    elif kind in ("not",):
        atoms_of(node[1], acc)
    elif kind in ("and", "or", "imp", "iff"):
        atoms_of(node[1], acc)
        atoms_of(node[2], acc)
    return acc


def evaluate(node: Any, env: dict[str, bool]) -> bool:
    kind = node[0]
    if kind == "const":
        return node[1]
    if kind == "atom":
        return env[node[1]]
    if kind == "not":
        return not evaluate(node[1], env)
    a = evaluate(node[1], env)
    if kind == "and":
        return a and evaluate(node[2], env)
    if kind == "or":
        return a or evaluate(node[2], env)
    b = evaluate(node[2], env)
    if kind == "imp":
        return (not a) or b
    if kind == "iff":
        return a == b
    raise LogicError(f"Unknown node {kind}")


def _assignments(atoms: list[str]):
    for values in itertools.product((False, True), repeat=len(atoms)):
        yield dict(zip(atoms, values))


def _truth_values(node: Any) -> list[bool]:
    atoms = sorted(atoms_of(node))
    if len(atoms) > MAX_ATOMS:
        raise LogicError(f"Too many atoms ({len(atoms)} > {MAX_ATOMS})")
    return [evaluate(node, env) for env in _assignments(atoms)]


# ---------------------------------------------------------------------------
# Public checks (accept formula strings or ASTs)
# ---------------------------------------------------------------------------
def _ast(f: Any) -> Any:
    return parse(f) if isinstance(f, str) else f


def is_tautology(f: Any) -> bool:
    return all(_truth_values(_ast(f)))


def is_satisfiable(f: Any) -> bool:
    return any(_truth_values(_ast(f)))


def is_contradiction(f: Any) -> bool:
    return not is_satisfiable(f)


self_consistent = is_satisfiable


def entails(f: Any, g: Any) -> bool:
    """f |= g  <=>  (f -> g) is a tautology."""
    return is_tautology(("imp", _ast(f), _ast(g)))


def equivalent(f: Any, g: Any) -> bool:
    return is_tautology(("iff", _ast(f), _ast(g)))


def consistent(f: Any, g: Any) -> bool:
    """Both can be true at once  <=>  (f & g) is satisfiable."""
    return is_satisfiable(("and", _ast(f), _ast(g)))


def countermodel(f: Any) -> dict[str, bool] | None:
    """An assignment that makes f FALSE, or None if f is a tautology."""
    node = _ast(f)
    atoms = sorted(atoms_of(node))
    if len(atoms) > MAX_ATOMS:
        raise LogicError(f"Too many atoms ({len(atoms)} > {MAX_ATOMS})")
    for env in _assignments(atoms):
        if not evaluate(node, env):
            return env
    return None


# ---------------------------------------------------------------------------
# Pairwise analysis used by the arbitration game
# ---------------------------------------------------------------------------
@dataclass
class PairAnalysis:
    ok: bool                       # formalisation parsed and was checkable
    a_consistent: bool = True      # A does not contradict itself
    b_consistent: bool = True
    compatible: bool = True        # A & B satisfiable
    a_entails_b: bool = False      # (A -> B) tautology
    b_entails_a: bool = False
    equivalent: bool = False       # (A <-> B) tautology
    error: str | None = None
    atoms: dict[str, str] | None = None
    formula_a: str | None = None
    formula_b: str | None = None

    @property
    def relation(self) -> str:
        if not self.ok:
            return "unknown"
        if not self.a_consistent and not self.b_consistent:
            return "both-self-contradictory"
        if not self.a_consistent:
            return "A-self-contradictory"
        if not self.b_consistent:
            return "B-self-contradictory"
        if self.equivalent:
            return "equivalent"
        if not self.compatible:
            return "contradictory"
        if self.a_entails_b:
            return "A-stronger"   # A says everything B says, and more
        if self.b_entails_a:
            return "B-stronger"
        return "compatible-independent"


def analyse_pair(formula_a: str, formula_b: str,
                 atoms: dict[str, str] | None = None) -> PairAnalysis:
    try:
        fa, fb = parse(formula_a), parse(formula_b)
        a_ok, b_ok = is_satisfiable(fa), is_satisfiable(fb)
        pa = PairAnalysis(ok=True, a_consistent=a_ok, b_consistent=b_ok,
                          atoms=atoms, formula_a=formula_a, formula_b=formula_b)
        if a_ok and b_ok:
            pa.compatible = consistent(fa, fb)
            pa.a_entails_b = entails(fa, fb)
            pa.b_entails_a = entails(fb, fa)
            pa.equivalent = pa.a_entails_b and pa.b_entails_a
        return pa
    except LogicError as e:
        return PairAnalysis(ok=False, error=str(e), atoms=atoms,
                            formula_a=formula_a, formula_b=formula_b)

# ============================================================================
# providers.py
# ============================================================================
"""
Model providers. Everything goes through OpenAI-compatible chat endpoints, so
NVIDIA NIM, Groq, Hugging Face, OpenRouter, Ollama, etc. all work by changing
base_url + api_key + model name via environment variables.

Players in the game:
  A      first answering model
  B      second answering model
  J      judge (defaults to A's provider/model; set JUDGE_* to use a 3rd model)
"""


import json
import os
import re

from openai import OpenAI

NIM_BASE = "https://integrate.api.nvidia.com/v1"

CFG: dict[str, dict] = {
    "A": {
        "base_url": os.getenv("MODEL_A_BASE_URL", NIM_BASE),
        "api_key": os.getenv("MODEL_A_API_KEY") or os.getenv("NVIDIA_API_KEY", ""),
        "model": os.getenv("MODEL_A_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1.5"),
        "label": os.getenv("MODEL_A_LABEL", "Nemotron"),
    },
    "B": {
        "base_url": os.getenv("MODEL_B_BASE_URL", NIM_BASE),
        "api_key": os.getenv("MODEL_B_API_KEY") or os.getenv("NVIDIA_API_KEY", ""),
        "model": os.getenv("MODEL_B_NAME", "openai/gpt-oss-120b"),
        "label": os.getenv("MODEL_B_LABEL", "GPT-OSS"),
    },
}
CFG["J"] = {
    "base_url": os.getenv("JUDGE_BASE_URL", CFG["A"]["base_url"]),
    "api_key": os.getenv("JUDGE_API_KEY") or CFG["A"]["api_key"],
    "model": os.getenv("JUDGE_NAME", CFG["A"]["model"]),
    "label": "Judge",
}

_clients: dict[str, OpenAI] = {}


def _client(key: str) -> OpenAI:
    if key not in _clients:
        c = CFG[key]
        _clients[key] = OpenAI(base_url=c["base_url"], api_key=c["api_key"])
    return _clients[key]


class ProviderError(RuntimeError):
    pass


def chat(key: str, messages: list[dict], temperature: float = 0.2,
         max_tokens: int = 1500) -> str:
    """One chat completion against player `key`. Raises ProviderError."""
    c = CFG[key]
    if not c["api_key"]:
        raise ProviderError(f"{c['label']}: no API key configured")
    try:
        resp = _client(key).chat.completions.create(
            model=c["model"], messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - surface any SDK/network failure
        raise ProviderError(f"{c['label']}: {type(e).__name__}: {e}") from e


def chat_json(key: str, messages: list[dict], temperature: float = 0.0,
              max_tokens: int = 2000) -> dict:
    """chat() + tolerant JSON extraction (strips fences / leading prose)."""
    raw = chat(key, messages, temperature, max_tokens)
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.S)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise ProviderError(f"{CFG[key]['label']}: response was not valid JSON: {raw[:200]!r}")

# ============================================================================
# prompts.py
# ============================================================================
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

# ============================================================================
# game.py
# ============================================================================
import types as _types
logic = _types.SimpleNamespace(analyse_pair=analyse_pair)
prompts = _types.SimpleNamespace(ANSWER_SYSTEM=ANSWER_SYSTEM, FORMALISE_SYSTEM=FORMALISE_SYSTEM,
    CRITIQUE_SYSTEM=CRITIQUE_SYSTEM, REVISE_SYSTEM=REVISE_SYSTEM, JUDGE_SYSTEM=JUDGE_SYSTEM)
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


import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any


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


# ============================================================================
# app.py (Gradio UI)
# ============================================================================


import gradio as gr

# Hugging Face free accounts can now only create ZeroGPU Spaces, and ZeroGPU
# refuses to start unless at least one function is decorated with
# @spaces.GPU. This app never touches a GPU, so register a placeholder that is
# never called; no GPU time is ever allocated. Harmless elsewhere (ImportError).
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return "ok"
except Exception:  # noqa: BLE001 - not on ZeroGPU, or spaces unavailable
    pass



def _md_table(g) -> str:
    if not g.payoff:
        return ""
    cols = [j["order"] for j in g.judge_rounds]
    head = "| candidate | " + " | ".join(f"judge {c}" for c in cols) + " | min (maximin) |\n"
    head += "|---|" + "---|" * (len(cols) + 1) + "\n"
    rows = ""
    for c in ("A", "B"):
        rows += f"| {CFG[c]['label']} ({c}) | " + " | ".join(str(v) for v in g.payoff[c]) \
                + f" | **{g.maximin.get(c)}** |\n"
    return head + rows


def render(g, show_log: bool) -> str:
    if not show_log:
        return g.final
    la, lb = CFG["A"]["label"], CFG["B"]["label"]
    out = [g.final, "", "<details><summary>Game log: drafts, logic checks, payoff matrix</summary>", ""]
    out += [f"**Selection:** `{g.selection}` · **confidence:** {g.confidence} · "
            f"**mode:** {g.mode} · **latency:** {g.latency_s}s", ""]
    out += ["#### Log"] + [f"{i+1}. {line}" for i, line in enumerate(g.log)] + [""]
    if g.logic.get("atoms"):
        out += ["#### Atoms (from formalisation)"]
        out += [f"- `{k}`: {v}" for k, v in g.logic["atoms"].items()] + [""]
    if g.payoff:
        out += ["#### Payoff matrix", "", _md_table(g), ""]
        for j in g.judge_rounds:
            for c in ("A", "B"):
                if j["errors"].get(c):
                    out += [f"- judge {j['order']} errors in {CFG[c]['label']}: " + "; ".join(map(str, j["errors"][c]))]
        out += [""]
    out += [f"#### Draft A — {la}", "", g.answers.get("A", ""), ""]
    out += [f"#### Draft B — {lb}", "", g.answers.get("B", ""), ""]
    for p, other in (("A", "B"), ("B", "A")):
        if p in g.critiques:
            out += [f"#### {CFG[p]['label']}'s critique of {CFG[other]['label']}", "", g.critiques[p], ""]
    for p in ("A", "B"):
        if p in g.revisions:
            r = g.revisions[p]
            out += [f"#### {CFG[p]['label']} move: {r['action']}", "", f"_{r.get('rebuttal','')}_", ""]
            if r["action"] == "REVISE":
                out += [r["answer"], ""]
    if g.merged and g.selection != "merge":
        out += ["#### Judge's merged answer (not selected)", "", g.merged, ""]
    if g.errors:
        out += ["#### Errors"] + [f"- {e}" for e in g.errors] + [""]
    out += ["</details>"]
    return "\n".join(out)


def respond(message, history, mode, show_log, temperature):
    if not message or not message.strip():
        return ""
    g = play(message, history, mode=mode, temperature=temperature)
    return render(g, show_log)


DESCRIPTION = f"""
Each prompt is answered by **{CFG['A']['label']}** (`{CFG['A']['model']}`) and
**{CFG['B']['label']}** (`{CFG['B']['model']}`). The answers are formalised in
propositional logic and checked exactly (self-consistency, compatibility,
entailment, equivalence). Unless they are logically equivalent, the models
critique each other, choose to DEFEND or REVISE, and a judge scores both
candidates twice (order swapped). The payoff matrix is resolved by **maximin**;
a near-tie returns the judge's merged answer.
**Calls per turn:** quick ≈ 5 · full ≈ 9 (free-tier rate limits apply).
"""

demo = gr.ChatInterface(
    fn=respond,
    title="Dual-Model Arbitration Chat",
    description=DESCRIPTION,
    additional_inputs=[
        gr.Radio(["full", "quick"], value="full", label="Mode (full = critique + revision rounds)"),
        gr.Checkbox(value=True, label="Show game log"),
        gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Temperature"),
    ],
    examples=[
        ["What is the derivative of x^x?"],
        ["Is 1 a prime number? Explain briefly."],
        ["Explain precision vs recall with a medical-test example."],
    ],
)

if __name__ == "__main__":
    demo.launch()
