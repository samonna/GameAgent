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

from __future__ import annotations

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
