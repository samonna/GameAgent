from arbiter import logic as L


def test_tautologies():
    assert L.is_tautology("p | ~p")
    assert L.is_tautology("(p -> q) <-> (~p | q)")
    assert L.is_tautology("((p -> q) & (q -> r)) -> (p -> r)")
    assert not L.is_tautology("p -> q")


def test_satisfiability():
    assert L.is_satisfiable("p & q")
    assert not L.is_satisfiable("p & ~p")
    assert L.is_contradiction("(p -> q) & p & ~q")


def test_entailment_and_equivalence():
    assert L.entails("p & q", "p")
    assert not L.entails("p", "p & q")
    assert L.equivalent("~(p & q)", "~p | ~q")   # De Morgan
    assert L.consistent("p", "q")
    assert not L.consistent("p & q", "~q")


def test_pair_analysis_relations():
    assert L.analyse_pair("p1 & p2", "p1 & p2").relation == "equivalent"
    assert L.analyse_pair("p1 & p2", "p1").relation == "A-stronger"
    assert L.analyse_pair("p1", "p1 & p2").relation == "B-stronger"
    assert L.analyse_pair("p1 & p2", "p1 & ~p2").relation == "contradictory"
    assert L.analyse_pair("p1 & ~p1", "p2").relation == "A-self-contradictory"
    assert L.analyse_pair("p1", "p2").relation == "compatible-independent"
    assert L.analyse_pair("p1 &&& p2", "p1").relation == "unknown"


def test_syntax_variants():
    assert L.is_tautology("!p || p")
    assert L.is_tautology("TRUE")
    assert L.is_contradiction("FALSE")
    assert L.countermodel("p -> q") == {"p": True, "q": False}
