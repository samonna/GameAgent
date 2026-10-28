"""
Offline evaluation: does arbitration beat the best single model?

Usage:
  python eval.py data/eval_set.jsonl [--mode full|quick]

Input: JSONL with {"question": "...", "answer": "<reference>"} per line.
Grades A's draft, B's draft, and the game's final answer against the reference
using the judge model as a yes/no grader, then prints accuracy for each.
"""

import argparse
import json
import sys

from arbiter import play
from arbiter.providers import ProviderError, chat

GRADER = ("You grade whether a candidate answer is correct given a reference answer. "
          "Equivalent phrasing counts as correct. Reply with exactly one word: CORRECT or WRONG.")


def grade(question: str, candidate: str, reference: str) -> bool:
    try:
        r = chat("J", [
            {"role": "system", "content": GRADER},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nCANDIDATE:\n{candidate}"},
        ], 0.0, 5)
        return r.strip().upper().startswith("CORRECT")
    except ProviderError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--mode", default="full", choices=["full", "quick"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    tally = {"A": 0, "B": 0, "final": 0}
    selections = {}
    for i, row in enumerate(rows, 1):
        g = play(row["question"], mode=args.mode)
        cands = {"A": g.answers.get("A", ""), "B": g.answers.get("B", ""), "final": g.final}
        marks = {k: grade(row["question"], v, row["answer"]) for k, v in cands.items()}
        for k, ok in marks.items():
            tally[k] += int(ok)
        selections[g.selection] = selections.get(g.selection, 0) + 1
        print(f"[{i}/{len(rows)}] A={marks['A']} B={marks['B']} final={marks['final']} "
              f"sel={g.selection} rel={g.logic.get('relation','-')} {g.latency_s}s", file=sys.stderr)

    n = len(rows)
    print("\n=== accuracy ===")
    for k, v in tally.items():
        print(f"{k:>6}: {v}/{n} = {v/n:.1%}")
    print("selection counts:", selections)


if __name__ == "__main__":
    main()
