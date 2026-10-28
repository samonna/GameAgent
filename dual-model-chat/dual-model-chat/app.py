"""Gradio chat UI for the dual-model arbitration game. Run: python app.py"""

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

from arbiter import play
from arbiter.providers import CFG


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
