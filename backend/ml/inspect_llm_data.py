"""
Step 2 check for LLM-written training data. Run from backend/:  python ml/inspect_llm_data.py [--split train]

Prints (and saves to ml/results/llm_data_report_<split>.txt):
  * rows per label and per kind, questions covered, questions with fewer than 4 variants
  * word-length distribution per label: the length ranges must overlap, otherwise length is a shortcut
  * style tells per label: share of answers with markdown, bullet points, "As a ..." openers, refusals
  * 3 random answers per label to read by hand
  * questions that also appear in validation.csv (must be zero - leakage check)
Read the samples before training. If one label has a visibly different style, regenerate it.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TELLS = {"markdown": r"[*#`]|\n-\s", "as_a_opener": r"^\s*(as an?|sure|certainly|great question)\b",
         "refusal": r"\b(i can'?t|i cannot|as an ai)\b"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    path = ROOT / "data" / "processed" / f"llm_{a.split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run generate_llm_dataset.py first")
    df = pd.read_csv(path)
    df["words"] = df.answer.str.split().str.len()
    out = [f"{len(df)} rows, {df.question.nunique()} questions",
           "\nRows per label:\n" + df.label.value_counts().to_string(),
           "\nRows per kind:\n" + df.kind.value_counts().to_string(),
           f"\nQuestions with < 4 variants: {(df.groupby('question').size() < 4).sum()}",
           "\nWord count per label:\n" + df.groupby("label").words.describe()[["min", "25%", "50%", "75%", "max"]].to_string()]
    tell = pd.DataFrame({k: df.answer.str.contains(p, flags=re.I, regex=True) for k, p in TELLS.items()})
    tell["label"] = df.label
    out.append("\nStyle tells (share of answers):\n" + tell.groupby("label").mean().round(3).to_string())
    other = "validation" if a.split == "train" else "train"
    leak = set(df.question) & set(pd.read_csv(ROOT / "data" / "processed" / f"{other}.csv").question)
    out.append(f"\nQuestions shared with {other}.csv (must be 0): {len(leak)}")
    for lab, g in df.groupby("label"):
        out.append(f"\n--- sample: {lab} ---")
        for _, r in g.sample(min(3, len(g)), random_state=1).iterrows():
            out.append(f"Q: {r.question[:120]}\nA ({r.words} w): {r.answer[:400]}\n")
    text = "\n".join(out)
    print(text)
    (ROOT / "backend" / "ml" / "results" / f"llm_data_report_{a.split}.txt").write_text(text)
