"""
Step 1 of the ML pipeline: build the answer-quality training CSV.

The Hugging Face dataset (shimogerald/interview-coach-dataset) has only
question -> reference answer pairs. It has NO human quality labels.

Labelling methodology (simple + deterministic):
  The reference answer is treated as the "Excellent" answer.
  Weaker answers are derived from it by removing content:

    Excellent : the reference answer, capped at ~130 words (a realistic length
                for a spoken/typed interview answer; the raw references are
                often 250+ words which no candidate would write)
    Good      : leading ~55% of that answer (whole sentences when possible)
    Average   : leading ~28% of that answer
    Poor      : either a few random words from the answer wrapped in a vague
                opener, or a vague "I don't really know" style answer

  Random choices are seeded from an MD5 hash of the question, so re-running
  the script always produces exactly the same CSV.

Version 2 changes (see README §6):
  * a 'reference' column is kept so question/reference-aware features can be computed
  * lengths vary inside every class (Excellent 60-130 words, Good 40-65%, Average 18-35%)
  * Poor rotates through six failure types instead of two:
      vague (sampled words), dont_know, other_question (a REAL correct answer to a different
      question - long and fluent, so length alone cannot rate it well), keyword_stuffing,
      restatement (the question echoed back), filler (long, content-free text)
  * an extra Poor 'other_question' row and an extra 'Good' short-correct row are added per question
    so length is not a reliable shortcut; classes stay balanced (Poor 2x is offset by Good/Excellent
    variants - see build_rows)
  Still synthetic. LLM-written answers (generate_llm_dataset.py) are NOT merged here: they stay in
  llm_train.csv and are appended in memory at training time with --augment (see ml/training_data.py).

  train.csv / validation.csv are the project's authoritative dataset. This script refuses to overwrite them
  unless --force is given.

Run:  python ml/prepare_dataset.py        (from the backend/ folder)
"""
import hashlib
import math
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

VAGUE_OPENERS = [
    "I am not very sure but I think it is something like",
    "I don't remember exactly, maybe it is about",
    "It is a concept in programming, I think it has to do with",
    "Not sure, it is probably related to",
]
VAGUE_TEMPLATES = [
    "I don't know.",
    "I do not know the answer to this.",
    "I do not know.",
    "No idea.",
    "I am not sure, I have not studied this.",
    "Sorry, I don't remember this topic.",
    "I think I know it but I cannot explain.",
    "Not sure about this one.",
    "I have no idea, sorry.",
    "I have heard of it but I cannot explain it.",
    "I don't know much about this. It is something we study in computer science.",
    "I think it is a topic used in software. I have heard about it but I cannot explain.",
    "It is related to {kw}. I am not sure about the details.",
    "Maybe it is used in many applications. I would have to look it up.",
]
STOP = set("the a an of to in and or is are for with on how what why can you explain describe your that this it be as by from at".split())


def load_split(name):
    """Load a split from local parquet if present, else download with `datasets`."""
    path = RAW / f"{name}-00000-of-00001.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from datasets import load_dataset  # lazy import (only needed for download)
    ds = load_dataset("shimogerald/interview-coach-dataset")
    df = ds[name].to_pandas()
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def extract_pairs(df):
    pairs = []
    for msgs in df["messages"]:
        q = next((m["content"] for m in msgs if m["role"] == "user"), None)
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if q and a:
            pairs.append((q.strip(), a.strip()))
    return pairs


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


MAX_WORDS = 130


def leading_words(text, target):
    """First ~target words; prefers ending on a sentence boundary."""
    words = text.split()
    target = max(4, min(target, len(words)))
    kept, count = [], 0
    for s in sentences(text):
        n = len(s.split())
        if count + n <= target * 1.15:
            kept.append(s)
            count += n
        else:
            break
    if count >= target * 0.6:          # sentence-aligned version is close enough
        return " ".join(kept)
    return " ".join(words[:target])    # otherwise cut at word level


def leading_fraction(text, frac):
    return leading_words(text, math.ceil(len(text.split()) * frac))


def poor_answer(question, answer, rng, use_sampled):
    if use_sampled:
        words = [w for w in answer.split() if w.lower().strip(".,;:()") not in STOP]
        k = max(4, int(len(words) * 0.08))
        k = min(k, len(words))
        idx = sorted(rng.sample(range(len(words)), k))
        return rng.choice(VAGUE_OPENERS) + " " + " ".join(words[i] for i in idx) + "."
    kws = [w for w in re.findall(r"[A-Za-z]+", question) if w.lower() not in STOP and len(w) > 3]
    kw = rng.choice(kws) if kws else "this topic"
    return rng.choice(VAGUE_TEMPLATES).format(kw=kw)


FILLERS = [
    "This is a very important concept and it is used in many places. There are many aspects to it and it depends on "
    "the situation and the use case. Every engineer should understand it because it comes up often in projects.",
    "In general there are advantages and disadvantages, and it really depends on the requirements. It is widely used "
    "in the industry and it is a key topic, so it is good to know about it for real-world work and for interviews.",
    "Well, it is a broad topic. Different people approach it differently, and the best choice depends on many factors. "
    "I would say it is important to consider all the factors carefully and pick what works best in each case.",
]


def keyword_stuffing(question, answer, rng):
    kws = list(dict.fromkeys(w.strip(".,;:()`*") for w in (question + " " + answer).split()
                             if w.lower().strip(".,;:()`*") not in STOP and len(w) > 3))
    rng.shuffle(kws)
    return " ".join(kws[: rng.randint(8, 16)])


def restatement(question, rng):
    q = re.sub(r"\s+", " ", question).strip().rstrip("?.")
    return rng.choice(["Basically, ", "So the question is ", "Yes, ", ""]) + q[0].lower() + q[1:] + \
        rng.choice([". That is the answer.", ", and that is how it works.", ". It is an important concept."])


def poor_variant(kind, q, base, other_answer, rng):
    if kind == "vague":
        return poor_answer(q, base, rng, use_sampled=True)
    if kind == "dont_know":
        return poor_answer(q, base, rng, use_sampled=False)
    if kind == "other_question":
        return leading_words(other_answer, rng.randint(40, MAX_WORDS))
    if kind == "keyword_stuffing":
        return keyword_stuffing(q, base, rng)
    if kind == "restatement":
        return restatement(q, rng)
    return rng.choice(FILLERS)


POOR_KINDS = ["vague", "dont_know", "keyword_stuffing", "restatement", "filler"]


def build_rows(pairs, split):
    """Per question: 2 Excellent, 2 Good, 2 Average, 2 Poor (one is always 'other_question')."""
    rows = []
    for i, (q, a) in enumerate(pairs):
        seed = int(hashlib.md5(q.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)
        full = leading_words(a, MAX_WORDS)
        other = pairs[(i + 1 + rng.randrange(len(pairs) - 1)) % len(pairs)][1]
        add = lambda ans, label, kind: rows.append((q, a, ans, label, kind))
        add(full, "Excellent", "reference")
        add(leading_words(a, rng.randint(60, 100)), "Excellent", "reference_short")
        add(leading_fraction(full, rng.uniform(0.40, 0.65)), "Good", "truncated")
        add(leading_words(full, rng.randint(15, 30)), "Good", "short_opening")
        add(leading_fraction(full, rng.uniform(0.18, 0.35)), "Average", "truncated")
        # half on-topic, half drifting into another question's answer = partially relevant
        add(leading_words(full, rng.randint(12, 25)) + " " + leading_words(other, rng.randint(15, 35)),
            "Average", "half_relevant")
        add(poor_variant("other_question", q, full, other, rng), "Poor", "other_question")
        kind = POOR_KINDS[seed % len(POOR_KINDS)]
        add(poor_variant(kind, q, full, other, rng), "Poor", kind)
    df = pd.DataFrame(rows, columns=["question", "reference", "answer", "label", "kind"])
    df["split"] = split
    return df


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--force", action="store_true", help="overwrite the existing train.csv / validation.csv")
    if (OUT / "train.csv").exists() and not _ap.parse_args().force:
        raise SystemExit("data/processed/train.csv already exists and is the authoritative dataset - not rebuilding. "
                         "Use --force only if you really intend to regenerate it.")
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation"]:
        pairs = extract_pairs(load_split(split))
        df = build_rows(pairs, split)
        df.drop(columns="split").to_csv(OUT / f"{split}.csv", index=False)
        print(f"{split}: {len(pairs)} questions -> {len(df)} labelled rows")
        print(df["label"].value_counts().to_string(), "\n")
