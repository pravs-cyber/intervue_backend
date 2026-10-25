"""
Reference answers for the built-in TECHNICAL bank questions - generated, then reviewed by a person, then applied.
Nothing reaches the database without status 'approved'. Run from backend/:

  python ml/fill_bank_references.py generate [--max-tokens 800] [--delay 2]   needs AI_API_KEY
  python ml/fill_bank_references.py generate --regenerate-flagged [--dry-run]
        also regenerates rows that are still 'pending' AND have warning flags; the old text is kept in
        previous_reference. Approved, rejected and unflagged rows are never touched. --dry-run lists them only.
  python ml/fill_bank_references.py generate --question "URL" [--dry-run]
        regenerates exactly ONE pending row, chosen by a case-insensitive substring of its question (flagged or
        not). Stops with the list of matches if the text matches zero or several rows; refuses approved/rejected.
        -> data/sources/bank_references.csv (TRACKED): one row per technical bank question, status 'pending',
           plus automatic warning flags. Rows already 'approved' or 'rejected' are never regenerated;
           'pending' rows with an empty reference are retried.
  python ml/fill_bank_references.py report        flags and status counts (no API calls)
  (edit the CSV: set status to approved / rejected, fix wording if needed, add reviewer_note)
  python ml/fill_bank_references.py apply         writes APPROVED references into the database
init_db() also applies approved references automatically, so a fresh database picks them up.

HR bank questions are excluded: the ML evaluation is validated for technical answers only.
Flags are hints for the reviewer, not decisions. A bad reference makes good answers look off-topic.
"""
import argparse
import datetime
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from database.question_bank import TECHNICAL  # noqa: E402
from services.constants import LEVELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "sources" / "bank_references.csv"
COLS = ["skill", "difficulty", "question", "reference", "status", "flags", "generated_by", "generated_at", "reviewer_note",
        "previous_reference"]
STATUSES = {"pending", "approved", "rejected"}
STOP = set("the a an of and or to in on for with is are be what how why when which explain describe give one "
           "difference between your you its it this that use used using do does".split())


def flags_for(question: str, reference: str) -> str:
    r = (reference or "").strip()
    if not r:
        return "empty"
    f, words = [], r.split()
    if not 50 <= len(words) <= 160:
        f.append(f"length {len(words)} words (asked 80-120)")
    if re.search(r"(^|\n)\s*([-*•]|\d+\.)\s|\*\*|^#", r):
        f.append("markdown/list formatting")
    letters = [c for c in r if c.isalpha()]
    if letters and sum(not c.isascii() for c in letters) / len(letters) > 0.2:
        f.append("non-English text")
    if re.search(r"\b(as an ai|i cannot|i can't|i'm sorry|language model)\b", r, re.I):
        f.append("refusal/AI self-reference")
    qk = {w for w in re.findall(r"[a-z][a-z0-9+#-]+", question.lower()) if w not in STOP and len(w) > 2}
    hit = {w for w in qk if w in r.lower()}
    if qk and len(hit) / len(qk) < 0.5:
        f.append(f"mentions only {len(hit)}/{len(qk)} question key terms")
    if r.count("\n\n") >= 2:
        f.append("several paragraphs")
    return "; ".join(f)


def load():
    if CSV.exists():
        df = pd.read_csv(CSV, keep_default_na=False, dtype=str)
        for c in COLS:                                     # files written before a column existed
            if c not in df.columns:
                df[c] = ""
        bad = set(df.status) - STATUSES
        assert not bad, f"unknown status values in {CSV.name}: {bad}"
        return df
    return pd.DataFrame(columns=COLS)


def bank_rows():
    return [{"skill": s, "difficulty": LEVELS[d], "question": q} for s, d, q in TECHNICAL]


def save(df):
    tmp = CSV.with_suffix(".tmp")
    df[COLS].to_csv(tmp, index=False)
    tmp.replace(CSV)


def to_regenerate(df, regenerate_flagged=False):
    """Pending rows with no reference, plus (optionally) pending rows that carry warning flags."""
    pending = df.status == "pending"
    empty = df.reference.str.strip() == ""
    flagged = df["flags"].str.strip() != ""
    return df.index[pending & (empty | (flagged & regenerate_flagged))]


def select_question(df, text):
    """Index of the ONE row whose question contains `text` (case-insensitive); raises otherwise."""
    hits = df.index[df.question.str.lower().str.contains(text.lower(), regex=False)]
    if len(hits) != 1:
        listing = "\n".join(f"  [{df.at[i, 'status']}] {df.at[i, 'question']}" for i in hits) or "  (none)"
        raise SystemExit(f"--question {text!r} matches {len(hits)} rows; it must match exactly one:\n{listing}")
    i = hits[0]
    if df.at[i, "status"] != "pending":
        raise SystemExit(f"--question {text!r} selects a row with status '{df.at[i, 'status']}'; only pending rows "
                         f"are regenerated. Set it back to pending in the CSV first if you really mean it.")
    return i


def generate(gen=None, max_tokens=800, delay=2.0, sleep=time.sleep, log=print, regenerate_flagged=False,
             dry_run=False, question=None):
    from ml.generate_llm_dataset import RateLimited, call_with_retry
    from services import llm_service
    gen = gen or (lambda system, user, max_tokens: llm_service._chat(system, user, max_tokens=max_tokens))
    df = load()
    known = {r.question: i for i, r in df.iterrows()}
    for row in bank_rows():                                    # add any new bank questions as empty pending rows
        if row["question"] not in known:
            df = pd.concat([df, pd.DataFrame([{**row, **{c: "" for c in COLS if c not in row}, "status": "pending"}])],
                           ignore_index=True)
    todo = [select_question(df, question)] if question else to_regenerate(df, regenerate_flagged)
    if dry_run:
        for i in todo:
            why = ("selected by --question" if question else
                   "empty" if not df.at[i, "reference"].strip() else f"flagged: {df.at[i, 'flags']}")
            log(f"  would regenerate ({why}): {df.at[i, 'question'][:70]}")
        log(f"{len(todo)} row(s) would be regenerated (dry run, nothing written)")
        return 0, 0
    done = failed = 0
    for n, i in enumerate(todo, 1):
        q, skill = df.at[i, "question"], df.at[i, "skill"]
        try:
            ref = call_with_retry(gen, llm_service.REFERENCE_SYSTEM, f"Skill: {skill}\nQuestion: {q}", sleep=sleep,
                                  log=log, max_tokens=max_tokens)
        except RateLimited as e:
            log(f"stopping: rate limited ({e}); rerun later - progress is saved")
            break
        except Exception as e:
            log(f"  skip ({e}): {q[:60]}")
            failed += 1
            continue
        if df.at[i, "reference"].strip():                 # regenerating: keep the old text for the reviewer
            df.at[i, "previous_reference"] = df.at[i, "reference"]
        df.loc[i, ["reference", "flags", "generated_by", "generated_at"]] = [
            ref.strip(), flags_for(q, ref), config.AI_MODEL, datetime.datetime.now().isoformat(timespec="seconds")]
        done += 1
        save(df)
        log(f"{n}/{len(todo)} ok{'  FLAGS: ' + df.at[i, 'flags'] if df.at[i, 'flags'] else ''}  {q[:60]}")
        if delay:
            sleep(delay)
    save(df)
    return done, failed


def report():
    df = load()
    print(f"{len(df)} technical bank questions in {CSV.name}: " + str(df.status.value_counts().to_dict()))
    flagged = df[df["flags"].str.strip() != ""]
    print(f"rows with warning flags: {len(flagged)}")
    for _, r in flagged.iterrows():
        print(f"  [{r.status}] {r.question[:70]}\n      -> {r['flags']}")


def apply(conn=None):
    """Write approved references to the bank questions in the DB; returns number of rows updated."""
    df = load()
    ok = df[(df.status == "approved") & (df.reference.str.strip() != "")]
    if conn is None:
        from database.db import get_conn, init_db
        init_db()
        with get_conn() as c:
            return apply(c)
    n = 0
    for _, r in ok.iterrows():
        n += conn.execute("UPDATE questions SET reference=? WHERE source='bank' AND type='technical' AND question=?",
                          (r.reference.strip(), r.question)).rowcount
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "report", "apply"])
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--regenerate-flagged", action="store_true",
                    help="also regenerate pending rows that have warning flags (old text kept in previous_reference)")
    ap.add_argument("--dry-run", action="store_true", help="with generate: list the rows that would be generated")
    ap.add_argument("--question", type=str, default=None,
                    help="regenerate only the ONE pending row whose question contains this text (case-insensitive)")
    a = ap.parse_args()
    if a.cmd == "generate":
        if not config.AI_API_KEY and not a.dry_run:
            sys.exit("Set AI_API_KEY in .env first (or write references by hand in the CSV).")
        if a.question and a.regenerate_flagged:
            sys.exit("Use either --question or --regenerate-flagged, not both.")
        d, f = generate(max_tokens=a.max_tokens, delay=a.delay, regenerate_flagged=a.regenerate_flagged,
                        dry_run=a.dry_run, question=a.question)
        if a.dry_run:
            sys.exit(0)
        print(f"\ngenerated {d}, failed {f}. Next: review {CSV} (set status approved/rejected), then: apply")
    elif a.cmd == "report":
        report()
    else:
        print(f"applied {apply()} approved references to the database")
