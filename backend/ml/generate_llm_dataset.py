"""
OPTIONAL step 3b: add LLM-written answers that truncation cannot produce. Needs AI_API_KEY in .env.
Run from backend/:  python ml/generate_llm_dataset.py --split train --limit 300
                    python ml/generate_llm_dataset.py --split validation
Then re-run prepare_dataset.py - it merges data/processed/llm_<split>.csv automatically.
Repair an existing file (no API calls): python ml/generate_llm_dataset.py --split train --repair
Reruns only generate missing (question, kind) pairs; every row's reference is copied from its source question.
Rate limits (HTTP 429): Retry-After is honoured, otherwise waits 5/15/30/60 s; a question that stays limited is
skipped and resumed on the next run; 3 rate-limited questions in a row stop the run (quota used up).
    python ml/generate_llm_dataset.py --split train --limit 300 --dry-run     # what is missing, no API calls
    python ml/generate_llm_dataset.py --split train --limit 300 --delay 2

To avoid the model learning the PROMPT instead of the content, every answer comes from ONE prompt with
the same student persona; only the one-line instruction changes. Inspect a sample before training.
Generated once, offline, saved to CSV: the LLM never grades live answers.
"""
import argparse
import datetime
import email.utils
import hashlib
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from services import llm_service  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PERSONA = ("You are a final-year computer science student in a job interview. Answer in plain spoken English, "
           "no headings, no bullet points, no markdown. Write {length} words. {instruction}")
VARIANTS = [  # (kind, label, instruction)
    ("paraphrase_correct", "Excellent", "Give a correct and complete answer in your own words. Do not copy phrasing from any textbook."),
    ("concise_correct", "Good", "Give a correct answer covering only the single most important point."),
    ("partial", "Average", "Get the basic idea right but leave out or blur the most important detail."),
    ("confident_wrong", "Poor", "Sound confident and use the right technical terms, but make the central claim factually wrong."),
]
LENGTHS = ["25-40", "50-80", "90-120"]  # varied inside every class so length is not a shortcut


KINDS = {"llm_" + k: label for k, label, _ in VARIANTS}
COLUMNS = ["question", "reference", "answer", "label", "kind"]


def _blank(x):
    return x is None or (isinstance(x, float) and pd.isna(x)) or not str(x).strip()


def reference_lookup(src: pd.DataFrame) -> dict:
    """question -> reference, taken from the ORIGINAL (non-LLM) rows of the split, never from generated rows."""
    orig = src[~src.get("kind", pd.Series("", index=src.index)).astype(str).str.startswith("llm_")]
    orig = orig[~orig.reference.map(_blank)]
    return dict(orig.drop_duplicates("question")[["question", "reference"]].itertuples(index=False))


def repair(df: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """Copy each row's reference from its source question, restore label from kind, drop duplicate
    (question, kind) rows keeping the first one with a non-empty answer. Valid rows are left unchanged."""
    df = df.reindex(columns=COLUMNS).copy()
    df["reference"] = df.question.map(refs).where(df.question.map(refs).notna(), df.reference)
    known = df.kind.isin(KINDS)
    df.loc[known, "label"] = df.loc[known, "kind"].map(KINDS)
    df["_empty"] = df.answer.map(_blank)
    df = df.sort_values("_empty", kind="stable").drop_duplicates(["question", "kind"], keep="first")
    return df.sort_index().drop(columns="_empty")


def write(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp")
    df.reindex(columns=COLUMNS).to_csv(tmp, index=False)
    tmp.replace(path)                      # atomic: a crash never leaves a half-written CSV


BACKOFF = [5, 15, 30, 60]          # seconds between 429 retries when there is no Retry-After header
MAX_WAIT = 120                     # a Retry-After longer than this usually means a daily cap: give up on the pair
STOP_AFTER_SKIPPED_QUESTIONS = 3   # consecutive questions skipped for rate limits -> stop the run (quota exhausted)


class RateLimited(Exception):
    """All retries for one API call hit HTTP 429."""


def _retry_after(exc):
    """Seconds from a Retry-After header (seconds or HTTP date), or None."""
    resp = getattr(exc, "response", None)
    value = resp.headers.get("retry-after") if resp is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
            return max(0.0, (when - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


def _is_429(exc):
    return isinstance(exc, httpx.HTTPStatusError) and exc.response is not None and exc.response.status_code == 429


DEFAULT_MAX_TOKENS = 800          # reasoning models (gpt-oss) need room for hidden reasoning + the answer


def call_with_retry(chat, system, user, sleep=time.sleep, backoff=BACKOFF, max_wait=MAX_WAIT, log=print,
                    max_tokens=DEFAULT_MAX_TOKENS):
    """Call the LLM; on HTTP 429 wait (Retry-After, else backoff) and retry. Raises RateLimited when exhausted.
    Any other error is raised unchanged."""
    for attempt in range(len(backoff) + 1):
        try:
            return chat(system, user, max_tokens=max_tokens)
        except Exception as e:
            if not _is_429(e):
                raise
            if attempt == len(backoff):
                raise RateLimited("still rate-limited after retries") from e
            wait = _retry_after(e)
            wait = backoff[attempt] if wait is None else wait
            if wait > max_wait:
                raise RateLimited(f"Retry-After {wait:.0f}s exceeds {max_wait}s (daily limit?)") from e
            log(f"    429 rate limit - waiting {wait:.0f}s (retry {attempt + 1}/{len(backoff)})")
            sleep(wait)


def plan(split, limit=None, root: Path = ROOT):
    """Load source + existing output; return (refs, questions, kept_rows, done_pairs, out_path)."""
    src = pd.read_csv(root / "data" / "processed" / f"{split}.csv")
    refs = reference_lookup(src)
    qs = list(refs)[:limit] if limit else list(refs)        # --limit = number of QUESTIONS
    out_path = root / "data" / "processed" / f"llm_{split}.csv"
    have = repair(pd.read_csv(out_path), refs) if out_path.exists() else pd.DataFrame(columns=COLUMNS)
    done = {(q, k) for q, k, a in zip(have.question, have.kind, have.answer) if not _blank(a)}
    have = have[[(q, k) in done for q, k in zip(have.question, have.kind)]]   # empty answers get regenerated
    return refs, qs, have, done, out_path


def missing_pairs(qs, done):
    return [(q, "llm_" + k) for q in qs for k, _, _ in VARIANTS if (q, "llm_" + k) not in done]


def generate(split: str, limit=None, root: Path = ROOT, chat=None, save_every=20, delay=2.0,
             sleep=time.sleep, backoff=BACKOFF, max_wait=MAX_WAIT, log=print, max_tokens=DEFAULT_MAX_TOKENS):
    chat = chat or llm_service._chat
    refs, qs, have, done, out_path = plan(split, limit, root)
    existing = sum((q, "llm_" + k) in done for q in qs for k, _, _ in VARIANTS)
    todo_total = len(missing_pairs(qs, done))
    stats = {"generated": 0, "skipped": 0, "existing": existing, "target": len(qs) * len(VARIANTS)}
    new, skipped_in_a_row = [], 0
    for n, q in enumerate(qs, 1):
        pending = [(i, v) for i, v in enumerate(VARIANTS) if (q, "llm_" + v[0]) not in done]
        if not pending:
            continue
        h = int(hashlib.md5(q.encode()).hexdigest(), 16)
        rate_limited = False
        for i, (kind, label, instr) in pending:
            if rate_limited:                       # rest of this question: skip, resume on the next run
                stats["skipped"] += 1
                continue
            length = LENGTHS[(h + i) % len(LENGTHS)]
            try:
                ans = call_with_retry(chat, PERSONA.format(length=length, instruction=instr),
                                      f"Interview question: {q}", sleep=sleep, backoff=backoff, max_wait=max_wait, log=log,
                                      max_tokens=max_tokens)
            except RateLimited as e:
                log(f"    skip question (rate limited: {e})")
                stats["skipped"] += 1
                rate_limited = True
                continue
            except Exception as e:                 # other errors: skip this pair, keep going
                log(f"    skip {kind}: {e}")
                stats["skipped"] += 1
                continue
            if _blank(ans):
                log(f"    skip {kind}: empty answer")
                stats["skipped"] += 1
                continue
            new.append({"question": q, "reference": refs[q], "answer": ans, "label": label, "kind": "llm_" + kind})
            stats["generated"] += 1
            if delay:
                sleep(delay)
        log(f"question {n}/{len(qs)} | generated {stats['generated']}/{todo_total} | skipped {stats['skipped']}")
        skipped_in_a_row = skipped_in_a_row + 1 if rate_limited else 0
        if new and (n % save_every == 0 or rate_limited):
            have, new = pd.concat([have, pd.DataFrame(new)], ignore_index=True), []
            write(have, out_path)
        if skipped_in_a_row >= STOP_AFTER_SKIPPED_QUESTIONS:
            log(f"Stopping: {skipped_in_a_row} questions in a row were rate-limited (quota likely used up). "
                f"Rerun later - it resumes from the missing pairs.")
            stats["skipped"] += len(missing_pairs(qs[n:], done))
            break
    if new:
        have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
    have = repair(have, refs)
    write(have, out_path)
    stats["total_rows"] = len(have)
    return out_path, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--limit", type=int, default=None, help="number of QUESTIONS to cover (each needs 4 answers)")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to wait after each successful API call")
    ap.add_argument("--max-wait", type=float, default=MAX_WAIT, help="longest Retry-After to honour, in seconds")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="completion budget per answer (reasoning models use part of it before answering)")
    ap.add_argument("--dry-run", action="store_true", help="report missing (question, kind) pairs; no API calls")
    ap.add_argument("--repair", action="store_true", help="fix references/labels/duplicates in the existing CSV; no API calls")
    args = ap.parse_args()
    if args.repair:
        path = ROOT / "data" / "processed" / f"llm_{args.split}.csv"
        refs = reference_lookup(pd.read_csv(ROOT / "data" / "processed" / f"{args.split}.csv"))
        before = pd.read_csv(path)
        after = repair(before, refs)
        write(after, path)
        print(f"Repaired {path}: {len(before)} -> {len(after)} rows, "
              f"{int(before.reference.map(_blank).sum())} empty references fixed, "
              f"{int(after.reference.map(_blank).sum())} still empty (question not in {args.split}.csv)")
        return
    if args.dry_run:
        _, qs, have, done, out_path = plan(args.split, args.limit)
        miss = missing_pairs(qs, done)
        by_kind = pd.Series([k for _, k in miss], dtype=str).value_counts().to_dict()
        print(f"Questions in scope: {len(qs)}  (target pairs: {len(qs) * len(VARIANTS)})")
        print(f"Existing complete pairs: {len(qs) * len(VARIANTS) - len(miss)}   rows in {out_path.name}: {len(have)}")
        print(f"Missing pairs: {len(miss)}  by kind: {by_kind}")
        print(f"Questions with something missing: {len({q for q, _ in miss})}")
        return
    if not config.AI_API_KEY:
        sys.exit("Set AI_API_KEY in .env first.")
    path, st = generate(args.split, args.limit, delay=args.delay, max_wait=args.max_wait, max_tokens=args.max_tokens)
    print(f"\nGenerated: {st['generated']}/{st['target']}")
    print(f"Skipped/rate-limited: {st['skipped']}")
    print(f"Existing: {st['existing']}")
    print(f"Total rows: {st['total_rows']}  ({path})")


if __name__ == "__main__":
    main()
