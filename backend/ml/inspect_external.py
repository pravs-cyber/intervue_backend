"""
Inspect an external short-answer-grading dataset BEFORE any import or training. Read-only.
Run from backend/:   python ml/inspect_external.py sprag
Writes ml/results/external_<name>_report.txt and ml/results/external_<name>_questions.csv (one row per question
with its heuristic topic, so the topic assignment can be checked and corrected by hand).

Each dataset gets a small adapter that returns one normalised frame:
    question_id, question, reference, answer, score, score_max, graders (list of raw grader scores)
Topics are assigned by keyword rules (TOPIC_RULES) - a heuristic for deciding fit, not a ground truth.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
OUT = Path(__file__).resolve().parent / "results"

# first matching rule wins; order matters (specific before general)
TOPIC_RULES = [
    ("SQL", r"\b(sql|select|join|where clause|having|group by|query|queries)\b"),
    ("DBMS", r"\b(database|dbms|normali[sz]|transaction|acid|primary key|foreign key|index(ing)?|relational|schema)\b"),
    ("OS", r"\b(operating system|process(es)?|thread|deadlock|schedul|semaphore|mutex|paging|virtual memory|kernel)\b"),
    ("Computer Networks", r"\b(network|tcp|udp|ip address|http|dns|osi|router|protocol|packet|socket)\b"),
    ("OOP", r"\b(class(es)?|object|inheritance|polymorphism|encapsulation|abstraction|constructor|method overrid|interface|oop)\b"),
    ("DSA", r"\b(array|linked list|stack|queue|tree|graph|heap|hash|sort(ing)?|search(ing)?|recursion|complexity|big[- ]o|algorithm|binary)\b"),
    ("ML/AI", r"\b(machine learning|regression|classification|cluster|overfit|neural|model|training|accuracy|precision|recall|feature|supervised|histogram)\b"),
    ("Programming", r"\b(python|java|function|variable|loop|list|tuple|dictionary|string|exception|module|lambda|mutable|numpy|pandas|syntax|def)\b"),
]


def topic(text: str) -> str:
    t = (text or "").lower()
    for name, pat in TOPIC_RULES:
        if re.search(pat, t):
            return name
    return "Other CS"


# ------------------------------------------------------------------ adapters
def load_sprag():
    d = EXT / "sprag"
    q = pd.read_csv(d / "Question_data.csv")
    resp = pd.concat([pd.read_csv(d / "PyResponses.csv"), pd.read_csv(d / "MLResponses.csv")], ignore_index=True)
    df = resp.merge(q, on="QuestionID", how="left")
    comb = pd.read_csv(d / "refstdcombined.csv")
    max_by_q = comb.groupby("QuestionID").max_score.max()            # per-question maximum as supplied
    out = pd.DataFrame({
        "question_id": df.QuestionID, "question": df.QuestionText, "reference": df["Reference Answer"],
        "answer": df.StudentAnswer, "score": (df.Score1 + df.Score2) / 2,
        "score_max": df.QuestionID.map(max_by_q), "grader1": df.Score1, "grader2": df.Score2,
        "question_type": df.QuestionType,
        "course": df.QuestionID.str.extract(r"^([A-Za-z]+)", expand=False).map({"PythonQ": "Python programming", "BasicML": "Basic ML / data analysis"})})
    notes = [f"responses: PyResponses {len(pd.read_csv(d / 'PyResponses.csv'))} + MLResponses "
             f"{len(pd.read_csv(d / 'MLResponses.csv'))}; questions file {len(q)}; refstdcombined.csv {len(comb)} rows",
             f"responses without a matching question: {int(df.QuestionText.isna().sum())}"]
    return out, notes


ADAPTERS = {"sprag": load_sprag}


def _filled(s):
    return s.notna() & (s.astype(str).str.strip() != "")


def report(name):
    df, notes = ADAPTERS[name]()
    words = df.answer.fillna("").astype(str).str.split().str.len()
    blank = ~_filled(df.answer)
    nonanswer = df.answer.fillna("").astype(str).str.strip().str.lower().str.match(
        r"^(-+|\.+|na|n/a|nil|none|no answer.*|not attempted|i don'?t know.*|dont know.*|\?+)$")
    qs = df.drop_duplicates("question_id")[["question_id", "question", "reference"]].copy()
    # question text only: reference answers mention incidental terms ("query", "process") and mislead the rules
    qs["topic"] = qs.question.fillna("").map(topic)
    qs["answers"] = qs.question_id.map(df.question_id.value_counts())
    qs["score_max"] = qs.question_id.map(df.groupby("question_id").score_max.max())
    qs["mean_score"] = qs.question_id.map(df.groupby("question_id").score.mean().round(2))
    df = df.merge(qs[["question_id", "topic"]], on="question_id", how="left")
    lines = [f"===== {name} =====", *notes,
             f"questions: {qs.question_id.nunique()}   answers: {len(df)}",
             f"questions missing text: {int((~_filled(qs.question)).sum())}   missing reference: {int((~_filled(qs.reference)).sum())}",
             f"answers blank: {int(blank.sum())}   non-answers ('-', 'no answer', 'I don't know'...): {int((nonanswer & ~blank).sum())}",
             f"duplicate (question_id, answer) rows: {int(df.duplicated(['question_id', 'answer']).sum())}",
             "\nAnswers per question: " + qs.answers.describe()[["min", "25%", "50%", "mean", "75%", "max"]].round(1).to_string().replace("\n", " | "),
             "\nScore scale (score_max per question):\n" + qs.score_max.value_counts().sort_index().to_string(),
             "\nGrader agreement: exact " + f"{(df.grader1 == df.grader2).mean():.2f}, within 1 point {((df.grader1 - df.grader2).abs() <= 1).mean():.2f}, "
             f"Pearson r {df.grader1.corr(df.grader2):.2f}",
             "\nAverage score distribution (as fraction of the question's max):\n" +
             pd.cut(df.score / df.score_max, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0]).value_counts().sort_index().to_string(),
             "\nRaw average score counts:\n" + df.score.value_counts().sort_index().to_string(),
             "\nAnswer length (words): " + words.describe()[["min", "25%", "50%", "mean", "75%", "max"]].round(1).to_string().replace("\n", " | "),
             "\nTopics (heuristic) - questions / answers:\n" +
             pd.DataFrame({"questions": qs.topic.value_counts(), "answers": df.topic.value_counts()}).fillna(0).astype(int).to_string()]
    if "course" in df:
        lines.append("\nCourse (from question IDs) x heuristic topic (questions):\n" +
                     pd.crosstab(df.drop_duplicates("question_id").course, df.drop_duplicates("question_id").topic).to_string())
    if "question_type" in df:
        lines.append("\nQuestion types:\n" + df.drop_duplicates("question_id").question_type.value_counts().to_string())
    lines.append("\nSample questions per topic:")
    for t, g in qs.groupby("topic"):
        lines.append(f"  [{t}]")
        for _, r in g.head(4).iterrows():
            lines.append(f"    {r.question_id}: {str(r.question)[:110]}")
    lines.append("\nSample answers by score band (fraction of max):")
    df["band"] = pd.cut(df.score / df.score_max, [-0.01, 0.25, 0.6, 0.85, 1.0], labels=["low", "mid-low", "mid-high", "top"])
    for b, g in df.groupby("band", observed=True):
        lines.append(f"  --- {b} ---")
        for _, r in g.sample(min(3, len(g)), random_state=3).iterrows():
            lines.append(f"    Q: {str(r.question)[:90]}\n    ref: {str(r.reference)[:110]}\n    A ({r.score}/{r.score_max}): {str(r.answer)[:160]}")
    text = "\n".join(lines)
    OUT.mkdir(exist_ok=True)
    (OUT / f"external_{name}_report.txt").write_text(text)
    qs.to_csv(OUT / f"external_{name}_questions.csv", index=False)
    print(text)


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "sprag")
