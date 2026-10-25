"""
LLM provider abstraction (any OpenAI-compatible /chat/completions API).
Configured ONLY through .env: AI_API_KEY, AI_MODEL (+ optional AI_BASE_URL).

The LLM never decides the answer category or score - that comes from the Random Forest.
The LLM is used for: questions, follow-ups, feedback, improvement suggestions.
If no key is configured (or a call fails) we fall back to simple templates and
flag the result with source="template" so the UI can show it honestly.
"""
import json
import re
from typing import Optional

import httpx

import config


def is_configured() -> bool:
    return bool(config.AI_API_KEY)


class EmptyCompletion(RuntimeError):
    """The API call succeeded but returned no answer text."""


def _chat(system: str, user: str, max_tokens: int = 600) -> str:
    resp = httpx.post(
        f"{config.AI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.AI_API_KEY}"},
        json={
            "model": config.AI_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=40,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    if not content:
        # e.g. reasoning models (gpt-oss) that spend max_tokens on hidden reasoning and return no text
        raise EmptyCompletion(f"empty message.content (finish_reason={choice.get('finish_reason')!r}, "
                              f"max_tokens={max_tokens})")
    return content


def _json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


# ---------- questions ----------
def generate_question(skill: str, difficulty: str, asked: list) -> Optional[str]:
    if not is_configured():
        return None
    try:
        avoid = "; ".join(asked[-8:]) or "none"
        return _chat(
            "You are a technical interviewer for fresher software roles. Reply with ONE interview question only, no preface.",
            f"Skill: {skill}. Difficulty: {difficulty}. Do not repeat these questions: {avoid}. "
            "The question must be answerable in a short paragraph.",
            max_tokens=120,
        ).strip('"')
    except Exception:
        return None


def generate_followup(question: str, answer: str, skill: str) -> Optional[str]:
    if not is_configured():
        return None
    try:
        return _chat(
            "You are an interviewer. The candidate answered very well. Ask ONE harder follow-up question that digs deeper. Reply with the question only.",
            f"Skill: {skill}\nQuestion: {question}\nAnswer: {answer}",
            max_tokens=120,
        ).strip('"')
    except Exception:
        return None


REFERENCE_SYSTEM = ("You are an expert interviewer. Write a correct, complete model answer of 80-120 words. "
                    "Plain prose, no headings or lists.Write in English.")


def generate_reference(question: str, skill: str, max_tokens: int = 800, raise_errors: bool = False) -> Optional[str]:
    """A short model answer, used only as input to the reference-aware features.
    max_tokens leaves room for reasoning models (gpt-oss) that think before answering."""
    if not is_configured():
        return None
    try:
        return _chat(REFERENCE_SYSTEM, f"Skill: {skill}\nQuestion: {question}", max_tokens=max_tokens)
    except Exception:
        if raise_errors:
            raise
        return None


def generate_resume_questions(resume: dict, n: int) -> Optional[list]:
    """Returns [{"question":..., "skill":...}] or None."""
    if not is_configured():
        return None
    try:
        raw = _chat(
            "You are an interviewer. Write interview questions based on this candidate's resume. "
            'Reply ONLY with JSON: [{"question": "...", "skill": "<one of DSA,OOP,DBMS,SQL,OS,CN,Java,Python,Communication>"}]',
            f"Number of questions: {n}\nSkills: {resume['skills']}\nProjects: {resume['projects']}\n"
            f"Education: {resume['education']}\nExperience: {resume['experience']}",
            max_tokens=900,
        )
        items = _json(raw)
        return [i for i in items if i.get("question")][:n]
    except Exception:
        return None


# ---------- feedback ----------
def _template_feedback(label: str, skill: str, answer: str) -> dict:
    words = len(answer.split())
    well = {
        "Excellent": ["Your answer was detailed and well covered.", "You explained the concept thoroughly."],
        "Good": ["You identified the main concept correctly.", "Your explanation was relevant."],
        "Average": ["You touched on the basic idea."],
        "Poor": ["You attempted the question."],
    }[label]
    improve = {
        "Excellent": ["Keep practising harder variations of this topic."],
        "Good": ["Add a practical example.", "Explain the key differences or trade-offs more clearly."],
        "Average": ["Cover more details of the concept.", "Add an example and mention trade-offs."],
        "Poor": ["Revise the basic definition of this topic.", "Structure the answer: definition, how it works, example."],
    }[label]
    if words < 25 and label != "Poor":
        improve.append("Your answer is short - elaborate more.")
    return {
        "summary": f"Predicted answer category: {label.upper()} (from the ML classifier).",
        "went_well": well,
        "missing": [],
        "improve": improve,
        "recommended_topic": f"{skill} fundamentals",
        "evaluation_consistency": "not_checked",
        "source": "template",
    }


def generate_feedback(question: str, answer: str, label: str, skill: str, reference: Optional[str] = None) -> dict:
    if not is_configured():
        return _template_feedback(label, skill, answer)
    try:
        raw = _chat(
            "You are an interview coach. An ML classifier has already assigned the answer a category. You do NOT "
            "change the category. Explain it, specifically to this answer: what was correct, which important points "
            "were missing, and how to improve. Separately, say whether the category looks consistent with the answer "
            "('consistent') or questionable ('questionable'); this is logged for review and never overrides the model. "
            "Reply ONLY with JSON: "
            '{"summary": "2 sentences", "went_well": [".."], "missing": [".."], "improve": [".."], '
            '"recommended_topic": "short topic", "evaluation_consistency": "consistent|questionable"}',
            f"Skill: {skill}\nQuestion: {question}\n" + (f"Reference answer: {reference}\n" if reference else "") +
            f"Candidate answer: {answer}\nPredicted category: {label}",
            max_tokens=600,
        )
        data = _json(raw)
        return {
            "summary": str(data.get("summary", "")),
            "went_well": [str(x) for x in data.get("went_well", [])][:4],
            "missing": [str(x) for x in data.get("missing", [])][:4],
            "improve": [str(x) for x in data.get("improve", [])][:4],
            "recommended_topic": str(data.get("recommended_topic", f"{skill} fundamentals")),
            "evaluation_consistency": "questionable" if data.get("evaluation_consistency") == "questionable" else "consistent",
            "source": "llm",
        }
    except Exception:
        fb = _template_feedback(label, skill, answer)
        fb["source"] = "template"
        return fb
