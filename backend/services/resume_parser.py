"""Deliberately simple resume parsing: keyword matching + section headings."""
import io
import re

from services.constants import SKILLS

KNOWN_SKILLS = ["python", "java", "c++", "c", "javascript", "typescript", "react", "next.js", "node.js", "sql", "mysql",
                "postgresql", "mongodb", "sqlite", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "fastapi",
                "flask", "django", "spring boot", "git", "docker", "html", "css", "tailwind", "machine learning",
                "data structures", "algorithms", "oop", "dbms", "operating systems", "computer networks", "aws"]
# resume keyword -> one of our 8 tracked skills (others count as Communication)
SKILL_MAP = {"python": "Python", "pandas": "Python", "numpy": "Python", "flask": "Python", "django": "Python",
             "fastapi": "Python", "scikit-learn": "Python", "java": "Java", "spring boot": "Java",
             "sql": "SQL", "mysql": "SQL", "postgresql": "SQL", "sqlite": "SQL", "mongodb": "DBMS", "dbms": "DBMS",
             "data structures": "DSA", "algorithms": "DSA", "oop": "OOP", "operating systems": "OS",
             "computer networks": "CN"}
HEADINGS = {"skills": r"(technical )?skills", "projects": r"(academic )?projects?", "education": r"education",
            "experience": r"(work )?experience|internships?"}


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


def parse_resume(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines()]
    sections, current = {k: [] for k in HEADINGS}, None
    for line in lines:
        if not line:
            continue
        clean = re.sub(r"[^a-zA-Z ]", "", line).strip().lower()
        hit = next((k for k, pat in HEADINGS.items() if re.fullmatch(pat, clean)), None)
        if hit:
            current = hit
        elif re.fullmatch(r"[A-Z &]{4,30}", line):  # some other ALL-CAPS heading ends the section
            current = None
        elif current:
            sections[current].append(line)
    lower = text.lower()
    skills = [s for s in KNOWN_SKILLS if re.search(rf"(?<![a-z+#]){re.escape(s)}(?![a-z+#])", lower)]

    def titles(items, n):
        out = [re.sub(r"^[-•*\u2022\s]+", "", i) for i in items if len(i) < 90]
        return [o for o in out if o][:n]

    return {"skills": skills, "projects": titles(sections["projects"], 4),
            "education": titles(sections["education"], 3), "experience": titles(sections["experience"], 3)}


def summary(resume: dict) -> str:
    return (f"Skills: {', '.join(resume.get('skills', []))}. Projects: {'; '.join(resume.get('projects', []))}. "
            f"Education: {'; '.join(resume.get('education', []))}. Experience: {'; '.join(resume.get('experience', []))}.")


def resume_candidates(resume: dict) -> list[dict]:
    """Template questions built from the parsed resume (used when no LLM is available)."""
    out = []
    for s in resume.get("skills", [])[:8]:
        skill = SKILL_MAP.get(s, "Communication")
        out.append({"question": f"You listed {s} on your resume. What is it, and where have you used it?", "skill": skill, "difficulty": 1, "type": "resume"})
        out.append({"question": f"Explain one concept in {s} that you know well and how you applied it in a project.", "skill": skill, "difficulty": 2, "type": "resume"})
    for p in resume.get("projects", []):
        out.append({"question": f"Walk me through your project \"{p}\". What problem did it solve and what was your role?", "skill": "Communication", "difficulty": 2, "type": "resume"})
        out.append({"question": f"What was the hardest technical challenge in \"{p}\", and how would you improve or scale it?", "skill": "Communication", "difficulty": 3, "type": "resume"})
    for e in resume.get("education", [])[:1]:
        out.append({"question": f"Which subject from your studies ({e}) did you enjoy most, and why?", "skill": "Communication", "difficulty": 1, "type": "resume"})
    for x in resume.get("experience", []):
        out.append({"question": f"Describe your responsibilities in \"{x}\" and what you learned from it.", "skill": "Communication", "difficulty": 2, "type": "resume"})
    return out
