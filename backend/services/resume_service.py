"""Very simple resume parser: keyword skills + section splitting. No ML."""
import io
import re

KNOWN_SKILLS = [
    "Python", "Java", "C++", "C", "JavaScript", "TypeScript", "React", "Next.js", "Node.js", "SQL", "MySQL",
    "PostgreSQL", "MongoDB", "DBMS", "DSA", "OOP", "Operating Systems", "Computer Networks", "Machine Learning",
    "Deep Learning", "Pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch", "FastAPI", "Django", "Flask",
    "Spring Boot", "Git", "Docker", "HTML", "CSS", "Tailwind",
]
SECTIONS = {
    "projects": r"^(projects?|academic projects?|personal projects?)\s*:?$",
    "education": r"^(education|academics?|qualifications?)\s*:?$",
    "experience": r"^(experience|work experience|internships?|professional experience)\s*:?$",
    "skills": r"^(skills|technical skills)\s*:?$",
}


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def parse_resume(text: str) -> dict:
    lines = [l.strip(" \t•-–*▪●") for l in text.splitlines()]
    sections = {k: [] for k in SECTIONS}
    current = None
    for line in lines:
        if not line:
            continue
        for name, pat in SECTIONS.items():
            if re.match(pat, line, re.I):
                current = name
                break
        else:
            if current:
                sections[current].append(line)
    lowered = text.lower()
    skills = [s for s in KNOWN_SKILLS if re.search(r"(?<![A-Za-z])" + re.escape(s.lower()) + r"(?![A-Za-z+#])", lowered)]
    # project titles = short lines (likely headings) inside the Projects section
    projects = [l for l in sections["projects"] if len(l.split()) <= 12][:5] or sections["projects"][:3]
    return {
        "skills": skills[:15],
        "projects": projects,
        "education": sections["education"][:4],
        "experience": sections["experience"][:5],
    }


def fallback_questions(parsed: dict, n: int) -> list:
    """Template questions when no LLM key is set."""
    qs = []
    for p in parsed["projects"][:3]:
        qs.append({"question": f"Walk me through your project '{p[:80]}'. What was your role and what challenges did you face?", "skill": "Communication"})
    for s in parsed["skills"]:
        qs.append({"question": f"You listed {s} on your resume. Explain a core concept in {s} and where you have used it.",
                   "skill": s if s in ("Java", "Python", "SQL", "DBMS", "DSA", "OOP") else "Communication"})
    if parsed["experience"]:
        qs.append({"question": "Tell me about your experience or internship and what you learned from it.", "skill": "Communication"})
    if parsed["education"]:
        qs.append({"question": "Which subject from your degree interests you the most, and why?", "skill": "Communication"})
    qs.append({"question": "Which project on your resume are you proudest of, and why?", "skill": "Communication"})
    return qs[:n]
