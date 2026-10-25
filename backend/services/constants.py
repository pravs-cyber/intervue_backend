"""Single source of truth for skills, difficulty levels, scores and status thresholds."""

TECH_SKILLS = ["DSA", "OOP", "DBMS", "SQL", "OS", "CN", "Java", "Python"]
ALL_SKILLS = TECH_SKILLS + ["Communication"]
SKILL_NAMES = {"DSA": "Data Structures & Algorithms", "OOP": "Object-Oriented Programming", "DBMS": "DBMS",
               "SQL": "SQL", "OS": "Operating Systems", "CN": "Computer Networks", "Java": "Java",
               "Python": "Python", "Communication": "Communication"}

# Difficulty is stored as text; code works with the index 0..2.
LEVELS = ["Easy", "Medium", "Hard"]

# Predicted answer categories, ordered worst -> best, and the score each one stands for.
CATEGORIES = ["Poor", "Average", "Good", "Excellent"]
CATEGORY_SCORE = {"Poor": 25, "Average": 50, "Good": 75, "Excellent": 90}

# Adaptive rule: predicted category -> change in difficulty for the next question.
DIFFICULTY_DELTA = {"Poor": -1, "Average": 0, "Good": +1, "Excellent": +1}


def skill_name(skill: str) -> str:
    return SKILL_NAMES.get(skill, skill)


def skill_status(score: float) -> str:
    if score < 60:
        return "Needs Improvement"
    if score <= 75:
        return "Developing"
    return "Strong"
