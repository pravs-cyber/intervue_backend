"""Rule-based adaptive interviewing (no RL): the next question depends on the ML label."""


def next_step(label: str, skill: str, difficulty: int, topics: list[str]) -> tuple[str, int, str]:
    """Return (next_skill, next_difficulty, human-readable reason)."""
    def rotate():
        if skill in topics and len(topics) > 1:
            return topics[(topics.index(skill) + 1) % len(topics)]
        return skill

    if label == "Poor":
        return skill, max(1, difficulty - 1), "Answer was Poor -> easier question on the same skill"
    if label == "Average":
        return rotate(), difficulty, "Answer was Average -> similar difficulty, next topic"
    if label == "Good":
        return rotate(), min(3, difficulty + 1), "Answer was Good -> slightly harder question"
    return skill, min(3, difficulty + 1), "Answer was Excellent -> harder follow-up on the same skill"
