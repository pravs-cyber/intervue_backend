"""Built-in question bank (used directly, and as the fallback when no LLM is configured)."""
import random
from typing import Optional

# difficulty: 1 easy, 2 medium, 3 hard
TECHNICAL = {
    "DSA": {
        1: ["What is the difference between an array and a linked list?", "Explain how a stack and a queue work and give one use case of each."],
        2: ["Explain how binary search works and why it needs sorted input. What is its time complexity?", "How does a hash table handle collisions? Compare chaining and open addressing."],
        3: ["Explain how you would detect a cycle in a linked list and prove why your approach works.", "Compare Dijkstra's algorithm and Bellman-Ford. When would you choose each?"],
    },
    "OOP": {
        1: ["What are the four pillars of object-oriented programming? Explain each briefly.", "What is the difference between a class and an object?"],
        2: ["Explain the difference between method overloading and method overriding.", "What is the difference between an abstract class and an interface?"],
        3: ["Explain the 'favour composition over inheritance' principle with an example.", "What are the SOLID principles? Pick two and show how violating them causes problems."],
    },
    "DBMS": {
        1: ["What is a primary key and how is it different from a foreign key?", "What is normalization and why is it used?"],
        2: ["Explain ACID properties in a database transaction.", "What is an index? How does it speed up queries and what is its cost?"],
        3: ["Explain transaction isolation levels and the anomalies each one prevents.", "Compare 2NF, 3NF and BCNF with a short example of a table that violates 3NF."],
    },
    "SQL": {
        1: ["What is the difference between WHERE and HAVING in SQL?", "What is the difference between DELETE, TRUNCATE and DROP?"],
        2: ["Explain the different types of JOINs with an example.", "How would you find the second highest salary from an Employee table?"],
        3: ["How would you find duplicate rows in a table and delete all but one of them?", "Explain window functions such as ROW_NUMBER and RANK and when to use them."],
    },
    "OS": {
        1: ["What is the difference between a process and a thread?", "What is the purpose of an operating system?"],
        2: ["What is a deadlock? What are the four necessary conditions for it?", "Explain virtual memory and paging."],
        3: ["Compare FCFS, Round Robin and Shortest Job First scheduling. Which gives the best average waiting time?", "How do semaphores and mutexes differ, and how would you solve the producer-consumer problem?"],
    },
    "CN": {
        1: ["What is the difference between TCP and UDP?", "What is an IP address and what is the role of DNS?"],
        2: ["Explain the OSI model layers with one example protocol per layer.", "What happens when you type a URL in the browser and press Enter?"],
        3: ["Explain the TCP three-way handshake and how TCP achieves reliable delivery and congestion control.", "How does HTTPS/TLS establish a secure connection?"],
    },
    "Java": {
        1: ["What is the difference between JDK, JRE and JVM?", "What is the difference between == and .equals() in Java?"],
        2: ["Explain how HashMap works internally in Java.", "What is the difference between checked and unchecked exceptions?"],
        3: ["Explain Java memory management: heap, stack and garbage collection.", "How does multithreading work in Java? Explain synchronized, volatile and the risk of race conditions."],
    },
    "Python": {
        1: ["What is the difference between a list and a tuple in Python?", "What are Python's mutable and immutable data types? Give examples."],
        2: ["Explain how decorators work in Python with a simple example.", "What is the difference between a shallow copy and a deep copy?"],
        3: ["Explain the Global Interpreter Lock (GIL) and how it affects multithreading and multiprocessing.", "Explain generators and iterators and why they are memory-efficient."],
    },
}

HR = {
    1: ["Tell me about yourself.", "What are your greatest strengths?"],
    2: ["Describe a challenging project you worked on and how you handled it.", "How do you handle pressure and tight deadlines?", "What is your biggest weakness and how are you working on it?"],
    3: ["Tell me about a time you disagreed with a teammate. How did you resolve it?", "Where do you see yourself in five years, and how does this role fit that plan?", "Describe a failure and what you learned from it."],
}

_rng = random.Random()


def _fresh(candidates: list[dict], asked_texts: set[str]):
    return [c for c in candidates if c["question"] not in asked_texts]


def bank_candidates(itype: str, skills: list[str]) -> list[dict]:
    out = []
    if itype == "hr":
        for d, qs in HR.items():
            out += [{"question": q, "skill": "Communication", "difficulty": d, "type": "hr"} for q in qs]
    else:
        for s in skills:
            for d, qs in TECHNICAL.get(s, {}).items():
                out += [{"question": q, "skill": s, "difficulty": d, "type": "technical"} for q in qs]
    return out


def pick_from_bank(itype: str, skills: list[str], skill: str, difficulty: int, asked_texts: set[str]) -> Optional[dict]:
    """Best match first: same skill+difficulty, then nearest difficulty, then any skill near that difficulty."""
    pool = _fresh(bank_candidates(itype, skills), asked_texts)
    if not pool:
        return None
    if itype == "hr":
        skill = "Communication"
    order = sorted(pool, key=lambda c: (c["skill"] != skill, abs(c["difficulty"] - difficulty), _rng.random()))
    return order[0]
