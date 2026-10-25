"""Seed question bank used when no LLM is configured (and as a fallback).
Format: (skill, difficulty 0=Easy 1=Medium 2=Hard, question)"""

TECHNICAL = [
    ("DSA", 0, "What is the difference between an array and a linked list?"),
    ("DSA", 0, "Explain what a stack is and give one real-world use of it."),
    ("DSA", 1, "Explain how binary search works and what its time complexity is."),
    ("DSA", 1, "What is the difference between BFS and DFS, and when would you use each?"),
    ("DSA", 2, "Explain how a hash table handles collisions and how that affects lookup complexity."),
    ("DSA", 2, "How would you detect a cycle in a linked list, and why does Floyd's algorithm work?"),

    ("OOP", 0, "What are the four pillars of object-oriented programming?"),
    ("OOP", 0, "What is the difference between a class and an object?"),
    ("OOP", 1, "Explain the difference between method overloading and overriding."),
    ("OOP", 1, "What is the difference between an abstract class and an interface?"),
    ("OOP", 2, "Explain the principle 'favor composition over inheritance' with an example."),
    ("OOP", 2, "What are the SOLID principles? Explain any two with examples."),

    ("DBMS", 0, "What is a primary key and how is it different from a foreign key?"),
    ("DBMS", 0, "What is normalization and why is it needed?"),
    ("DBMS", 1, "Explain the ACID properties of a transaction."),
    ("DBMS", 1, "What is the difference between a clustered and a non-clustered index?"),
    ("DBMS", 2, "Explain transaction isolation levels and the anomalies each one prevents."),
    ("DBMS", 2, "What is a deadlock in a database, and how can it be prevented or resolved?"),

    ("SQL", 0, "What is the difference between WHERE and HAVING in SQL?"),
    ("SQL", 0, "What is the difference between DELETE, TRUNCATE and DROP?"),
    ("SQL", 1, "Explain the different types of SQL joins with examples."),
    ("SQL", 1, "How would you find the second highest salary from an Employee table?"),
    ("SQL", 2, "How do window functions differ from GROUP BY aggregations?"),
    ("SQL", 2, "How would you optimize a slow SQL query? What would you check first?"),

    ("OS", 0, "What is an operating system and what are its main functions?"),
    ("OS", 0, "What is the difference between a process and a thread?"),
    ("OS", 1, "Explain the four conditions necessary for a deadlock."),
    ("OS", 1, "What is paging and how does it differ from segmentation?"),
    ("OS", 2, "Compare CPU scheduling algorithms such as Round Robin and Shortest Job First."),
    ("OS", 2, "Explain how virtual memory and page replacement algorithms work."),

    ("CN", 0, "What is the difference between TCP and UDP?"),
    ("CN", 0, "What is an IP address and how do IPv4 and IPv6 differ?"),
    ("CN", 1, "What happens when you type a URL into a browser and press Enter?"),
    ("CN", 1, "What is the OSI model? Briefly describe its layers."),
    ("CN", 2, "Explain how the TCP three-way handshake and congestion control work."),
    ("CN", 2, "How does DNS resolution work, and how does caching affect it?"),

    ("Java", 0, "What is the difference between JDK, JRE and JVM?"),
    ("Java", 0, "What is the difference between == and equals() in Java?"),
    ("Java", 1, "Explain the difference between ArrayList and LinkedList in Java."),
    ("Java", 1, "What is the difference between checked and unchecked exceptions in Java?"),
    ("Java", 2, "How does HashMap work internally in Java?"),
    ("Java", 2, "Explain how garbage collection works in Java."),

    ("Python", 0, "What is the difference between a list and a tuple in Python?"),
    ("Python", 0, "What are mutable and immutable types in Python? Give examples."),
    ("Python", 1, "What are decorators in Python and how do they work?"),
    ("Python", 1, "Explain the difference between a generator and a regular function in Python."),
    ("Python", 2, "Explain Python's Global Interpreter Lock and its impact on multithreading."),
    ("Python", 2, "How does memory management and reference counting work in Python?"),
]

HR = [
    ("Communication", 0, "Tell me about yourself."),
    ("Communication", 0, "What are your strengths?"),
    ("Communication", 0, "Why do you want this role?"),
    ("Communication", 1, "Describe a challenging project you worked on."),
    ("Communication", 1, "How do you handle pressure and tight deadlines?"),
    ("Communication", 1, "Tell me about a time you faced a conflict in a team and how you resolved it."),
    ("Communication", 2, "Where do you see yourself in five years?"),
    ("Communication", 2, "Tell me about a failure and what you learned from it."),
    ("Communication", 2, "Why should we hire you over other candidates?"),
]
