from typing import List, Optional
from pydantic import BaseModel, Field


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=254)
    password: str = Field(min_length=8, max_length=128)
    target_role: str = Field(default="", max_length=100)


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    target_role: str = Field(default="", max_length=100)


class InterviewIn(BaseModel):
    type: str = Field(pattern="^(technical|hr|resume)$")
    topics: List[str] = Field(default=[], max_length=10)
    num_questions: int = Field(default=10, ge=3, le=15)
    resume_text: str = Field(default="", max_length=20000)


class AnswerIn(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)


class EvaluateIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=4000)
    reference: Optional[str] = Field(default=None, max_length=4000)
    skill: Optional[str] = Field(default="General", max_length=50)
