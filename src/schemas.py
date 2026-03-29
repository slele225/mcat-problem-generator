"""Pydantic models for question/passage data and LLM response parsing."""

from pydantic import BaseModel, field_validator
from typing import Optional


class DiscreteQuestion(BaseModel):
    """A single discrete (non-CARS) MCAT question."""
    question_id: str
    topic_id: str
    section: str
    content_category: str
    topic_group: str
    topic: str
    subtopics_tested: list[str]
    stem: str
    choices: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str  # "A", "B", "C", or "D"
    explanation: str
    difficulty: str  # "easy", "medium", "hard"
    skill_tested: str  # AAMC SIRS skill (Skill 1-4)
    validation: dict  # {"adversarial_pass": bool, "blind_solve_pass": bool}

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v):
        if set(v.keys()) != {"A", "B", "C", "D"}:
            raise ValueError(f"choices must have keys A,B,C,D, got {set(v.keys())}")
        return v


class CARSQuestion(BaseModel):
    """A single CARS question tied to a passage."""
    question_id: str
    skill_type: str  # "Foundations of Comprehension", "Reasoning Within the Text", "Reasoning Beyond the Text"
    stem: str
    choices: dict[str, str]
    correct_answer: str
    explanation: str

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v


class CARSPassage(BaseModel):
    """A CARS passage with its associated questions."""
    passage_id: str
    passage_text: str
    word_count: int
    subject: str
    questions: list[CARSQuestion]
    validation: dict


# --- Models for parsing raw LLM output (before validation) ---

class RawDiscreteQuestion(BaseModel):
    """Raw question as returned by the generation LLM."""
    stem: str
    choices: dict[str, str]
    correct_answer: str
    explanation: str
    difficulty: str
    subtopics_tested: list[str] = []
    skill_tested: str = ""

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        v = v.strip().upper()
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v


class RawCARSPassage(BaseModel):
    """Raw passage as returned by the generation LLM."""
    passage_text: str
    subject: str


class RawCARSQuestion(BaseModel):
    """Raw CARS question as returned by the generation LLM."""
    skill_type: str
    stem: str
    choices: dict[str, str]
    correct_answer: str
    explanation: str

    @field_validator("correct_answer")
    @classmethod
    def validate_answer(cls, v):
        v = v.strip().upper()
        if v not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer must be A/B/C/D, got {v}")
        return v


class AdversarialReview(BaseModel):
    """Result of adversarial review."""
    passed: bool
    issues: list[str] = []
    reasoning: str = ""


class BlindSolveResult(BaseModel):
    """Result of blind solve attempt."""
    chosen_answer: str
    confidence: str  # "high", "medium", "low"
    reasoning: str = ""
