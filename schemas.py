"""Pydantic request and response models."""

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    """Incoming student question."""
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be empty or whitespace-only.")
        return v.strip()


class ChatResponse(BaseModel):
    """Chatbot reply sent back to the student."""
    question: str
    answer: str
    matched_faq: str | None = None   # the FAQ question that was matched
    confidence: float                 # similarity score (0–1)


class HealthResponse(BaseModel):
    status: str
    version: str
