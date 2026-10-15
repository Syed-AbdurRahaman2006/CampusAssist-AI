"""CampusAssist AI — FastAPI application entry point."""

from fastapi import FastAPI

from config import APP_TITLE, APP_DESCRIPTION, APP_VERSION
from schemas import ChatRequest, ChatResponse, HealthResponse
from chatbot import get_faqs, find_answer

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


@app.get("/")
def root():
    """Welcome message and basic project info."""
    return {
        "project": APP_TITLE,
        "description": APP_DESCRIPTION,
        "version": APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """API health check."""
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/faqs")
def faqs():
    """Return all available FAQ entries."""
    return get_faqs()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Accept a student question and return the best matching FAQ answer."""
    answer, matched_faq, confidence = find_answer(req.question)
    return ChatResponse(
        question=req.question,
        answer=answer,
        matched_faq=matched_faq,
        confidence=confidence,
    )
