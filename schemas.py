from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    age_mode: str = "student"
    language: str = "English"

class ChatResponse(BaseModel):
    answer: str

class SummaryResponse(BaseModel):
    summary: str