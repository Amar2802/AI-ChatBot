from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class CandidateSchema(BaseModel):
    id: int
    question: str
    score: float

class ChatResponseChunk(BaseModel):
    session_id: Optional[str] = None
    message_id: Optional[int] = None
    mode: Optional[str] = None
    similarity: Optional[float] = None
    candidates: Optional[List[CandidateSchema]] = None
    chunk: str
    done: bool
