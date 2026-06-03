from pydantic import BaseModel, Field, field_validator
from typing import Optional

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: int
    rating: int = Field(..., description="Rating must be 1 (Helpful) or -1 (Unhelpful)")
    note: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, value: int) -> int:
        if value not in (1, -1):
            raise ValueError("Rating must be either 1 or -1")
        return value
