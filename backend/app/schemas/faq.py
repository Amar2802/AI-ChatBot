from pydantic import BaseModel, Field
from typing import List

class FAQItem(BaseModel):
    question: str = Field(..., min_length=3)
    answer: str = Field(..., min_length=3)

class BulkFAQRequest(BaseModel):
    faqs: List[FAQItem]
