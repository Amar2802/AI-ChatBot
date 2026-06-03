from fastapi import APIRouter
from app.api.v1 import chat, feedback, faq

api_router = APIRouter()

# Register routes at root level for seamless frontend compatibility
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(feedback.router, tags=["Feedback"])
api_router.include_router(faq.router, tags=["FAQ"])
