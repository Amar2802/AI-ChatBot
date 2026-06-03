# Import Base and all database models so that Base.metadata registers them
from app.core.database import Base
from app.models.session import Session
from app.models.message import Message
from app.models.feedback import Feedback
from app.models.faq import FAQ
