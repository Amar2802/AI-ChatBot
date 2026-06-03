from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from app.core.database import Base

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1 for thumbs-up, -1 for thumbs-down
    note = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Check constraint for allowed ratings
    __table_args__ = (
        CheckConstraint("rating IN (1, -1)", name="check_valid_rating"),
    )

    # Relationships
    session = relationship("Session", back_populates="feedbacks")
    message = relationship("Message", back_populates="feedbacks")
