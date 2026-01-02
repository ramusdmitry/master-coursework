from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, Boolean
from sqlalchemy.sql import func
from app.database import Base
import json


class RequestHistory(Base):
    __tablename__ = "request_history"

    id = Column(Integer, primary_key=True, index=True)
    request_type = Column(String, nullable=False)  # "json" or "image"
    request_data = Column(JSON, nullable=True)  # для JSON запросов
    request_headers = Column(JSON, nullable=True)  # для заголовков
    response_data = Column(JSON, nullable=True)
    processing_time_ms = Column(Float, nullable=False)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    input_size = Column(Integer, nullable=True)  # размер входных данных (токены для текста, размер изображения)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "request_type": self.request_type,
            "request_data": self.request_data,
            "response_data": self.response_data,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "error_message": self.error_message,
            "input_size": self.input_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


