from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class ForwardRequestJSON(BaseModel):
    """Схема для JSON запроса без изображений"""
    data: List[List[float]]  # временное окно с фичами
    asset: Optional[str] = "BTC"  # актив


class ForwardResponseJSON(BaseModel):
    """Ответ для JSON запроса"""
    prediction: int  # 0 или 1
    probabilities: Dict[str, float]  # вероятности классов
    asset: str


class ForwardResponseImage(BaseModel):
    """Ответ для запроса с изображением"""
    image_base64: str
    prediction: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Схема для ошибок"""
    error: str


class HistoryItem(BaseModel):
    """Элемент истории запросов"""
    id: int
    request_type: str
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    input_size: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
        # Для Pydantic v2
        orm_mode = True


class HistoryResponse(BaseModel):
    """Ответ для GET /history"""
    total: int
    items: List[HistoryItem]


class StatsResponse(BaseModel):
    """Ответ для GET /stats"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time_ms: float
    processing_time_quantiles: Dict[str, float]  # mean, 50%, 95%, 99%
    input_size_stats: Optional[Dict[str, Any]] = None  # статистика по размерам входных данных


class UserCreate(BaseModel):
    username: str
    password: str
    is_admin: bool = False


class UserResponse(BaseModel):
    id: int
    username: str
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True
        # Для Pydantic v2
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None

