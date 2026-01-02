from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header, status, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List, Union
import time
import base64
import numpy as np
from datetime import datetime, timedelta

from app.database import get_db, Base, engine
from app import models, schemas
from app.ml_model import get_ml_service
from app.auth import get_current_admin_user, get_password_hash, create_access_token, authenticate_user
from app.schemas import Token, UserCreate, UserResponse
import os
from dotenv import load_dotenv

load_dotenv()

# Создание таблиц
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ML Service для прогнозирования криптовалют",
    description="Сервис для прогнозирования направления цены криптовалют на основе временных рядов",
    version="1.0.0"
)


@app.post("/forward")
async def forward(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    POST /forward - универсальный эндпоинт для обработки запросов
    
    Поддерживает два формата:
    1. JSON запрос (без изображений) - данные в теле запроса
    2. multipart/form-data (с изображением) - параметр image, дополнительные параметры в заголовках
    """
    content_type = request.headers.get("content-type", "").lower()
    
    # Определяем тип запроса по Content-Type
    if "multipart/form-data" in content_type:
        # Обработка запроса с изображением
        form = await request.form()
        image_file = form.get("image")
        
        if image_file is None:
            return JSONResponse(
                status_code=400,
                content={"error": "bad request"}
            )
        
        # Читаем файл
        image_data = await image_file.read()
        
        # Получаем заголовки
        x_asset = request.headers.get("X-Asset")
        x_window_size = request.headers.get("X-Window-Size")
        
        return await _forward_image(image_data, x_asset, x_window_size, db)
    else:
        # Обработка JSON запроса
        try:
            body = await request.json()
            json_request = schemas.ForwardRequestJSON(**body)
            return await _forward_json(json_request, db)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": "bad request"}
            )


async def _forward_json(
    request: schemas.ForwardRequestJSON,
    db: Session
):
    """Обработка JSON запроса"""
    start_time = time.time()
    
    try:
        # Валидация данных
        if not request.data or len(request.data) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "bad request"}
            )
        
        # Получаем ML сервис
        ml_service = get_ml_service()
        
        # Выполняем предсказание
        prediction, probabilities = ml_service.predict(request.data)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Сохраняем в историю
        history_item = models.RequestHistory(
            request_type="json",
            request_data={"data_length": len(request.data), "asset": request.asset},
            response_data={"prediction": prediction, "probabilities": probabilities},
            processing_time_ms=processing_time_ms,
            success=True,
            input_size=len(request.data) * len(request.data[0]) if request.data else None
        )
        db.add(history_item)
        db.commit()
        
        return schemas.ForwardResponseJSON(
            prediction=prediction,
            probabilities=probabilities,
            asset=request.asset or "BTC"
        )
        
    except ValueError as e:
        error_message = str(e)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Сохраняем ошибку в историю
        history_item = models.RequestHistory(
            request_type="json",
            request_data={"data_length": len(request.data) if request.data else 0},
            response_data=None,
            processing_time_ms=processing_time_ms,
            success=False,
            error_message=error_message,
            input_size=len(request.data) * len(request.data[0]) if request.data else None
        )
        db.add(history_item)
        db.commit()
        
        return JSONResponse(
            status_code=403,
            content={"error": "модель не смогла обработать данные"}
        )
    except Exception as e:
        error_message = str(e)
        processing_time_ms = (time.time() - start_time) * 1000
        
        history_item = models.RequestHistory(
            request_type="json",
            request_data={"data_length": len(request.data) if request.data else 0},
            response_data=None,
            processing_time_ms=processing_time_ms,
            success=False,
            error_message=error_message
        )
        db.add(history_item)
        db.commit()
        
        return JSONResponse(
            status_code=403,
            content={"error": "модель не смогла обработать данные"}
        )


async def _forward_image(
    image_data: bytes,
    x_asset: Optional[str],
    x_window_size: Optional[str],
    db: Session
):
    """Обработка запроса с изображением"""
    start_time = time.time()
    
    try:
        if not image_data:
            return JSONResponse(
                status_code=400,
                content={"error": "bad request"}
            )
        
        # Получаем ML сервис
        ml_service = get_ml_service()
        
        # Обрабатываем изображение
        image_base64, metadata = ml_service.predict_image(image_data)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Сохраняем в историю
        headers_data = {}
        if x_asset:
            headers_data["X-Asset"] = x_asset
        if x_window_size:
            headers_data["X-Window-Size"] = x_window_size
        
        history_item = models.RequestHistory(
            request_type="image",
            request_headers=headers_data,
            response_data={"metadata": metadata},
            processing_time_ms=processing_time_ms,
            success=True,
            input_size=len(image_data)
        )
        db.add(history_item)
        db.commit()
        
        return schemas.ForwardResponseImage(
            image_base64=image_base64,
            metadata=metadata
        )
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        
        history_item = models.RequestHistory(
            request_type="image",
            response_data=None,
            processing_time_ms=processing_time_ms,
            success=False,
            error_message=str(e),
            input_size=len(image_data) if 'image_data' in locals() else None
        )
        db.add(history_item)
        db.commit()
        
        return JSONResponse(
            status_code=403,
            content={"error": "модель не смогла обработать данные"}
        )


@app.get("/history", response_model=schemas.HistoryResponse)
async def get_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    GET /history - получение истории всех запросов
    """
    total = db.query(models.RequestHistory).count()
    items = db.query(models.RequestHistory)\
        .order_by(models.RequestHistory.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return schemas.HistoryResponse(
        total=total,
        items=[schemas.HistoryItem.model_validate(item) for item in items]
    )


@app.delete("/history")
async def delete_history(
    token: str = Header(..., alias="X-Delete-Token"),
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin_user)
):
    """
    DELETE /history - удаление истории запросов (PRO)
    Требует токен подтверждения в заголовке X-Delete-Token
    """
    # Проверка токена подтверждения
    expected_token = os.getenv("DELETE_CONFIRM_TOKEN", "DELETE_CONFIRM_TOKEN")
    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Неверный токен подтверждения"
        )
    
    # Удаление всех записей истории
    deleted_count = db.query(models.RequestHistory).delete()
    db.commit()
    
    return {"message": f"Удалено записей: {deleted_count}"}


@app.get("/stats", response_model=schemas.StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """
    GET /stats - статистика запросов (PRO)
    Возвращает среднее время обработки, квантили, характеристики входных данных
    """
    # Общая статистика
    total_requests = db.query(models.RequestHistory).count()
    successful_requests = db.query(models.RequestHistory)\
        .filter(models.RequestHistory.success == True)\
        .count()
    failed_requests = total_requests - successful_requests
    
    # Статистика времени обработки
    processing_times = db.query(models.RequestHistory.processing_time_ms)\
        .filter(models.RequestHistory.processing_time_ms.isnot(None))\
        .all()
    
    if processing_times:
        times = [t[0] for t in processing_times]
        avg_time = np.mean(times)
        quantiles = {
            "mean": float(avg_time),
            "50%": float(np.percentile(times, 50)),
            "95%": float(np.percentile(times, 95)),
            "99%": float(np.percentile(times, 99))
        }
    else:
        quantiles = {
            "mean": 0.0,
            "50%": 0.0,
            "95%": 0.0,
            "99%": 0.0
        }
    
    # Статистика размеров входных данных
    input_sizes = db.query(models.RequestHistory.input_size)\
        .filter(models.RequestHistory.input_size.isnot(None))\
        .all()
    
    input_size_stats = None
    if input_sizes:
        sizes = [s[0] for s in input_sizes]
        input_size_stats = {
            "mean": float(np.mean(sizes)),
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
            "median": float(np.median(sizes))
        }
    
    return schemas.StatsResponse(
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        average_processing_time_ms=quantiles["mean"],
        processing_time_quantiles=quantiles,
        input_size_stats=input_size_stats
    )


@app.post("/token", response_model=Token)
async def login_for_access_token(
    username: str,
    password: str,
    db: Session = Depends(get_db)
):
    """Получение JWT токена для авторизации"""
    user = authenticate_user(db, username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username, "is_admin": user.is_admin},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
    admin_user: models.User = Depends(get_current_admin_user)
):
    """Создание нового пользователя (только для администраторов)"""
    # Проверка существования пользователя
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Пользователь с таким именем уже существует"
        )
    
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        hashed_password=hashed_password,
        is_admin=user.is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "ML Service для прогнозирования криптовалют",
        "version": "1.0.0",
        "endpoints": {
            "POST /forward": "Предсказание (JSON или multipart/form-data с изображением)",
            "GET /history": "История запросов",
            "DELETE /history": "Удаление истории (требует авторизации)",
            "GET /stats": "Статистика запросов",
            "POST /token": "Получение JWT токена",
            "POST /users": "Создание пользователя (требует авторизации)"
        }
    }

