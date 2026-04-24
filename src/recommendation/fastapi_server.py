# fastapi_server.py
# REST API сервер для Java Monolith приложения
# Расположение: src/recommendation/fastapi_server.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
import sys
import os
import joblib
import pandas as pd
import numpy as np

# Добавляем путь для импорта predict.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем нашу модель
from predict import get_recommendation_advanced, load_advanced_model

# ============================================================
# НАСТРОЙКИ
# ============================================================

app = FastAPI(
    title="Blood Donation AI Recommendation API",
    description="API для персонализированных рекомендаций по донации крови",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Настройка CORS для Java Monolith
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на URL вашего Java приложения
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# МОДЕЛИ ДАННЫХ (DTO)
# ============================================================

class DonorRequest(BaseModel):
    """Запрос от Java приложения с данными донора"""
    age: int = Field(..., ge=18, le=65, description="Возраст донора (18-65 лет)")
    gender: int = Field(..., ge=0, le=1, description="Пол: 0=женщина, 1=мужчина")
    blood_type: str = Field(..., pattern="^(A|B|AB|O)[+-]$", description="Группа крови: A+, A-, B+, B-, AB+, AB-, O+, O-")
    height_cm: float = Field(..., ge=100, le=250, description="Рост в сантиметрах")
    weight_kg: float = Field(..., ge=30, le=200, description="Вес в килограммах")
    hemoglobin: float = Field(..., ge=8, le=20, description="Уровень гемоглобина (г/дл)")
    ferritin: Optional[float] = Field(None, ge=5, le=500, description="Уровень ферритина (мкг/л)")
    prev_donations: int = Field(0, ge=0, le=100, description="Количество предыдущих донаций")
    avg_interval_days: Optional[int] = Field(None, ge=30, le=365, description="Средний интервал между донациями (дни)")
    low_hgb_history: int = Field(0, ge=0, le=1, description="Были ли проблемы с низким гемоглобином: 0=нет, 1=да")

    @validator('hemoglobin')
    def validate_hemoglobin(cls, v, values):
        gender = values.get('gender')
        if gender == 1 and v < 11:
            raise ValueError(f'Гемоглобин {v} слишком низкий для мужчины (минимум 11 для донации)')
        if gender == 0 and v < 10.5:
            raise ValueError(f'Гемоглобин {v} слишком низкий для женщины (минимум 10.5 для донации)')
        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "gender": 1,
                "blood_type": "O+",
                "height_cm": 180,
                "weight_kg": 75,
                "hemoglobin": 15.2,
                "ferritin": 120,
                "prev_donations": 5,
                "avg_interval_days": 95,
                "low_hgb_history": 0
            }
        }


class DonorResponse(BaseModel):
    """Ответ API с рекомендацией"""
    success: bool
    donor_id: Optional[int] = None
    next_donation_days: int
    ready_soon: bool
    readiness_level: str  # green, yellow, red
    readiness_text: str
    health_advice: str
    confidence: float
    bmi: float
    bmi_category: str
    predicted_at: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "next_donation_days": 90,
                "ready_soon": True,
                "readiness_level": "green",
                "readiness_text": "✅ Готов к донации",
                "health_advice": "Вы можете сдавать кровь. Хорошие показатели.",
                "confidence": 0.92,
                "bmi": 23.1,
                "bmi_category": "normal"
            }
        }


class BatchRequest(BaseModel):
    """Пакетный запрос для нескольких доноров"""
    donors: List[DonorRequest]


class BatchResponse(BaseModel):
    """Пакетный ответ"""
    success: bool
    total: int
    recommendations: List[DonorResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Ответ проверки здоровья сервиса"""
    status: str
    model_loaded: bool
    model_accuracy_days: Optional[float] = None
    version: str
    timestamp: datetime


# ============================================================
# ЗАГРУЗКА МОДЕЛИ ПРИ СТАРТЕ
# ============================================================

model_loaded = False
model_accuracy = None


@app.on_event("startup")
async def startup_event():
    """Загружаем модель при запуске сервера"""
    global model_loaded, model_accuracy
    try:
        _, metadata, _, _ = load_advanced_model()
        model_loaded = True
        model_accuracy = metadata.get('mae', None)
        print("✅ Модель загружена успешно!")
        print(f"   Точность модели: ±{model_accuracy} дней" if model_accuracy else "")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        model_loaded = False


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", tags=["Health"])
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "Blood Donation AI Recommendation",
        "version": "2.0.0",
        "status": "running",
        "docs": "/api/docs"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Проверка состояния сервиса"""
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_accuracy_days=model_accuracy,
        version="2.0.0",
        timestamp=datetime.now()
    )


@app.post("/api/recommend", response_model=DonorResponse, tags=["Recommendations"])
async def get_recommendation(donor: DonorRequest):
    """
    Получить персонализированную рекомендацию для донора

    - **Java Monolith** вызывает этот эндпоинт с данными донора
    - **ИИ модель** анализирует параметры и возвращает рекомендацию
    - **Результат** можно сохранить в PostgreSQL или вернуть пользователю
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        # Подготовка параметров (заполняем значения по умолчанию)
        ferritin = donor.ferritin if donor.ferritin is not None else 80
        avg_interval = donor.avg_interval_days if donor.avg_interval_days is not None else 90

        # Вызов нашей модели
        result = get_recommendation_advanced(
            age=donor.age,
            gender=donor.gender,
            blood_type=donor.blood_type,
            height_cm=donor.height_cm,
            weight_kg=donor.weight_kg,
            hemoglobin=donor.hemoglobin,
            ferritin=ferritin,
            prev_donations=donor.prev_donations,
            avg_interval_days=avg_interval,
            low_hgb_history=donor.low_hgb_history
        )

        # Формируем ответ
        readiness_text_map = {
            'green': '✅ Готов к донации',
            'yellow': '⚠️ Требуется осторожность',
            'red': '🔴 Донация не рекомендуется'
        }

        return DonorResponse(
            success=True,
            next_donation_days=result['next_donation_days'],
            ready_soon=result['ready_soon'],
            readiness_level=result['readiness_level'],
            readiness_text=readiness_text_map.get(result['readiness_level'], 'Неизвестно'),
            health_advice=result['health_advice'],
            confidence=result.get('confidence', 0.85),
            bmi=result['bmi'],
            bmi_category=result['bmi_category']
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/recommend/batch", response_model=BatchResponse, tags=["Recommendations"])
async def get_batch_recommendations(batch: BatchRequest):
    """
    Получить рекомендации для нескольких доноров (пакетная обработка)

    Полезно для:
    - Загрузки истории доноров
    - Массовых рассылок
    - Отчётов
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    import time
    start_time = time.time()

    recommendations = []
    errors = 0

    for donor in batch.donors:
        try:
            ferritin = donor.ferritin if donor.ferritin is not None else 80
            avg_interval = donor.avg_interval_days if donor.avg_interval_days is not None else 90

            result = get_recommendation_advanced(
                age=donor.age,
                gender=donor.gender,
                blood_type=donor.blood_type,
                height_cm=donor.height_cm,
                weight_kg=donor.weight_kg,
                hemoglobin=donor.hemoglobin,
                ferritin=ferritin,
                prev_donations=donor.prev_donations,
                avg_interval_days=avg_interval,
                low_hgb_history=donor.low_hgb_history
            )

            readiness_text_map = {
                'green': '✅ Готов к донации',
                'yellow': '⚠️ Требуется осторожность',
                'red': '🔴 Донация не рекомендуется'
            }

            recommendations.append(DonorResponse(
                success=True,
                next_donation_days=result['next_donation_days'],
                ready_soon=result['ready_soon'],
                readiness_level=result['readiness_level'],
                readiness_text=readiness_text_map.get(result['readiness_level'], 'Неизвестно'),
                health_advice=result['health_advice'],
                confidence=result.get('confidence', 0.85),
                bmi=result['bmi'],
                bmi_category=result['bmi_category']
            ))
        except Exception as e:
            errors += 1
            print(f"Ошибка при обработке донора: {e}")

    processing_time = (time.time() - start_time) * 1000

    return BatchResponse(
        success=errors == 0,
        total=len(recommendations),
        recommendations=recommendations,
        processing_time_ms=round(processing_time, 2)
    )


@app.get("/api/stats", tags=["Stats"])
async def get_stats():
    """Получить статистику работы модели"""
    global model_accuracy

    return {
        "model_loaded": model_loaded,
        "model_accuracy_days": model_accuracy,
        "features_used": [
            "age", "gender", "blood_type", "bmi", "hemoglobin",
            "ferritin", "prev_donations", "avg_interval_days", "low_hgb_history"
        ],
        "api_version": "2.0.0"
    }


# ============================================================
# ЗАПУСК СЕРВЕРА
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ЗАПУСК FASTAPI СЕРВЕРА ДЛЯ JAVA MONOLITH")
    print("=" * 60)
    print("\nСервер будет доступен по адресу: http://localhost:8000")
    print("Документация API: http://localhost:8000/api/docs")
    print("\nДля Java Monolith используйте:")
    print("  POST http://localhost:8000/api/recommend")
    print("=" * 60)

    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload при изменениях кода (только для разработки)
        log_level="info"
    )