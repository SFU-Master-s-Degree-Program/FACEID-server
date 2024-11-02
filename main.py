# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
import numpy as np

from database import SessionLocal, engine, Base
import models
from utils import extract_embedding, serialize_embedding, deserialize_embedding

# Создаём таблицы в базе данных
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Система Распознавания Лиц для Контроля Доступа")

# Добавляем CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники (не рекомендуется для продакшена)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки
)


# Зависимость для получения сессии базы данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic модели
class EmployeeBase(BaseModel):
    first_name: str
    last_name: str


class EmployeeCreate(EmployeeBase):
    pass


class EmployeeResponse(EmployeeBase):
    id: int

    class Config:
        orm_mode = True


# Новый эндпоинт для полной очистки базы данных
@app.delete("/clear_database/")
async def clear_database(db: Session = Depends(get_db)):
    # Удаляем все таблицы
    Base.metadata.drop_all(bind=engine)
    # Пересоздаём таблицы
    Base.metadata.create_all(bind=engine)

    return JSONResponse(content={"message": "База данных очищена."})


@app.post("/register/")
async def register_employee(
        first_name: str = Form(...),
        last_name: str = Form(...),
        files: List[UploadFile] = File(...),  # Используем тип List[UploadFile] для нескольких файлов
        db: Session = Depends(get_db)
):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Необходимо загрузить хотя бы одну фотографию.")

    # Создаём нового сотрудника
    employee = models.Employee(
        first_name=first_name,
        last_name=last_name
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)

    # Обрабатываем каждую фотографию
    for file in files:
        image_bytes = await file.read()
        embedding = extract_embedding(image_bytes)
        if embedding is None:
            continue  # Можно также решить, как обрабатывать фотографии без обнаруженных лиц

        serialized_embedding = serialize_embedding(embedding)
        emp_embedding = models.Embedding(
            employee_id=employee.id,
            embedding=serialized_embedding
        )
        db.add(emp_embedding)

    db.commit()

    return JSONResponse(content={"message": "Сотрудник успешно зарегистрирован.", "employee_id": employee.id})


@app.post("/recognize/")
async def recognize_employee(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Читаем изображение
    image_bytes = await file.read()

    # Извлекаем эмбеддинг
    embedding = extract_embedding(image_bytes)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Лицо не обнаружено на изображении.")

    # Получаем все эмбеддинги из базы данных
    embeddings = db.query(models.Embedding).all()

    if not embeddings:
        raise HTTPException(status_code=404, detail="Нет зарегистрированных сотрудников.")

    # Подготовка данных для сравнения
    stored_embeddings = np.array([deserialize_embedding(emp.embedding).flatten() for emp in embeddings])
    query_embedding = embedding.flatten()

    # Нормализация эмбеддингов для косинусного расстояния
    stored_norms = np.linalg.norm(stored_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        raise HTTPException(status_code=400, detail="Не удалось извлечь эмбеддинг из изображения.")

    # Косинусное расстояние
    cosine_sim = np.dot(stored_embeddings, query_embedding) / (stored_norms * query_norm)

    # Находим лучший матч
    best_match_idx = np.argmax(cosine_sim)
    best_similarity = cosine_sim[best_match_idx]
    best_embedding = embeddings[best_match_idx]

    # Устанавливаем порог для распознавания (можно настроить)
    threshold = 0.8  # Значение близкое к 1 означает высокую схожесть

    if best_similarity >= threshold:
        employee = db.query(models.Employee).filter(models.Employee.id == best_embedding.employee_id).first()
        full_name = f"{employee.first_name} {employee.last_name}"
        return JSONResponse(content={
            "matched": True,
            "name": full_name,
            "similarity": float(best_similarity)
        })
    else:
        return JSONResponse(content={"matched": False, "detail": "Сотрудник не распознан."})


@app.get("/employees/", response_model=List[EmployeeResponse])
async def get_employees(db: Session = Depends(get_db)):
    employees = db.query(models.Employee).all()
    return employees
