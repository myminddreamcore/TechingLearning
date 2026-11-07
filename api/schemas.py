
from typing import List

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str = Field(..., description="Название класса")
    probability: float = Field(..., description="Вероятность класса, 0–1")
    percent: float = Field(..., description="Вероятность класса, %")


class PredictionOut(BaseModel):
    filename: str = Field(..., description="Имя загруженного файла")
    label: str = Field(..., description="Результат распознавания — наиболее вероятный класс")
    confidence: float = Field(..., description="Точность (уверенность модели) для этого класса, 0–1")
    confidence_percent: float = Field(..., description="Точность (уверенность модели), %")
    low_confidence: bool = Field(..., description="Уверенность ниже порога: результату стоит доверять с осторожностью")
    probabilities: List[ClassProbability] = Field(..., description="Вероятности классов по убыванию")
    inference_ms: float = Field(..., description="Время работы модели, мс")
    backend: str = Field(..., description="Чем выполнена модель")


class ModelInfo(BaseModel):
    labels: List[str]
    input_width: int
    input_height: int
    backend: str
    resize_mode: str
    min_confidence: float
    model_file: str


class Health(BaseModel):
    status: str
    model_loaded: bool
