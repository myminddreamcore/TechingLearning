
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from . import config
from .classifier import ImageClassifier, ImageDecodeError, decode_image
from .schemas import Health, ModelInfo, PredictionOut

classifier: Optional[ImageClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = ImageClassifier(config.MODEL_PATH, config.LABELS_PATH, config.BACKEND, config.RESIZE_MODE, config.MIN_CONFIDENCE)
    yield


app = FastAPI(title="API распознавания изображений", version="1.0",
              description="Принимает изображение и возвращает результат распознавания (класс) и точность — уверенность модели.",
              lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    return {"service": "API распознавания изображений", "docs": "/docs", "predict": "POST /predict (multipart/form-data, поле file)"}


@app.get("/health", response_model=Health, tags=["Служебные"], summary="Проверка работы API")
def health():
    return Health(status="ok", model_loaded=classifier is not None)


@app.get("/model/info", response_model=ModelInfo, tags=["Служебные"], summary="Сведения о модели: классы, размер входа, рантайм")
def model_info():
    return classifier.info()


@app.post("/predict", response_model=PredictionOut, tags=["Распознавание"],
          summary="Распознать изображение: результат и точность")
async def predict(file: UploadFile = File(..., description="Изображение (JPEG, PNG, BMP, GIF, WEBP…)"),
                  top_k: Optional[int] = Query(None, ge=1, le=100, description="Сколько лучших классов вернуть")):
    limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(400, "Файл пустой")
    if len(data) > limit:
        raise HTTPException(413, f"Файл больше {config.MAX_UPLOAD_MB:g} МБ")
    try:
        image = decode_image(data)
    except ImageDecodeError as e:
        raise HTTPException(400, str(e))
    result = await run_in_threadpool(classifier.predict, image, top_k)      #вычисления не блокируют сервер
    return PredictionOut(filename=file.filename or "image", **result)
