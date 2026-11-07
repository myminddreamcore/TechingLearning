# Распознавание изображений: API (FastAPI) + приложение (Tkinter)

Готовая модель `model/model.tflite`.
Ничего не обучается — API только запускает готовую модель.

## Запуск (PyCharm или терминал, из корня проекта)
1. Откройте папку `ml_image_app` как проект, создайте виртуальное окружение, выполните `pip install -r requirements.txt`.
2. Окно 1 — сервер API: `uvicorn api.main:app --reload`
   Документация и проба запросов в браузере: http://127.0.0.1:8000/docs
3. Окно 2 — приложение: `python -m app.main` (справка: F1 или `python -m app.main --help`).
   Нажмите «Выбрать изображение…» — результат и точность появятся справа.

## API
| Метод | Адрес | Что делает |
|---|---|---|
| POST | `/predict` | принимает изображение (multipart, поле `file`; необязательный `?top_k=`), возвращает класс и точность |
| GET | `/model/info` | классы, размер входа, чем выполняется модель |
| GET | `/health` | проверка работы |

Пример ответа `/predict`:
```json
{"filename": "sun.png", "label": "sun", "confidence": 0.9731, "confidence_percent": 97.31, "low_confidence": false,
 "probabilities": [{"label": "sun", "probability": 0.9731, "percent": 97.31}, ...], "inference_ms": 92.4, "backend": "numpy (встроенный)"}
```
Пример из терминала: `curl -F "file=@samples/sun_sample.png" http://127.0.0.1:8000/predict`

Ошибки: 400 — файл не изображение или пустой; 413 — файл больше `MAX_UPLOAD_MB`.

## Что важно знать
- **«Точность распознавания»** = уверенность модели (вероятность выбранного класса). Если уверенность ниже `MIN_CONFIDENCE` (0.6), API возвращает `low_confidence: true`, приложение показывает предупреждение.

