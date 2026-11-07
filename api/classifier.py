import io
import threading
import time
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 60_000_000#защита от огромных изображений, которые съедят память


class ImageDecodeError(ValueError):
    ""


def decode_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))#открываем картинку из байтов в памяти, без временных файлов
        img.load()
    except Exception as e:
        raise ImageDecodeError("Файл не является поддерживаемым изображением (JPEG, PNG, BMP, GIF, WEBP…)") from e
    img = ImageOps.exif_transpose(img)#поворачиваем по EXIF, иначе фото с телефона будут боком
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):#режимы с прозрачностью
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))#белая подложка под прозрачные пиксели
        bg.paste(rgba, mask=rgba.split()[3])#вклеиваем только непрозрачные части
        return bg
    return img.convert("RGB")


class _OfficialRuntime:

    def __init__(self, model_path: str):
        Interp = None
        for mod, name in (("ai_edge_litert.interpreter", "Interpreter"), ("tflite_runtime.interpreter", "Interpreter"),
                          ("tensorflow.lite", "Interpreter")):
            try:
                Interp = getattr(__import__(mod, fromlist=[name]), name)#пробуем импортировать конкретный класс из модуля
                self.name = mod.split(".")[0]
                break
            except Exception:
                continue
        if Interp is None:
            raise ImportError("официальный TFLite-рантайм не установлен")
        self.it = Interp(model_path=str(model_path))
        self.it.allocate_tensors()#выделяем память под тензоры — обязательно до первого invoke
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]#берём первый вход и первый выход
        self.input_shape = [int(i) for i in self.inp["shape"]]#shape приходит как кортеж numpy-чисел, приводим к int

    def run(self, x: np.ndarray) -> np.ndarray:
        self.it.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))#приводим к типу входа модели (float32)
        self.it.invoke()
        return self.it.get_tensor(self.out["index"])


class _NumpyRuntime:
    name = "numpy (встроенный)"

    def __init__(self, model_path: str):
        from .tflite_numpy import Interpreter
        self.it = Interpreter(Path(model_path).read_bytes())#читаем .tflite целиком в память и отдаём интерпретатору
        self.input_shape = self.it.input_shape

    def run(self, x: np.ndarray) -> np.ndarray:
        return self.it.run(x)


class ImageClassifier:
    def __init__(self, model_path, labels_path=None, backend: str = "auto", resize_mode: str = "stretch",
                 min_confidence: float = 0.6):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
        self.resize_mode, self.min_confidence = resize_mode, min_confidence
        self.labels = self._load_labels(labels_path)
        self.runtime = self._load_runtime(backend)
        self.backend_name = self.runtime.name
        _, self.height, self.width, _ = self.runtime.input_shape#формат NHWC: (batch, height, width, channels)
        self._lock = threading.Lock()#один инстанс интерпретатора нельзя дёргать из нескольких потоков одновременно
        out = self.runtime.run(np.zeros((1, self.height, self.width, 3), np.float32))#пробный прогон, чтобы узнать число классов
        if out.shape[-1] != len(self.labels):
            raise ValueError(f"Число классов модели ({out.shape[-1]}) не совпадает с числом меток ({len(self.labels)})")

    def _load_labels(self, labels_path) -> List[str]:
        if labels_path and Path(labels_path).exists():
            text = Path(labels_path).read_text(encoding="utf-8")
        else:
            with zipfile.ZipFile(self.model_path) as z:
                text = z.read("labels.txt").decode("utf-8")
        labels = [l.strip() for l in text.splitlines() if l.strip()]
        if not labels:
            raise ValueError("Файл меток пуст")
        return labels

    def _load_runtime(self, backend: str):
        if backend in ("auto", "litert"):
            try:
                return _OfficialRuntime(self.model_path)
            except Exception as e:
                raise ValueError("Файл меток пуст")
        return _NumpyRuntime(self.model_path)

    def preprocess(self, img: Image.Image) -> np.ndarray:
        if self.resize_mode == "crop":
            img = ImageOps.fit(img, (self.width, self.height), Image.BILINEAR)#crop: заполняет кадр с обрезкой лишнего
        else:
            img = img.resize((self.width, self.height), Image.BILINEAR)#stretch: просто растягиваем, меняя пропорции
        return np.asarray(img, dtype=np.float32)[None, ...]#добавляем batch-измерение: (H,W,3) -> (1,H,W,3)

    def predict(self, img: Image.Image, top_k: Optional[int] = None) -> dict:
        x = self.preprocess(img)
        t0 = time.perf_counter()#монотонные часы высокого разрешения — для замеров времени
        with self._lock:
            probs = np.asarray(self.runtime.run(x), dtype=np.float64).ravel()#ravel: (1,N) -> (N,)
        ms = (time.perf_counter() - t0) * 1000
        order = np.argsort(-probs)#минус — чтобы сортировка шла по убыванию вероятности
        ranked = [{"label": self.labels[i], "probability": round(float(probs[i]), 6), "percent": round(float(probs[i]) * 100, 2)}
                  for i in order]
        best = ranked[0]
        return {"label": best["label"], "confidence": best["probability"], "confidence_percent": best["percent"],
                "low_confidence": best["probability"] < self.min_confidence,
                "probabilities": ranked[:top_k] if top_k else ranked,
                "inference_ms": round(ms, 1), "backend": self.backend_name}

    def info(self) -> dict:
        return {"labels": self.labels, "input_width": self.width, "input_height": self.height, "backend": self.backend_name,
                "resize_mode": self.resize_mode, "min_confidence": self.min_confidence, "model_file": self.model_path.name}