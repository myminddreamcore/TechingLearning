
import mimetypes
from pathlib import Path

import requests


class ApiError(Exception):
    ""


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: int = 60):
        self.base, self.timeout = base_url.rstrip("/"), timeout

    def _request(self, method: str, path: str, **kw):
        try:
            r = requests.request(method, self.base + path, timeout=self.timeout, **kw)
        except requests.ConnectionError:
            raise ApiError(f"Не удаётся подключиться к API ({self.base}). Запущен ли сервер: uvicorn api.main:app ?")
        except requests.Timeout:
            raise ApiError("Сервер слишком долго отвечает (таймаут).")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            if isinstance(detail, list):
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            raise ApiError(f"Ошибка {r.status_code}: {detail}")
        return r.json()

    def health(self):
        return self._request("GET", "/health")

    def info(self):
        return self._request("GET", "/model/info")

    def predict(self, path: str, top_k: int = None):
        p = Path(path)
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        with open(p, "rb") as f:
            files = {"file": (p.name, f, mime)}
            return self._request("POST", "/predict", files=files, params={"top_k": top_k} if top_k else None)
