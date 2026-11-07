
LABELS_RU = {"earth": "Земля", "sun": "Солнце", "moon": "Луна"}


def ru(label: str) -> str:
    return LABELS_RU.get(label.lower(), label)


def summarize(result: dict) -> dict:
    rows = [(ru(p["label"]), p["label"], float(p["percent"])) for p in result["probabilities"]]
    return {
        "title": ru(result["label"]),
        "original": result["label"],
        "confidence_text": f"Точность распознавания: {result['confidence_percent']:.1f}%",
        "confidence": float(result["confidence_percent"]),
        "warning": ("Модель не уверена в ответе: изображение мало похоже на известные классы."
                    if result.get("low_confidence") else ""),
        "rows": rows,
        "details": f"Время: {result['inference_ms']:.0f} мс · выполнено: {result['backend']}",
    }
