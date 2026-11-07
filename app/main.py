import argparse
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .api_client import ApiClient, ApiError
from .formatting import summarize

PREVIEW = (460, 460)
FILETYPES = [("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"), ("Все файлы", "*.*")]
HELP = ("Как пользоваться:\n"
        "1. Запустите API:  uvicorn api.main:app --reload\n"
        "2. Нажмите «Выбрать изображение…» и укажите файл — распознавание запустится автоматически.\n"
        "3. Справа появятся результат распознавания и точность по каждому классу.\n"
        "Кнопка «Распознать» повторяет распознавание, «Очистить» сбрасывает окно.")


class App(tk.Tk):
    def __init__(self, api_url: str):
        super().__init__()
        self.title("Распознавание изображений")
        self.geometry("1000x640")
        self.minsize(900, 560)
        self.api = ApiClient(api_url)
        self.path = None
        self.photo = None
        self.q = queue.Queue()
        self.bars = []
        self._build_ui()
        self.after(100, self._poll)
        self.check_connection()

    def run_async(self, func, on_ok, on_error=None):
        def work():
            try:
                self.q.put((on_ok, func(), None, on_error))
            except ApiError as e:
                self.q.put((on_ok, None, str(e), on_error))
            except Exception as e:
                self.q.put((on_ok, None, f"Непредвиденная ошибка: {e}", on_error))
        threading.Thread(target=work, daemon=True).start()#daemon: поток не держит процесс после закрытия окна

    def _poll(self):
        try:
            while True:
                on_ok, result, error, on_error = self.q.get_nowait()
                if error:
                    (on_error or self._show_error)(error)
                else:
                    on_ok(result)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        self.url_entry = ttk.Entry(top, width=34)
        self.url_entry.insert(0, self.api.base)
        self.url_entry.pack(side="left", padx=6)
        ttk.Button(top, text="Проверить связь", command=self.check_connection).pack(side="left")
        self.conn_label = ttk.Label(top, text="")
        self.conn_label.pack(side="left", padx=12)
        ttk.Button(top, text="Справка", command=lambda: messagebox.showinfo("Справка", HELP)).pack(side="right")

        body = ttk.Frame(self, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)
        left = ttk.LabelFrame(body, text="Изображение", padding=8)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, width=PREVIEW[0], height=PREVIEW[1], bg="#f2f2f2", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack()
        self.canvas.create_text(PREVIEW[0] // 2, PREVIEW[1] // 2, text="Изображение не выбрано", fill="#888", font=("", 12), tags="hint")#tags="hint" — чтобы потом можно было удалить подсказку по тегу
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Выбрать изображение…", command=self.choose).pack(side="left")
        self.recognize_btn = ttk.Button(btns, text="Распознать", command=self.recognize, state="disabled")
        self.recognize_btn.pack(side="left", padx=8)
        ttk.Button(btns, text="Очистить", command=self.clear).pack(side="left")
        self.file_label = ttk.Label(left, text="", wraplength=440, foreground="#555")
        self.file_label.pack(anchor="w")

        right = ttk.LabelFrame(body, text="Результат распознавания", padding=12)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.title_var = tk.StringVar(value="—")
        ttk.Label(right, textvariable=self.title_var, font=("", 30, "bold")).pack(anchor="w")
        self.orig_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.orig_var, foreground="#777").pack(anchor="w")
        self.conf_var = tk.StringVar(value="Точность распознавания: —")
        ttk.Label(right, textvariable=self.conf_var, font=("", 13)).pack(anchor="w", pady=(14, 4))
        self.conf_bar = ttk.Progressbar(right, maximum=100, length=360)
        self.conf_bar.pack(anchor="w")
        self.warn_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.warn_var, foreground="#b00", wraplength=380, justify="left").pack(anchor="w", pady=6)
        ttk.Label(right, text="Вероятности по классам:").pack(anchor="w", pady=(10, 4))
        self.rows_frame = ttk.Frame(right)
        self.rows_frame.pack(anchor="w", fill="x")
        self.details_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.details_var, foreground="#777").pack(anchor="w", pady=(14, 0))

        self.status = tk.StringVar(value="Выберите изображение для распознавания")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=3).pack(fill="x", side="bottom")

    def _sync_url(self):
        url = self.url_entry.get().strip()
        if url and url.rstrip("/") != self.api.base:
            self.api = ApiClient(url)

    def check_connection(self):
        self._sync_url()
        self.conn_label.configure(text="проверка…", foreground="#555")
        self.run_async(self.api.info, self._connected,
                       lambda msg: self.conn_label.configure(text="нет связи с API", foreground="#b00") or self.status.set(msg))

    def _connected(self, info):
        classes = ", ".join(info["labels"])
        self.conn_label.configure(text=f"API работает (классы: {classes})", foreground="#060")

    def choose(self):
        path = filedialog.askopenfilename(title="Выберите изображение", filetypes=FILETYPES)
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()
        except Exception:
            messagebox.showerror("Ошибка", "Не удалось открыть файл как изображение.")
            return
        self.path = path
        img.thumbnail(PREVIEW)#уменьшает на месте, сохраняя пропорции
        self.photo = ImageTk.PhotoImage(img)#ссылку нужно хранить: иначе картинка исчезнет
        self.canvas.delete("all")
        self.canvas.create_image(PREVIEW[0] // 2, PREVIEW[1] // 2, image=self.photo)
        self.file_label.configure(text=os.path.basename(path))
        self.recognize_btn.configure(state="normal")
        self._reset_result()
        self.recognize()#сразу запускаем распознавание после выбора файла

    def recognize(self):
        if not self.path:
            return
        self._sync_url()
        path = self.path
        self.status.set("Распознавание…")
        self.recognize_btn.configure(state="disabled")
        self.run_async(lambda: self.api.predict(path), self._show_result, self._recognition_failed)

    def _show_result(self, result):
        s = summarize(result)
        self.title_var.set(s["title"])
        self.orig_var.set(f"({s['original']})")
        self.conf_var.set(s["confidence_text"])
        self.conf_bar["value"] = s["confidence"]
        self.warn_var.set(s["warning"])
        self.details_var.set(s["details"])
        for w in self.rows_frame.winfo_children():
            w.destroy()#пересоздаём строки с нуля, чтобы не накапливать старые виджеты
        for name, _orig, percent in s["rows"]:
            row = ttk.Frame(self.rows_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name, width=10).pack(side="left")
            bar = ttk.Progressbar(row, maximum=100, length=220)
            bar["value"] = percent
            bar.pack(side="left", padx=6)
            ttk.Label(row, text=f"{percent:.1f}%", width=8).pack(side="left")
        self.recognize_btn.configure(state="normal")
        self.status.set("Готово")

    def _recognition_failed(self, message):
        self.recognize_btn.configure(state="normal" if self.path else "disabled")
        self.status.set(message)
        messagebox.showerror("Ошибка распознавания", message)

    def _show_error(self, message):
        self.status.set(message)
        messagebox.showerror("Ошибка", message)

    def _reset_result(self):
        self.title_var.set("—")
        self.orig_var.set("")
        self.conf_var.set("Точность распознавания: —")
        self.conf_bar["value"] = 0
        self.warn_var.set("")
        self.details_var.set("")
        for w in self.rows_frame.winfo_children():
            w.destroy()

    def clear(self):
        self.path = None
        self.photo = None
        self.canvas.delete("all")
        self.canvas.create_text(PREVIEW[0] // 2, PREVIEW[1] // 2, text="Изображение не выбрано", fill="#888", font=("", 12))
        self.file_label.configure(text="")
        self.recognize_btn.configure(state="disabled")
        self._reset_result()
        self.status.set("Выберите изображение для распознавания")


def main():
    ap = argparse.ArgumentParser(description="Приложение для распознавания изображений (работает через API)", epilog=HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)#RawDescriptionHelpFormatter — чтобы переносы строк в HELP сохранились
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://127.0.0.1:8000"), help="адрес API-сервера")
    App(ap.parse_args().api_url).mainloop()


if __name__ == "__main__":
    main()