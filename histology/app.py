"""Український інтерфейс відкриття та перегляду зображення."""
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, UnidentifiedImageError
from histology.image_io import load_image


class Application(tk.Tk):
    """Вікно першого етапу: оригінал і відомості про файл."""

    def __init__(self, config: dict):
        super().__init__()
        self.config_data = config
        self.original = None
        self.photo = None
        self.title('Histology — аналіз зображень тканин | Частина 1')
        self.geometry('1050x760')
        toolbar = ttk.Frame(self, padding=12)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='Відкрити зображення', command=self.open_image).pack(side='left')
        ttk.Label(toolbar, text='Частина 1: завантаження та перегляд').pack(side='right')
        self.info = tk.StringVar(value='Вибери гістологічне зображення PNG, JPEG, BMP або TIFF.')
        ttk.Label(self, textvariable=self.info, padding=12, wraplength=980).pack(fill='x')
        self.preview = ttk.Label(self, anchor='center')
        self.preview.pack(fill='both', expand=True, padx=12, pady=12)
        ttk.Label(self, text='Алгоритми виділення ядер будуть додані в наступних частинах.',
                  padding=12).pack(fill='x')

    def open_image(self):
        """Вибрати файл; помилка не видаляє попереднє відкрите зображення."""
        filename = filedialog.askopenfilename(title='Вибери зображення тканини', filetypes=[
            ('Зображення', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'), ('Усі файли', '*.*')])
        if not filename:
            return
        try:
            image = load_image(Path(filename), self.config_data['max_pixels'])
            # Зменшуємо лише копію для екрана. Оригінал зберігає всі пікселі.
            preview = image.copy()
            preview.thumbnail((self.config_data['preview_width'],
                               self.config_data['preview_height']), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
        except (OSError, ValueError, UnidentifiedImageError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            messagebox.showerror('Не вдалося відкрити зображення', str(error))
            return
        self.original = image
        self.photo = photo  # Посилання потрібне, щоб Tkinter не видалив картинку.
        self.preview.configure(image=self.photo)
        self.info.set(f'{Path(filename).name} | {image.width} × {image.height} пікселів | RGB. '
                      'Для багатосторінкового TIFF відкрито перший кадр.')
