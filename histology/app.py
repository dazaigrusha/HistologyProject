"""Перегляд оригіналу, ручне розділення H&E та експорт каналів."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, UnidentifiedImageError
from histology.image_io import load_image
from histology.stains import separate_stains, transmittance_image


class Application(tk.Tk):
    """Вікно другої частини; обчислення виконуються поза потоком інтерфейсу."""

    def __init__(self, config: dict):
        super().__init__()
        self.config_data = config
        self.original = None
        self.source_path = None
        self.channels = {}
        self.maps = None
        self.photos = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.title('Histology — аналіз зображень тканин | Частина 2')
        self.geometry('1100x780')
        toolbar = ttk.Frame(self, padding=12)
        toolbar.pack(fill='x')
        self.open_button = ttk.Button(toolbar, text='Відкрити зображення', command=self.open_image)
        self.open_button.pack(side='left')
        self.run_button = ttk.Button(toolbar, text='Розділити H&E', command=self.run_separation,
                                     state='disabled')
        self.run_button.pack(side='left', padx=8)
        self.save_button = ttk.Button(toolbar, text='Зберегти канали', command=self.save_channels,
                                      state='disabled')
        self.save_button.pack(side='left')
        self.info = tk.StringVar(value='Вибери зображення тканини із забарвленням H&E.')
        ttk.Label(self, textvariable=self.info, padding=12, wraplength=1040).pack(fill='x')
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=12, pady=8)
        ttk.Label(self, text='H: гематоксилін | E: еозин. Темніші ділянки — більше барвника. '
                  'Це ще не маска ядер.', padding=12).pack(fill='x')
        self.protocol('WM_DELETE_WINDOW', self.close)

    def close(self):
        """Закрити вікно та звільнити виконавець фонових завдань."""
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def show_tabs(self):
        """Показати копії для перегляду без зміни розміру вихідних даних."""
        for tab in self.notebook.tabs():
            self.notebook.nametowidget(tab).destroy()
        self.photos = []
        for name, image in {'Оригінал': self.original, **self.channels}.items():
            preview = image.copy()
            preview.thumbnail((self.config_data['preview_width'],
                               self.config_data['preview_height']), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
            self.photos.append(photo)
            frame = ttk.Frame(self.notebook)
            ttk.Label(frame, image=photo, anchor='center').pack(fill='both', expand=True)
            self.notebook.add(frame, text=name)

    def open_image(self):
        """Відкрити файл; скинути старі канали тільки після успішного читання."""
        filename = filedialog.askopenfilename(title='Вибери зображення тканини', filetypes=[
            ('Зображення', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'), ('Усі файли', '*.*')])
        if not filename:
            return
        try:
            image = load_image(Path(filename), self.config_data['max_pixels'])
        except (OSError, ValueError, UnidentifiedImageError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            messagebox.showerror('Не вдалося відкрити зображення', str(error))
            return
        self.original = image
        self.source_path = Path(filename)
        self.channels = {}
        self.maps = None
        self.show_tabs()
        self.run_button.configure(state='normal')
        self.save_button.configure(state='disabled')
        self.info.set(f'{self.source_path.name} | {image.width} × {image.height} пікселів | RGB. '
                      'Для багатосторінкового TIFF відкрито перший кадр.')

    def run_separation(self):
        """Запустити розділення з оригіналу, не з екранної копії."""
        if self.original is None or self.future is not None:
            return
        self.open_button.configure(state='disabled')
        self.run_button.configure(state='disabled')
        self.save_button.configure(state='disabled')
        self.info.set('Виконується розділення H&E…')
        self.future = self.executor.submit(separate_stains, np.asarray(self.original),
                                           self.config_data['stains'])
        self.after(80, self.check_result)

    def check_result(self):
        """Забрати результат у головному потоці; Tkinter не викликається з worker."""
        if not self.future.done():
            self.after(80, self.check_result)
            return
        try:
            self.maps = self.future.result()
            self.channels = {
                'H — гематоксилін': Image.fromarray(transmittance_image(self.maps[0])),
                'E — еозин': Image.fromarray(transmittance_image(self.maps[1])),
            }
            self.show_tabs()
            self.notebook.select(1)
            self.info.set('Готово. Перемикай вкладки для порівняння. '
                          'Підрахунок ядер буде додано в наступних частинах.')
        except (ValueError, MemoryError) as error:
            messagebox.showerror('Помилка розділення', str(error))
            self.info.set('Не вдалося виконати розділення. Спробуй менше зображення.')
        finally:
            self.future = None
            self.open_button.configure(state='normal')
            self.run_button.configure(state='normal')
            self.save_button.configure(state='normal' if self.maps is not None else 'disabled')

    def save_channels(self):
        """Зберегти PNG, вихідні float-карти NPY та параметри в нову теку запуску."""
        if self.maps is None:
            return
        directory = filedialog.askdirectory(title='Вибери теку для результатів')
        if not directory:
            return
        # Унікальна тека запобігає перезапису попередніх запусків.
        from tempfile import mkdtemp
        try:
            destination = Path(mkdtemp(prefix='he_', dir=directory))
            for name, channel in zip(('hematoxylin', 'eosin'), self.maps):
                Image.fromarray(transmittance_image(channel)).save(destination / f'{name}.png')
                np.save(destination / f'{name}_od.npy', channel, allow_pickle=False)
            metadata = {'source': self.source_path.name, 'size': list(self.original.size),
                        'method': 'two_stain_od_least_squares_clipped',
                        'stains': self.config_data['stains'],
                        'png': 'uint8 transmittance: round(255*exp(-concentration))',
                        'npy': 'float32 nonnegative concentration, original resolution'}
            (destination / 'parameters.json').write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        except (OSError, ValueError) as error:
            messagebox.showerror('Помилка збереження', f'{error}\nМожливі часткові файли.')
            return
        self.info.set(f'Канали та параметри збережено: {destination}')
