"""Інтерфейс H&E, сегментації, попереднього підрахунку та експорту."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, UnidentifiedImageError
from histology.analysis import distance_image, label_image, marker_image
from histology.texture import histogram_image
from histology.texture_export import write_texture
from histology.metrics import evaluate_result
from histology.batch import run_batch
from histology.workflow import analyze_image, split_image, texture_image
from histology.export import save_results
from histology.image_io import load_image
from histology.stains import separate_stains, transmittance_image


def save_texture_only(directory: Path, texture: dict) -> Path:
    """Зберегти gray-LBP без необхідності запускати H&E."""
    from tempfile import mkdtemp
    destination = Path(mkdtemp(prefix='lbp_', dir=directory))
    write_texture(destination, texture)
    return destination


class Application(tk.Tk):
    """Вікно шостої частини; усі звернення до Tkinter — в головному потоці."""

    def __init__(self, config: dict):
        super().__init__()
        self.config_data = config
        self.original = None
        self.source_path = None
        self.channels = {}
        self.maps = None
        self.result = None
        self.texture = None
        self.photos = []
        self.views = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.job = None
        self.title('Histology — аналіз зображень тканин | Частина 6')
        self.geometry('1140x850')
        toolbar = ttk.Frame(self, padding=12)
        toolbar.pack(fill='x')
        self.open_button = ttk.Button(toolbar, text='Відкрити зображення', command=self.open_image)
        self.open_button.pack(side='left')
        self.run_button = ttk.Button(toolbar, text='1. Розділити H&E', command=self.run_separation)
        self.run_button.pack(side='left', padx=8)
        self.segment_button = ttk.Button(toolbar, text='2. Знайти кандидатів', command=self.run_segmentation)
        self.segment_button.pack(side='left')
        self.watershed_button = ttk.Button(toolbar, text='3. Watershed', command=self.run_watershed)
        self.watershed_button.pack(side='left', padx=8)
        self.texture_button = ttk.Button(toolbar, text='4. Текстура LBP', command=self.run_texture)
        self.texture_button.pack(side='left')
        self.save_button = ttk.Button(toolbar, text='Зберегти результати', command=self.save_channels)
        self.save_button.pack(side='left', padx=8)
        extra_toolbar = ttk.Frame(self, padding=(12, 0))
        extra_toolbar.pack(fill='x')
        self.evaluate_button = ttk.Button(extra_toolbar, text='Порівняти з еталоном', command=self.run_evaluation)
        self.evaluate_button.pack(side='left')
        self.batch_button = ttk.Button(extra_toolbar, text='Обробити вибірку CSV', command=self.run_batch)
        self.batch_button.pack(side='left', padx=8)
        self.metric_text = tk.StringVar(value='Метрики: еталон ще не вибрано.')
        ttk.Label(extra_toolbar, textvariable=self.metric_text).pack(side='left', padx=8)
        self.info = tk.StringVar(value='Вибери зображення тканини із забарвленням H&E.')
        ttk.Label(self, textvariable=self.info, padding=12, wraplength=1080).pack(fill='x')
        self.summary = tk.StringVar(value='Кандидатів: —')
        ttk.Label(self, textvariable=self.summary, padding=(12, 0)).pack(fill='x')
        self.view_name = tk.StringVar()
        self.view_selector = ttk.Combobox(self, state='readonly', textvariable=self.view_name)
        self.view_selector.pack(fill='x', padx=12, pady=4)
        self.view_selector.bind('<<ComboboxSelected>>', lambda event: self.render_view())
        self.preview = ttk.Label(self, anchor='center')
        self.preview.pack(fill='both', expand=True, padx=12, pady=8)
        table_frame = ttk.Frame(self, padding=(12, 0))
        table_frame.pack(fill='x')
        columns = ('id', 'area_px', 'perimeter_px', 'circularity', 'centroid_x', 'centroid_y', 'touches_border')
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=4)
        for column, title in zip(columns, ('№', 'Площа, пікселі²', 'Периметр, пікселі', 'Округлість', 'Центр X', 'Центр Y', 'На краю')):
            self.table.heading(column, text=title)
            self.table.column(column, width=130, anchor='center')
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side='left', fill='x', expand=True)
        scrollbar.pack(side='right', fill='y')
        ttk.Label(self, text='Watershed розділяє області за автоматичними маркерами. '
                  'Це кандидати в ядра; точність на реальній вибірці ще не оцінена.', padding=12).pack(fill='x')
        self.update_buttons()
        self.protocol('WM_DELETE_WINDOW', self.close)

    def close(self):
        """Закрити вікно; активне числове завдання завершиться у фоні."""
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def update_buttons(self):
        """Увімкнути лише дії, для яких уже підготовлено дані."""
        idle = self.future is None
        for button, enabled in ((self.open_button, idle),
                                (self.run_button, idle and self.original is not None),
                                (self.segment_button, idle and self.maps is not None),
                                (self.watershed_button, idle and self.result is not None),
                                (self.texture_button, idle and self.original is not None
                                 and (self.config_data['texture']['source'] == 'gray' or self.maps is not None)
                                 and (self.config_data['texture']['scope'] == 'whole' or self.result is not None)),
                                (self.evaluate_button, idle and self.result is not None),
                                (self.batch_button, idle),
                                (self.save_button, idle and (self.maps is not None or self.texture is not None))):
            button.configure(state='normal' if enabled else 'disabled')

    def show_tabs(self):
        """Перебудувати вкладки з екранними копіями; оригінальні дані не змінювати."""
        previous = self.view_name.get()
        images = {'Оригінал': self.original, **self.channels}
        if self.result is not None:
            for title, key in (('Оцу', 'raw_mask'), ('Відкриття', 'opened_mask'), ('Відібрані', 'mask')):
                images[title] = Image.fromarray(self.result[key].astype(np.uint8) * 255)
            if self.result.get('stage') == 'watershed':
                images['Відстань'] = distance_image(self.result['distance'])
                images['Маркери'] = marker_image(self.result['distance'], self.result['markers'])
                images['До WS'] = self.result['before_overlay']
                images['Мітки WS'] = label_image(self.result['labels'])
                images['Після WS'] = self.result['overlay']
            else:
                images['Об’єкти'] = self.result['overlay']
        if self.texture is not None:
            images['LBP — вхід'] = Image.fromarray(self.texture['source_image'])
            images['LBP — коди'] = Image.fromarray(self.texture['codes'])
            images['LBP — гістограма'] = histogram_image(self.texture['histogram_counts'])
        self.views = images
        self.view_selector.configure(values=list(images))
        self.select_view(previous if previous in images else 'Оригінал')

    def select_view(self, name):
        """Вибрати результат у випадному списку."""
        self.view_name.set(name)
        self.render_view()

    def render_view(self):
        """Масштабувати лише поточну картинку для екрана."""
        if self.view_name.get() not in self.views:
            return
        preview = self.views[self.view_name.get()].copy()
        preview.thumbnail((self.config_data['preview_width'], self.config_data['preview_height']),
                          Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(preview)
        self.photos = [photo]
        self.preview.configure(image=photo)

    def show_objects(self):
        """Оновити лічильник і таблицю; ID відповідає labels.npy та objects.csv."""
        children = self.table.get_children()
        if children:
            self.table.delete(*children)
        if self.result is None:
            self.summary.set('Кандидатів: —')
            return
        for item in self.result['objects']:
            self.table.insert('', 'end', values=(item['id'], item['area_px'], item['perimeter_px'],
                              f"{item['circularity']:.3f}", f"{item['centroid_x']:.1f}", f"{item['centroid_y']:.1f}",
                              'так' if item['touches_border'] else 'ні'))
        counts = f"Кандидатів: {self.result['count']}"
        if self.result.get('stage') == 'watershed':
            counts = (f"До WS: {self.result['before_count']} → Після WS: {self.result['count']}"
                      f" | Маркерів: {self.result['marker_count']}")
        density = self.result['summary']['density_per_megapixel']
        self.summary.set(f"{counts} | Щільність: {density:.1f}/Мпіксель"
                         f" | Поріг: {self.result['effective_threshold']}")


    def open_image(self):
        """Відкрити файл; скинути старі результати після успішного читання."""
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
        self.maps = self.result = self.texture = None
        self.metric_text.set('Метрики: еталон ще не вибрано.')
        self.show_tabs()
        self.show_objects()
        self.update_buttons()
        self.info.set(f'{self.source_path.name} | {image.width} × {image.height} пікселів | RGB. '
                      'Для TIFF відкрито перший кадр.')

    def start_job(self, job: str, function, *args):
        """Передати роботу виконавцю, заблокувати кнопки та запланувати перевірку."""
        self.job = job
        self.future = self.executor.submit(function, *args)
        self.update_buttons()
        self.after(80, self.check_result)

    def run_separation(self):
        """Обчислити H/E з оригіналу, не з екранної копії."""
        if self.original is None or self.future is not None:
            return
        self.info.set('Виконується розділення H&E…')
        self.start_job('separate', separate_stains, np.asarray(self.original), self.config_data['stains'])

    def run_segmentation(self):
        """Запустити Оцу, очищення, відбір і підрахунок кандидатів."""
        if self.maps is None or self.future is not None:
            return
        self.info.set('Оцу → відкриття → компоненти зв’язності…')
        self.start_job('segment', analyze_image, np.asarray(self.original), self.maps[0],
                       self.config_data['segmentation'], self.config_data['measurements'])

    def run_watershed(self):
        """Розділити відкриту маску; повторний запуск не ділить попередні басейни."""
        if self.result is None or self.future is not None:
            return
        self.info.set('Карта відстаней → автоматичні маркери → Watershed…')
        self.start_job('watershed', split_image, np.asarray(self.original), self.result, self.config_data)

    def run_texture(self):
        """Розрахувати LBP для поточного джерела й актуальної сегментації."""
        if self.original is None or self.future is not None:
            return
        self.info.set('Обчислення LBP та гістограм…')
        self.start_job('texture', texture_image, np.asarray(self.original), self.maps,
                       self.result, self.config_data['texture'])

    def run_evaluation(self):
        """Вибрати еталон маски або instance-ID та порівняти поточний результат."""
        if self.result is None or self.future is not None:
            return
        filename = filedialog.askopenfilename(title='Еталон: бінарне зображення або instance-ID NPY',
            filetypes=[('Еталони', '*.png *.bmp *.tif *.tiff *.npy'), ('Усі файли', '*.*')])
        if not filename:
            return
        self.info.set('Порівняння з еталоном…')
        self.start_job('evaluation', evaluate_result, self.result, Path(filename), self.config_data['max_pixels'])

    def run_batch(self):
        """Вибрати опис вибірки й вихідну теку; виконати повний аналіз у фоні."""
        if self.future is not None:
            return
        filename = filedialog.askopenfilename(title='CSV зі стовпцями image,reference',
                                              filetypes=[('Вибірка CSV', '*.csv')])
        if not filename:
            return
        directory = filedialog.askdirectory(title='Тека для пакетного звіту')
        if not directory:
            return
        self.info.set('Пакетна обробка вибірки… Поточний відкритий кадр збережено у вікні.')
        self.start_job('batch', run_batch, Path(filename), Path(directory), self.config_data)

    def check_result(self):
        """Забрати результат фонового обчислення в головному потоці."""
        if not self.future.done():
            self.after(80, self.check_result)
            return
        try:
            result = self.future.result()
            if self.job == 'separate':
                self.maps = result
                self.result = self.texture = None
                self.metric_text.set('Метрики: еталон ще не вибрано.')
                self.channels = {'H': Image.fromarray(transmittance_image(self.maps[0])),
                                 'E': Image.fromarray(transmittance_image(self.maps[1]))}
                self.show_tabs()
                self.show_objects()
                self.select_view('H')
                self.info.set('H&E готово. Натисни «2. Знайти кандидатів».')
            elif self.job in ('segment', 'watershed'):
                self.result = result
                self.metric_text.set('Метрики: потрібно порівняти новий результат з еталоном.')
                self.texture = None
                self.show_tabs()
                self.show_objects()
                self.select_view(list(self.views)[-1])
                self.info.set('Готово. Зелені межі й номери — відібрані кандидати. '
                              'Периметр сітковий; щільність обчислена на всю площу кадру.')
            elif self.job == 'evaluation':
                self.result['evaluation'] = result
                current = result['rows'][-1]
                self.metric_text.set(f"Dice: {current['dice']:.3f} | IoU: {current['iou']:.3f}")
                self.info.set('Порівняння готове. Збережи результати для CSV/JSON метрик.')
            elif self.job == 'batch':
                self.info.set(f'Пакетну обробку завершено: {result}. Перевір batch_status.csv.')
            elif self.job == 'texture':
                self.texture = result
                self.texture['source_name'] = self.source_path.name
                self.show_tabs()
                self.select_view('LBP — гістограма')
                features = result['features']
                entropy = features['entropy_bits']
                entropy_text = 'немає центрів' if entropy is None else f'{entropy:.3f} біт'
                self.info.set(f"LBP: {features['valid_centers']} допустимих центрів | "
                              f"Ентропія: {entropy_text}. Це опис текстури, не діагноз.")
            else:
                self.info.set(f'Результати збережено: {result}')
        except Exception as error:  # Помилка worker має бути показана, а кнопки розблоковані.
            messagebox.showerror('Помилка операції', str(error))
            self.info.set('Операція не виконана. Для експорту можливі часткові файли. '
                          'Спробуй менше зображення або перевір параметри.')
        finally:
            self.future = None
            self.job = None
            self.update_buttons()

    def save_channels(self):
        """Зберегти поточні канали та, якщо є, результати сегментації у фоні."""
        if (self.maps is None and self.texture is None) or self.future is not None:
            return
        directory = filedialog.askdirectory(title='Вибери теку для результатів')
        if not directory:
            return
        self.info.set('Збереження результатів…')
        if self.maps is None:
            self.start_job('save', save_texture_only, Path(directory), self.texture)
        else:
            self.start_job('save', save_results, Path(directory), self.source_path.name,
                           self.original.size, self.maps, self.config_data, self.result, self.texture)
