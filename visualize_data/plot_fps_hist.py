# ================================================================
#  plot_fps_hist.py  –  гистограмма пропускной способности (FPS)
# ================================================================
"""
Функция plot_fps_hist

Читает CSV-файл со столбцами:
  • 'model' (названия моделей)
  • fps_col (значения FPS, кадр/сек)

• Рисует bar-чарт, где:
    • Ось X: модель (в том же порядке, что и в CSV)
    • Ось Y: FPS (в логарифмической шкале)
• Диапазон по оси Y ограничен ± margin (5 %) от [min; max] FPS
• Увеличивает шрифты до font_size (16 pt по умолчанию)
• Назначает цвет столбца по префиксу модели (см. GROUP_COLORS)
• Сохраняет итоговый график в PNG-файл, если указан параметр output_path

Параметры
---------
csv_path : str | Path
    Путь к CSV-файлу, содержащему как минимум две колонки:
    'model' и fps_col (например, 'fps').

fps_col : str, optional
    Название столбца в CSV для значений FPS. По умолчанию "fps".
    Если такого столбца нет — возбуждается ValueError.

font_size : int, optional
    Размер шрифта (в пунктах, pt) для всех подписей и меток. По умолчанию 16.

margin : float, optional
    Отношение для расширения диапазона по оси Y (±margin × 100 %).
    По умолчанию 0.05 (±5 %).

group_colors : dict[str, str], optional
    Словарь соответствия префикса имени модели → цвет.
    Если имя модели не начинается с ни одного префикса, используется DEFAULT_COLOR:
        {
            "XSNet": "tab:blue",
            "SNet":  "tab:orange",
            "MNet":  "tab:green",
            "LNet":  "tab:red",
        }
    DEFAULT_COLOR = "tab:gray"

output_path : str | Path | None, optional
    Если указан путь (например, "./plots/fps_hist.png"), итоговый график
    сохраняется в этот файл формата PNG. Если None (по умолчанию), 
    функция только отображает график и не сохраняет его.

Возвращает
---------
None
    Функция строит и показывает график, а при указании output_path
    дополнительно сохраняет его в файл.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ---------- цвета по группам моделей (по префиксу имени) ----------
GROUP_COLORS = {
    "XSNet": "tab:blue",
    "SNet":  "tab:orange",
    "MNet":  "tab:green",
    "LNet":  "tab:red",
}
DEFAULT_COLOR = "tab:gray"


def plot_fps_hist(
        csv_path: str | Path,
        fps_col: str = "fps",
        font_size: int = 16,
        margin: float = 0.05,                # ±5 % вокруг [min; max]
        group_colors: dict[str, str] = GROUP_COLORS,
        output_path: str | Path | None = None
) -> None:
    """
    Строит bar-чарт пропускной способности (FPS) для моделей и сохраняет его в PNG-файл, 
    если указан output_path.

    Параметры
    ----------
    csv_path : str | Path
        Путь к CSV-файлу, содержащему столбцы 'model' и fps_col.

    fps_col : str, optional
        Имя столбца в CSV с значениями FPS (кд/с). По умолчанию "fps".

    font_size : int, optional
        Размер шрифта (pt) для подписей и меток графика. По умолчанию 16.

    margin : float, optional
        Отношение для расширения диапазона оси Y вокруг [min; max]:
        диапазон Y = [min_fps*(1-margin), max_fps*(1+margin)]. По умолчанию 0.05.

    group_colors : dict[str, str], optional
        Словарь, связывающий префикс имени модели → цвет для баров.
        Если имя модели не начинается с ни одного ключа, используется DEFAULT_COLOR.

    output_path : str | Path | None, optional
        Путь для сохранения итогового графика в формате PNG. Пример:
        "./plots/fps_histogram.png". Если None, сохранения не происходит.
    """
    # -------- 1. Проверка и загрузка CSV --------
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл '{csv_path}' не найден.")

    # Считываем CSV в pandas.DataFrame
    df = pd.read_csv(csv_path)

    # Проверяем наличие столбца fps_col
    if fps_col not in df.columns:
        raise ValueError(f"В файле '{csv_path.name}' отсутствует столбец '{fps_col}'.")

    # Сохраняем порядок моделей и их значение FPS
    models  = df["model"].tolist()       # список моделей в исходном порядке
    fps_val = df[fps_col].tolist()       # список значений FPS

    # -------- 2. Функция назначения цвета по имени модели --------
    def get_color(name: str) -> str:
        """
        Возвращает цвет для модели по её имени: 
        если имя начинается с одного из ключей group_colors, 
        возвращаем соответствующий цвет, иначе DEFAULT_COLOR.
        """
        for pref, col in group_colors.items():
            if name.startswith(pref):
                return col
        return DEFAULT_COLOR

    # Формируем список цветов для каждого бара
    colors = [get_color(m) for m in models]

    # -------- 3. Определяем диапазон по оси Y (лог. шкала) --------
    min_fps = min(fps_val)
    max_fps = max(fps_val)
    ymin = max(min_fps * (1 - margin), 1e-3)  # нижняя граница (не ниже 1e-3, чтобы log не ушёл в -inf)
    ymax = max_fps * (1 + margin)            # верхняя граница

    # -------- 4. Настройка шрифта (font.size) --------
    plt.rcParams.update({"font.size": font_size})

    # -------- 5. Построение bar-чарта --------
    width = max(8, 0.5 * len(models))  # автоматическая ширина: минимум 8, иначе 0.5×число моделей
    fig, ax = plt.subplots(figsize=(width, 6))

    # Строим столбцы: X — models, Y — fps_val, цвета — из colors
    ax.bar(models, fps_val, color=colors, edgecolor="black")

    # Переводим Y в логарифмическую шкалу
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)

    # Подписи осей и заголовок
    ax.set_ylabel("FPS, кадр/сек (шкала лог.)")
    ax.set_title("Гистограмма пропускной способности моделей")

    # Поворот подписей по оси X для читаемости
    ax.set_xticklabels(models, rotation=45, ha="right")

    # -------- 6. Легенда по цветам групп --------
    legend_elems = [
        plt.Line2D([0], [0], marker='s', linestyle='', color=col, markersize=10, label=grp)
        for grp, col in group_colors.items()
    ]
    ax.legend(handles=legend_elems, title="Группа моделей")

    plt.tight_layout()

    # -------- 7. Сохранение графика в PNG (если указан output_path) --------
    if output_path is not None:
        output_path = Path(output_path).expanduser()
        # Создаём недостающие директории
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        # Сохраняем фигуру с dpi=300 для хорошего качества
        fig.savefig(output_path, format="png", dpi=300)
        print(f"Гистограмма FPS сохранена в '{output_path}'.")

    # -------- 8. Отображение графика (plt.show()) и очистка фигуры --------
    plt.show()
    plt.close(fig)
