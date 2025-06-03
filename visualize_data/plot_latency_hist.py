# ================================================================
#  plot_latency_hist.py  –  гистограмма задержек моделей (бар-чарт)
# ================================================================
"""
Функция plot_latency_hist

Читает общий CSV-файл со столбцами:
  • 'model' (имя модели),
  • 'latency_ms' (задержка в миллисекундах).

• Рисует bar-чарт: ось X — оригинальный порядок моделей из CSV, 
  ось Y — задержка (ms) в логарифмической шкале.
• Диапазон по Y ограничен ±5 % вокруг [min(latency); max(latency)].
• Шрифт (font.size) устанавливается в 16 pt.
• Цвета столбцов назначаются по префиксу модели (см. GROUP_COLORS).
• Сохраняет итоговый график в PNG-файл.

Параметры
---------
csv_path : str | Path
    Путь к CSV-файлу с колонками 'model' и latency_col.

latency_col : str, optional (по умолчанию "latency_ms")
    Имя столбца в CSV, содержащего значения задержки (ms). Если столбца с 
    таким именем нет, будет выброшено исключение ValueError.

font_size : int, optional (по умолчанию 16)
    Размер шрифта для подписей и чисел на графике (в пунктах, pt).

margin : float, optional (по умолчанию 0.05)
    Отступ по вертикали (± margin × 100 %). Например, margin=0.05
    означает, что ось Y будет простираться от (min_lat * 0.95) 
    до (max_lat * 1.05).

group_colors : dict[str, str], optional
    Словарь, сопоставляющий префикс модели → цвет (любые допустимые имена 
    цветов matplotlib). По умолчанию используется GROUP_COLORS:
        {
            "XSNet": "tab:blue",
            "SNet":  "tab:orange",
            "MNet":  "tab:green",
            "LNet":  "tab:red",
        }
    Если имя модели не начинается ни с одного из ключей в group_colors, 
    столбец будет окрашен в DEFAULT_COLOR ("tab:gray").

output_path : str | Path, optional
    Путь (полный путь + имя файла) для сохранения итогового графика в формате PNG.
    Если None, то график только отображается, но не сохраняется.
    Пример: "./plots/latency_histogram.png" или Path("results/latency.png").
    Если директории в пути нет, она будет создана автоматически.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Цвета по группам моделей (по префиксу имени)
GROUP_COLORS = {
    "XSNet": "tab:blue",
    "SNet":  "tab:orange",
    "MNet":  "tab:green",
    "LNet":  "tab:red",
}
DEFAULT_COLOR = "tab:gray"


def plot_latency_hist(
        csv_path: str | Path,
        latency_col: str = "latency_ms",
        font_size: int = 16,
        margin: float = 0.05,        # ±5 % вокруг [min; max]
        group_colors: dict[str, str] = GROUP_COLORS,
        output_path: str | Path | None = None
) -> None:
    """
    Строит bar-чарт задержек моделей и сохраняет его в PNG-файл, если указан параметр output_path.

    Параметры
    ----------
    csv_path : str | Path
        Путь к CSV-файлу с обязательными столбцами:
        'model' (имена моделей) и latency_col (численные значения задержки, мс).

    latency_col : str, optional
        Имя столбца в CSV, содержащего задержки (мс). По умолчанию "latency_ms".
        Если в CSV нет такого столбца, возбуждается ValueError.

    font_size : int, optional
        Размер шрифта (в pt) для всех надписей на графике. По умолчанию 16.

    margin : float, optional
        Отношение для расширения диапазона по оси Y вокруг [min_latency; max_latency].
        Диапазон Y будет от min_latency * (1 - margin) до max_latency * (1 + margin).
        По умолчанию 0.05 (±5 %).

    group_colors : dict[str, str], optional
        Словарь соответствия префикса имени модели → цвет столбца. 
        Если имя модели не совпадает ни с одним ключом, используется DEFAULT_COLOR.

    output_path : str | Path | None, optional
        Если указать путь (например, "./plots/latency.png"), итоговый график
        будет сохранён в этот файл формата PNG. Если None, график только
        выводится на экран без сохранения.
    """
    # -------- 1. Проверка и загрузка CSV --------
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл '{csv_path}' не найден.")

    # Читаем CSV в DataFrame pandas
    df = pd.read_csv(csv_path)

    # Проверяем, что столбец latency_col присутствует
    if latency_col not in df.columns:
        raise ValueError(f"В файле '{csv_path.name}' отсутствует столбец '{latency_col}'.")

    # Извлекаем список моделей (в том порядке, как в CSV) и значения задержек
    models   = df["model"].tolist()
    lat_vals = df[latency_col].tolist()

    # -------- 2. Функция для назначения цвета столбцу --------
    def get_color(name: str) -> str:
        """
        Возвращает цвет для модели по её имени. 
        Проходим по всем ключам в group_colors; если префикс совпадает,
        используем соответствующий цвет, иначе DEFAULT_COLOR.
        """
        for pref, col in group_colors.items():
            # сравниваем начало строки name с ключом pref
            if name.startswith(pref):
                return col
        return DEFAULT_COLOR

    # Формируем список цветов для каждой модели
    colors = [get_color(m) for m in models]

    # -------- 3. Определяем диапазон по оси Y (логарифмическая шкала) --------
    min_lat = min(lat_vals)
    max_lat = max(lat_vals)
    ymin = min_lat * (1 - margin)
    ymax = max_lat * (1 + margin)

    # -------- 4. Настройка глобального шрифта --------
    plt.rcParams.update({"font.size": font_size})

    # -------- 5. Построение bar-чарта --------
    # Автоматически увеличиваем ширину фигуры, если моделей много:
    width = max(8, 0.5 * len(models))
    fig, ax = plt.subplots(figsize=(width, 6))

    # Строим столбцы: X — модели, Y — задержки, цвета — из colors
    bars = ax.bar(models, lat_vals, color=colors, edgecolor="black")

    # Логарифмическая шкала по оси Y
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)

    # Подписи осей и заголовок
    ax.set_ylabel("Задержка, мс (лог. шкала)")
    ax.set_title("Гистограмма задержек моделей")

    # Поворот подписей по X для читаемости: 45°, выравнивание по правому краю
    ax.set_xticklabels(models, rotation=45, ha="right")

    # -------- 6. Добавляем легенду (по цветам групп) --------
    # Формируем элементы легенды: прямоугольник соответствующего цвета и метка
    legend_elems = [
        plt.Line2D([0], [0], marker='s', linestyle='', color=col, markersize=16, label=grp)
        for grp, col in group_colors.items()
    ]
    ax.legend(handles=legend_elems, title="Группа моделей")

    plt.tight_layout()

    # -------- 7. Сохранение графика в PNG (если указан output_path) --------
    if output_path is not None:
        output_path = Path(output_path).expanduser()
        # Создаём все необходимые директории, если их нет
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем фигуру в файл формата PNG с разрешением 300 dpi
        fig.savefig(output_path, format="png", dpi=300)
        # Опционально: можно вывести сообщение об успешном сохранении
        print(f"Гистограмма задержек сохранена в '{output_path}'.")

    # -------- 8. Отображение графика --------
    # Если не требуется интерактивный вывод, строку plt.show() можно опустить.
    plt.show()
    plt.close(fig)  # Закрываем фигуру, чтобы освободить память
