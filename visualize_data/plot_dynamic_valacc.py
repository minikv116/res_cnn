
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_dynamics(
    logs_dir: str | Path,
    category: str,
    output_dir: str | Path,
    *,
    figsize: tuple[int, int] = (9, 5),
    train_loss_col: str = "train_loss",
    val_loss_col: str = "val_loss",
    train_acc_col: str = "train_acc",
    val_acc_col: str = "val_acc",
) -> None:
    """
    Строит два отдельных графика (Loss и Accuracy) для всех моделей указанной
    категории (XS / S / M / L) и сохраняет их в виде PNG-файлов в указанную папку.

    Параметры
    ----------
    logs_dir : str | Path
        Путь к каталогу, в котором находятся *.csv-файлы с метриками обученных моделей.
        Каждый CSV-файл должен содержать столбцы с именами train_loss_col, val_loss_col,
        train_acc_col, val_acc_col (можно переопределить при помощи параметров).
    category : str
        Метка категории моделей для фильтрации по префиксу имени файла.
        Допустимые значения: 'XS', 'S', 'M', 'L'. Будет обрабатываться файл, если
        его имя (без расширения) начинается с префикса category.lower() + "net".
        Пример: для category='XS' выбираются файлы, у которых имя начинается с 'xsnet'.
    output_dir : str | Path
        Путь к директории, где будут сохранены PNG-файлы с графиками.
        Если директория не существует, она будет создана автоматически.
    figsize : tuple[int, int], optional
        Размер фигур (ширина, высота) для графиков. По умолчанию (9, 5).
    train_loss_col : str, optional
        Имя столбца в CSV с потерями на обучении. По умолчанию "train_loss".
    val_loss_col : str, optional
        Имя столбца в CSV с потерями на валидации. По умолчанию "val_loss".
    train_acc_col : str, optional
        Имя столбца в CSV с точностью на обучении (в диапазоне [0, 1]). По умолчанию "train_acc".
        Значение будет автоматически преобразовано в проценты (умножением на 100).
    val_acc_col : str, optional
        Имя столбца в CSV с точностью на валидации (в диапазоне [0, 1]). По умолчанию "val_acc".
        Значение будет автоматически преобразовано в проценты (умножением на 100).

    Возвращаемое значение
    ---------------------
    None
        Функция сохраняет файлы PNG и не возвращает значения.
    """
    # Преобразуем строки или Path в pathlib.Path для удобства
    logs_dir = Path(logs_dir)
    output_dir = Path(output_dir)

    # Проверяем, что директория с логами существует и является директорией
    if not logs_dir.exists() or not logs_dir.is_dir():
        raise FileNotFoundError(f"Папка с логами '{logs_dir}' не найдена или не является директорией.")

    # Создаем папку для сохранения графиков (если её ещё нет)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Формируем префикс для фильтрации CSV-файлов по категории
    prefix = category.lower() + "net"  # 'XS' → 'xsnet', 'S' → 'snet' и т.д.

    # Получаем все CSV-файлы в указанной папке
    csv_files = sorted(logs_dir.glob("*.csv"))

    # Если в папке нет CSV, сообщаем об ошибке
    if not csv_files:
        raise FileNotFoundError(f"В папке {logs_dir} не найдено CSV-файлов для построения графиков.")

    # Оставляем только те файлы, чьё имя (без расширения) начинается с нужного префикса
    selected = [f for f in csv_files if f.stem.lower().startswith(prefix)]
    if not selected:
        raise ValueError(
            f"Не найдено файлов для категории «{category}». "
            f"Ожидался префикс {prefix} в начале имени файла."
        )

    # ─── 1. Строим график Loss ───────────────────────────────────────────────────
    # Создаем фигуру и ось для потерь
    fig_loss, ax_loss = plt.subplots(figsize=figsize)

    # Для каждого отфильтрованного CSV-файла:
    for csv_path in selected:
        # Загружаем историю обучения в DataFrame pandas
        hist = pd.read_csv(csv_path)
        model_name = csv_path.stem  # Имя модели (файл без расширения)

        # Извлекаем массив значений потерь на обучении и валидации
        train_loss = hist[train_loss_col].to_numpy()
        val_loss   = hist[val_loss_col].to_numpy()

        # Рисуем кривые:
        #   - сплошная линия для train_loss
        #   - пунктирная линия для val_loss
        ax_loss.plot(train_loss, label=f"{model_name} train")
        ax_loss.plot(val_loss,   linestyle="--", label=f"{model_name} val")

    # Добавляем подписи, сетку и легенду
    ax_loss.set_title("Динамика функции потерь")  # Заголовок графика
    ax_loss.set_xlabel("Эпоха")                    # Метка оси X
    ax_loss.set_ylabel("Потери")                   # Метка оси Y
    ax_loss.grid(True)                             # Включаем сетку
    ax_loss.legend()                               # Показываем легенду

    # Подгоняем раскладку, чтобы метки не были обрезаны
    fig_loss.tight_layout()

    # Сохраняем график в PNG-файл: "<категория>_loss.png"
    loss_filename = output_dir / f"{category}_loss.png"
    fig_loss.savefig(loss_filename, format="png", dpi=300)
    plt.close(fig_loss)  # Закрываем фигуру, освобождаем память

    # ─── 2. Строим график Accuracy ────────────────────────────────────────────────
    # Создаем фигуру и ось для отображения точности моделей
    fig_acc, ax_acc = plt.subplots(figsize=figsize)

    # Для каждого отфильтрованного CSV-файла:
    for csv_path in selected:
        hist = pd.read_csv(csv_path)
        model_name = csv_path.stem

        # Извлекаем значения точности и переводим их в проценты
        train_acc = (hist[train_acc_col] * 100).to_numpy()
        val_acc   = (hist[val_acc_col]   * 100).to_numpy()

        # Рисуем кривые:
        #   - сплошная линия для train_acc (%)
        #   - пунктирная линия для val_acc (%)
        ax_acc.plot(train_acc, label=f"{model_name} train")
        ax_acc.plot(val_acc,   linestyle="--", label=f"{model_name} val")

    # Добавляем подписи, сетку и легенду
    ax_acc.set_title("Динамика точности")   # Заголовок графика
    ax_acc.set_xlabel("Эпоха")               # Метка оси X
    ax_acc.set_ylabel("Точность, %")         # Метка оси Y (в процентах)
    ax_acc.grid(True)                        # Включаем сетку
    ax_acc.legend()                          # Показываем легенду

    fig_acc.tight_layout()

    # Сохраняем график в PNG-файл: "<категория>_accuracy.png"
    acc_filename = output_dir / f"{category}_accuracy.png"
    fig_acc.savefig(acc_filename, format="png", dpi=300)
    plt.close(fig_acc)  # Закрываем фигуру после сохранения
