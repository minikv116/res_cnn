from __future__ import annotations
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, precision_score,
    recall_score, f1_score
)

from models import MODEL_ZOO

def evaluate_model_metrics(
    model_name: str,
    weights_dir: str | Path,
    loader,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    class_names: list[str] | None = None
) -> dict[str, float]:
    """
    Оценивает модель по заданному DataLoader и сохраняет результаты (матрицу ошибок) в PNG-файл.

    Параметры
    ----------
    model_name : str
        Ключ из словаря MODEL_ZOO, определяющий архитектуру модели.

    weights_dir : str | Path
        Папка, содержащая файл с весами модели в формате <model_name>_best.pth.

    loader : torch.utils.data.DataLoader
        Даталоадер (валидационный или тестовый) для проведения инференса и оценки.

    output_dir : str | Path
        Папка, в которую будут сохранены результаты (PNG с матрицей ошибок).
        Если папка не существует, будет создана автоматически.

    device : str | torch.device, optional
        Устройство для проведения инференса ('cpu' или 'cuda:0', 'cuda:1' и т.д.).
        По умолчанию 'cpu'.

    class_names : list[str] | None, optional
        Список меток классов (строки). Если None, будут использованы номера классов '0', '1', '...' 
        в порядке возрастания. Длина списка должна совпадать с числом уникальных классов.

    Что делает
    ----------
    1. Проверяет наличие model_name в MODEL_ZOO и создаёт модель на устройстве device.
    2. Загружает state_dict из файла <weights_dir>/<model_name>_best.pth.
    3. Прогоняет весь даталоадер, собирает истинные метки (y_true) и предсказания (y_pred).
    4. Считает confusion matrix и нормирует её по строкам (доли).
    5. Вычисляет метрики: Accuracy, Precision, Recall, F1 (average='macro').
    6. Строит тепловую карту нормированной матрицы ошибок:
       - Подписывает оси классами (class_names).
       - Записывает численные значения (с точностью до двух знаков).
       - Сохраняет график в файл <output_dir>/<model_name>_confusion_matrix.png.
    7. Печатает в консоль табличку с метриками (Accuracy, Precision, Recall, F1).
    8. Возвращает словарь с ключами "accuracy", "precision", "recall", "f1".

    Возвращаемое значение
    ---------------------
    dict[str, float]
        Словарь, содержащий рассчитанные метрики:
        {
            "accuracy": float,   # значение accuracy (0–1)
            "precision": float,  # значение precision (macro, 0–1)
            "recall": float,     # значение recall (macro, 0–1)
            "f1": float          # значение F1-score (macro, 0–1)
        }
    """

    # 2. Загрузка модели и весов ------------------------------------------
    model = MODEL_ZOO[model_name]().to(device).eval()
    weights_path = Path(weights_dir) / f"{model_name}_best.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Файл весов '{weights_path}' не найден.")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 3. Сбор предсказаний (y_true, y_pred) --------------------------------
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 4. Вычисление нормированной confusion matrix ------------------------
    n_classes = len(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    # Нормируем по строкам: делим каждую строку на сумму по ней (clip(min=1) предотвращает деление на 0)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    # 5. Вычисление метрик ------------------------------------------------
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 6. Визуализация и сохранение матрицы ошибок --------------------------
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError(f"Длина class_names ({len(class_names)}) не совпадает с числом классов ({n_classes}).")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 8))
    im = plt.imshow(cm_norm, cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Матрица ошибок — {model_name}")
    plt.xlabel("Предсказанные классы")
    plt.ylabel("Истинные классы")
    plt.xticks(np.arange(n_classes), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(n_classes), class_names)

    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_norm[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            plt.text(j, i, f"{value:.2f}", ha='center', va='center',
                     color=text_color, fontsize=14)

    plt.tight_layout()
    cm_filename = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_filename, format="png", dpi=300)
    plt.close()

    # 7. Вывод метрик в консоль -------------------------------------------
    print(f"Метрики для модели '{model_name}':")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F1-score : {f1 * 100:.2f}%")

    # 8. Возвращаем словарь с рассчитанными метриками ---------------------
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
