from __future__ import annotations
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def save_prediction_grid_from_loader(
    model: torch.nn.Module,
    test_loader: DataLoader,
    num_examples: int,
    output_path: str | Path,
    device: str | torch.device = "cpu"
) -> None:
    """
    Сохраняет сетку предсказаний модели, используя DataLoader, в виде PNG-файла.

    Функция берёт первые `num_examples` изображений из `test_loader`,
    выполняет инференс, строит сетку и сохраняет результат в PNG-файл.
    Неправильные предсказания (где предсказанный класс != истинного) 
    отмечаются красным цветом заголовка.

    Параметры
    ----------
    model : torch.nn.Module
        Экземпляр PyTorch-модели (с загруженными весами). Модель будет
        переведена в режим eval() и на устройство `device`.

    test_loader : torch.utils.data.DataLoader
        DataLoader для тестового набора. Из него будут последовательно 
        взяты батчи, пока не накопится `num_examples` примеров.

    num_examples : int
        Количество примеров, которые нужно взять из начала `test_loader`.
        Например, если num_examples=12, будут собраны первые 12 изображений 
        и их меток.

    output_path : str | Path
        Путь (включая имя файла) для сохранения PNG-изображения. Если 
        директория не существует, она будет создана автоматически.
        Пример: "./plots/predictions_grid.png".

    device : str | torch.device, optional
        Устройство для инференса ("cpu" или "cuda:0"/"cuda:1"). По умолчанию "cpu".

    Пример использования
    ---------------------
        model = MNet5()
        model.load_state_dict(torch.load("./best_models/MNet5_best.pth"))
        save_prediction_grid_from_loader(
            model=model,
            test_loader=test_loader,
            num_examples=12,
            output_path="./plots/MNet5_predictions.png",
            device="cuda:0"  # или "cpu"
        )
        # В итоге появится файл "./plots/MNet5_predictions.png"
    """
    # 1. Подготовка устройства и модели ------------------------------------
    # Преобразуем device к torch.device, если передана строка
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()  # Переводим модель в режим инференса

    # 2. Сбор первых num_examples изображений и меток -----------------------
    images_list = []
    labels_list = []
    collected = 0

    # Итерируем по батчам тестового DataLoader
    with torch.no_grad():
        for batch in test_loader:
            # Предполагается, что batch = (inputs, targets)
            inputs, targets = batch

            # Если размерность inputs = [B, C, H, W], targets = [B]
            batch_size = inputs.size(0)

            # Если собранных примеров + батч_size превышает num_examples,
            # берём только нужное количество из текущего батча
            need = num_examples - collected
            if need <= 0:
                break

            if batch_size <= need:
                # Берём весь батч
                images_list.append(inputs)
                labels_list.append(targets)
                collected += batch_size
            else:
                # Берём только первые need элементов текущего батча
                images_list.append(inputs[:need])
                labels_list.append(targets[:need])
                collected += need
                break  # Достаточно изображений, выходим из цикла

    # Проверяем, что получили хотя бы num_examples
    if collected < num_examples:
        raise ValueError(
            f"В test_loader не удалось собрать {num_examples} примеров, "
            f"собрано только {collected}."
        )

    # Конкатенируем списки батчей в один тензор: [num_examples, C, H, W]
    images = torch.cat(images_list, dim=0)  # форма: [num_examples, C, H, W]
    labels = torch.cat(labels_list, dim=0)  # форма: [num_examples]

    # 3. Инференс -----------------------------------------------------------
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)                 # [num_examples, num_classes]
        preds = torch.argmax(outputs, dim=1)     # [num_examples]
        preds = preds.cpu().numpy()              # переводим в numpy

    labels = labels.cpu().numpy()                # истинные метки np.ndarray

    # 4. Словарь эмоций на русском -----------------------------------------
    emotion_map = {
        0: 'нейтральное',
        1: 'радость',
        2: 'грусть',
        3: 'гнев',
        4: 'отвращение',
        5: 'страх',
        6: 'удивление'
    }

    # 5. Подготовка сетки ---------------------------------------------------
    # Вычисляем количество строк и столбцов для сетки:
    # Попробуем получить сетку максимально близкую к квадратной
    cols = min(num_examples, 6)  # максимум 6 столбцов, чтобы не было слишком широкого графика
    rows = (num_examples + cols - 1) // cols  # округление вверх

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(
        'Пример распознанных эмоций\n'
        'Истинная метка (черным) и предсказанная (красным, если неверно)',
        fontsize=20
    )

    # 6. Отрисовка каждого примера в ячейке ---------------------------------
    for idx in range(rows * cols):
        # Определяем, в какой ячейке расположить изображение
        row_idx = idx // cols
        col_idx = idx % cols

        # Если идем за предел num_examples, скрываем пустые оси
        if idx >= num_examples:
            axes[row_idx, col_idx].axis('off')
            continue

        ax = axes[row_idx, col_idx]
        img = images[idx].cpu().numpy()  # np.ndarray формы [C, H, W] или [1, H, W] для grayscale

        # Если C > 1, предполагаем RGB или похожее, иначе squeeze до [H, W]
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)  # [1, H, W] → [H, W]

        true_lbl = emotion_map[int(labels[idx])]
        pred_lbl = emotion_map[int(preds[idx])]
        title_color = 'black' if true_lbl == pred_lbl else 'red'

        # Если изображение grayscale, показываем cmap='gray'
        cmap = 'gray' if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap, interpolation='bicubic')
        ax.axis('off')
        ax.set_title(
            f"{idx + 1}\nИст: {true_lbl}\nПред: {pred_lbl}",
            fontsize=14,
            color=title_color
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # оставляем место для suptitle

    # 7. Сохранение в PNG ---------------------------------------------------
    output_path = Path(output_path).expanduser()
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format="png", dpi=300)
    plt.close(fig)
    print(f"Изображение с предсказаниями сохранено в '{output_path}'")
