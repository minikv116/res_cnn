# ================================================================
#  Сводная таблица по моделям — summary_metrics.csv 
#  Универсальная функция summarize_models
# ---------------------------------------------------------------
"""
Usage example
-------------
from summary_metrics import summarize_models
from my_models import MODEL_ZOO
from dataloaders import create_dataloaders

_, _, test_loader = create_dataloaders('./data/fer2013.csv', batch_size=128)

df = summarize_models(
    models_dir='./models_trained_pth_csv',
    model_zoo=MODEL_ZOO,
    test_loader=test_loader,
    device='cuda:0',    # или 'cpu'
    in_shape=(1, 48, 48)
)
print(df.head())
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List
import warnings

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from ptflops import get_model_complexity_info


__all__ = [
    'summarize_models',
]

# ---------------------------------------------------------------
# Helper: count parameters manually (int)
# ---------------------------------------------------------------

def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ---------------------------------------------------------------
# Helper: compute MACs/FLOPs via ptflops
# ---------------------------------------------------------------

def _calc_flops(model: nn.Module, in_shape: Tuple[int, int, int], device: str | torch.device) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ptflops uses deprecated API warnings
        macs, _ = get_model_complexity_info(
            model.to(device),
            in_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
    # ptflops returns MACs; FLOPs≈2×MACs for Conv/FC layers
    return float(macs) * 2.0

# ---------------------------------------------------------------
# Evaluation on test set (accuracy & f1)
# ---------------------------------------------------------------

def _evaluate(model: nn.Module, loader: DataLoader, device: str | torch.device) -> Tuple[float, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            preds = torch.argmax(out, dim=1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

# ---------------------------------------------------------------
# Main function
# ---------------------------------------------------------------

def summarize_models(
    models_dir: str | Path,
    model_zoo: Dict[str, Callable[[], nn.Module]],
    test_loader: Optional[DataLoader] = None,
    device: str | torch.device = 'cpu',
    in_shape: Tuple[int, int, int] = (1, 48, 48),
    outfile: str = 'summary_metrics.csv'
) -> pd.DataFrame:
    """Генерирует CSV‑отчёт с ключевыми метриками для каждой модели.

    Parameters
    ----------
    models_dir : Path
        Папка, где лежат *<model>_metrics.csv* и веса *<model>_best.pth*.
    model_zoo : dict[str, Callable[[], nn.Module]]
        Ordered‑словарь **имя → конструктор модели**.
        Порядок используется для финальной сортировки строк.
    test_loader : DataLoader | None
        Если передан, будет рассчитан `test_acc` и `f1`.
    device : str | torch.device
        Устройство для инференса при вычислении FLOPs и метрик.
    in_shape : (C, H, W)
        Размер входного тензора для FLOPs.
    outfile : str
        Имя итогового CSV в *models_dir*.

    Returns
    -------
    pd.DataFrame
        Таблица со столбцами:
        ['model', 'best_epoch', 'val_acc', 'test_acc', 'f1', 'params', 'flops']
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Папка {models_dir} не найдена")

    rows: List[dict] = []

    for model_name, constructor in model_zoo.items():
        print(f'Gather metrics for model {model_name}')
        metrics_csv = models_dir / f"{model_name}_metrics.csv"
        weights_pth = models_dir / f"{model_name}_best.pth"

        if not metrics_csv.exists():
            print(f"[WARN] CSV метрик не найден для {model_name}: {metrics_csv}")
            continue
        if not weights_pth.exists():
            print(f"[WARN] Файл весов не найден для {model_name}: {weights_pth}")
            continue

        # ---- 1. Извлекаем лучшую эпоху и val_acc ----
        df_metrics = pd.read_csv(metrics_csv)
        idx_best = df_metrics['val_acc'].idxmax()
        best_epoch = int(df_metrics.loc[idx_best, 'epoch'])
        best_val_acc = float(df_metrics.loc[idx_best, 'val_acc'])

        # ---- 2. Загружаем модель и веса ----
        model = constructor()
        state = torch.load(weights_pth, map_location='cpu')
        model.load_state_dict(state)
        model.to(device)

        # ---- 3. Считаем параметры и FLOPs ----
        params = _count_parameters(model)
        flops = _calc_flops(model, in_shape, device)

        # ---- 4. Тестовые метрики (если есть loader) ----
        test_acc = f1 = float('nan')
        if test_loader is not None:
            test_acc, f1 = _evaluate(model, test_loader, device)

        rows.append({
            'model': model_name,
            'best_epoch': best_epoch,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'f1': f1,
            'params': params,
            'flops': flops
        })

        # освобождаем GPU память, если нужно
        del model
        torch.cuda.empty_cache()

    # ---- 5. Собираем DataFrame и сохраняем ----
    df_summary = pd.DataFrame(rows)

    # Сортировка в том же порядке, как в model_zoo
    df_summary['order'] = df_summary['model'].apply(lambda m: list(model_zoo.keys()).index(m))
    df_summary = df_summary.sort_values('order').drop(columns=['order'])

    outfile_path = models_dir / outfile
    df_summary.to_csv(outfile_path, index=False)
    print(f"[INFO] Итоговый CSV сохранён → {outfile_path}")

    return df_summary
