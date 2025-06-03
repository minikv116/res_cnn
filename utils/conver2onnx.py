# ================================================================
# Конвертер моделей в формат ONNX
# ---------------------------------------------------------------
# Скрипт/функция convert_to_onnx(
#     models_dir: str | Path,   # папка с <model>_best.pth
#     onnx_dir:   str | Path,   # куда сохранять *.onnx
#     model_zoo:  Dict[str, Callable[[], nn.Module]],
#     input_shape: Tuple[int, int, int, int] = (1, 1, 48, 48),
#     device: str | torch.device = "cpu",
#     opset: int = 12
# )
# ---------------------------------------------------------------
# Основные шаги:
# 1. Находит все *_best.pth в models_dir.
# 2. Для каждого имени загружает соответствующий класс из model_zoo.
# 3. Экспортирует в ONNX (ir_version >= 7).
# 4. Сохраняет имена вида <model>.onnx в onnx_dir.
# ---------------------------------------------------------------
# Пример вызова:
# >>> from models_library import MODEL_ZOO
# >>> convert_to_onnx('./best_models', './onnx_models', MODEL_ZOO, device='cuda:0')
# ================================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Callable, Tuple
import importlib

import torch
import torch.nn as nn

__all__ = ["convert_to_onnx"]


def convert_to_onnx(
    models_dir: str | Path,
    onnx_dir: str | Path,
    model_zoo: Dict[str, Callable[[], nn.Module]],
    input_shape: Tuple[int, int, int, int] = (1, 1, 48, 48),
    device: str | torch.device = "cpu",
    opset: int = 12,
) -> None:
    """Экспорт всех *_best.pth в ONNX.

    Args:
        models_dir: каталог, содержащий файлы весов вида ``<model>_best.pth``.
        onnx_dir:   выходной каталог для *.onnx.
        model_zoo:  словарь «имя модели → конструктор nn.Module».
        input_shape: форма dummy‑входа (N, C, H, W).
        device:     устройство для инициализации модели.
        opset:      версия ONNX‑opset.
    """
    models_dir = Path(models_dir)
    onnx_dir   = Path(onnx_dir)
    onnx_dir.mkdir(exist_ok=True, parents=True)

    weight_files = sorted(models_dir.glob('*_best.pth'))
    if not weight_files:
        raise FileNotFoundError(f"No *_best.pth files found in {models_dir}")

    # Подготовим фиктивный вход
    dummy_input = torch.randn(*input_shape, device=device)

    for w_path in weight_files:
        # Извлекаем <model> из «<model>_best.pth»
        model_name = w_path.stem[:-5] if w_path.stem.endswith('_best') else w_path.stem
        if model_name not in model_zoo:
            print(f"[WARN] Model '{model_name}' not found in model_zoo → skip")
            continue

        print(f"[INFO] Exporting {model_name} …", flush=True)
        model: nn.Module = model_zoo[model_name]()
        model.load_state_dict(torch.load(w_path, map_location=device))
        model.eval().to(device)

        onnx_path = onnx_dir / f"{model_name}.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
        print(f"      → saved to {onnx_path.relative_to(Path.cwd()) if onnx_path.is_relative_to(Path.cwd()) else onnx_path}")

