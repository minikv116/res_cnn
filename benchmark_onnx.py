# ================================================================
# Скрипт‑бенчмарк ONNX‑моделей на Raspberry Pi 5 (CPU)
# -------------------------------------------------------
# usage (пример):
#   python benchmark_onnx.py /home/pi/onnx_models results_pi5.csv
# -------------------------------------------------------
# • Аргументы:
#     models_dir   — каталог с *.onnx
#     outfile_csv  — имя CSV (в том же каталоге, если путь не указан)
# • Для каждой модели выполняет:
#     ‑ 20 «прогревочных» прогонов
#     ‑ 200 измерений времени (сек)
#     ‑ latency  = median(time) * 1000  (мс)
#     ‑ fps      = mean(1 / time)
# • Сохраняет summary CSV с полями: model, latency_ms, fps
# ================================================================
# Пример запуска с ограничением по производительности 
#   systemd-run --user --scope -p AllowedCPUs=0 -p CPUQuota=100% \
#       python ./benchmark_onnx.py models results_pi5.csv   
#=================================================================
from __future__ import annotations
import argparse
import time
from pathlib import Path
from statistics import median, mean

import numpy as np
import pandas as pd
import onnxruntime as ort


# ---------- 1. Функции ------------------------------------------------------

def infer_session(session: ort.InferenceSession,
                  input_name: str,
                  warmup_iters: int = 20,
                  bench_iters: int = 200) -> tuple[float, float]:
    """Возвращает (latency_ms, fps)."""
    # Получаем входную форму модели и создаём фиктивный тензор одного изображения
    _, c, h, w = session.get_inputs()[0].shape  # shape: [N,C,H,W]; N может быть None
    dummy = np.random.rand(1, c, h, w).astype(np.float32)

    # Warm‑up
    for _ in range(warmup_iters):
        _ = session.run(None, {input_name: dummy})

    times: list[float] = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        times.append(time.perf_counter() - t0)

    lat_ms = median(times) * 1000.0
    fps = mean(1.0 / np.array(times))
    return lat_ms, fps


def benchmark_models(models_dir: Path,
                     outfile: Path | None = None,
                     warmup: int = 20,
                     iters: int = 200) -> pd.DataFrame:
    models = sorted(models_dir.glob("*.onnx"))
    if not models:
        raise FileNotFoundError(f"В каталоге {models_dir} нет .onnx файлов")

    results: list[dict[str, float | str]] = []
    print(f"[INFO] Найдено моделей: {len(models)}\n")

    for model_path in models:
        model_name = model_path.stem
        print(f"Проверка модели: {model_name} …", end=" ")
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        lat_ms, fps = infer_session(sess, input_name, warmup, iters)
        print(f"latency={lat_ms:.2f} ms | fps={fps:.2f}")
        results.append({
            "model": model_name,
            "latency_ms": round(lat_ms, 3),
            "fps": round(fps, 3)
        })

    df = pd.DataFrame(results)
    # Сохраняем CSV рядом с моделями, если путь не указан
    if outfile is None:
        outfile = models_dir / "summary_latency.csv"
    else:
        outfile = Path(outfile)
        if not outfile.is_absolute():
            outfile = models_dir / outfile
    df.to_csv(outfile, index=False)
    print(f"\n[INFO] Результаты сохранены → {outfile.relative_to(Path.cwd())}")
    return df


# ---------- 2. CLI ----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ONNX models on Raspberry Pi 5 (CPU)")
    p.add_argument("models_dir", type=Path, help="Папка с .onnx моделями")
    p.add_argument("outfile",    nargs="?", default=None,
                   help="Имя выходного CSV (по умолчанию summary_latency.csv)")
    p.add_argument("--warmup", type=int, default=20, help="Количество прогревочных итераций")
    p.add_argument("--iters",  type=int, default=200, help="Количество измерений")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_models(args.models_dir, args.outfile, args.warmup, args.iters)
