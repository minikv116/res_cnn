# ================================================================
#  merge_metrics_and_perf.py  –  объединяем сводные метрики
# ================================================================
"""
Функция merge_metrics_perf:

1. Читает CSV с «лучшими метриками» (summary_metrics.csv,
   полученный select_best_models) и CSV с результатами
   ONNX-бенчмарка (latency, fps, fps@2, …).

2. Выполняет left-join по столбцу 'model', добавляя в таблицу
   latency/fps-колонки.

3. Сохраняет объединённый файл (по умолчанию combined_metrics.csv)
   в ту же папку, где находится файл best_csv_path.

Использование:
------------------------------------------------------------------
combined_df = merge_metrics_perf(
    best_csv_path  = "./best_models/summary_metrics.csv",
    perf_csv_path  = "./onnx_models/results_pi5.csv",
    out_csv_path   = None            # ← сохранить рядом с best_csv
)
print(combined_df.head())
------------------------------------------------------------------
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def merge_metrics_perf(
        best_csv_path: str | Path,
        perf_csv_path: str | Path,
        out_csv_path: str | Path | None = None
) -> pd.DataFrame:
    best_csv_path = Path(best_csv_path).expanduser()
    perf_csv_path = Path(perf_csv_path).expanduser()

    # --- чтение -------------------------------------------------
    best_df = pd.read_csv(best_csv_path)
    perf_df = pd.read_csv(perf_csv_path)

    # --- проверка уникальности 'model' -------------------------
    if best_df["model"].duplicated().any():
        raise ValueError("В best-метриках встречаются дубли model")
    if perf_df["model"].duplicated().any():
        raise ValueError("В perf-CSV встречаются дубли model")

    # --- объединяем (left-join) --------------------------------
    merged = best_df.merge(perf_df, on="model", how="left")

    # --- сохранение --------------------------------------------
    if out_csv_path is None:
        out_csv_path = best_csv_path.parent / "combined_metrics.csv"
    out_csv_path = Path(out_csv_path).expanduser()
    merged.to_csv(out_csv_path, index=False)

    print(f"[✓] Сохранён объединённый CSV → {out_csv_path}")
    return merged
