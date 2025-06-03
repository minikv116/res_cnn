# ================================================================
#     •  Отбор лучших моделей из нескольких опытов
#     • Из указанных каталогов ищет summary_metrics.csv
#     • Для каждой модели выбирает вариант с максимальным test_acc
#     • Копирует веса *.pth и CSV метрик в ./best_models
#     • Формирует объединённый summary_metrics.csv
# ---------------------------------------------------------------
# Использование:
#   MODEL_DIRS = ["./run1", "./run2", "./models_trained_pth_csv"]
#   select_best_models(MODEL_DIRS, out_dir="./best_models")
# ================================================================
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def select_best_models(dirs: Iterable[str | Path], out_dir: str | Path = "./best_models") -> pd.DataFrame:
    """Выбирает лучшие модели по test_acc из нескольких каталогов.

    Параметры
    ----------
    dirs : Iterable[str | Path]
        Список путей, в каждом из которых ожидается summary_metrics.csv
    out_dir : str | Path
        Папка, куда копируются лучшие .pth, *_metrics.csv и где
        создаётся итоговый summary_metrics.csv

    Возвращает
    ---------
    pd.DataFrame
        Сводная таблица лучших моделей
    """
    dirs = [Path(d) for d in dirs]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []

    # 1. Собираем все summary_metrics.csv
    for d in dirs:
        csv_path = d / "summary_metrics.csv"
        if not csv_path.exists():
            print(f"[WARN] Файл {csv_path} не найден, пропускаю.")
            continue
        df = pd.read_csv(csv_path)
        df["source_dir"] = str(d.resolve())
        frames.append(df)
        print(f"[INFO] Добавлен {csv_path} ({len(df)} моделей)")

    if not frames:
        raise FileNotFoundError("Ни одного summary_metrics.csv не найдено …")

    all_df = pd.concat(frames, ignore_index=True)

    # 2. Выбираем запись с максимальным test_acc для каждой модели
    sort_cols = ["model", "val_acc"]
    all_df_sorted = all_df.sort_values(sort_cols, ascending=[True, False])
    best_df = all_df_sorted.groupby("model", as_index=False).first()

    # 3. Копируем соответствующие файлы
    for _, row in best_df.iterrows():
        src_dir = Path(row["source_dir"])
        model_name = row["model"]

        # пути исходных файлов
        pth_src  = src_dir / f"{model_name}_best.pth"
        csv_src  = src_dir / f"{model_name}_metrics.csv"

        # пути назначения
        pth_dst = out_dir / pth_src.name
        csv_dst = out_dir / csv_src.name

        for src, dst in [(pth_src, pth_dst), (csv_src, csv_dst)]:
            if src.exists():
                shutil.copy2(src, dst)
                print(f"[COPY] {src} → {dst}")
            else:
                print(f"[WARN] Не найден файл {src}")

    # 4. Сохраняем объединённый summary_metrics.csv
    summary_path = out_dir / "summary_metrics.csv"
    best_df.to_csv(summary_path, index=False)
    print(f"[OK] Итоговый CSV сохранён: {summary_path}")

    return best_df
