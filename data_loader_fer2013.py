# ================================================================
#        • Загрузчики FER‑2013 с учётом «нелица» 
#        • Фиксированный random‑seed по умолчанию
#        • Датасет возвращает (img, label)  → совместимо с train‑loops
#        • Debug‑блок запускается напрямую без переменных окружения
# ================================================================
from __future__ import annotations
import os, random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------- 0. Глобальный seed ----------------------------
DEFAULT_SEED = 42

def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ---------------- 1. Индексы «нелица» ---------------------------
NONFACE_IDX = [
     59, 2059, 2171, 2809, 3262, 3931, 4275, 5082, 5439, 5722, 5881, 6102, 6458, 6699,
     7172, 7496, 7527, 7629, 8030, 8423, 8737, 8856, 9026, 9500, 9673, 9679, 10023,
     10423, 10657, 11244, 11286, 11295, 11846, 12289, 12352, 13148, 13402, 13697,
     13839, 13988, 14279, 15144, 15553, 15835, 15838, 15894, 16540, 17081, 19238,
     19632, 20222, 20712, 20817, 21817, 22198, 22314, 22407, 22927, 23596, 23894,
     24053, 24441, 24891, 25219, 25603, 25909, 26383, 26860, 26897, 28601, 29447,
     29557, 30002, 30981, 31127, 31825, 34334, 35121, 35469, 35632, 35743
]
NONFACE_CSV = Path('nonface_indices.csv')
if not NONFACE_CSV.exists():
    pd.Series(sorted(set(NONFACE_IDX))).to_csv(NONFACE_CSV, index=False, header=False)
    try:
        rel = NONFACE_CSV.relative_to(Path.cwd())
        print(f"[INFO] Сохранён список нелиц → {rel}")
    except ValueError:
        print(f"[INFO] Сохранён список нелиц → {NONFACE_CSV}")

NONFACE_SET = set(NONFACE_IDX)

# ---------------- 2. Датасет ------------------------------------
class FER2013DatasetFiltered(Dataset):
    """FER‑2013 с возможностью исключить нелица."""
    def __init__(self, csv_file: str | Path, usage: str,
                 transform=None, skip_nonface: bool = True):
        df = pd.read_csv(csv_file)
        # сохраняем исходный индекс в новом столбце 'orig_idx'
        df = df.reset_index(names='orig_idx')
        df = df[df['Usage'] == usage].reset_index(drop=True)
        if skip_nonface:
            df = df[~df['orig_idx'].isin(NONFACE_SET)].reset_index(drop=True)
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ').reshape(48, 48)
        if self.transform:
            img = self.transform(img)
        label = int(row['emotion'])
        return img, label

# ---------------- 3. Трансформации -------------------------------
train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------------- 4. worker init ---------------------------------

def worker_init_fn(worker_id):
    worker_seed = DEFAULT_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------- 5. Фабрика DataLoader --------------------------

def create_dataloaders(csv_path: str | Path,
                       batch_size: int = 128,
                       num_workers: int = 4,
                       skip_nonface: bool = True,
                       seed: int = DEFAULT_SEED) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Возвращает train/val/test DataLoader‑ы."""
    set_seed(seed)

    train_ds = FER2013DatasetFiltered(csv_path, 'Training', transform=train_tfms, skip_nonface=skip_nonface)
    val_ds   = FER2013DatasetFiltered(csv_path, 'PublicTest', transform=test_tfms, skip_nonface=skip_nonface)
    test_ds  = FER2013DatasetFiltered(csv_path, 'PrivateTest', transform=test_tfms, skip_nonface=skip_nonface)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True)

    return train_loader, val_loader, test_loader

