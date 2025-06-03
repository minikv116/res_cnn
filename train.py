
# ================================================================
#  Функция обучения группы моделей
#  (SGD + ReduceLROnPlateau — сохранение .pth и .csv)
# ================================================================
from pathlib import Path
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


def accuracy(out, y):
    """Top-1 accuracy."""
    preds = torch.argmax(out, dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, epoch, epochs, device):
    """
    Обучение за одну эпоху с единственным tqdm-прогресс-баром.
    Возвращает средние loss и accuracy по эпохе.
    """
    model.train()
    running_loss, running_acc = 0.0, 0.0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1:02d}/{epochs:02d}', leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_acc  += accuracy(out, y) * x.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Валидация / тест: средние loss и accuracy."""
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out  = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        acc_sum  += accuracy(out, y) * x.size(0)
    n = len(loader.dataset)
    return loss_sum / n, acc_sum / n


def train_models(models_list,
                 out_dir: str | Path,
                 train_loader,
                 val_loader,
                 test_loader,
                 device: torch.device,
                 epochs: int = 110,
                 lr_init: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 sched_factor: float = 0.5,
                 sched_patience: int = 5,
                 sched_min_lr: float = 1e-6):
    """Обучает несколько моделей и сохраняет лучшие веса (.pth) и метрики (.csv).

    Args:
        models_list (list[tuple[str, nn.Module]]): список (имя, модель).
        out_dir (str | Path): каталог для сохранения .pth и .csv.
        train_loader, val_loader, test_loader: dataloaders.
        device: torch.device, где идёт обучение.
        epochs (int, optional): число эпох. Defaults to 150.
        lr_init (float, optional): начальный learning rate. Defaults to 0.1.
        momentum (float, optional): momentum для SGD. Defaults to 0.9.
        weight_decay (float, optional): L2‑регуляризация. Defaults to 1e-4.
        sched_factor (float): во сколько раз снижать LR. Defaults to 0.5.
        sched_patience (int): сколько эпох ждать без улучшения. Defaults to 5.
        sched_min_lr (float): минимально допустимый LR. Defaults to 1e-6.

    Returns:
        list[dict]: истории обучения (loss/acc) каждой модели.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_histories = []

    for name, model in models_list:
        print(f"\n===== [ {name} ] training on {device} =====")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=lr_init,
                              momentum=momentum,
                              weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",          # минимизируем val‑loss
            factor=sched_factor,  # снижение LR в 'factor' раз
            patience=sched_patience,
            min_lr=sched_min_lr
        )

        best_state = copy.deepcopy(model.state_dict())
        best_val   = -1.0
        hist = {
            'model': name,
            'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  []
        }

        for epoch in range(epochs):
            tl, ta = train_one_epoch(model, train_loader, criterion,
                                     optimizer, epoch, epochs, device)
            vl, va = evaluate(model, val_loader, criterion, device)
            # ReduceLROnPlateau требует метрику; используем val_loss
            scheduler.step(vl)

            hist['train_loss'].append(tl);  hist['val_loss'].append(vl)
            hist['train_acc'].append(ta);   hist['val_acc'].append(va)

            print(f"Epoch {epoch+1:03d}/{epochs:03d} | "
                  f"Train: loss={tl:.3f} acc={ta:.3f} | "
                  f"Val: loss={vl:.3f} acc={va:.3f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

            # Сохраняем лучшие веса по val-accuracy
            if va > best_val:
                best_val = va
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, out_dir / f"{name}_best.pth")

        # ── Тестирование лучшей модели ───────────────────────────
        model.load_state_dict(best_state)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        hist['test_loss'] = test_loss
        hist['test_acc']  = test_acc
        all_histories.append(hist)

        # ── Сохранение истории в CSV ─────────────────────────────
        pd.DataFrame({
            'epoch': range(1, epochs + 1),
            'train_loss': hist['train_loss'],
            'val_loss':   hist['val_loss'],
            'train_acc':  hist['train_acc'],
            'val_acc':    hist['val_acc']
        }).to_csv(out_dir / f"{name}_metrics.csv", index=False)

        print(f">>> {name}: BEST val_acc={best_val:.3f} | test_acc={test_acc:.3f}")

    return all_histories