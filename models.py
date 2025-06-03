import torch
import torch.nn as nn


# ================================================================
# XSNet
# ================================================================
# XSNet3
# ================================================================
class XSNet3(nn.Module):
    """
    XSNet
    -------------
    Архитектура:
      Conv3x3-16 → BN → ReLU → MaxPool2
      Conv3x3-32 → BN → ReLU → MaxPool2
      Conv3x3-64 → BN → ReLU
      AdaptiveAvgPool(1×1)
      FC-256 → BN → ReLU → Dropout(0.3)
      FC-7
    Параметры:
      num_classes (int): число выходных классов (по умолчанию 7).
    Forward(x):
      x – тензор формы (B,1,48,48); возвращает (B,num_classes).
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),  # 48×48
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 24×24

            nn.Conv2d(16, 32, 3, padding=1, bias=False), # 24×24
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 12×12

            nn.Conv2d(32, 64, 3, padding=1, bias=False), # 12×12
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)              # 1×1
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # 64
            nn.Linear(64, 256, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# XSNet4
# ================================================================
class XSNet4(nn.Module):
    """Аналог XSNet3, но с дополнительной Conv 64 каналов."""
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),  # 48
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 24

            nn.Conv2d(16, 32, 3, padding=1, bias=False), # 24
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False), # 24
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 12

            nn.Conv2d(64, 64, 3, padding=1, bias=False), # 12
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# XSNet5
# ================================================================
class XSNet5(nn.Module):
    """Добавляет ещё два сверточных блока (64 и 128 каналов)."""
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 24

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 12

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# XSNet6
# ================================================================
class XSNet6(nn.Module):
    """Самый глубокий XS-вариант: 6 сверточных слоёв, 128 каналов финально."""
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 24

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 12

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)

# ================================================================
# SNet
# ================================================================
# SNet3
# ================================================================
class SNet3(nn.Module):
    """
    Conv5x5-64  → MaxPool2
    Conv5x5-128 → MaxPool2
    Conv3x3-256
    AdaptiveAvgPool(1×1)
    FC-1024 → Dropout(0.3) → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        # --- Сверточная «голова»
        self.features = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=5, padding=2, bias=False),  # 48×48
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # 24×24

            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # 12×12

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        # --- Завершающая часть
        self.pool = nn.AdaptiveAvgPool2d(1)                          # 1×1
        self.classifier = nn.Sequential(
            nn.Flatten(),                                            # 256
            nn.Linear(256, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# SNet4
# ================================================================
class SNet4(nn.Module):
    """
    Conv7x7-32  → MaxPool2
    Conv5x5-64
    Conv3x3-128 → MaxPool2
    Conv3x3-256
    AdaptiveAvgPool(1×1)
    FC-1024 → Dropout → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3, bias=False),   # 48×48
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # 24×24

            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # 12×12

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                            # 256
            nn.Linear(256, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# SNet5
# ================================================================
class SNet5(nn.Module):
    """
    Conv5x5-32  → MaxPool2
    Conv5x5-64
    Conv3x3-128 → MaxPool2
    Conv3x3-128
    Conv3x3-256
    AdaptiveAvgPool → FC-1024 → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),   # 48×48
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # 24×24

            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # 12×12

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                            # 256
            nn.Linear(256, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# SNet6
# ================================================================
class SNet6(nn.Module):
    """
    Conv3x3-32  → MaxPool2
    Conv3x3-64
    Conv3x3-64
    Conv3x3-128 → MaxPool2
    Conv3x3-128
    Conv3x3-256
    AdaptiveAvgPool → FC-1024 → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),   # 48×48
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # 24×24

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                         # 12×12

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                            # 256
            nn.Linear(256, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# MNet
# ================================================================
# MNet3
# ================================================================
class MNet3(nn.Module):
    """
    Conv7x7-96 → MaxPool2
    Conv5x5-192 → MaxPool2
    Conv3x3-384
    AdaptiveAvgPool → FC-1024 → Dropout → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96,  kernel_size=7, padding=3, bias=False),   # 48
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 24

            nn.Conv2d(96, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 12

            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 384
            nn.Linear(384, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# MNet4
# ================================================================
class MNet4(nn.Module):
    """
    Conv7x7-64  → MaxPool2
    Conv5x5-128
    Conv3x3-256 (stride 2)
    Conv3x3-512
    AdaptiveAvgPool → FC-1024 → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=7, padding=3, bias=False),   # 48
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 24

            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),               # 12

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 512
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# MNet5
# ================================================================
class MNet5(nn.Module):
    """
    Conv5x5-64  → MaxPool2
    Conv5x5-128
    Conv3x3-128
    Conv3x3-256 (stride 2)
    Conv3x3-512
    AdaptiveAvgPool → FC-1024 → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=5, padding=2, bias=False),   # 48
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 24

            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),               # 12

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 512
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# MNet6
# ================================================================
class MNet6(nn.Module):
    """
    Conv3x3-64  → MaxPool2
    Conv3x3-64
    Conv3x3-128
    Conv3x3-128 (stride 2)
    Conv3x3-256
    Conv3x3-512
    AdaptiveAvgPool → FC-1024 → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=3, padding=1, bias=False),   # 48
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 24

            nn.Conv2d(64, 64,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),               # 12

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 512
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)



# ================================================================
# LNet
# ================================================================
# LNet3
# ================================================================
class LNet3(nn.Module):
    """
    Архитектура:
        Conv7×7-128 → BN → ReLU → MaxPool2
        Conv5×5-256 → BN → ReLU → MaxPool2
        Conv3×3-512 → BN → ReLU
        AdaptiveAvgPool(1×1)
        FC-2048 → BN → ReLU → Dropout(0.3)
        FC-1024 → BN → ReLU → Dropout(0.3)
        FC-7
    Параметры:
        num_classes (int): количество выходных категорий (по умолчанию 7).
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()

        # ── Сверточная часть ─────────────────────────────────────
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7, padding=3, bias=False),   # 48×48
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 24×24

            nn.Conv2d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 12×12

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # ── Завершающая часть ────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)                           # 1×1 карта
        self.classifier = nn.Sequential(
            nn.Flatten(),                                             # 512 признаков
            nn.Linear(512, 2048, bias=False),
            nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# LNet4
# ================================================================
class LNet4(nn.Module):
    """
    Conv7×7-64  → MP2
    Conv5×5-128
    Conv3×3-256 (stride 2)
    Conv3×3-512
    ↓
    AdaptiveAvgPool → FC-2048 → FC-1024 → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, bias=False),    # 48
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 24

            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),               # 12

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2048, bias=False),
            nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# LNet5
# ================================================================
class LNet5(nn.Module):
    """
    Conv5×5-64  → MP2
    Conv5×5-128
    Conv3×3-256
    Conv3×3-256 (stride 2)
    Conv3×3-512
    ↓
    AdaptiveAvgPool → FC-2048 → FC-1024 → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, bias=False),    # 48
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 24

            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),               # 12

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2048, bias=False),
            nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)


# ================================================================
# LNet6
# ================================================================
class LNet6(nn.Module):
    """
    Conv3×3-64  → MP2
    Conv3×3-64
    Conv3×3-128
    Conv3×3-128 (stride 2)
    Conv3×3-256
    Conv3×3-512
    ↓
    AdaptiveAvgPool → FC-2048 → FC-1024 → FC-7
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),    # 48
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 24

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),               # 12

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2048, bias=False),
            nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)
