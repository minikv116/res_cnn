import torch
from pathlib import Path
from visualize_data.plot_confusion_matrix import plot_cf_and_evaluate_model_metrics
from visualize_data.plot_dynamic_valacc import plot_training_dynamics
from visualize_data.plot_latency_hist import plot_latency_hist
from visualize_data.plot_fps_hist import plot_fps_hist
from data_loader_fer2013 import create_dataloaders


if __name__ == "__main__":

    GROUP_COLORS = {
        "XSNet": "tab:blue",
        "SNet":  "tab:orange",
        "MNet":  "tab:green",
        "LNet":  "tab:red",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")


    train_loader, val_loader, test_loader = create_dataloaders('./data/fer2013.csv', batch_size=128, num_workers=8, skip_nonface=True)


    # Пример вызова:
    plot_training_dynamics(
        logs_dir="./logs",      # Папка с CSV-файлами
        category="XS",          # Фильтр файлов по префиксу 'xsnet'
        output_dir="./plots/XS" # Папка, куда сохранять PNG-файлы
    )


    metrics = plot_cf_and_evaluate_model_metrics(
        model_name="MyNet",
        weights_dir="./best_models",
        loader=test_loader,
        output_dir="./metrics_plots",
        device="cuda:0",  # или "cpu"
        class_names=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    )


    csv_file = Path("./best_models/combined_metrics.csv")

    plot_latency_hist(
        csv_path     = csv_file,
        latency_col  = "latency_ms",             # имя столбца с задержкой
        font_size    = 16,                       # размер шрифта
        margin       = 0.05,                     # ±5 % запас по оси Y
        group_colors = GROUP_COLORS,             # словарь цветов по префиксу
        output_path  = "./plots/latency_hist.png"  # сохраняем в файл PNG
    )
    # После запуска появится файл: "./plots/latency_hist.png"


    # Построим гистограмму FPS и сохраним её в файл:
    plot_fps_hist(
        csv_path    = csv_file,
        fps_col     = "fps",                      # столбец с FPS
        font_size   = 16,                         # размер шрифта
        margin      = 0.05,                       # ±5 % диапазон по Y
        group_colors= GROUP_COLORS,               # цвета по префиксу модели
        output_path = "./plots/fps_histogram.png" # сохраняем график
    )
    # В результате появится файл "./plots/fps_histogram.png"
