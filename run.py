from models import XSNet3, XSNet4, XSNet5, XSNet6, SNet3, SNet4, SNet5, SNet6, MNet3, MNet4, MNet5, MNet6, LNet3, LNet4, LNet5, LNet6, MODEL_ZOO
from train import train_models
from data_loader_fer2013 import create_dataloaders
from utils.get_trainerd_summary import summarize_models
from utils.best_model import select_best_models
from utils.conver2onnx import convert_to_onnx
from utils.merge_metrics import merge_metrics_perf



# Установка и проверка доступности GPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")





if __name__ == "__main__":

    # ================================================================
    #  Список моделей для обучения
    # ================================================================
    models_to_train = [
        ('XSNet3', XSNet3()),
        ('XSNet4', XSNet4()),
        ('XSNet5', XSNet5()),
        ('XSNet6', XSNet6()),
        ('SNet3', SNet3()),
        ('SNet4', SNet4()),
        ('SNet5', SNet5()),
        ('SNet6', SNet6()),
        ('MNet3', MNet3()),
        ('MNet4', MNet4()),
        ('MNet5', MNet5()),
        ('MNet6', MNet6()),
        ('LNet3', LNet3()),
        ('LNet4', LNet4()),
        ('LNet5', LNet5()),
        ('LNet6', LNet6())
    ] 

    train_loader, val_loader, test_loader = create_dataloaders('./data/fer2013.csv', batch_size=128, num_workers=8, skip_nonface=True)


    out_dir_base = './series_result_'
    series_n = 11
    model_dirs_list = []
    for s_ser in range(1, series_n):
        out_dir_current = f'{out_dir_base}{s_ser}'
        model_dirs_list.append (out_dir_current)

        # train all models
        train_models(models_list=models_to_train,
                    out_dir=out_dir_current,
                    train_loader= train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device)

        # get summary csv
        summarize_models(
            models_dir=out_dir_current,
            model_zoo=MODEL_ZOO,
            test_loader=test_loader,
            device='cuda:0',    # или 'cpu'
            in_shape=(1, 48, 48)
        )
    

    out_dir = "./best_models"
    onnx_dir = "./onnx_models"
    # select only best val_acc models drom trained models
    select_best_models(model_dirs_list, out_dir=out_dir) 

    convert_to_onnx(
        models_dir=out_dir,
        onnx_dir=onnx_dir,
        model_zoo=MODEL_ZOO,
        device="cpu",
    )


    # merge all metrics to single file
    # summary metrics of best models + performance metrics
    # do if performce test done

    #best_csv_path  = "./best_models/summary_metrics.csv",
    #perf_csv_path  = "./onnx_models_v0/results_pi5.csv",
    #out_csv_path   = "./best_models/summary_metrics_final.csv" 

    #combined_df = merge_metrics_perf(
    #    best_csv_path=best_csv_path,
    #    perf_csv_path=perf_csv_path,
    #    out_csv_path=out_csv_path          
    #)
    #print(combined_df.head())
