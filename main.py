import torch
import wandb
import os
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_utils import GuitarDataModule, CNNClassifier, GuitarDataset
from pytorch_lightning.callbacks import ModelCheckpoint
#from suck_my_balls import load_files_from_directories, extract_mel_mfcc_multible_files, dataset_combine_multible_files

################HYPERPARAMETERS#####################
HOME = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME, "Guitar_Data_Mel_Spectrograms_Adjusted_Clipping") #Directory containing images of the mel spectograms
#DATA_DIR = os.path.join(HOME, "Guitar_Data_MFCCs") #Directory containing images of the MFFCs
CSV_DIR = "./metadata.csv" #CSV File with labels
####################################################
sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'test_loss'
        },
    'parameters': {
        'batch_size': {'values': [32, 64, 128]},
        'lr': {'values': [0.00001]},
        'optimizer': {'values':['Adam','SGD','Adadelta','Adagrad','AdamW','NAdam','RAdam','RMSprop','Rprop']}
     }
    # 'parameters': {
    #     'batch_size': {'values': [64]},
    #     'lr': {'values': [0.001]},
    #     'optimizer': {'values':['RAdam']}
    #  }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="HOLA-PEPSICOLA")
####################################################33

def main():
    run = wandb.init()
    lr  =  wandb.config.lr
    bs = wandb.config.batch_size
    optimizer= wandb.config.optimizer
    run.name = f"Lr: {str(lr)} | BatchSize: {str(bs)} | Optimizer: {optimizer}"

    torch.set_float32_matmul_precision('medium') #To utilize Tensor Cores in the NVIDIA A40. Less precise multiplications, but faster processing time
    
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Metric to monitor
    mode='min',          # 'min' if the metric should be minimized, 'max' if maximized
    save_top_k=1         # Save only the best model
    )
    train_loader = GuitarDataModule(DATA_DIR, CSV_DIR, batch_size=bs)
    trainer = pl.Trainer(max_epochs=200, accelerator='gpu', devices=1, log_every_n_steps=5, callbacks=[checkpoint_callback])
    model = CNNClassifier(optimizer,lr)

    trainer.fit(model, train_loader)
    trainer.test(model, train_loader)

    torch.cuda.empty_cache() #Release GPU memory, or else CUDA will run out of memory when sweeping

    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)
    print("End of main")
wandb.agent(sweep_id=sweep_id, function=main)
if __name__ == "__main__":
    main()