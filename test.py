import torch

import os
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_utils import GuitarDataModule, CNNClassifier, GuitarDataset

################HYPERPARAMETERS#####################
HOME = os.path.expanduser('~')
#DATA_DIR = os.path.join(HOME, "Guitar_Data_Mel_Spectograms") #Directory containing images of the mel spectograms
DATA_DIR = os.path.join(HOME, "FenderUS_Test_Data/Mel_Spectrograms") #Directory containing images of the mel spectograms
CSV_DIR = "./metadata_jesper_testdata.csv" #CSV File with labels
CHECKPOINT_DIR = "/home/ubuntu/AVS8_CNN/lightning_logs/version_131_best_model/checkpoints/epoch=91-step=1932.ckpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
####################################################

def main():

    torch.set_float32_matmul_precision('medium') #To utilize Tensor Cores in the NVIDIA A40. Less precise multiplications, but faster processing time
    model = CNNClassifier.load_from_checkpoint(CHECKPOINT_DIR, only_test_set=True)
    model.eval()

    train_loader = GuitarDataModule(DATA_DIR, CSV_DIR, 64)
    trainer = pl.Trainer(max_epochs=1, accelerator='gpu', devices=1, log_every_n_steps=5)

    trainer.test(model, train_loader)

if __name__ == "__main__":
    main()