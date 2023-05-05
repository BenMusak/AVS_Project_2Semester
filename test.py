import torch

import os
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_utils import GuitarDataModule, CNNClassifier, GuitarDataset

################HYPERPARAMETERS#####################
HOME = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME, "Guitar_Data_Mel_Spectograms") #Directory containing images of the mel spectograms
#DATA_DIR = os.path.join(HOME, "Guitar_Data_MFCCs") #Directory containing images of the MFFCs
CSV_DIR = "./metadata.csv" #CSV File with labels
BATCH_SIZE = 128
CHECKPOINT_DIR = "./lightning_logs/version_77/checkpoints/epoch=0-step=9.ckpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
####################################################

def main():

    torch.set_float32_matmul_precision('medium') #To utilize Tensor Cores in the NVIDIA A40. Less precise multiplications, but faster processing time

    model = CNNClassifier().load_from_checkpoint(CHECKPOINT_DIR)
    model.eval()

    train_loader = GuitarDataModule(DATA_DIR, CSV_DIR, BATCH_SIZE)
    trainer = pl.Trainer(max_epochs=1, accelerator='gpu', devices=1, log_every_n_steps=5)

    trainer.test(model, train_loader)

if __name__ == "__main__":
    main()