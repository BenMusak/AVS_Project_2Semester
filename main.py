import torch

import os
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_utils import GuitarDataModule, CNNClassifier, GuitarDataset
#from suck_my_balls import load_files_from_directories, extract_mel_mfcc_multible_files, dataset_combine_multible_files


HOME = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME, "Guitar_Data_Mel_Spectograms") #Directory containing images of the mel spectograms
#DATA_DIR = os.path.join(HOME, "Guitar_Data_MFCCs") #Directory containing images of the MFFCs
CSV_DIR = os.path.join(HOME, "AVS8_CNN/metadata.csv") #CSV File with labels

BATCH_SIZE = 128

def main():
    print("Running CNN Classifier on images in: ", DATA_DIR)

    torch.set_float32_matmul_precision('medium') #To utilize Tensor Cores in the NVIDIA A40. Less precise multiplications, but faster processing time

    train_loader = GuitarDataModule(DATA_DIR, CSV_DIR, BATCH_SIZE)
    trainer = pl.Trainer(max_epochs=25, accelerator='gpu', devices=1, log_every_n_steps=10)
    model = CNNClassifier()

    trainer.fit(model, train_loader)
    trainer.test(model, train_loader)

    print("End of main")
if __name__ == "__main__":
    main()