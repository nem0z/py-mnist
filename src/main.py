import gzip
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset

DATASETS_PATH = "./datasets/"
TRAIN_LABEL = "train-labels-idx1-ubyte.gz"
TRAIN_IMAGE = "train-images-idx3-ubyte.gz"
T10K_LABEL = "t10k-labels-idx1-ubyte.gz"
T10K_IMAGE = "t10k-images-idx3-ubyte.gz"

train_dataset = Dataset()
t10k_dataset = Dataset()
    
train_dataset.load(DATASETS_PATH+TRAIN_LABEL, DATASETS_PATH+TRAIN_IMAGE)
t10k_dataset.load(DATASETS_PATH+T10K_LABEL, DATASETS_PATH+T10K_IMAGE)
    
train_dataset.plot_labels()
t10k_dataset.plot_labels()

input("Press [enter] to quit...")

