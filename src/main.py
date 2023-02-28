import gzip
import numpy as np
from utils.parse import parse_label, parse_image
DATASETS_PATH = "./datasets/"
TRAIN_LABEL = "train-labels-idx1-ubyte.gz"
TRAIN_IMAGE = "train-images-idx3-ubyte.gz"
T10K_LABEL = "t10k-labels-idx1-ubyte.gz"
T10K_IMAGE = "t10k-images-idx3-ubyte.gz"
    
train_lables = parse_label(DATASETS_PATH + TRAIN_LABEL)
train_images = parse_image(DATASETS_PATH + TRAIN_IMAGE)

labels = parse_label(DATASETS_PATH + T10K_LABEL)
images = parse_image(DATASETS_PATH + T10K_IMAGE)
    
    
print(len(train_images), len(train_lables))
print(len(images), len(labels))
