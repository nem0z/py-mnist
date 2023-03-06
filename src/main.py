import numpy as np

from dataset import Dataset
from model import Model
from utils.display import display_one, display_many

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

means = train_dataset.mean()
display_many([img for img in means.values()], 2, 5)

model = Model(train_dataset, t10k_dataset)

model.train()
model.test()

accuracy = model.accuracy()
print(f"Accuracy: {accuracy:.4f}")

model.plot_sample(False)
model.plot_confusion_matrix()

