import numpy as np
import matplotlib.pyplot as plt

from utils.parse import parse_label, parse_image

class Dataset:
    def __init__(self, labels=[], images=[]):
        self.labels = labels
        self.data = images
        
    def load(self, label_path, images_path):
        self.labels = parse_label(label_path)
        self.images = parse_image(images_path)
        
    def plot_labels(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.bar(unique, counts)
        plt.show(block = False)