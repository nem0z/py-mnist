import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
        plt.show()
        
    def reshape(self):
        return self.images.reshape(len(self.images), len(self.images[0]) * len(self.images[0][0]))
        
    def mean(self):
        unique_labels = np.unique(self.labels)

        mean_images = {}
        for label in unique_labels:
            label_images = self.images[self.labels == label]
            stacked_images = np.stack(label_images)
            mean_image = np.mean(stacked_images, axis=0)
            mean_images[label] = mean_image
            
        return mean_images
        