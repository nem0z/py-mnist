import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

class Model:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.predicted_labels = []

        
    def train(self):
        self.knn.fit(self.train_set.reshape(), self.train_set.labels)
        
    def test(self):
        self.predicted_labels = self.knn.predict(self.test_set.reshape())
        return self.predicted_labels
        
    def accuracy(self):
        return np.mean(self.predicted_labels == self.test_set.labels)
    
    def plot_sample(self, ok):
        sample = [i for i in range(len(self.predicted_labels)) if (self.predicted_labels[i] == self.test_set.labels[i]) == ok]
        
        _, axes = plt.subplots(nrows=3, ncols=5, figsize=(8, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(self.test_set.images[sample[i]], cmap="gray")
            ax.set_title(f"Predicted: {self.predicted_labels[sample[i]]}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.test_set.labels, self.predicted_labels)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
            xticklabels=np.arange(10), yticklabels=np.arange(10),
            xlabel="Predicted label", ylabel="True label")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        plt.show()
        