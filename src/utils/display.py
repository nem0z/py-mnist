import matplotlib.pyplot as plt

def display_one(img, label=""):
    plt.imshow(img, cmap="gray")
    plt.title = f"Display image of {label}"
    plt.show()

def display_many(imgs, x, y):
    _, axs = plt.subplots(x, y)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img, cmap="gray")
    plt.show()

            