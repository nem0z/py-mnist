import matplotlib.pyplot as plt

def display_one(img, title=""):
    plt.imshow(img, cmap="gray")
    plt.title = f"Display image of {title}"
    plt.show()

def display_many(imgs, x, y, title=""):
    fig, axs = plt.subplots(x, y)
    fig.suptitle(title)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img, cmap="gray")
    plt.show()

            