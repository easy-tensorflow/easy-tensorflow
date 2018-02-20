import matplotlib.pyplot as plt
import numpy as np


def plot_images(original_images, noisy_images, reconstructed_images):
    """
    Create figure of original and reconstructed image.
    :param original_image: original images to be plotted, (?, img_h*img_w)
    :param noisy_image: original images to be plotted, (?, img_h*img_w)
    :param reconstructed_image: reconstructed images to be plotted, (?, img_h*img_w)
    """
    num_images = original_images.shape[0]
    fig, axes = plt.subplots(num_images, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=.1, wspace=0)

    img_h = img_w = np.sqrt(original_images.shape[-1]).astype(int)
    for i, ax in enumerate(axes):
        # Plot image.
        ax[0].imshow(original_images[i].reshape((img_h, img_w)), cmap='gray')
        ax[1].imshow(noisy_images[i].reshape((img_h, img_w)), cmap='gray')
        ax[2].imshow(reconstructed_images[i].reshape((img_h, img_w)), cmap='gray')

        # Remove ticks from the plot.
        for sub_ax in ax:
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])

    for ax, col in zip(axes[0], ["Original Image", "Noisy Image", "Reconstructed Image"]):
        ax.set_title(col)

    fig.tight_layout()
    plt.show(block=False)


def plot_max_active(x):
    """
    Plots the images that are maximally activating the hidden units
    :param x: numpy array of size [input_dim, num_hidden_units]
    """
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(9, 9))
    fig.subplots_adjust(hspace=.1, wspace=0)
    img_h = img_w = np.sqrt(x.shape[0]).astype(int)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(x[:, i].reshape((img_h, img_w)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show(block=False)
