import matplotlib.pyplot as plt


def show_image_and_target(images, targets, show: bool = True):
    """Show a grid of 25 images with labels."""
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(25):
        img = images[i].squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(str(int(targets[i])))
        axes[i].axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    return fig
