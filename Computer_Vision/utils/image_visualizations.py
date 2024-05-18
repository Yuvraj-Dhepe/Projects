import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def plot(
    imgs, subplots_kwargs=None, imshow_kwargs=None, subplot_adjust_kwargs=None
):
    if subplots_kwargs is None:
        subplots_kwargs = {}
    if imshow_kwargs is None:
        imshow_kwargs = {}

    if subplot_adjust_kwargs is None:
        subplot_adjust_kwargs = {}

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, **subplots_kwargs)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), **imshow_kwargs)
        axs[0, i].axis("on")  # Show axis
    plt.subplots_adjust(**subplot_adjust_kwargs)
    plt.tight_layout()
    plt.show()
