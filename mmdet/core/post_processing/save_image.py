import numpy as np
import os
import imageio


def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    """
    Converts a Tensor arrey into a numpy image array
    """
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """
    Save a numpy image to the disk
    """

    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)


def mkdirs(paths):
    """
    Create empty directories if they don't exist
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """
    Create a single empty directory if it didn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)