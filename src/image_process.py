import numpy as np
from scipy.misc import toimage


def merge_imgs(data, row=10, col=10, transfrom=True):
    """
    Merge Images to one matrix
    :param data: numpy array
    :param row: int
    :param col: int
    :param transfrom: bool, transfer the numpy array into PIL Image if true.
    :return: a numpy array of PIL Image contain multiply images
    """
    assert len(data) >= row * col, "Shape doesn't fit"
    imgs = []
    for i in range(col):
        imgs.append(np.concatenate([data[row * i + j] for j in range(row)], axis=1))  # merge into one column
    if len(imgs[0].shape) > 2:
        imgs = np.concatenate(imgs, axis=2)
    else:
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.squeeze(imgs)
    if transfrom:
        imgs = toimage(imgs)
    return imgs