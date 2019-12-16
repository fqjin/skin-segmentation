"""Image Processing functions"""
import cv2
import numpy as np


def get_otsu(img):
    """Applies Otsu's Thresholding Method

    Uses the cv2 library.

    Args:
        img: uint8 or float array
            float arrays will be normalized

    Returns:
        uint8 array with values either 0 or 255
    """
    if img.dtype == np.uint8:
        pass
    elif np.issubdtype(img.dtype, np.floating):
        img = img.copy()
        img -= img.min()
        img *= 255.0 / img.max()
        img = np.rint(img).astype(np.uint8)
    else:
        raise NotImplementedError('dtype not accepted')
    th, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def truncate(img, int_p):
    """Aligns and truncates an image

    Columns of the input image are truncated at
    input coordinates. The bottom half is aligned
    flat across the top. The image is also trimmed
    to the shortest column.

    Args:
        img: input image
        int_p: integer coordinates for each column

    Returns:
        newimg: truncated image
        newh: truncated height
    """
    h, w = img.shape
    assert len(int_p) == w
    newh = h - max(int_p)
    newimg = np.empty((newh, w), dtype=img.dtype)
    for i, pos in enumerate(int_p):
        newimg[:, i] = img[pos:pos + newh, i]
    return newimg, newh
