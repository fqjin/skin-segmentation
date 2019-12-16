import os
import pytest
os.chdir(os.path.dirname(__file__))


def test_get_otsu_uint8():
    import cv2
    import numpy as np
    from thresholding.img_proc import get_otsu

    a = cv2.imread('test_img/phantom1.png', 0)
    otsu = get_otsu(a)

    assert np.array_equal(otsu, otsu.astype(bool) * 255)
    assert otsu[50, 100] == 0
    assert otsu[150, 200] == 255


def test_get_otsu_float():
    import numpy as np
    from thresholding.img_proc import get_otsu

    a = np.random.randn(100, 100)
    a[:50] -= 3
    a[50:] += 3
    otsu = get_otsu(a)

    assert np.array_equal(otsu, otsu.astype(bool) * 255)
    assert 0 <= otsu[:50].mean() < 1.0
    assert 254.0 < otsu[50:].mean() <= 255.0


def test_truncate():
    import numpy as np
    from thresholding.img_proc import truncate

    a = np.random.randn(100, 8)
    int_p = np.array([1, 2, 3, 4, 5, 3, 3, 3])
    b, h = truncate(a, int_p=int_p)

    assert b.shape[0] == h
    assert b.shape[1] == a.shape[1]
    assert np.array_equal(b[:, 0], a[int_p[0]:int_p[0]+h, 0])
    assert np.array_equal(b[:, 4], a[int_p[4]:int_p[4]+h, 4])
