import os
import pytest
os.chdir(os.path.dirname(__file__))


def test_get_tops_clean():
    """Test `get_tops` with a clean mask"""
    import cv2
    from thresholding.mask_proc import get_tops

    mask = cv2.imread('test_img/mask_clean.png', 0)
    h, w = mask.shape
    x, tops = get_tops(mask)

    assert len(x) == w
    assert len(tops) == w
    assert x[99] == 99
    assert tops[45] == 67


def test_get_tops_null():
    """Test `get_tops` with a null mask"""
    import cv2
    from thresholding.mask_proc import get_tops
    from thresholding.mask_proc import TopNotFoundError

    mask = cv2.imread('test_img/mask_null.png', 0)

    with pytest.raises(TopNotFoundError):
        get_tops(mask)


def test_prune_tops_linear():
    """Test `prune_tops` on linear data"""
    import numpy as np
    from thresholding.mask_proc import prune_tops

    x = np.arange(10)
    tops = np.arange(10) * 3 + 2
    tops[5] *= 2
    newx, newtops = prune_tops(x, tops)

    assert np.all(newx == np.delete(x, 5))
    assert np.all(newtops == np.delete(tops, 5))


def test_prune_tops_rand():
    """Test `prune_tops` on random data"""
    import numpy as np
    from thresholding.mask_proc import prune_tops

    x = np.arange(200)
    tops = np.random.randint(40, 250, size=200)
    newx, newtops = prune_tops(x, tops)

    assert len(newx) == len(newtops)
    assert len(newx) < 160
    assert np.mean(tops) == pytest.approx(np.mean(newtops), rel=0.1)
    # Above test will stochastically fail, with probability 1 in 10000
    assert np.std(tops) > np.std(newtops)


def test_fill_holes():
    """Test `fill_holes`"""
    import numpy as np
    from thresholding.mask_proc import fill_holes

    a = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    assert np.all(fill_holes(a) == np.ones((3, 3)))
    assert np.all(fill_holes(np.logical_not(a), mode=1) == np.zeros((3, 3)))



