import os
import sys
filePath = os.path.dirname(os.path.abspath(__file__))
os.chdir(filePath)
sys.path.insert(0, os.path.join(filePath, '..'))

import cv2
import numpy as np
img = cv2.imread('test_img/native_res_skin.png', 0)


def test_integration_threshold():
    from thresholding.img_proc import get_otsu, truncate
    from thresholding.mask_proc import get_tops, prune_tops, fill_holes

    h, w = img.shape
    x_range = np.arange(w)

    # Detect top edge
    otsu = get_otsu(img)
    otsu = fill_holes(otsu, mode=1)
    x, tops = get_tops(otsu, start_depth=50, end_depth=500)
    x, tops = prune_tops(x, tops, degree=3)
    p = np.interp(x_range, x, tops)

    # Detect bottom edge
    newimg, newh = truncate(img, np.rint(p).astype(int))
    newotsu = get_otsu(newimg)
    newotsu = fill_holes(newotsu, mode=0)
    x, bots = get_tops(np.logical_not(newotsu), start_depth=50, end_depth=500)
    x, bots = prune_tops(x, bots, degree=1)
    bots += np.take(p, x)
    q = np.interp(x_range, x, bots)

    assert 200 < p.mean() < 250
    assert 350 < q.mean() < 400


def test_integration_network():
    import torch
    from network.segnet_deploy import load_model, color_log_p

    device = 'cpu'
    h = img.shape[0] - img.shape[0] % 64
    w = img.shape[1] - img.shape[1] % 4
    img_torch = torch.from_numpy(img[:h, :w])
    img_torch = img_torch.to(device).float()

    m = load_model((h, w), device=device)
    with torch.no_grad():
        c, p = m(img_torch[None, None])
    c = c[0].cpu().numpy()
    p = p[0].cpu().numpy()
    heatmap = color_log_p(p)
    c *= 1024

    assert 200 < c[0].mean() < 250
    assert 350 < c[1].mean() < 400
    assert heatmap[230, 128, 0] > 200
    assert heatmap[230, 128, 1] < 20
    assert heatmap[376, 128, 0] < 100
    assert heatmap[376, 128, 1] > 200
    assert np.array_equal(heatmap[1000, 128], [0, 0, 0])
