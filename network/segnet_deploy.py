import os
import numpy as np
import torch
from network.seg_net_1d import SegNet1D


def load_model(input_size, axial_depth=19.69, top_depth=5.0, device='cpu'):
    """Returns trained model in full image mode

    Args:
        input_size: shape of input image
        axial_depth: axial depth in mm
        top_depth: expected top depth in mm
        device: defaults to 'cpu'
    """
    # Original aspect ratio:
    #   Axial depth: 19.69 mm
    #   Average top depth: 5 mm
    model = SegNet1D(input_size=input_size, c_mult=6, e_fact=5)
    model.arange *= axial_depth / 19.69
    model.arange -= (top_depth - 5.0) / 19.69
    model.full_image_mode = True

    abspath = os.path.abspath(os.path.dirname(__file__))
    weight_path = os.path.join(abspath, 'models', 'deep_epox80s1_BACKUP.pt')
    state_dict = torch.load(weight_path, map_location=device)
    state_dict['arange'] = model.arange  # Don't overwrite
    state_dict['coord.xrange'] = model.coord.xrange  # Don't overwrite
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def color_log_p(p, cutoff=8.0):
    """Convert probabilty heatmap to red/cyan image

    This function does not perform softmax and does NOT
    check whether input probabilities sum to one.

    Args:
        p: input probability for top and bottom.
            Should be of shape (2, h, w)
        cutoff: Power of 10 down to cutoff signal.
            Defaults to 8.0

    Raises ValueError if input probability not in [0, 1]
    """
    if np.any(p < 0.0):
        raise ValueError('Probability cannot be less than 0')
    if np.any(p > 1.0):
        raise ValueError('Probability cannot be more than 1')
    assert p.shape[0] == 2
    heatmap = np.log10(np.stack([p[0], p[1], p[1]], axis=-1))
    heatmap = np.maximum(heatmap + cutoff, 0) / cutoff
    heatmap = (255.0 * heatmap).astype(np.uint8)
    return heatmap
