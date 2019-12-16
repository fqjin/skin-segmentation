import os
import pytest
os.chdir(os.path.dirname(__file__))


def test_load_model():
    """Test load pre-trained model"""
    from network.segnet_deploy import load_model
    m = load_model((64, 16), axial_depth=20.0, top_depth=3.0, device='cpu')


def test_color_log_p():
    """Test heatmap to log image conversion"""
    import warnings
    import numpy as np
    from network.segnet_deploy import color_log_p

    # log(0.0) = -Inf maps to 0 in the output
    p = np.array([[
        [0.0, 0.0],
        [1e-4, 1e-4],
    ], [
        [1e-9, 1.0],
        [1e-9, 1.0]
    ]])
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    heatmap = color_log_p(p, cutoff=8.0)

    assert np.array_equal(heatmap[0, 0], [0, 0, 0])
    assert np.array_equal(heatmap[0, 1], [0, 255, 255])
    assert np.array_equal(heatmap[1, 0], [127, 0, 0])
    assert np.array_equal(heatmap[1, 1], [127, 255, 255])


def test_color_log_p_error():
    """Test heatmap to log image conversion errors"""
    import numpy as np
    from network.segnet_deploy import color_log_p

    p = np.full((2, 2, 2), 0.5)

    p[0, 0, 0] = -0.1
    with pytest.raises(ValueError):
        color_log_p(p, cutoff=8.0)

    p[0, 0, 0] = 1.1
    with pytest.raises(ValueError):
        color_log_p(p, cutoff=8.0)
