"""Extract boundaries from a binary mask input"""
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes


class TopNotFoundError(Exception):
    """Raised when no top is found by `get_tops`"""
    pass


def get_tops(mask, start_depth=40, end_depth=250):
    """Detections boundary transition in mask

    For each column, start at `start_depth` and search
    for the first `True` value in mask. If not found,
    then that column index excluded in output.

    Args:
        mask: input image mask, any dtype with bool eval
        start_depth: starting pixel depth to begin detection.
            Defaults to 40 (approx 1.5 mm).
        end_depth: ending pixel depth to stop detection.
            Defaults to 250 (approx 9 mm).

    Returns:
        x: list of column indices
        tops: list of row indices associated with each column
            index in x, corresponding to the boundary.

    Raises:
        TopNotFoundError: when top not found in any column
    """
    h, w = mask.shape
    end_depth = min(end_depth, h)
    x = []
    tops = []
    for i, column in enumerate(mask.T):
        for j in range(start_depth, end_depth):
            if column[j]:  # accepts any dtype with bool eval
                x.append(i)
                tops.append(j)
                break
    if not x:
        raise TopNotFoundError(
            'Failed to find tops. Depth from {} to {}'.format(start_depth, end_depth))
    return x, tops


def polyfit_tops(x, tops, degree=1):
    """Fits a polynomial to a detected edge.

    If only 1 data point, returns a constant.
    Numpy issues a RankWarning if ill conditioned.

    Args:
        x: list of horizontal (column) indices
        tops: list of vertical positions of edge
        degree: degree of polynomial to fit.
            Defaults to 1 (linear).

    Returns:
        np.poly1d object
    """
    if len(tops) == 1:
        return np.poly1d(tops)
    return np.poly1d(np.polyfit(x, tops, degree))


def prune_tops(x, tops, degree=1):
    """Fits a line to detected edge and prunes all data points
    greater than 1 standard deviation away from the fit.

    Args:
        x: list of horizontal (column) indices
        tops: list of vertical positions of edge
        degree: degree of polynomial to fit.
            Defaults to 1 (linear).

    Returns:
        newx: pruned x
        newtops: pruned tops
    """
    # TODO: L1 regression instead of L2
    p = polyfit_tops(x, tops, degree)
    err = p(x) - tops
    sigma = np.std(err)
    if sigma < 1e-9:  # If data is perfect fit
        return x, tops
    newx = []
    newtops = []
    for i, abserr in enumerate(np.abs(err)):
        if abserr <= sigma:
            newx.append(x[i])
            newtops.append(tops[i])
    return newx, newtops


def fill_holes(mask, mode=0):
    """Fills either dots or holes in a mask

    Uses `scipy.ndimage.morphology.binary_fill_holes`
    Fills holes (0) if `mode=0`. Removes dots (1) if `mode=1`.

    Args:
        mask: input mask array, any dtype with binary bool eval
        mode: 0 for holes, 1 for dots. Defaults to 0.

    Returns:
        filled mask
    """
    if not mode:
        return binary_fill_holes(mask)
    else:
        return np.logical_not(binary_fill_holes(np.logical_not(mask)))
