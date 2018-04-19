from ._pavlidis import pavlidis as pavlidis_impl

import numpy as np

def pavlidis(array, row_seed, column_seed):
    '''Follow the contour of an object clockwise

    :param array: a boolean or similar (0 = background) array. The object that
                  is to be connected must be 4-connected.
    :param row_seed: the row coordinate of the first point on the contour
    :param column_seed: the column coordinate of the first point on the contour
    :returns: an N x 2 array of the row and column of each point on the contour
    '''
    return pavlidis_impl(array.astype(np.uint8), row_seed, column_seed)