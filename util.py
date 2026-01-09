import numpy as np
from collections import deque
import operator
def batch_get(arr,idxs):
    """Get a list of indexes from an array"""
    if isinstance(arr,(list,deque)):
        return np.array(operator.itemgetter(*idxs)(arr))
    else:
        return arr[idxs]