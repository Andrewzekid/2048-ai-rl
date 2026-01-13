import numpy as np
from collections import deque
import operator
import pydash as ps
def batch_get(arr,idxs):
    """Get a list of indexes from an array"""
    if isinstance(arr,(list,deque)):
        return np.array(operator.itemgetter(*idxs)(arr))
    else:
        return arr[idxs]
    
def set_attr(obj,attr_dict,keys=None):
    if keys is not None:
        attr_dict = ps.pick(attr_dict,keys)
    for key,val in attr_dict:
        setattr(obj,key,val)
    return obj