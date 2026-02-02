import numpy as np
from collections import deque
import operator
from typing import List
import torch
import pydash as ps
device = "cuda" if torch.cuda.is_available() else "cpu"
def batch_get(arr,idxs):
    """Get a list of indexes from an array"""
    if isinstance(arr,(list,deque)):
        return np.array(operator.itemgetter(*idxs)(arr))
    elif torch.is_tensor(arr[0]):
        #idxs must be of type list or np.array, cannot be cuda tensor
        return batch_get_tensor(arr,idxs)
    else:
        return arr[idxs]
    
def set_attr(obj,attr_dict,keys=None):
    if keys is not None:
        attr_dict = ps.pick(attr_dict,keys)
    # print("Attributes: ",attr_dict)
    for key,val in attr_dict.items():
        setattr(obj,key,val)
    return obj

def get_class_name(obj, lower=False):
    '''Get the class name of an object'''
    class_name = obj.__class__.__name__
    if lower:
        class_name = class_name.lower()
    return class_name


def batch_get_tensor(lis:List[torch.Tensor],idxs:list):
    """Gets a list of indexes from a list of tensors
    :param lis List of torch tensors
    :param idxs list or numpy.array
    """
    return operator.itemgetter(*idxs)(lis)
