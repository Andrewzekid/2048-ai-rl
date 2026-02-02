import numpy as np
from collections import deque
import operator
from typing import List
import torch
import pydash as ps
device = "cuda" if torch.cuda.is_available() else "cpu"
def batch_get(arr,idxs,dim=0):
    """Get a list of indexes from an array"""
    if isinstance(arr,(list,deque)):
        first = arr[0]
        if torch.is_tensor(first):
            #idxs must be of type list or np.array, cannot be cuda tensor. if it is a list of tensors, return a batch_get_tensor
            return batch_get_tensor(arr,idxs)
        elif isinstance(first,(int,float)): #list of ints or floats
            return torch.tensor(operator.itemgetter(*idxs)(arr),device=device,dtype=torch.float32)
        else:
            return np.array(operator.itemgetter(*idxs)(arr))
    elif torch.is_tensor(arr):
        return torch.index_select(arr,dim=dim,index=idxs)            
    else: #np array
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
    res = operator.itemgetter(*idxs)(lis)
    if torch.is_tensor(res):
        return res
    elif torch.is_tensor(res[0]): #possible errors: one iteger output
        return torch.stack(res)
    else:
        return torch.tensor(res,device=device)

