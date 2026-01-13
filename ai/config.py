#KEY constants
import numpy as np
import torch
MAXTILE = 32768
BATCH_SIZE = 10000
MAXREWARD = 18 #Max reward is 2^18
GRID_SIZE = 4
FOLDER = "./data"
SAVE_FOLDER = "./ckpt"
EPSILON = 0.95 #Exploration / Explotation tradeoff
GAMMA = 0.995 #Future rewards discount rate
BUFFER_SIZE = 20000
POLICY = "boltzmann"
UNIQUE_ENCODINGS = 16
available_setting = {
    "max_tile":32768,
    "batch_size":512,
    "max_reward":18,
    "grid_size":4,
    "folder":FOLDER,
    "save_folder":SAVE_FOLDER,
    "buffer_size":BUFFER_SIZE,
    "board_enc_length":4,
    "unique_encodings":UNIQUE_ENCODINGS,
    "all_tiles": torch.tensor([0] + [2**x for x in range(1,UNIQUE_ENCODINGS+1)]),
    "epsilon":EPSILON,
    "epsilon_end":0.01,
    "steps":10000,
    "gamma":GAMMA,
    "policy":POLICY
}
class Config(dict):
    """Handles parameter configuration"""
    def __init__(self,d=None):
        super().__init__()
        if d is None:
            d = {}
        for k,v in d.items():
            self[k] = v
    def __getitem__(self,key):
        if key not in available_setting:
            raise Exception(f"Key {key} not in available_setting")
        return super().__getitem__(key)
    def __setitem__(self,key,value):
        if key not in available_setting:
            raise Exception(f"Key {key} not in available setting")
        return super().__setitem__(key,value)
    def get(self,key,default=None):
        try:
            return self[key]
        except KeyError as e:
            return default
        except Exception as e:
            return e

config = Config(available_setting)
def conf():
    return config