#Define constants
import numpy as np
from collections import deque
from pathlib import Path
import shutil
from typing import List,Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.multiprocessing as mp

from ai.replay import Buffer
from ai.decay import LinearDecay
from ai.policy import policy_factory
from ai.priority import PrioritizedExperienceReplay
from ai.agent import CNN
from ai.config import conf
from ai.dqn import DQN
import ai.util as util
from torch.utils.tensorboard import SummaryWriter
#initialize configuration

SAVE_FOLDER = "./ckpt"
LOG_FOLDER = "./data"
class Trainer:
    """Trainer class responsible for training the 2048 ai"""
    def __init__(self,config,**kwargs):
        #TODO: create a config file / kwargs to take in all of the arguments, current impl is messy
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_config()
        self.policy = policy_factory(self.action_selection,epsilon_start=self.epsilon,
        epsilon_end=self.epsilon_end, maxsteps=self.steps,trainer=self)
        self.sumwriter = SummaryWriter()

        #Agent configuration
        self.net.loss_fn = nn.SmoothL1Loss() #parameterize this later
        self.net.optimizer = optim.SGD(self.agent.parameters(),lr=0.0001)
        self.net.buffer =PrioritizedExperienceReplay(memory_spec=self.memory_spec,body=self.body)

        #create the save folder if it does not exist
        save_folder_path = str(Path(SAVE_FOLDER).resolve()) #convert to abspath
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

    def log(self,mode:str,loss:float=None,steps:int=0,score:float=None):
        """logs the test_loss avgScore to the tensorboard
        Args:
        :param mode (str), test or training
        :param test_loss (float) test loss
        :param avgScore (float) average score over test playouts
        :param content (str) content to add to the log file
        :param test_steps (int) number of test steps
        """
        if mode=="test":
            sumwriter.add_scalar("Loss/test",loss,steps)
            sumwriter.add_scalar("Mean Reward",score,steps)
        elif mode == "train":
            sumwriter.add_scalar("Loss/train",loss,steps)
    


    def load_config(self):
        """Loads in the configuration for the trainer class"""
        for attr,val in self.config.items():
            setattr(self,attr,val)
    
    def decay(self):
        self.epsilon = self.policy.decay_fn.decay(self.epsilon)

    
    def parallelize(self,func,args: tuple):
        """Parallizes training using the hogwild algorithm
        :param args (tuple) tuple of arguments to pass into the function to parallize
        """
        workers = []
        num_workers = 4
        for _rank in range(num_workers):
            w = mp.Process(target=func, args=args)
            w.start()
            workers.append(w)
            for w in workers:
                w.join()

    def collect_data(self,n:int=100000):
        """Collects N pieces of training data and adds it to the Replay buffer
        :param n (int) number of moves to simulate
        """
        raise NotImplementedError
    

        self.buffer.add_experience(s_oh,a,r,s1_oh,done)
    
    def move(self,gb,policy) -> Tuple[torch.Tensor,int,int,torch.Tensor]:
        """perform one move in the game and return a tuple of (s,a,r,s2)
        Args:
        :param gb GameBoard object
        :param policy Policy object
        Returns:
            Tuple of (s,a,r,s2)
        """
        s = gb.board
        a = policy.choice(gb) 
        s1,_,r = gb.MOVES[a](s)
        r = r.item()
         #remove the pytorch tensor
        s1 = gb.add_new_tile(s1)
        gb.board = s1
        gb.score += r
        done = 0 if gb.has_valid_move() else 1
        s_oh = self.one_hot(s)
        s1_oh = self.one_hot(s1)
        return (s_oh,a,r,s1_oh,done)

   

