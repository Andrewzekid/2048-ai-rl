#Define constants
import numpy as np
import torch
from collections import deque
from pathlib import Path
import shutil
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

from ai.replay import Buffer
from ai.decay import LinearDecay
from ai.policy import policy_factory
from ai.priority import PrioritizedExperienceReplay
from ai.agent import RLAgent
from ai.config import conf
import ai.util as util

#initialize configuration
Config = conf()
SAVE_FOLDER = "./ckpt"
LOG_FOLDER = "./data"
class Trainer:
    """Trainer class responsible for training the 2048 ai"""
    def __init__(self,config=Config,**kwargs):
        #TODO: create a config file / kwargs to take in all of the arguments, current impl is messy
        self.config = config
        self.load_config()
        self.policy = policy_factory(self.action_selection,epsilon_start=self.epsilon,
        epsilon_end=self.epsilon_end, maxsteps=self.steps,trainer=self)

        self.agent = kwargs.get("agent")
        self.targNet = kwargs.get("targNet")
       
        if not(self.agent or self.targNet): #Manual initialization if both are not provided
            self.init_nets()
        
        self.loss_fn = nn.SmoothL1Loss() #parameterize this later
        self.optimizer = optim.SGD(self.agent.parameters(),lr=0.0001)

        #create the save folder if it does not exist
        save_folder_path = str(Path(SAVE_FOLDER).resolve()) #convert to abspath
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)
        
        self.log_path = Path(self.log_path)
        if not(self.log_path.exists()):
            #create the log folder if it does not exist
            fp = open(str(self.log_path.resolve()),"w")
            fp.write("")
            fp.close()

        self.num_workers = int(os.cpu_count() * 0.75) #Only use a portion of cpus to avoid black screen
        #Gameplay queue
        self.buffer = PrioritizedExperienceReplay(memory_spec=self.memory_spec,body=self.body)
        

    def load_config(self):
        """Loads in the configuration for the trainer class"""
        for attr,val in self.config.items():
            setattr(self,attr,val)
    
    def decay(self):
        self.epsilon = self.policy.decay_fn.decay(self.epsilon)

    def one_hot(self,board:List[List[int]]) -> torch.Tensor:
        """Generates a one hot encoding of the board
        board: List[List[int]] Game board
        Returns:
            torch.Tensor (16,4,4) one hot encoding of the game board
        """
        unique_encodings = self.unique_encodings #There are log2(max_tile) + 1 different tiles. Include 0 for the +1
        all_tiles = self.all_tiles
        encoded = torch.zeros((unique_encodings,self.grid_size,self.grid_size),dtype=torch.float32) #make an NxNxE one hot encoding
        # print(f"[INFO] OH encoding params: ue {unique_encodings} all_tiles {all_tiles}")
        for i in range(len(all_tiles)):
            tile = all_tiles[i]
            found_rows,found_cols = torch.where(board == tile) #Returns two arrays with the indicies of the rows and columns where matches were found
            # print(f"[INFO] tile {tile} found at {found_rows} and col {found_cols}")
            for j in range(len(found_rows)):
                encoded[i,found_rows[j],found_cols[j]] = 1
        return encoded

    def save(self,filename:str):
        """Save the pytorch model into a file
        Args:
            filename(str): (name of the pytorch model weights file)
        """
        path = Path(self.save_folder) / filename
        torch.save(self.agent.state_dict(),str(path))
    
    def load(self,filename:str):
        """Load the pytorch model weights from ckpt
        :param filename name of the ckpt file
        """
        path = Path(self.save_folder) / filename
        if os.path.exists(path):
            state_dict = torch.load(path)
            self.agent.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"{path} is not a valid pytorch checkpoint object!")

    def train_step(self,qnet,batch) -> float:
        """Performs one gradient descent step on the TD error
        Returns: loss (float), loss from the current training step
        :param batch batch of training data sampled from the experience buffer
        """
        self.optimizer.zero_grad()
        loss = self.calc_q_loss(qnet,batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def parallelize(self,func,args: tuple):
        """Parallizes an operation using the hogwild algorithm
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

    
    def test_step(self,batch) -> float:
        """Conducts one test step and returns the loss"""
        #add self.eval()
        loss = self.calc_q_loss(qnet=self.agent,batch=batch)
        return loss.item()
    
    def train_mode(self):
        self.agent.train()

    def calc_q_loss(self,qnet,batch):
        """Calculates the Q learning loss for the current batch
        :param qnet (RLAgent) q network to use for predictions
        :param batch batch of data to train on
        """
        states = batch["states"]
        next_states = batch["next_states"]
        q_preds= qnet(states) #Calculate ai prime
        with torch.inference_mode():
            next_targ_q = self.targNet(next_states) #action selection in the next state
            next_q_preds = qnet(next_states)
        action_q_preds = q_preds.gather(-1,batch["actions"].long().unsqueeze(-1)).squeeze(-1)
        sp_actions = next_q_preds.argmax(dim=-1,keepdim=True) #calculate max ai prime
        targ_q_sp = next_targ_q.gather(-1,sp_actions).squeeze(-1)
        y = self.gamma * (1-batch["done"]) * targ_q_sp + batch["rewards"]
        q_loss = self.loss_fn(action_q_preds,y)
        #Add prioritized experience replay code
        if "Prioritized" in util.get_class_name(self.buffer):
            errors = (y - action_q_preds.detach()).abs().cpu().numpy()
            self.buffer.update_priorities(errors)
        return q_loss

    def update_params(self):
        """Synchronizes the Target Q network and the Q networks parameters"""
        params = self.agent.state_dict()
        self.targNet.load_state_dict(params)
    
    def init_nets(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.targNet = RLAgent().to(device)
        self.agent = RLAgent().to(device)
    
    def train(self):
        self.targNet.train()
        self.agent.train()
    
    def eval(self):
        self.targNet.eval()
        self.agent.eval()
    
    def logging(self,content:str):
        """Implements logging behavior for training"""
        with open(str(self.log_path.resolve()),"a") as f:
            f.write(content)

    def visualize(self):
        """Visualize Training Results including train_loss and average_score"""
        log_folder_path = self.log_path
        fp = str(log_folder_path.resolve())
        data = pd.read_csv(fp,header=None,names=["train_loss","average_score"])
        steps = np.arange(0,len(data),1)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(steps,data["train_loss"])
        ax1.set_xlabel("Train Steps")
        ax1.set_ylabel("Train Loss")
        ax1.set_title("Train Loss over time")
        
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(steps,data["average_score"])
        ax2.set_xlabel("Train Steps")
        ax2.set_ylabel("Average Reward")
        ax2.set_title("Reward over time")
        plt.subplots_adjust(hspace=1)
        plt.show()

    

