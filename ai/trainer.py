#Define constants
import numpy as np
import torch
from collections import deque
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from decay import LinearDecay
from policy import policy_factory
import shutil
import os
import torch.nn.functional as F
from replay import Buffer
from typing import List
from agent import RLAgent
from config import conf
import util
#initialize configuration
Config = conf()
SAVE_FOLDER = "./ckpt"
class Trainer:
    """Trainer class responsible for training the 2048 ai"""
    def __init__(self,config=Config,**kwargs):
        #TODO: create a config file / kwargs to take in all of the arguments, current impl is messy
        self.config = config
        self.load_config()
        self.policy = policy_factory(self.action_selection,epsilon_start=self.epsilon,
        epsilon_end=self.epsilon_end, maxsteps=self.steps,trainer=self)

        agent = kwargs.get("agent")
        targNet = kwargs.get("targNet")
        if agent and targNet:
            self.agent = agent
            self.targNet = targNet
        else:
            self.init_nets()

        self.loss_fn = nn.MSELoss() #parameterize this later
        self.optimizer = optim.SGD(self.agent.parameters(),lr=0.001)

        #create the save folder if it does not exist
        save_folder_path = str(Path(SAVE_FOLDER)) #convert to abspath
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        #Gameplay queue
        self.buffer = Buffer(memory_spec=self.memory_spec,body=self.body)
        

    def load_config(self):
        """Loads in the configuration for the trainer class"""
        for attr,val in self.config.items():
            setattr(self,attr,val)
    
    def decay(self):
        self.epsilon = self.decay_fn.decay(self.epsilon)

    def one_hot(self,board:List[List[int]]):
        """Generates a one hot encoding of the board
        board: List[List[int]] Game board
        Returns:
            List[List[int]] one hot encoding of the game board
        """
        unique_encodings = self.unique_encodings #There are log2(max_tile) + 1 different tiles. Include 0 for the +1
        all_tiles = self.all_tiles
        encoded = torch.zeros((unique_encodings,self.grid_size,self.grid_size)) #make an NxNxE one hot encoding
        # print(f"[INFO] OH encoding params: ue {unique_encodings} all_tiles {all_tiles}")
        for i in range(len(all_tiles)):
            tile = all_tiles[i]
            found_rows,found_cols = torch.where(board == tile) #Returns two arrays with the indicies of the rows and columns where matches were found
            # print(f"[INFO] tile {tile} found at {found_rows} and col {found_cols}")
            for j in range(len(found_cols)):
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

    def train_step(self) -> float:
        """Performs one gradient descent step on the TD error
        Returns: loss (float), loss from the current training step
        """
        #ADD         self.agent.train()
        self.optimizer.zero_grad()
        #ADD BATCH SAMPLING CODE
        batch = trainer.buffer.sample()
        loss = self.calc_q_loss(batch)
        self.optimizer.step()
        return loss

    def calc_q_loss(self,batch):
        """Calculates the Q learning loss for the current batch"""
        states = batch["states"]
        next_states = batch["next_states"]
        q_preds= self.agent(states) #Calculate ai prime
        with torch.inference_mode():
            next_targ_q = self.targNet(next_states) #action selection in the next state
            next_q_preds = self.agent(next_states)
        action_q_preds = q_preds.gather(-1,batch["actions"].long().unsqueeze(-1)).squeeze(-1)
        sp_actions = next_q_preds.argmax(dim=-1,keepdim=True) #calculate max ai prime
        targ_q_sp = next_targ_q.gather(-1,sp_actions).squeeze(-1)
        y = trainer.gamma * (1-batch["dones"]) * targ_q_sp + batch["rewards"]
        q_loss = self.loss_fn(action_q_preds,y)
        q_loss.backward()
        #Add prioritized experience replay code
        if "Prioritized" in util.get_class_name(self.buffer):
            errors = (y - action_q_preds.detach()).abs().cpu().numpy()
            self.buffer.update_priorities(errors)
        return loss

    def update_params(self):
        """Synchronizes the Target Q network and the Q networks parameters"""
        params = self.targNet.state_dict()
        self.agent.load_state_dict(params)
    
    def init_nets(self):
        self.targNet = RLAgent()
        self.agent = RLAgent()
    

