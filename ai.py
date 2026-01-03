#Define constants
import numpy as np
import torch
from collections import deque
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import shutil
import os
import torch.nn.functional as F
from typing import List
from config import conf
#initialize configuration
Config = conf()
BUFFER_SIZE = 512
SAVE_FOLDER = "./ckpt"
class Trainer:
    """Trainer class responsible for training the 2048 ai"""
    def __init__(self,config=Config,**kwargs):
        #TODO: create a config file / kwargs to take in all of the arguments, current impl is messy
        if config:
            self.config = config
            self.load_config()
        else:
            raise Exception("Error loading in configuration for the trainer class!")
        
        self.agent = kwargs.get("agent")
        self.loss_fn = nn.L1Loss(reduction="mean") #parameterize this later
        self.optimizer = optim.SGD(self.agent.parameters(),lr=0.001)

        #create the save folder if it does not exist
        save_folder_path = str(Path(SAVE_FOLDER)) #convert to abspath
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        #Gameplay queue
        self.buffer = Buffer(buffer_size=self.buffer_size)

    def load_config(self):
        """Loads in the configuration for the trainer class"""
        for attr,val in self.config.items():
            setattr(self,attr,val)

    def one_hot(self,board:List[List[int]]):
        """Generates a one hot encoding of the board
        board: List[List[int]] Game board
        Returns:
            List[List[int]] one hot encoding of the game board
        """
        unique_encodings = self.unique_encodings #There are log2(max_tile) + 1 different tiles. Include 0 for the +1
        all_tiles = self.all_tiles
        encoded = np.zeros((unique_encodings,self.grid_size,self.grid_size)) #make an NxNxE one hot encoding
        # print(f"[INFO] OH encoding params: ue {unique_encodings} all_tiles {all_tiles}")
        for i in range(len(all_tiles)):
            tile = all_tiles[i]
            found_rows,found_cols = np.where(board == tile) #Returns two arrays with the indicies of the rows and columns where matches were found
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

    def to_bin(self,s1: List[List[int]],s2:List[List[int]],a1:int,r1:int):
        """Converts an episode containing (s1,a1,r1,s2) into binary format"""
        a1_bin = f"{np.bin(a1)[2:]:02d}"
        r1_bin = f"{np.bin(r1)[2:]:018d}" #edit to factor in the max reward
        s1_bin = board_to_bin(s1)
        s2_bin = board_to_bin(s2)
        return (s1_bin,a1_bin,r1_bin,s2_bin)
    
    def board_to_bin(self,board:List[List[int]]):
        """Converts board representation to binary"""
        binstr = ""
        for rown in range(self.grid_size):
            for coln in range(self.grid_size):
                val = all_tiles.index(board[rown][coln])
                binstr += f"{np.bin(val)[2:]:04d}"
        return binstr
    
    def load_data(self,binfile:str):
        """Loads in training data from a binary file"""
        path = os.path.join(self.FOLDER,binfile)
        with open(path,"r") as f:
            s_t = self.load_board(f.read())
            a_t = f.read()
            r_t = int(f.read(),2)
            s_t_plusone = self.load_board(f.read())
            yield (s_t,a_t,r_t,s_t_plusone)
    
    def load_board(self,bin:str) -> List[List[int]]:
        """Given a binary string corresponding to the board, load in the original values"""
        board = np.zeros((self.CELL_COUNT,self.CELL_COUNT),dtype="int")
        idx = list[range(0,len(s_t) - 3,4)]
        cntr = 0
        for i in range(self.CELL_COUNT):
            for j in range(self.CELL_COUNT):
                board[i][j] = int(bin[idx[cntr]:idx[cntr+1]],2)
                cntr +=1
        return board

    def train_step(self) -> float:
        """Performs one gradient descent step on the TD error
        Returns: loss (float), loss from the current training step
        """
        self.agent.train()
        self.optimizer.zero_grad()
        y = self.buffer.v_s1 + self.buffer.r
        q = self.buffer.v_s
        loss = self.loss_fn(q,y)
        loss.backward()
        self.optimizer.step()
        return loss

class Buffer(nn.Module):
    """Class to keep track of the training data"""
    def __init__(self,buffer_size=BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.clear()
    
    def clear(self):
        self.v_s = torch.tensor([],dtype=torch.float32,requires_grad=True) #Immediate reward and long term reward from the future state
        self.r = torch.tensor([],dtype=torch.float32,requires_grad=True)
        self.v_s1 = torch.tensor([],dtype=torch.float32,requires_grad=True)
    
    def add_data(self,v_t:int,r_t:int,v_t_1:int):
        """Adds data to the buffer"""
        self.v_s.add(v_t)
        self.r.add(r_t)
        self.v_s1.add(v_t_1)


    

            
class RLAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=16,kernel_size=(2,2),out_channels=256)
        self.conv2 = nn.Conv2d(in_channels=256,kernel_size=(2,2),out_channels=512)
        self.dense1 = nn.Linear(512 * 2 * 2,1024)
        self.dense2 = nn.Linear(1024,256)
        self.output = nn.Linear(256,1)
    def forward(self,x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c2 = c2.view(-1,512 *2 * 2)
        d1 = F.relu(self.dense1(c2))
        d2 = F.relu(self.dense2(d1))
        output = self.output(d2)
        return output

