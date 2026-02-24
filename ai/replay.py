from ai.memory import Memory
import ai.util as util
import numpy as np
from pathlib import Path
from collections import deque
import torch
from typing import List
import string
import json
import pdb
class Buffer(Memory):
    """Class to keep track of the training data"""
    def __init__(self,memory_spec,body):
        super().__init__(memory_spec,body)
        util.set_attr(self,memory_spec,keys=[
            "use_cer",
            "batch_size",
            "max_size",
            "save_folder",
            "save_file"
        ])
        #TODO: add memory spec save file and save folder
        self.batch_idxs = None
        self.size = 0
        self.seen_size = 0
        self.head = -1
        self.ns_idx_offset = self.body.env.num_envs if body['env']['is_venv'] else 1
        self.ns_buffer = deque(maxlen=self.ns_idx_offset)

        #TODO: save folder implementation
        self.data_keys = ["states","actions","rewards","next_states","done"]
        self.reset()
        #Error handling for save_folder and save_file
        if not(self.save_folder.exists()): #Pathlib.path object
            try:
                self.save_folder.mkdir()
            except Exception as e:
                print(e)
        
        if not(self.save_file.exists()):
            with open(str(self.save_file.resolve()),"w") as f:
                f.write("") #Create a new file
        
    
    def __len__(self) -> int:
        """Returns the number of experiences stored"""
        return len(list(filter(lambda x: x is not None, self.states)))
    
    def reset(self):
        for k in self.data_keys:
            # if k != "next_states":
            setattr(self,k,[None] * self.max_size)
        self.ns_buffer.clear()
        self.size = 0
        self.head = -1
    

    def sample_next_state(self,head,max_size,ns_idx_offset,batch_idxs,states,ns_buffer):
        """Guard for out of bounds sampling of next state
        Args:
        :param batch_idxs np.ndarray(batch_size,)
        :param head (int)
        :param max_size (int)
        :param ns_idx_offset (int)
        :param states (List[np.ndarray])
        :param ns_buffer deque()
        """
        #Assume batch_idxs is a torch tensor
        ns_batch_idxs = (batch_idxs + ns_idx_offset) % max_size
        mask = (head < ns_batch_idxs) & (ns_batch_idxs <= head + ns_idx_offset)
        buffer_ns_locs = torch.where(mask)[0]
        to_replace = buffer_ns_locs.numel() != 0
        if to_replace:
            buffer_idx = ns_batch_idxs[buffer_ns_locs] - head - 1
            ns_batch_idxs[buffer_ns_locs] = 0
        ns_batch_idxs = ns_batch_idxs % max_size
        batch =util.batch_get_tensor(states,ns_batch_idxs)
        if to_replace:
            batch_ns = util.batch_get_tensor(ns_buffer,buffer_idx)
            batch[buffer_ns_locs] = batch_ns
        return batch
        

    def sample_idxs(self,batch_size):
        batch_idxs = torch.randint(low=0,high=self.size,size=batch_size,dtype=torch.int64).to(self.device)
        if self.use_cer:
            batch_idxs[-1] = self.head
        return batch_idxs

    def batch_get(self,attr,batch_idxs):
        """Gets a series of sampled data"""
        return torch.gather(attr,-1,batch_idxs)
    
    def add_experience(self,state,action,reward,next_state,done):
        #switch to float 16?
        self.head = (self.head + 1) % self.max_size
        self.states[self.head] = state
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.done[self.head] = done
        self.ns_buffer.append(next_state)
        if self.size < self.max_size:
            self.size += 1
        self.seen_size += 1
        # trainer = self.trainer
        # trainer.to_train = trainer.to_train or (self.head % trainer.training_frequency == 0)

    def sample(self):
        """Samples a portion of (SARS) tuples from the buffer.
        Sample results
        s: (states, torch.Tensor(batch_size,16,4,4))
        a: actions, torch.Tensor(batch_size,)
        r: rewards, torch.Tensor(batch_size,)
        s': new states, torch.Tensor(batch_size,16,4,4)
        priorities: priorities in the sumtree, torch.Tensor(batch_size,)
        dones: whether state is terminal state or not torch.Tensor(batch_size,)
        """
        self.batch_idxs = self.sample_idxs(self.batch_size)
        batch = {}
        for k in self.data_keys:
          
            if k == "next_states":
                batch[k] = self.sample_next_state(self.head,self.max_size,self.ns_idx_offset,self.batch_idxs,self.states,self.ns_buffer)
            else:
                batch[k] = util.batch_get(getattr(self,k),self.batch_idxs)
        return batch

    def update(self,state,action,reward,next_state,done):
        """Adds data to the buffer"""
        self.add_experience(self,state,action,reward,next_state,done)

    
    def save_data(self,save_keys:List[str]=["done","actions","rewards","states","next_states"] ):
        """
        Saves the SARS training data into the save folders.
        Args:
        :param save_keys keys to save
        """
        time = datetime.datetime.now()
        alphabet = string.ascii_lowercase + string.ascii_uppercase()
        time_str = time.strftime("%Y-%m-%d-")
        uuid = "".join(random.choice(alphabet,k=8))
        self.save_tensors(save_keys)
        self.save_config()
        with open(str(self.save_file.resolve()),"a") as f:
            f.write(time_str + "-" + uuid) #Write the newest uuid into the save files
        return uuid

    def save_tensors(self,save_keys:List[str]):
        """Saves the data into pytorch tensors. The tensors will be located in the save folder
        Args:
        save_keys (List[str]): list of strings corresponding to the names of the keys to save.
         For the states and new states, they should not be included because states and new states are saved as binary
        """

        #ADD create new folder functionality
        new_folder = self.save_folder / uuid #todo, add a new place to store the latest saved uuids?
        
        try:
            new_folder.mkdir()
        except Exception as e:
            print(f"Failed to create new folder to save pytorch tensors")
            print(e)
        
        for k in save_keys:
            filename = f"{uuid}-{k}.pth" #Create a random uuid
            filepath = str((new_folder / filename).resolve())
            idxs = torch.arange(0,self.size,device=self.device)
            tensor = util.batch_get(arr=getattr(self,k),idxs=idxs) 
            torch.save(filepath,tensor)
        return uuid
    
    def load_data(self,uuid:str,save_keys:List[str]=["done","actions","rewards","states","next_states"]):
        """Loads in the most recently saved data
        Args:
        :param uuid (str) uuid of the save file
        :param save_keys List[str] names of the tensors to save
        """
        save_folder = self.save_folder / uuid
        self.load_config()
        for k in save_keys:
            file_name = uuid + "-" + k + ".pth"
            fp = str((save_folder / file_name).resolve())
            try:
                data = torch.load(fp)
                if k in ["states","next_states"]:
                    #Need to decompose stack of tensors into list of tensors
                    #Need a config file with key parameters saved from last time, e.g, self.head, self.batch_size,self.size...etc
                    split = torch.tensor_split(data,self.size,dim=0) 
                    loaded = list(map(list,split)) #Map all to list
                else:
                    #need to convert tensors into the list
                    size = data.shape[0] #length of the tensor
                    diff = self.max_size - size
                    loaded = list(data) + [None]*diff
                setattr(self,k,loaded)
            except Exception as e:
                print("Error loading ",k," from filepath",fp)
                print(e)
    
    def save_config(self,save_folder:Path,keys:List[str]=["size","seen_size","head","batch_size","max_size"]):
        """Saves the configuration of the buffer to continue training
        Args:
        :param keys List[str] configuration parameters to save
        :param save_folder Path pathlib Path object, path to the save_folder
        """
        config = {k:getattr(self,k) for k in keys}
        fp = str((save_folder / "config.json").resolve())
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(e)

    
    def load_config(self,save_folder:Path):
        """Loads the configuration file to continue training
        Args:
        :param save_folder Path pathlib Path object to the save_folder
        """
        fp = str((save_folder / "config.json").resolve())
        try:
            with open(fp,"r") as f:
                config = json.load(f)
            for k,v in config:
                setattr(self,k,v) 
        except Exception as e:
            print(e)