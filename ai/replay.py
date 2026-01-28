from ai.memory import Memory
import ai.util as util
import numpy as np
from collections import deque
import torch
class Buffer(Memory):
    """Class to keep track of the training data"""
    def __init__(self,memory_spec,body):
        super().__init__(memory_spec,body)
        util.set_attr(self,memory_spec,keys=[
            "use_cer",
            "batch_size",
            "max_size",
        ])
        #TODO: add memory spec
        self.batch_idxs = None
        self.size = 0
        self.seen_size = 0
        self.head = -1
        self.ns_idx_offset = self.body.env.num_envs if body['env']['is_venv'] else 1
        self.ns_buffer = deque(maxlen=self.ns_idx_offset)

        self.data_keys = ["states","actions","rewards","next_states","done"]
        self.reset()
    
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
        ns_batch_idxs = (batch_idxs + ns_idx_offset) % max_size
        mask = (head < ns_batch_idxs) & (ns_batch_idxs <= head + ns_idx_offset)
        buffer_ns_locs = torch.where(mask)[0]
        to_replace = buffer_ns_locs.numel() != 0
        if to_replace:
            buffer_idx = ns_batch_idxs[buffer_ns_locs] - head - 1
            ns_batch_idxs[buffer_ns_locs] = 0
        ns_batch_idxs = ns_batch_idxs % max_size
        print(f"ns_batch_idxs: {ns_batch_idxs}")
        batch = util.batch_get(states,ns_batch_idxs)
        if to_replace:
            batch_ns = util.batch_get(ns_buffer,buffer_idx) #torch tensor supports indexing with deque?
            batch[buffer_ns_locs] = batch_ns
        return batch
        

    def sample_idxs(self,batch_size):
        batch_idxs = torch.randint(self.size,size=batch_size,dtype=torch.int64).to(self.device)
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
        """Samples a portion of (SARS) tuples from the buffer"""
        self.batch_idxs = self.sample_idxs(self.batch_size)
        batch = {}
        for k in self.data_keys:
            if k == "next_states":
                batch[k] = torch.tensor(self.sample_next_state(self.head,self.max_size,self.ns_idx_offset,
                self.batch_idxs,self.states,self.ns_buffer),dtype=torch.float32,device=self.device)
            else:
                batch[k] = torch.tensor(util.batch_get(getattr(self,k),self.batch_idxs),dtype=torch.float32,device=self.device)
        return batch

    def update(self,state,action,reward,next_state,done):
        """Adds data to the buffer"""
        self.add_experience(self,state,action,reward,next_state,done)

