from ai.replay import Buffer
from ai.memory import Memory
import torch
import ai.util as util
import numpy as np
import random
class PrioritizedExperienceReplay(Buffer):
    def __init__(self,memory_spec,body):
        print(f"Creating PER with spec {memory_spec}")
        util.set_attr(self,memory_spec,keys=[
            "use_cer",
            "batch_size",
            "max_size",
            "alpha",
            "epsilon"
        ])
        super().__init__(memory_spec,body)
        #TODO: add memory spec
        self.epsilon = torch.full((1,),self.epsilon,device=self.device)
        self.alpha = torch.full((1,),self.alpha,device=self.device)

        self.data_keys = ["states","actions","rewards","next_states","done","priorities"]
        self.reset()
    
    def reset(self):
        super().reset()
        self.tree = SumTree(capacity=self.max_size)
    

    def add_experience(self, state, action, reward, next_state, done, error=100000):
        '''
        Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.
        '''
        super().add_experience(state, action, reward, next_state, done)
        priority = self.get_priority(error)
        self.priorities[self.head] = priority
        self.tree.add(priority, self.head)

    def get_priority(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        return torch.pow(error + self.epsilon, self.alpha).squeeze()

    def sample_idxs(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        batch_idxs = torch.zeros(batch_size,dtype=torch.long,device=self.device)
        tree_idxs = torch.zeros(batch_size,dtype=torch.long,device=self.device)

        for i in range(batch_size):
            s = random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx

        batch_idxs = batch_idxs.long()
        self.tree_idxs = tree_idxs
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.head
        return batch_idxs

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self.get_priority(errors)
        assert len(priorities) == self.batch_idxs.numel(), f"Error between size of batch_idxs and priorities batch_idxs size: {self.batch_idxs.numel()} size of priorities: {len(priorities)}"
        for idx, p in zip(self.batch_idxs, priorities):
            self.priorities[idx] = p
        for p, i in zip(priorities, self.tree_idxs):
            self.tree.update(i, p)

class SumTree:
    '''
    Helper class for PrioritizedReplay

    This implementation is, with minor adaptations, Jaromír Janisch's. The license is reproduced below.
    For more information see his excellent blog series "Let's make a DQN" https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

    MIT License

    Copyright (c) 2018 Jaromír Janisch

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    '''
    write = 0

    def __init__(self, capacity):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1,dtype=torch.float32,device=self.device)  # Stores the priorities and sums of priorities
        self.indices = torch.zeros(capacity,dtype=torch.long,device=self.device)  # Stores the indices of the experiences

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.indices[indexIdx])

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Prio: {self.tree[j]}')