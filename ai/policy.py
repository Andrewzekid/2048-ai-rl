import random
from typing import Tuple,List
import ai.util as util
from ai.decay import ExponentialDecay
import numpy as np
from torch.distributions import Categorical
from abc import abstractmethod
import torch
class Policy:
    def __init__(self,epsilon_start,epsilon_end,maxsteps,trainer,num_actions=4):
            #EP-Greedy Decay
        self.epsilon = epsilon_start
        self.trainer = trainer
        self.decay_fn = ExponentialDecay(epsilon_start=epsilon_start,epsilon_end=epsilon_end,maxsteps=maxsteps)
        self.num_actions = 4
    
    def decay(self):
        self.epsilon = self.decay_fn(self.epsilon)
    
    def calc_q_vals(self,gb):
        """Calculates the Q values for all possible actions in the current game state
        :param gb Gameboard object
        """
        s = gb.board
        trainer = self.trainer
        valid_moves = gb.get_valid_moves(s)
        with torch.inference_mode():
            trainer.agent.eval()
            q_values = trainer.agent(trainer.one_hot(s)).squeeze(0).numpy()
            q = util.batch_get(q_values,valid_moves)

            q_valid = np.zeros(self.num_actions)
            q_valid[valid_moves] = q #Returns num_actions Q values
        return q_valid

    @abstractmethod
    def choice(self):
        """Chooses an action a"""
        raise NotImplementedError
    

class EpsilonGreedyPolicy(Policy):
    def choice(self,gb) -> int:
        """Given a list of valid actions, choose one to take based on the epsilon greedy policy
        :param gb GameBoard object
        Returns: action (0-num_actions)
        """ 
        q_values = self.calc_q_vals(gb)
        if random.random() < self.epsilon:
            #Random action
            a = random.randint(0,self.num_actions)
        else:
            #Greedy action
            a = torch.argmax(q_values)
        return a
    
class BoltzmannPolicy(Policy):
    def choice(self,gb) -> int:
        """Choose an action according to the boltzmann policy.
        :param gb Gameboard object
        Returns: action (0-num_actions)
        """
        q_values = torch.tensor(self.calc_q_vals(gb) / self.epsilon)
        logits = torch.softmax(q_values,dim=-1)
        pd = Categorical(logits)
        a = pd.sample().item()
        return a



def policy_factory(policy_name:str,epsilon_start,epsilon_end,maxsteps,trainer):
    """Factory for a policy, boltzmann = BoltzmannPolicy epgreedy = EpsilonGreedyPolicy"""
    if policy_name == "boltzmann":
        return BoltzmannPolicy(epsilon_start,epsilon_end,maxsteps,trainer)
    elif policy_name == "epgreedy":
        return EpsilonGreedyPolicy(epsilon_start,epsilon_end,maxsteps,trainer)
    else:
        raise ValueError("Policy Name: %s is not a valid policy!".format(policy_name))
     