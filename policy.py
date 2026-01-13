import random
from typing import Tuple,List
from torch.distributions import Categorical
from abc import abstractmethod
class Policy:
    def __init__(self,decay,epsilon_start,epsilon_end,maxsteps,trainer,num_actions=4):
            #EP-Greedy Decay
        self.epsilon = epsilon_start
        self.trainer = trainer
        self.decay_fn = ExponentialDecay(epsilon_start=epsilon_start,epsilon_end=epsilon_end,maxsteps=maxsteps)
        self.num_actions = 4
    
    def decay(self):
        self.epsilon = self.decay_fn(self.epsilon)
    
    def calc_q_vals(self) -> torch.tensor:
        """Calculates the Q values for all possible actions in the current game state"""
        gb = trainer.gb
        s = gb.board
        valid_moves = gb.get_valid_moves()
        with torch.inference_mode():
            trainer.agent.eval()
            q = util.batch_get(trainer.agent(s).numpy(),valid_moves)
            q_values = np.zeros(self.num_actions)
            q_values[valid_moves] = q #Returns 4 Q values
        return q_values

    @abstractmethod
    def choice(self):
        """Chooses an action a"""
        raise NotImplementedError
    

class EpsilonGreedyPolicy(Policy):
    def choice(self) -> int:
        """Given a list of valid actions, choose one to take based on the epsilon greedy policy
        Returns: action (0-num_actions)
        """ 
        q_values = self.calc_q_vals()
        if random.random() < self.epsilon:
            #Random action
            a = random.randint(0,self.num_actions)
        else:
            #Greedy action
            a = torch.argmax(q_values)
        return a
    
class BoltzmannPolicy(Policy):
    def choice(self) -> int:
        """Choose an action according to the boltzmann policy.
        Returns: action (0-num_actions)
        """
        q_values = self.calc_q_vals() / self.epsilon
        logits = torch.softmax(q_values)
        pd = Categorical(logits)
        a = pd.sample().item()
        return a



def policy_factory(policy_name:str,decay,epsilon_start,epsilon_end,maxsteps,trainer):
    """Factory for a policy, boltzmann = BoltzmannPolicy epgreedy = EpsilonGreedyPolicy"""
    if policy_name == "boltzmann":
        return BoltzmannPolicy(decay,epsilon_start,epsilon_end,maxsteps,trainer)
    elif policy_name == "epgreedy":
        return EpsilonGreedyPolicy(decay,epsilon_start,epsilon_end,maxsteps,trainer)
    else:
        raise ValueError("Policy Name: %s is not a valid policy!".format(policy_name))
     