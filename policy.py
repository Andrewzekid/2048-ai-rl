import random
from typing import Tuple,List
class Policy:
    def __init__(self,decay,epsilon_start,epsilon_end,maxsteps,trainer):
            #EP-Greedy Decay
        self.epsilon = epsilon_start
        self.trainer = trainer
        self.decay_fn = ExponentialDecay(epsilon_start=epsilon_start,epsilon_end=epsilon_end,maxsteps=maxsteps)
    
    def decay(self):
        self.epsilon = self.decay_fn(self.epsilon)
    
    def choice(self):
        gb = trainer.gb
        s = gb.board
        valid_moves = gb.get_valid_moves(s) 
        terminal = False
        s_t = trainer.one_hot(s)
        if random.random() < self.epsilon:
            chosen_a = random.choice(valid_moves)
        else:
        
        move_func = gb.MOVES[chosen_a]
        s_1,move_made,score = eval_move(move_func,s)
        gb.board = s_1
        terminal = gb.has_valid_move()
        trainer.buffer.add_experience(s_t,a,score,s_1,terminal)
    
    def eval_move(self,move_func,s:List[List[int]]) -> Tuple[List[List[int]],bool,int]:
        """Given a move_function, return the new state and extra score after move execution"""
        return move_func(s)

class EpsilonGreedyPolicy(Policy):
    def choice(self,actions):
        """Given a list of valid actions, choose one to take based on the epsilon greedy policy"""  
        
            # print(f"[info] s_t: {s_t}")

            
                    v = trainer.gamma * trainer.agent(torch.tensor(ns)).unsqueeze(-1) + gains
                    max_idx = torch.argmax(v)
                    gb.board = ns_unencoded[max_idx]
                    terminal = gb.has_valid_move()   
        if random.random() < self.epsilon:
            #Random action
            a = random.choice(valid_moves)
            move_func = gb.MOVES[a]
            s_1,move_made,score = move_func(s)
            gb.board = s_1
            terminal = gb.has_valid_move()
            trainer.buffer.add_experience(s_t,a,score,s_1,terminal)
        else:
            #Greedy action
            with torch.inference_mode():
                trainer.agent.eval()
                gains = []
                ns = []
                ns_unencoded = []

                for a in valid_moves:
                    move_func = gb.MOVES[a]
                    s_1,move_made,score = move_func(s)
                    gains.append(score)
                    ns.append(trainer.one_hot(s_1))
                    ns_unencoded.append(s_1)

class BoltzmannPolicy(EpsilonGreedyPolicy):

    def choice()
    raise NotImplementedError

def policy_factory(policy_name:str,decay,epsilon_start,epsilon_end,maxsteps,trainer):
    """Factory for a policy, boltzmann = BoltzmannPolicy epgreedy = EpsilonGreedyPolicy"""
    if policy_name == "boltzmann":
        return BoltzmannPolicy(decay,epsilon_start,epsilon_end,maxsteps,trainer)
    elif policy_name == "epgreedy":
        return EpsilonGreedyPolicy(decay,epsilon_start,epsilon_end,maxsteps,trainer)
    else:
        raise ValueError("Policy Name: %s is not a valid policy!".format(policy_name))
     