from gamenv import GameBoard
import math
import numpy as np
    # gb = GameBoard()
class MonteCarloTreeSearch:
    """Monte Carlo Tree Search algorithm to play 2048. TODO: add dqn value function into action selection"""

    def __init__(self,state=None,parent=None,action=None,gb=None):
        self.gb = gb #Gameboard object with utility classes
        self.state = state
        self.parent = parent
        self.action = action
        self.untried_actions = self.gb.get_avaliable_moves(self.state)
        self.children = []
        self.wins = 0
        self.visits = 0
    
    def is_terminal(self):
        """Checks if the game is finished"""
        return bool(len(self.gb.get_available_moves(self.state)))
    
    def expand(self):
        """Expands an action at random"""
        raise NotImplementedError
    
    def get_children(self):
        """Adds the possible child nodes to self.children"""
        raise NotImplementedError
        
    
    def get_best_child(self,c=1.4) -> int:
        """Gets the best child out of all children using the uct formula. Returns the index of the best child
        :param c (float) exploration constant
        """
        for child in self.children:
            if child.visits == 0:
                return child
            
        def ucb(child,c=1.4):
            exploitation = child.wins / child.visits
            exploration= c * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration
        
        return max(self.children,key=ucb)
        
        

    



