from gamenv import GameBoard
import math
import numpy as np
import ai.util as util
    # gb = GameBoard()


class MCTSNode(Node):
    """Monte Carlo Tree Search algorithm to play 2048. TODO: add dqn value function into action selection"""
    def __init__(self,state=None,parent=None,action=None,gb=None,agent=None,score=0,max_search_depth=20):
        """Initializes a Node for the MCTS tree
        :param state game board
        :parent parent node
        :param action (int) action taken to get to the current node
        :param gb GameBoard object
        :param agent RLAgent object
        """
        self.gb = gb #Gameboard object with utility classes
        self.agent = agent
        self.state = state
        self.parent = parent
        self.action = action
        self.score = score
        self.action_to_str = {0:"right",1:"left",2:"up",3:"down"}
        self.untried_actions = self.gb.get_valid_moves(self.state)
        self.children = []
        self.wins = 0
        self.max_search_depth = max_search_depth #Max 20 move rollout
        self.visits = 0
    
    def __repr__(self):
        """Object representation for the current node"""
        move = self.action_to_str[self.action]
        actions = []
        for action in self.untried_actions:
            actions.append(self.action_to_str[action])
        return f"[M: {move}, V:{self.score}  Untried Actions: {actions}]"
    
    def display_tree(self,depth=1,max_depth=3):
        """Display N moves of the search tree
        :param depth (How deep to display)
        """
        if display_depth <= depth:
            print(" | " * depth,repr(self))
            for child in self.children:
                child.display_tree(depth+1,max_depth=max_depth)
    @property
    def is_terminal(self):
        """Checks if the game is finished"""
        return bool(len(self.gb.get_available_moves(self.state)))
    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def expand(self):
        """Expands an action at random"""
        action = untried_actions.pop() #in the future, change this to selection via softmax probabilities with value function Q(s,a)
        #Take the action
        move = self.gb.MOVES[action]
        new_state,move_made,score = move(self.state)
        new_score = score + self.score
        new_state = self.gb.add_new_tile(gb) #Stochastic sampling to add a new tile. Cons: may ignore edge cases which would result in death
        #Return the new child node
        child = MCTSNode(state=new_state,parent=self,action=action,gb=self.gb,agent=self.agent,score=new_score,max_search_depth=self.max_search_depth)
        self.children.append(child)
        return child
      
    def _backpropagate(self,win:int):
        """Updates the parent node's wins and number of visits
        :param win (int) number of wins. 0 if the game is over and 1 if the game is still alive
        """
        self.visits += 1
        self.wins += win
        if self.parent:
            self.parent._backpropagate(win)
    
    def rollout(self,search_depth=1):
        """Simulates rollouts from the current node onwards"""
        if search_depth >= self.max_search_depth:
            win = 0 if self.is_terminal else 1
            self.wins += win
            self._backpropagate(win)
        else:
            if self.is_terminal:
                win = 0
                self._backpropagate(win)
            else:
                if not self.is_fully_expanded:
                    node = self.expand()
                    node.rollout(search_depth=search_depth+1)
        
    
    def get_best_child(self,c=1.4):
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
        
        

class MCTSNodeQ(MCTSNode):
    """MCTS Node with DQN inclusion"""
    def __init__(self,state=None,parent=None,action=None,gb=None,agent=None):
        super().__init__(state=state,parent=parent,action=action,gb=gb,agent=agent)
        with torch.inference_mode():
                self.q_values = util.batch_get(self.agent(self.gb.one_hot(self.state).unsqueeze(0)),self.untried_actions) #Only get the q values for the actions which can be taken
    
    def get_best_child(self,c=1.4):
        for child in self.children:
            if child.visits == 0:
                return child

        def ucb(child,c=1.4):
            """UCB including Q(s,a) values"""
            exploitation = child.wins / child.visits + self.q_values[child.action]/child.visits
            exploration= c * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration
        return max(self.children,key=ucb)

class MonteCarloTreeSearch:
    def __init__(self,state=None,parent=None,action=None,gb=None,agent=None,max_search_depth=20):
        self.state = state
        self.parent = parent
        self.action = action
        self.gb = gb
        self.agent = agent
        self.max_search_depth = max_search_depth
    
    def mcts_search(self,iterations=500):
        root = MCTSNode(self.state,self.parent,self.action,self.gb,self.agent,self.max_search_depth)
        node = root
        for _ in range(iterations):
            while not node.is_terminal and node.is_fully_expanded:
                #The node is not a terminal node and is fully expanded => get child
                node = node.get_best_child()
            if not node.is_terminal and not node.is_fully_expanded:
                node.rollout()
        
        return root.get_best_child()
