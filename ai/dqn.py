import torch
import ai.util as util
class DQN:
    def __init__(self,**kwargs):
        self.agent = kwargs.get("agent")
        self.targNet = kwargs.get("targNet")
       
        if not(self.agent or self.targNet): #Manual initialization if both are not provided
            self.init_nets()
    
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
            self.update_params() #Sync the target q network and q network
        else:
            raise FileNotFoundError(f"{path} is not a valid pytorch checkpoint object!")
    
    def train_mode(self):
        self.agent.train()
    
    def test_mode(self):
        self.targNet.eval()
        self.agent.eval()

    def calc_q_loss(self,batch,trainer):
        """Calculates the Q learning loss for the current batch
        :param batch batch of data to train on
        :param trainer (Trainer) trainer class
        """
        #Set config for buffer, gamema,optimizer,loss
        states = batch["states"]
        next_states = batch["next_states"]
        q_preds= self.agent(states) #Calculate ai prime
        with torch.inference_mode():
            next_targ_q = self.targNet(next_states) #action selection in the next state
            next_q_preds = self.agent(next_states)
        action_q_preds = q_preds.gather(-1,batch["actions"].long().unsqueeze(-1)).squeeze(-1)
        sp_actions = next_q_preds.argmax(dim=-1,keepdim=True) #calculate max ai prime
        targ_q_sp = next_targ_q.gather(-1,sp_actions).squeeze(-1)
        y = self.gamma * (1-batch["done"]) * targ_q_sp + batch["rewards"]
        q_loss = self.loss_fn(action_q_preds,y)
        #Add prioritized experience replay code
        if "Prioritized" in util.get_class_name(self.buffer):
            errors = (y - action_q_preds.detach()).abs()
            self.buffer.update_priorities(errors)
        return q_loss
    
    def test_step(self,batch) -> float:
        """Conducts one test step and returns the loss"""
        #add self.eval()
        self.test_mode()
        with torch.inference_mode():
            return self.calc_q_loss(batch).item()
    
    def train_step(self,batch) -> float:
        """Performs one gradient descent step on the TD error
        Returns: loss (float), loss from the current training step
        :param batch batch of training data sampled from the experience buffer
        """
        self.optimizer.zero_grad()
        loss = self.calc_q_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_params(self):
        """Synchronizes the Target Q network and the Q networks parameters"""
        params = self.agent.state_dict()
        self.targNet.load_state_dict(params)
    
    def init_nets(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.targNet = CNN().to(device)
        self.agent = CNN().to(device)
    
    
    def one_hot(self,board:torch.Tensor) -> torch.Tensor:
        """Generates a one hot encoding of the board
        board: torch.Tensor (4,4) Game board
        Returns:
            torch.Tensor (16,4,4) one hot encoding of the game board
        """
        unique_encodings = self.all_tiles.view(-1,1,1) #There are log2(max_tile) + 1 different tiles. Include 0 for the +1
        board = board.unsqueeze(0)
        # print(f"Device of board: {board.device} device of all_tiles: {unique_encodings.device}")
        enc = (unique_encodings == board).float()
        return enc
    
    def simulate(self,gb,n:int=5) -> tuple[float,int]:
        """Performs monte carlo simulation to evaluate the networks performance
        Args:
        :param n (int) number of games to play
        :param gb (GameBoard) gameboard object to perform simulations on
        Returns:
            average score during n games (float) and max score achieved (int)
        """
        self.test_mode()
        totalScore = 0
        maxScore = 0
        for i in range(n):
            while gb.has_valid_move():
                s = gb.board
                valid_actions = gb.get_valid_moves(s)
                with torch.inference_mode():
                    q = self.net.agent(self.one_hot(s).unsqueeze(0)).squeeze()
                    valid_actions_tensor = torch.tensor(valid_actions,device=device,dtype=torch.long)
                    idx = torch.argmax(util.batch_get(q,valid_actions_tensor))
                    action = valid_actions[idx]
                        #get the q values for the valid actions 
                move = gb.MOVES[action]
                sn,move_made,r = move(s)
                sn = gb.add_new_tile(sn)
                assert move_made, f"[ERROR] In Evaluation step, no move was made. Q values: {action}"
                gb.board = sn
                gb.score += r
            #Game over
            score = gb.score
            totalScore += score
            maxScore = max(maxScore,score)
            new_gb.reset()
        return round(totalScore/n,2),maxScore

    def eval(self,n:int=5) -> float:
        """Tests the model performance
        Args:
        :param n(int) number of batches to test for
        Returns:
            Test Loss in float
        """
        self.test_mode()
        test_loss = 0
        with torch.inference_mode():
            for i in tqdm(range(n),desc="Performing Evaluation Step: "):
                batch = self.buffer.sample()
                test_loss += self.test_step(batch)
        test_loss /= n #Avg loss per epoch per batch
        return test_loss
    
    def train(self,n:int=5) -> float:
        """Trains the model for n batches
        Args:
        :param n (int) number of batches to run the model through
        Returns:
            Train loss to 2 dp
        """
        self.train_mode()
        train_loss = 0
        for i in tqdm(range(n),desc="Training Progress"):
            batch = self.buffer.sample()
            train_loss += trainer.train_step(qnet=net,batch=batch)
        train_loss /= n
        return train_loss