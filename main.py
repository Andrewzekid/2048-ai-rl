
from ai.agent import RLAgent
from ai.trainer import Trainer
from gamenv import GameBoard
import numpy as np
import random
from ai.policy import policy_factory
import pdb
from torch import distributions
import logging
import torch.multiprocessing as mp
from tqdm import tqdm
import ai.util as util
import pdb
import torch
#Key Parameters
MAX_ITERATIONS = 4000000
BUFFER_SIZE = 200000
NUM_BATCHES = 4 #Number of batches to go through
START_SIZE = BUFFER_SIZE//2
POLICY = "boltzmann"
EPOCHS = 10 #How many updates per batch
if __name__ == "__main__":
    print("[INFO] Initializing Training... setting global variables")
    mp.set_start_method("spawn")
    trainer = Trainer()
    trainer.init_nets()

    gb = GameBoard()
    policy = trainer.policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collecting_data = True
    iterations = 0
    train_steps = 0

    nGames = 0
    Score = 0


    def collect_data(gb,trainer,policy):
        """performs one data collection step and adds it to the PER"""
        s = gb.board
        a = policy.choice(gb) 
        move_func = gb.MOVES[a]
        s1,_,r = move_func(s)
        s1 = gb.add_new_tile(s1)
        gb.board = s1
        gb.score += r
        done = 0 if gb.has_valid_move() else 1
        s_oh = trainer.one_hot(s).numpy()
        s1_oh = trainer.one_hot(s1).numpy()
        pdb.set_trace()
        trainer.buffer.add_experience(s_oh,a,r,s1_oh,done)



    #Main loop
    print("[INFO] Beginning gameplay \n Initializing Data Collection...")
    while iterations < MAX_ITERATIONS: #Check if gameover
        if (iterations % 4 == 0) and iterations > START_SIZE:
            print(f"Beginning train_step! Train step: {train_steps}")
            trainer.train_mode()
            #Parallel training 
            net = trainer.agent.share_memory()
            for i in tqdm(range(NUM_BATCHES),desc="Training Progress"):
                batch = trainer.buffer.sample()
                trainer.parallelize(trainer.train_step,args=(net,batch,))
            train_steps += 1

            #Add eval code for the message displaying every 50 steps
            if(train_steps % 25== 0):
                train_loss = 0
                #Print eval message and log after every 1000 iterations
                trainer.eval()
                for i in tqdm(range(4),desc="Performing Evaluation Step: "):
                    with torch.inference_mode():
                        batch = trainer.buffer.sample()
                        train_loss += trainer.test_step(batch)

                train_loss /= NUM_BATCHES #Avg loss per epoch per batch
                #Create new games
                num_Games = 4
                totalScore = 0
                maxScore = 0
                new_gb = GameBoard()
                for i in range(num_Games):
                    while new_gb.has_valid_move():
                        s = new_gb.board
                        valid_actions = new_gb.get_valid_moves(s)
                        with torch.inference_mode():
                            q = trainer.agent(trainer.one_hot(s).unsqueeze(0)).squeeze()
                            idx = torch.argmax(util.batch_get(q,valid_actions))
                            action = valid_actions[idx]
                             #get the q values for the valid actions
                        
                        move = new_gb.MOVES[action]
                        sn,move_made,r = move(s)
                        sn = gb.add_new_tile(sn)
                        assert move_made, f"[ERROR] In Evaluation step, no move was made. Q values: {action}"
                        new_gb.board = sn
                        new_gb.score += r
                    
                    #Game over
                    score = new_gb.score
                    totalScore += score
                    maxScore = max(maxScore,score)
                    new_gb.reset()

                avgScore = int(totalScore / num_Games)
                msg = f"EPOCH {train_steps} | Train Loss: {(train_loss):.2f} | Average Score for past {num_Games} games: {avgScore}  | Max Score: {maxScore}"    
                trainer.logging(f"{train_loss},{avgScore}") #Write as csv format
                print("[INFO] " + msg)
                
            if(train_steps %50 == 0):
                #Save the model weights every 10000 steps
                filename = f"{train_steps}.pth"
                trainer.save(filename) 
                print(f"[INFO] Saving model weights to {filename}")
                trainer.update_params() #sync targ Q net and Q net params
                print(f"[INFO] Synchronizing Q and target Q networks")
        else:
            #Check game continuation
            if gb.has_valid_move():
                #Collect data on multiple cpus
                collect_data(gb,trainer,policy)
            else:
                gb.game_over = True
                nGames += 1
                Score += gb.score
                gb.reset() #Restart the game board
            
        iterations+=1
        trainer.decay()
        #Logging
        num_data = len(trainer.buffer)
        if collecting_data and (num_data % 1000 == 0):
            print(f"[INFO] PER Collected {num_data}/{BUFFER_SIZE} Experiences! Games Played: {nGames} Avg Score: {int(Score / nGames)}")
            if num_data > BUFFER_SIZE:
                collecting_data = False





