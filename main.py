
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
    #Game params
    nGames = 0
    totalScore = 0
    maxScore = 0


    def collect_data(gb,trainer,policy):
        """performs one data collection step and adds it to the PER"""
        s = gb.board
        a = policy.choice(gb) 
        move_func = gb.MOVES[a]
        s1,_,r = move_func(s)
        gb.board = s1
        terminal = gb.has_valid_move()

        s_oh = trainer.one_hot(s).numpy()
        s1_oh = trainer.one_hot(s1).numpy()
        trainer.buffer.add_experience(s_oh,a,r,s1_oh,terminal)


    #Main loop
    print("[INFO] Beginning gameplay \n Initializing Data Collection...")
    while iterations < MAX_ITERATIONS: #Check if gameover
        if (iterations % 4 == 0) and iterations > START_SIZE:
            trainer.train_mode()
            #Parallel training 
            net = trainer.agent.share_memory()
            for i in tqdm(range(NUM_BATCHES),desc="Training Progress"):
                batch = trainer.buffer.sample()
                trainer.parallelize(trainer.train_step,args=(net,batch,))
            #Add eval code for the message displaying
            if(iterations % 50 == 0):
                print(f"[INFO] Beginning Evaluation")
                train_loss = 0
                #Print eval message and log after every 1000 iterations
                trainer.eval()
                for i in tqdm(range(NUM_BATCHES),desc="Testing Progress"):
                    with torch.inference_mode():
                        batch = trainer.buffer.sample()
                        train_loss += trainer.test_step(batch)

                train_loss /= NUM_BATCHES #Avg loss per epoch per batch
                train_steps += 1

                msg = f"EPOCH {train_steps} | Train Loss: {(train_loss):.2f} | Average Score for past 1k games: {(totalScore / nGames):.2f} | Number of Games: {nGames} | Max Score: {maxScore}"    
                trainer.logging(f"{train_loss},{int(totalScore/nGames)}") #Write as csv format
                print("[INFO] " + msg)
                trainer.update_params() #sync targ Q net and Q net params
                #Reset params
                nGames = 0
                totalScore = 0
                maxScore = 0

            if (iterations - START_SIZE + 10000) % 10000 == 0:
                #Save the model weights every 10000 steps
                filename = f"{train_steps}.pth"
                trainer.save(filename) 
                print(f"[INFO] Saving model weights to {filename}")
        else:
            #Collect data on multiple cpus
            collect_data(gb,trainer,policy)
            
        iterations+=1
        trainer.decay()
        #Check game continuation
        if not(gb.has_valid_move()):
            gb.game_over = True
            nGames += 1
            totalScore += gb.score
            maxScore = max(gb.score,maxScore)
            gb.reset() #Restart the game board
        
        #Logging
        num_data = len(trainer.buffer)
        if collecting_data and (num_data % 1000 == 0):
            print(f"[INFO] PER Collected {num_data}/{BUFFER_SIZE} Experiences!")
            if num_data > BUFFER_SIZE:
                collecting_data = False





