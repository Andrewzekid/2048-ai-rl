
from ai.agent import CNN
from ai.trainer import Trainer
from gamenv import GameBoard
import numpy as np
import random
from ai.policy import policy_factory
import pdb
from torch import distributions
import logging
from datetime import datetime
from dqn import DQN
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
import ai.util as util
from ai.logging import log
import pdb
import torch
#Key Parameters
MAX_ITERATIONS = 4000000
BUFFER_SIZE = 100000
NUM_BATCHES = 5 #Number of batches to go through
START_SIZE = BUFFER_SIZE//2
POLICY = "boltzmann"
LOAD = ""
  
def main():
    parser = argparse.ArgumentParser(prog="train",description="training AI")
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=100000,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--steps', '--s', type=int, default=400000,
                       help='Learning rate')
    args = parser.parse_args()
    config = conf(**args) #Initialize the config
    trainer = Trainer(config=config)
    #Load DQN
    dqn = DQN()
    dqn.init_nets()
    if LOAD:
        dqn.load(LOAD)
    trainer.net = dqn
    
    print("[INFO] Initializing Training... setting global variables")
    gb = GameBoard()
    policy = trainer.policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collecting_data = True
    iterations = 0
    train_steps = 0
    test_steps = 0
    nGames = 0
    Score = 0

    #Main loop
    print("[INFO] Beginning gameplay \n Initializing Data Collection...")
    while iterations < MAX_ITERATIONS: #Check if gameover
        if (iterations % 4 == 0) and iterations > START_SIZE:
            #Parallel training 
            train_loss = trainer.net.train()
            print(f"EPOCH {train_steps} | Train Loss {train_loss}")
            log(mode="train",loss=train_loss,steps=train_steps)
            train_steps += 1

            #Add eval code for the message displaying every 50 steps
            if(train_steps % 25== 0):
                test_loss = trainer.agent.eval()
                test_steps += 1
                avgScore,maxScore = trainer.agent.simulate()

                #Logging functionality
                now = datetime.now()
                msg = f"{now.strftime('%Y-%m-%d %H:%M:%S')} Test Epoch {test_steps} | Test Loss: {test_loss} | Average Score for past {num_Games} games: {avgScore}  | Max Score: {maxScore}"    
                trainer.log(mode="test",loss=test_loss,steps=test_steps,score=avgScore)
                print("[INFO] " + msg)
                
            if(train_steps %50 == 0):
                #Save the model weights every 10000 steps
                filename = f"{train_steps}.pth"
                trainer.agent.save(filename) 
                msg = f"[INFO] Saving model weights to {filename} \n Synchronizing Q and target Q networks"
                print(msg)
                trainer.update_params() #sync targ Q net and Q net params
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
        num_data = len(trainer.agent.buffer)
        if collecting_data and (num_data % 1000 == 0):
            print(f"[INFO] PER Collected {num_data}/{BUFFER_SIZE} Experiences! Games Played: {nGames} Avg Score: {int(Score / nGames)}")
            if num_data > BUFFER_SIZE:
                collecting_data = False
    print(f"[INFO] Finishing training... performing garbage collection")
    trainer.sumwriter.flush()
    trainer.sumwriter.close()

if __name__ == "__main__":
    main()