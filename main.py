
from ai.agent import RLAgent
from ai.trainer import Trainer
from gamenv import GameBoard
import numpy as np
import random
from ai.policy import policy_factory
import pdb
from torch import distributions
import logging
import torch
#Key Parameters
MAX_ITERATIONS = 100000000000
BUFFER_SIZE = 20000
NUM_BATCHES = 16 #Number of batches to go through
POLICY = "boltzmann"
EPOCHS = 10 #How many updates per batch
if __name__ == "__main__":
    print("[INFO] Initializing Training... setting global variables")
    trainer = Trainer()
    gb = GameBoard()
    policy = trainer.policy
    iterations = 0
    train_steps = 0
    #Game params
    nGames = 0
    totalScore = 0
    maxScore = 0
    #Main loop
    print("[INFO] Beginning gameplay")
    print("[INFO] Initializing Data Collection...")
    while iterations < MAX_ITERATIONS: #Check if gameover
        if (iterations % 1024 == 0) and iterations > BUFFER_SIZE:
            train_loss = 0
            trainer.train_mode()

            if train_steps % 10 == 0:
                #Save the game every 500 steps
                filename = f"{train_steps}.pth"
                trainer.save(filename) 
            for i in range(NUM_BATCHES):
                #Train step
                batch = trainer.buffer.sample()
                for epoch in range(EPOCHS): 
                    train_loss += (trainer.train_step(batch) / EPOCHS)

            train_loss /= NUM_BATCHES #Avg loss per epoch per batch
            train_steps += 1
            msg = f"EPOCH {train_steps} | Train Loss: {(train_loss):.2f} | Average Score: {(totalScore / nGames):.2f} | Number of Games: {nGames} | Max Score: {maxScore}"    
            print("[INFO] " + msg)
        else:
            s = gb.board
            a = policy.choice(gb) 
            move_func = gb.MOVES[a]
            s1,_,r = move_func(s)
            gb.board = s1
            terminal = gb.has_valid_move()

            s_oh = trainer.one_hot(s).numpy()
            s1_oh = trainer.one_hot(s1).numpy()
            trainer.buffer.add_experience(s_oh,a,r,s1_oh,terminal)

        iterations+=1
        trainer.decay()
        #Check game continuation
        if terminal:
            gb.game_over = True
            nGames += 1
            totalScore += gb.score
            maxScore = max(gb.score,maxScore)
            gb.reset() #Restart the game board



    

