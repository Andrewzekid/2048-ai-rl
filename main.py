
from ai import RLAgent,Trainer
from gamenv import GameBoard
import numpy as np
import random
from policy import policy_factory
import pdb
from torch import distributions
import logging
import torch
#Key Parameters
MAX_ITERATIONS = 100000
BUFFER_SIZE = 1024
POLICY = "boltzmann"
EPOCHS = 10 #How many epochs to train for for each batch
if __name__ == "__main__":
    print("[INFO] Initializing Training... setting global variables")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = RLAgent().to(device) 
    targNet = RLAgent().to(device) # Target Q network
    trainer = Trainer(agent=agent,targNet=targNet)
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
    while iterations < MAX_ITERATIONS: #Check if gameover
        if iterations % 1024 == 0:
            #Train step
            trainer.buffer.to_tensor()
            train_loss = 0
            train_steps += 1
            for epoch in range(EPOCHS):
                train_loss += trainer.train_step()
            #debug
            msg = f"Batch {iterations//1024} | Train Loss: {(train_loss / EPOCHS):.2f} | Average Score: {(totalScore / nGames):.2f} | Number of Games: {nGames} | Max Score: {maxScore}"    
            print("[INFO] " + msg)
            logging.info(msg)
            trainer.buffer.clear()
            #Add the model saving code
            if train_steps % 10 == 0:
                #Save the game every 500 steps
                filename = f"{train_steps}.pth"
                trainer.save(filename) 
        else:
            s = gb.board
            a = policy.choice(gb) 
            move_func = gb.MOVES[a]
            s1,_,r = move_func(s)
            gb.board = s1
            terminal = gb.has_valid_move()

            s_oh = trainer.one_hot(s)
            s1_oh = trainer.one_hot(s1)
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



    

