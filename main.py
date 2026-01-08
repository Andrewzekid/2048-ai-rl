from gameenv import GameBoard
from ai import RLAgent,Trainer
import numpy as np
import random
import pdb
from torch import distributions
import logging
import torch
#Key Parameters
MAX_ITERATIONS = 100000
BUFFER_SIZE = 1024
EPOCHS = 10 #How many epochs to train for for each batch
if __name__ == "__main__":
    print("[INFO] Initializing Training... setting global variables")
    logging.basicConfig(
        filename="training.log",
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gb = GameBoard()
    agent = RLAgent().to(device)
    trainer = Trainer(agent=agent)
    trainer.load("60.pth")
    iterations = 10241
    train_steps = 60
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
            terminal = False
            s,a,r,s_1,terminal = epsilon_greedy()
            trainer.buffer.add_experience(s,a,r,s_1,terminal)

        iterations+=1
        trainer.decay()
        #Check game continuation
        if terminal:
            gb.game_over = True
            nGames += 1
            totalScore += gb.score
            maxScore = max(gb.score,maxScore)
            gb.reset() #Restart the game board



    

def epsilon_greedy(gb):
    terminal = False
    s = gb.board
    s_oh = trainer.one_hot(s)
    valid_moves = gb.get_valid_moves(gb.board)
    if random.random() < trainer.epsilon:
        #Random action
        a = random.choice(valid_moves)
        move_func = gb.MOVES[a]
        s_1,_,r = move_func(s)
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
            v = trainer.gamma * trainer.agent(torch.tensor(ns)).unsqueeze(-1) + gains
            max_idx = torch.argmax(v)
            s_1 = ns[max_idx]
            s_1_oh = ns_unencoded[max_idx]
            a = valid_moves[max_idx]
            r = gains[max_idx]

    gb.board = s_1_oh
    terminal = gb.has_valid_move()

    return (s_oh,a,r,s_1,terminal)

def boltzmann(gb):
    terminal = False
    s = gb.board
    s_oh = trainer.one_hot(s)
    valid_moves = gb.get_valid_moves(gb.board)
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
        v = trainer.gamma * trainer.agent(torch.tensor(ns)).unsqueeze(-1) + gains
        v_exp = torch.exp(v) / trainer.epsilon
        boltzmann_pd = v_exp / (torch.sum(v_exp))
        pd = Categorical(boltzmann_pd)
        idx = pd.sample().item()

        s_1 = ns[idx]
        s_1_oh = ns_unencoded[idx]
        a = valid_moves[idx]
        r = gains[idx]

    gb.board = s_1_oh
    terminal = gb.has_valid_move()

    return (s_oh,a,r,s_1,terminal)