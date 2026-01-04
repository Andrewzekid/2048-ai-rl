from gameenv import GameBoard
from ai import RLAgent,Trainer
import numpy as np
import random
import pdb
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
    iterations = 1
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
            s_t = gb.board
            oh = trainer.one_hot(s_t)
            # print(f"[info] s_t: {s_t}")

            valid_moves = gb.get_valid_moves(gb.board)
            choices = []
            gains = []
            for a_t in valid_moves:
                move_func = gb.MOVES[a_t]
                s_t1,move_made,r_t = move_func(s_t)
                one_hot = torch.unsqueeze(trainer.one_hot(s_t1),0).to(device)
                v_t1 = agent(one_hot)
                v_t1_int = v_t1.item()

                choices.append((s_t,a_t,r_t,s_t1))
                gains.append(v_t1_int + r_t)


            s_t,a_t,r_t,s_t1 = choices[np.argmax(np.array(gains))]
            gb.board = gb.add_new_tile(s_t1)
            gb.score += r_t

            #Update the buffer
            one_hot = torch.unsqueeze(trainer.one_hot(s_t),0).to(device)
            v_t = agent(one_hot)
            # print(f"Adding data {v_t} {r_t} {v_t1} to the buffer")
            trainer.buffer.add_data(v_t,r_t,v_t1)

        iterations+=1

        #Check game continuation
        if not gb.has_valid_move():
            gb.game_over = True
            nGames += 1
            totalScore += gb.score
            maxScore = max(gb.score,maxScore)
            gb.reset() #Restart the game board



    

