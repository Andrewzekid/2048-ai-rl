from gameenv import GameBoard
from ai import RLAgent,Trainer
import random
#Key Parameters
MAX_ITERATIONS = 10000
BUFFER_SIZE = 1024
EPOCHS = 10 #How many epochs to train for for each batch
if __name__ == "__main__":
    agent = RLAgent()
    trainer = Trainer()
    iterations = 0
    train_steps = 0
    #Game params
    nGames = 0
    totalScore = 0
    maxScore = 0
    #Main loop
    while iterations < MAX_ITERATIONS: #Check if gameover
        if iterations % 1024 == 0:
            #Train step
            train_loss = 0
            train_steps += 1
            for epoch in range(EPOCHS):
                train_loss += trainer.train_step()
            print(f"Batch {iterations//1024} | Train Loss: {(train_loss / EPOCHS):.2f} | Average Score: {(totalScore / nGames):.2f} | Number of Games: {nGames} | Max Score: {maxScore}")
            trainer.buffer.clear()
            #Add the model saving code
            if train_steps % 500 == 0:
                #Save the game every 500 steps
                trainer.save() 
        else:
            s_t = gb.board
            valid_moves = gb.get_valid_moves(gb.board)
            choices = []
            gains = []
            for a_t in valid_moves:
                move_func = gb.MOVES[a_t]
                s_t1,move_made,r_t = move_func(s_t)
                v_t1 = agent(s_t1)

                choices.append((s_t,a_t,r_t,s_t1))
                gains.append(v_t1 + r_t)


            s_t,a_t,r_t,s_t1 = choices[np.argmax(gains)]
            gb.board = gb.add_new_tile(s_t1)

            #Update the buffer
            v_t = agent(s_t)
            trainer.buffer.add_data(v_t,r_t,v_t1)

        iterations+=1

        #Check game continuation
        gb.game_over = gb.has_valid_move()
        if gb.game_over:
            nGames += 1
            totalScore += gb.score
            maxScore = max(gb.score,maxScore)
            gb.reset() #Restart the game board


        #Update Buffer

        #Gradient Descent

        #CKPT

    

