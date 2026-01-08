import numpy as np
import torch
from ai import Trainer,RLAgent
import random
from typing import List
from ai import RLAgent
#File for the game code
CELL_COUNT = 4 #4x4
DISTRIBUTION = np.array([2,2,2,2,2,2,2,2,2,4])
class GameBoard:
    def __init__(self,cell_count=CELL_COUNT,distribution=DISTRIBUTION):
        self.CELL_COUNT = cell_count
        self.DISTRIBUTION = distribution
        self.MOVES = {"00":self.move_right,"01":self.move_left,"10":self.move_up,"11":self.move_down}
        self.reset()
    
    def reset(self):
        """Resets the game board and game parameters after the game is over"""
        self.board = self.initialize_game()
        self.score = 0
        self.game_over = False

    def initialize_game(self):
        NUMBER_OF_SQUARES = self.CELL_COUNT * self.CELL_COUNT
        board = torch.zeros((NUMBER_OF_SQUARES,), dtype=torch.float32)
        initial_twos = np.random.default_rng().choice(NUMBER_OF_SQUARES, 2, replace=False)
        board[initial_twos] = 2
        board = board.reshape((self.CELL_COUNT, self.CELL_COUNT))
        return board

    def push_right(self,board):
        new = torch.zeros((self.CELL_COUNT,self.CELL_COUNT),dtype=torch.float32)
        changed = False
        for row in range(self.CELL_COUNT):
            cntr = self.CELL_COUNT - 1
            for col in range(self.CELL_COUNT - 1,-1,-1):
                if board[row,col] != 0:
                    new[row,cntr] = board[row,col]
                    if cntr != col:
                        changed = True
                    cntr -=1
        return new,changed

    def merge_elements(self,board: np.array):
        changed = False
        score = 0
        for row in range(self.CELL_COUNT):
            cntr = self.CELL_COUNT - 1
            for col in range(self.CELL_COUNT - 1,0,-1):
                if board[row,col] != 0 and board[row,col] == board[row,col - 1]:
                    board[row,col] *= 2
                    score += board[row,col]
                    changed = True
                    board[row,col - 1] = 0
        return board,score,changed

    def move_right(self,board):
        board,has_pushed = self.push_right(board)
        board,score,has_merged = self.merge_elements(board)
        board,_ = self.push_right(board)
        move_made = has_pushed or has_merged
        return board,move_made,score

    def move_left(self,board):
        board = torch.rot90(board,k=2)
        board,has_pushed = self.push_right(board)
        board,score,has_merged = self.merge_elements(board)
        board,_ = self.push_right(board)
        board = torch.rot90(board,-2)
        move_made = has_pushed or has_merged
        return board,move_made,score

    def move_up(self,board):
        board = torch.rot90(board,-1)
        board,has_pushed = self.push_right(board)
        board,score,has_merged = self.merge_elements(board)
        board,_ = self.push_right(board)
        board = torch.rot90(board,1)
        move_made = has_pushed or has_merged
        return board,move_made,score

    def move_down(self,board):
        board = torch.rot90(board,1)
        board,has_pushed = self.push_right(board)
        board,score,has_merged = self.merge_elements(board)
        board,_ = self.push_right(board)
        board = torch.rot90(board,-1)
        move_made = has_pushed or has_merged
        return board,move_made,score

    def add_new_tile(self,board):
        """Adds a new tile according to a distribution"""
        #get empty tiles
        candidates = []
        for row in range(self.CELL_COUNT):
            for col in range(self.CELL_COUNT):
                #assume regular square
                if board[row,col] == 0:
                    candidates.append((row,col))
        chosen_num = self.DISTRIBUTION[random.randint(0,len(self.DISTRIBUTION) - 1)]
        row,col = candidates[random.randint(0,len(candidates) - 1)]
        board[row][col] = chosen_num
        return board


    def has_valid_move(self) -> bool:
        """Determines whether the player still has a valid move
        Returns: boolean value indicating if no valid moves (False) or has valid moves (True)
        """
        return bool(len(self.get_valid_moves(self.board)))

    def get_valid_moves(self,board) -> List[str]:
        """Get valid move
        Returns: List[str]: List of strings, with each string being a binary code for the move made
        """
        moves = [(self.move_right,"00"),(self.move_left,"01"),(self.move_up,"10"),(self.move_down,"11")] #RIGHT = 00, LEFT = 01, UP = 10, DOWN = 11
        valid_moves = []
        for move_func,bin_code in moves:
            copy = board.detach()
            new_board,move_made,score = move_func(copy)
            if move_made: valid_moves.append(bin_code)
        return valid_moves

    def display_board(self):
        """Prints the board to the terminal"""
        for rown in range(self.CELL_COUNT):
            print(*self.board[rown],sep="| ")
    
    def ai_mode(self,weights_file:str,n:int):
        """Runs the game in AI mode for evaluation of the AI.
        :param weights_file: name of the file keeping the ai weights
        :param n: number of turns to run the game for
        """
        agent = RLAgent()
        trainer = Trainer(agent=agent)
        trainer.load(weights_file)
        while not self.game_over:
            with torch.inference_mode():
                print("Game board: ")
                print(f"Score: {self.score}")
                self.display_board()
                valid_moves = self.get_valid_moves(self.board)
                move_to_bin = {"W":"10","A":"01","S":"11","D":"00"}
                choices = []
                gains = []
                for a_t in valid_moves:
                    move_func = self.MOVES[a_t]
                    s_t1,move_made,r_t = move_func(s_t)
                    one_hot = torch.unsqueeze(trainer.one_hot(s_t1),0).to(device)
                    v_t1 = trainer.agent(one_hot)
                    v_t1_int = v_t1.item()

                    choices.append((s_t,a_t,r_t,s_t1))
                    gains.append(trainer.gamma*v_t1_int + r_t)
                cont = input("Press any key to continue: ")

                s_t,a_t,r_t,s_t1 = choices[np.argmax(np.array(gains))]
                self.board = self.add_new_tile(s_t1)
                self.score += r_t

                #Update the buffer
                one_hot = torch.unsqueeze(trainer.one_hot(s_t),0).to(device)

                board,_,score = self.MOVES[move_bin](self.board)
                #Update the score and board state
                self.board = self.add_new_tile(board)
                self.score += score

            if not self.has_valid_move():
                print(f"Game Over! Final Score: {self.score}")
                self.game_over = True
                self.reset()
                break
            


    def player_mode(self):
        """Initializes the game loop for testing"""
        while not self.game_over:
            print("Game board: ")
            print(f"Score: {self.score}")
            self.display_board()
            valid_moves = self.get_valid_moves(self.board)
            is_valid = False
            move = ""
            move_to_bin = {"W":"10","A":"01","S":"11","D":"00"}
            while not is_valid:
                move = input("Enter a move: (WASD). W: Up A: Left S: Down D: Right: ").upper()
                if move in ["W","A","S","D"]:
                    move_bin = move_to_bin[move]
                    if move_bin in valid_moves:
                        is_valid = True
                        break
                print("Invalid Move!")
            board,_,score = self.MOVES[move_bin](self.board)
            #Update the score and board state
            self.board = self.add_new_tile(board)
            self.score += score

            if not self.has_valid_move():
                print(f"Game Over! Final Score: {self.score}")
                self.game_over = True
                self.reset()
                break
            



