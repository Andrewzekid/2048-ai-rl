import numpy as np
import torch
from ai.trainer import Trainer
from ai.agent import RLAgent
import random
from typing import List
import pdb
import ai.util as util
#File for the game code
CELL_COUNT = 4 #4x4
DISTRIBUTION = np.array([2,2,2,2,2,2,2,2,2,4])
class GameBoard:
    def __init__(self,cell_count=CELL_COUNT,distribution=DISTRIBUTION):
        self.CELL_COUNT = cell_count
        self.DISTRIBUTION = distribution
        self.MOVES = {0:self.move_right,1:self.move_left,2:self.move_up,3:self.move_down}
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
    

    def move(self,board,k):
        """Implements moving up/right/left/down
        :param board game board
        :param k (int) How many times to rotate 90 Left: 2, right: 0 UP: -1 Down: 1"""
        if k:
            board = torch.rot90(board,k)

        board,has_pushed = self.push_right(board)
        board,score,has_merged = self.merge_elements(board)
        board,_ = self.push_right(board)

        if k:
            board = torch.rot90(board,-k)

        move_made = has_pushed or has_merged
        return board,move_made,score

    def move_right(self,board):
        return self.move(board,k=0)

    def move_left(self,board):
        return self.move(board,k=-2)

    def move_up(self,board):
        return self.move(board,k=-1)

    def move_down(self,board):
        return self.move(board,k=1)

    def add_new_tile(self, board):
        """Adds a new tile (2 or 4) to a random empty cell on the board.
        
        Args:
            board: A 2D tensor representing the game board
            
        Returns:
            The updated board with a new tile added
            
        Raises:
            ValueError: If the board has no empty cells
        """
        # Find all empty positions (where value is 0)
        empty_positions = torch.argwhere(board == 0)
        
        if len(empty_positions) == 0:
            raise ValueError("Cannot add new tile: board is already full")
        
        # Randomly select an empty position
        position_idx = random.randrange(len(empty_positions))
        row, col = empty_positions[position_idx]
        
        # Choose tile value according to the distribution (typically 90% 2, 10% 4)
        tile_value = random.choice(self.DISTRIBUTION)
        
        # Place the tile
        board[row, col] = tile_value
        
        return board

    def has_move(self,board):
        """Copy of has_valid_move but takes in an extra board parameter"""
        return len(self.get_valid_moves(board)) != 0

    def has_valid_move(self) -> bool:
        """Determines whether the player still has a valid move
        Returns: boolean value indicating if no valid moves (False) or has valid moves (True)
        """
        return len(self.get_valid_moves(self.board)) != 0

    def get_valid_moves(self,board) -> List[str]:
        """Get valid move
        Returns: List[str]: List of strings, with each string being a binary code for the move made
        """
        moves = [(self.move_right,0),(self.move_left,1),(self.move_up,2),(self.move_down,3)] #RIGHT = 00, LEFT = 01, UP = 10, DOWN = 11
        valid_moves = []
        for move_func,bin_code in moves:
            copy = board.clone()
            new_board,move_made,score = move_func(copy)
            if move_made: valid_moves.append(bin_code)
        return valid_moves

    def display_board(self):
        """Prints the board to the terminal"""
        for rown in range(self.CELL_COUNT):
            for coln in range(self.CELL_COUNT):
                print(f"{int(self.board[rown][coln].item())} | ",end="")
            print()
    
    def ai_mode(self,weights_file:str):
        """Runs the game in AI mode for evaluation of the AI.
        :param weights_file: name of the file keeping the ai weights
        :param n: number of turns to run the game for
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = RLAgent().to(device)
        targNet = RLAgent().to(device)
        trainer = Trainer(agent=agent,targNet=targNet)
        trainer.load(weights_file)
        while self.has_valid_move():
            with torch.inference_mode():
                trainer.eval()
                print("Game board: ")
                print(f"Score: {self.score}")
                self.display_board()
                valid_moves = self.get_valid_moves(self.board)
                s_t = self.board
                move_to_bin = {2:"Up",1:"Left",3:"Down",0:"Right"}


                one_hot = torch.unsqueeze(trainer.one_hot(s_t),0).to(device)
                q_vals = trainer.agent(one_hot).squeeze()
                print(f"Available moves: {valid_moves} Q values: {q_vals}")
                q_valid = util.batch_get(q_vals,valid_moves)
                a = valid_moves[torch.argmax(q_valid)]
                move_func = self.MOVES[a]
                s_t1,_,r_t = move_func(s_t)
                
                #Update the buffer

                print(f"Move Chosen: {move_to_bin[a]}")
                #Update the score and board state
                cont = input("Press any key to continue: ")
                self.board = self.add_new_tile(s_t1)
                self.score += r_t

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
        
    
            



