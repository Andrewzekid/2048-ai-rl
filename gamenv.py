import numpy as np
import torch
from ai.trainer import Trainer
from ai.agent import CNN
import random
from typing import List,Tuple
import pdb
import ai.util as util
import torch
from ai.dqn import DQN
#File for the game code
CELL_COUNT = 4 #4x4
ROW_MASK = np.uint64(0xFFFF)
COL_MASK = np.uint64(0x000F000F000F000F)
DISTRIBUTION = np.array([2,2,2,2,2,2,2,2,2,4])
class GameBoard:
    def __init__(self,cell_count=CELL_COUNT,distribution=DISTRIBUTION):
        self.CELL_COUNT = cell_count
        self.DISTRIBUTION = distribution
        self.MOVES = {0:self.move_right,1:self.move_left,2:self.move_up,3:self.move_down}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reset()
    
    def reset(self):
        """Resets the game board and game parameters after the game is over"""
        self.board = self.initialize_game()
        self.score = 0.0
        self.game_over = False

    def initialize_game(self) -> torch.Tensor:
        """Initializes the gameboard with 2 random tiles and returns it.
        Returns:
        board torch.Tensor (4,4)
         """
        NUMBER_OF_SQUARES = self.CELL_COUNT * self.CELL_COUNT
        board = torch.zeros((NUMBER_OF_SQUARES,), dtype=torch.float32,device=self.device)
        initial_twos = np.random.default_rng().choice(NUMBER_OF_SQUARES, 2, replace=False)
        board[initial_twos] = 2
        board = board.reshape((self.CELL_COUNT, self.CELL_COUNT))
        return board

    def push_right(self,board) -> Tuple[torch.Tensor,bool]:
        """Push all squares to the right
        Returns:
        new game board after all squares are pushed to the right and a boolean indicating whether the gameboard has changed or not
        """
        new = torch.zeros((self.CELL_COUNT,self.CELL_COUNT),dtype=torch.float32,device=self.device)
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

    def merge_elements(self,board: torch.Tensor) -> Tuple[torch.Tensor,int,bool]:
        changed = False
        score = 0.0
        for row in range(self.CELL_COUNT):
            cntr = self.CELL_COUNT - 1
            for col in range(self.CELL_COUNT - 1,0,-1):
                if board[row,col] != 0 and board[row,col] == board[row,col - 1]:
                    board[row,col] *= 2
                    score += board[row,col]
                    changed = True
                    board[row,col - 1] = 0
        return board,score,changed
    

    def move(self,board,k) -> torch.Tensor:
        """Implements moving up/right/left/down
        :param board game board
        :param k (int) How many times to rotate 90 Left: 2, right: 0 UP: -1 Down: 1
        """
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

    def has_move(self,board: torch.Tensor) -> bool:
        """Determines whether a given gameboard has a valid move
        Returns:
        boolean value indicating whether or not there is a valid move
        """
        return len(self.get_valid_moves(board)) != 0

    def has_valid_move(self) -> bool:
        """Determines whether the current gameboard has a valid move
        Returns: boolean value indicating if no valid moves (False) or has valid moves (True)
        """
        return len(self.get_valid_moves(self.board)) != 0

    def get_valid_moves(self,board) -> List[int]:
        """Get valid move
        Returns: List[int]: List of integers, each corresponding to a move made
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
        dqn = DQN()
        dqn.init_nets()
        dqn.load(weights_file)
        trainer = Trainer()
        trainer.net = dqn

        while self.has_valid_move():
            with torch.inference_mode():
                trainer.net.test_mode()
                print("Game board: ")
                print(f"Score: {self.score}")
                self.display_board()
                valid_moves = self.get_valid_moves(self.board)
                s_t = self.board
                move_to_bin = {2:"Up",1:"Left",3:"Down",0:"Right"}


                one_hot = torch.unsqueeze(trainer.net.one_hot(s_t),0).to(device)
                q_vals = trainer.net.agent(one_hot).squeeze()
                print(f"Available moves: {valid_moves} Q values: {q_vals}")
                valid_moves_tensor = torch.tensor(valid_moves,device=self.device,dtype=torch.long)
                q_valid = util.batch_get(q_vals,valid_moves_tensor)
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
    
    def from_binary(self,binaries):
        """Converts binary representation into a board
        Args:
        :param binaries (batch_size,1) np.array, array of binaries
        """
        binaries = binaries.astype(np.uint64) #Convert to a consistent dtype
        positions = np.arange(0,self.CELL_COUNT*self.CELL_COUNT,4,dtype=np.uint64)
        #first reshape to batch_size,16
        ncells = self.CELL_COUNT * self.CELL_COUNT
        exponents = np.zeros_like((batch_size,ncells),dtype=np.uint64)
        for i,shift in enumerate(positions):
            exponents[:,i] = ((binaries[:] >> shift) & 0xF)
        board = np.where(exponents == 0,0,1 << exponents)
        board = board.reshape(batch_size,self.CELL_COUNT,self.CELL_COUNT) #will cause error if board is not CELL_COUNT X CELL_COUNT
        return board

    def to_binary(self,board) -> str:
        """Changes the board representation to a binary string for storage
        Args:
        :param board Batch of board tensors (batch_size,4,4)
        Returns tensor of binaries (batch_size,16)
        """
        batch_size = board.shape[0] #batch of boards
        board = board.numpy().astype(np.uint64) #Ensure that the data type of the board is consistent
        positions = np.array([
        [60, 56, 52, 48],  # Row 0
        [44, 40, 36, 32],  # Row 1
        [28, 24, 20, 16],  # Row 2
        [12,  8,  4,  0],  # Row 3
    ], dtype=np.uint64)
        positions_flat = positions.flatten()
        board_flat = board.flatten(start_dim=1)
        exponents = np.zeros_like(board_flat,dtype=np.uint64)
        nonzero_mask = board_flat > 0
        exponents[nonzero_mask] = np.log2(board_flat[nonzero_mask])
        binaries = np.zeros(batch_size,dtype=np.uint64)
        for i in range(16):
            shift_amounts = positions_flat[i]
            binaries |= (exponents[:,i] << shift_amounts) #for all batches (the shape is batch_size,16), shift the binary left by shift_amounts bits
        return binaries

    def save_binaries(self,s:torch.Tensor,a:torch.Tensor,r:int,s1):
        raise NotImplementedError

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
class BitBoard:
    NIBBLE_MASK = np.uint64(0xF)
    def merge_right(self):
        raise NotImplementedError
    
    def get_cell(self,row,col):
        pos = row *4 + col
        value = (self.board >> (pos * 4)) & self.NIBBLE_MASK
        return value
    
    def get_item(x,i:int):
        """Obtains an item from a specific index in the bitboard O(1) time
        Args:
        x: bitboard
        i: index
        """
        mask = 0xF << 4*((4-i-1))
        x = x & mask
        x = x >> 4*(4-i-1)
        if x== 0:
            return 0
        return 2**x
    
    def reverse_row(row):
        """Reverse the bits in a row
        Args:
        row: bits representing the current row
        """
        a_1 = (row & np.uint64(0xF000)) >> 12
        a_2 = (row & np.uint64(0x0F00)) >> 4
        a_3 = (row & np.uint64(0x00F0)) << 4
        a_4 = (row & np.uint64(0x000F)) << 12
        return a_1 | a_2 | a_3 | a_4
    
    def transpose_board(x):
        """Transpose the bits in a board
        Args:
        x: bitboard
        """
        a_1 = x & np.uint64(0xF0F00F0FF0F00F0F)
        a_2 = x & np.uint64(0x0000F0F00000F0F0)
        a_3 = x & np.uint64(0x0F0F00000F0F0000)
        a = a_1 | (a_2 << np.uint64(12)) | (a_3 >> np.uint64(12))
        b1 = a & np.uint64(0xFF00FF0000FF00FF)
        b2 = a & np.uint64(0x00FF00FF00000000)
        b3 = a & np.uint64(0x00000000FF00FF00)
        return b1 | (b2 >> np.uint64(24)) | (b3 << np.uint64(24))
    
    def merge_grid_row_right(row):
        """Merges a grid to the right
        Args:
        row: Row containing the binary numbers
        """
        pos = 3
        num = 0
        for j in range(3,-1,-1):
            if row[j] == 0:
                continue
            if num == 0:
                num = row[j]
            elif num == row[j]:
                #Merge operation
                row[pos] = num * 2
                pos -= 1
                num = 0
            else:
                #No merge operation, just move operation
                if row[pos] != num:
                    row[pos] = num
                pos -= 1
                num = row[j]
        if num != 0:
            if row[pos] != num:
                row[pos] = num
            pos -= 1
        for j in range(pos+1):
            if row[j] != 0:
                row[j] = 0
        return row
    def count_empty(x):
        """Uses bitwise operations to count the number of 1s in O(1) time
        Args:
        x: bitboard representation
        """
        m1 = np.uint64(0x3333333333333333)
        m2 = np.uint64(0x5555555555555555)
        m4 = np.uint64(0x0f0f0f0f0f0f0f0f)
        m8 = np.uint64(0x00ff00ff00ff00ff)
        m16 = np.uint64(0x0000ffff0000ffff)
        m32 = np.uint64(0x00000000ffffffff)
        x = (x & m1) + ((x >> np.uint64(1)) & m1)
        x = (x & m2) + ((x >> np.uint64(2)) & m2)
        x = (x & m4) + ((x >> np.uint64(4)) & m4)
        x = (x & m8) + ((x >> np.uint64(8)) & m8)
        x = (x & m16) + ((x >> np.uint64(16)) & m16)
        x = (x & m32) + ((x >> np.uint64(32)) & m32)
        return x

    

    
