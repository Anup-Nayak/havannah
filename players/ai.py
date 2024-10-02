import time
import math
import random
import numpy as np
from helper import *
from strategies.mcts4 import *
from strategies.mcts6 import *

from utils.brute import *
from utils.dim4 import if_dim_is_4
from utils.dim6 import if_dim_is_6

from debug import *

def count_filled_positions(board: np.array) -> int:
    return np.sum((board == 1) | (board == 2))

def count_unfilled_positions(board: np.array) -> int:
    return np.sum(board == 0) 

class AIPlayer:

    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.dim = 4
        self.dimension = 4

    def get_move(self, board: np.array) -> Tuple[int, int]:
        
        self.dim = board.shape[0]
        self.dimension = (board.shape[0] + 1) // 2       
        
        if(self.dimension == 4):
            if if_dim_is_4(board,self.player_number):
                return if_dim_is_4(board,self.player_number) 
            
            
        if(self.dimension == 6):
            if if_dim_is_6(board,self.player_number):
                return if_dim_is_6(board,self.player_number)
        
        
        if self.dimension == 4:
            return make_move_4(board,self.player_number)
        
        if self.dimension == 6:
            return make_move_6(board,self.player_number)

