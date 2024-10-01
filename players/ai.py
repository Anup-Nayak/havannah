import time
import math
import random
import numpy as np
from helper import *
from strategies.mcts import *

from debug import *

class AIPlayer:

    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.dim = 4
        self.dimension = 4

    def get_move(self, board: np.array) -> Tuple[int, int]:
        
        # dimension used for helper.py functions
        self.dim = board.shape[0]
        
        # actual dimension
        self.dimension = (board.shape[0] + 1) // 2
        # debug(self.player_number)
        return make_move(board,self.player_number)
        raise NotImplementedError('Whoops I don\'t know what to do')

