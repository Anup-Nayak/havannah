import time
import math
import random
import numpy as np
from helper import *
from strategies.mcts4 import *

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
        
        return make_move(board,self.player_number)



''' opening theory for dim 4'''

d1 = {
	(0,0) :[(1,2),(2,1)],
	(3,0):[(2,1),(4,2)],
	(6,3):[(4,2),(4,4)],
	(3,6):[(4,4),(2,5)],
	(0,6):[(2,5),(1,4)],
	(0,3):[(1,4),(1,2)]
}

def inbetween_marked(board,cell,corner,player_num):
	nbr_list = get_neighbours(dim=7,vertex=corner)
	nbr_list2 = get_neighbours(dim=7,vertex=cell)
	for nbr in nbr_list:
		cnt = 0
		if((nbr in nbr_list2)):
			if((board[nbr]==player_num)):
				cnt+=1
	if(board[corner]==player_num):
		cnt+=1
	if(cnt==0):
		return False,corner
	else:
		return True,None
		
def check_for_strat_opp(board,player_num):
	opp = (3-player_num)
	corners = get_all_corners(dim=7)
	
	prev_corner = corners[-1]
	for corner in corners:
		if((board[prev_corner]==opp) and(board[corner]==opp)):
			for nbr in d1[prev_corner]:
				if((nbr in d1[corner])):
					if(board[nbr]==player_num):
						print("already saved")
						continue
					elif(board[nbr]==0):
						print('saviour is here')
						a1,b1 = inbetween_marked(board,nbr,corner,player_num)
						a2,b2 = inbetween_marked(board,nbr,prev_corner,player_num)
						if(not a1 and not a2):
							return True,nbr
		prev_corner = corner
	prev_corner = corners[-1]

	for corner in corners:
		for cell in d1[prev_corner]:
			if((cell in d1[corner])):
				if(board[prev_corner]==opp and board[cell]==opp):
					a,b = inbetween_marked(board,cell,corner,player_num)
					if(not a): return True,b
				elif(board[corner]==opp and board[cell]==opp):
					a,b = inbetween_marked(board,cell,prev_corner,player_num)
					if(not a): return True,b
		prev_corner =corner
	return False,None

