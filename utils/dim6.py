import numpy as np
from helper import *

from utils.brute import * 

def if_dim_is_6(board,player_number):
    
    if count_filled_positions(board) >= 10:
        win,move1 = check_for_win(board,player_number)
        if win:
            return move1
        
        lose,move2 = check_for_loss(board,player_number)
        if lose:
            return move2
    
    if count_unfilled_positions(board) <= 60:
        win_in_2, move3 = check_mate_in_2(board, player_number)
        if win_in_2:
            return move3

        lose_in_2, move4 = check_loss_in_2(board, player_number) 
        
        if lose_in_2:
            return move4
    
    if count_unfilled_positions(board) <= 25:
        win_in_3, move5 = check_mate_in_3(board, player_number)
        if win_in_3:
            return move5

        lose_in_3, move6 = check_loss_in_3(board, player_number)
        if lose_in_3:
            return move6
