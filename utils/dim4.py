import numpy as np
from helper import *

from utils.brute import * 

def count_filled_positions(board: np.array) -> int:
    return np.sum((board == 1) | (board == 2))

def count_unfilled_positions(board: np.array) -> int:
    return np.sum(board == 0)


def if_dim_is_4(board,player_number):
    if  count_filled_positions(board) <=1:
        return first_move_strategy(board,player_number)
    
    win,move1 = check_for_win(board,player_number)
    if win:
        return move1
    
    lose,move2 = check_for_loss(board,player_number)
    if lose:
        return move2
    
    win_in_2, move3 = check_mate_in_2(board, player_number)
    if win_in_2:
        return move3

    lose_in_2, move4 = check_loss_in_2(board, player_number)
    if lose_in_2:
        return move4
    
    if count_filled_positions(board) >= 10:
        win_in_3, move5 = check_mate_in_3(board, player_number)
        if win_in_3:
            return move5

        # check if opponent can win in 3 moves
        lose_in_3, move6 = check_loss_in_3(board, player_number)
        if lose_in_3:
            return move6