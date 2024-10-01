from helper import *

def check_for_win(board, player_num):
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = player_num
        win, _ = check_win(temp_board, move, player_num)  
        if win:
            return True, move  
    return False, None

def check_for_loss(board, player_num):
    opponent = 3 - player_num  
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = opponent
        win, _ = check_win(temp_board, move, opponent)  
        if win:
            return True, move  
    return False, None  


