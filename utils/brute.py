from helper import *

def count_filled_positions(board: np.array) -> int:
    return np.sum((board == 1) | (board == 2))

def count_unfilled_positions(board: np.array) -> int:
    return np.sum(board == 0)


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

def check_forced_mate_in_2(board, player_num):
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = player_num
        opponent = 3 - player_num
        opponent_moves = get_valid_actions(temp_board)
        flag = True
        for opp_move in opponent_moves:
            temp_board_opp = temp_board.copy()
            temp_board_opp[opp_move] = opponent
            a,b = check_for_win(temp_board_opp,player_num)
            if(not a):
                flag = False
        if(flag):
            return True,move
    return False,None

def check_mate_in_2(board, player_num):
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = player_num

        opponent = 3 - player_num
        opponent_moves = get_valid_actions(temp_board)
        for opp_move in opponent_moves:
            temp_board_opp = temp_board.copy()
            temp_board_opp[opp_move] = opponent

            player_second_moves = get_valid_actions(temp_board_opp)
            for second_move in player_second_moves:
                temp_board_player = temp_board_opp.copy()
                temp_board_player[second_move] = player_num
                win, _ = check_win(temp_board_player, second_move, player_num)
                if win:
                    return True, move
    return False, None

def check_loss_in_2(board, player_num):
    opponent = 3 - player_num
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = opponent

        player_moves = get_valid_actions(temp_board)
        for player_move in player_moves:
            temp_board_player = temp_board.copy()
            temp_board_player[player_move] = player_num

            opponent_second_moves = get_valid_actions(temp_board_player)
            for opp_second_move in opponent_second_moves:
                temp_board_opp_second = temp_board_player.copy()
                temp_board_opp_second[opp_second_move] = opponent
                win, _ = check_win(temp_board_opp_second, opp_second_move, opponent)
                if win:
                    return True, move
    return False, None

def check_mate_in_3(board, player_num):
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = player_num

        opponent = 3 - player_num
        opponent_moves = get_valid_actions(temp_board)
        for opp_move in opponent_moves:
            temp_board_opp = temp_board.copy()
            temp_board_opp[opp_move] = opponent

            player_second_moves = get_valid_actions(temp_board_opp)
            for second_move in player_second_moves:
                temp_board_player = temp_board_opp.copy()
                temp_board_player[second_move] = player_num

                opponent_second_moves = get_valid_actions(temp_board_player)
                for opp_second_move in opponent_second_moves:
                    temp_board_opp_second = temp_board_player.copy()
                    temp_board_opp_second[opp_second_move] = opponent

                    player_third_moves = get_valid_actions(temp_board_opp_second)
                    for third_move in player_third_moves:
                        temp_board_player_third = temp_board_opp_second.copy()
                        temp_board_player_third[third_move] = player_num
                        win, _ = check_win(temp_board_player_third, third_move, player_num)
                        if win:
                            return True, move
    return False, None

def check_loss_in_3(board, player_num):
    opponent = 3 - player_num
    valid_moves = get_valid_actions(board)
    for move in valid_moves:
        temp_board = board.copy()
        temp_board[move] = opponent

        player_moves = get_valid_actions(temp_board)
        for player_move in player_moves:
            temp_board_player = temp_board.copy()
            temp_board_player[player_move] = player_num

            opponent_second_moves = get_valid_actions(temp_board_player)
            for opp_second_move in opponent_second_moves:
                temp_board_opp_second = temp_board_player.copy()
                temp_board_opp_second[opp_second_move] = opponent

                player_third_moves = get_valid_actions(temp_board_opp_second)
                for player_third_move in player_third_moves:
                    temp_board_player_third = temp_board_opp_second.copy()
                    temp_board_player_third[player_third_move] = player_num

                    opponent_third_moves = get_valid_actions(temp_board_player_third)
                    for opp_third_move in opponent_third_moves:
                        temp_board_opp_third = temp_board_player_third.copy()
                        temp_board_opp_third[opp_third_move] = opponent
                        win, _ = check_win(temp_board_opp_third, opp_third_move, opponent)
                        if win:
                            return True, move
    return False, None

def first_move_strategy(board: np.array, player_num: int) -> Tuple[int, int]:
    dim = board.shape[0]
    corners = get_all_corners(dim)

    if player_num == 1 and np.sum((board == 1) | (board == 2)) == 0:
        return corners[0]
    
    if player_num == 2:
        opponent_moves = np.argwhere(board == 1)
        if len(opponent_moves) > 0:
            opponent_move = tuple(opponent_moves[0])

            if opponent_move in corners:
                corner_edges = {
                    corners[0]: [corners[1], corners[5]],  
                    corners[1]: [corners[0], corners[2]],  
                    corners[2]: [corners[1], corners[3]],
                    corners[3]: [corners[2], corners[4]],  
                    corners[4]: [corners[3], corners[5]],  
                    corners[5]: [corners[0], corners[4]]
                }
                
                
                return corner_edges[opponent_move][0] 
            else:
                return corners[0]  


    valid_moves = get_valid_actions(board)
    return valid_moves[0] if valid_moves else None
