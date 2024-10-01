import numpy as np
from helper import get_valid_actions, get_neighbours, fetch_remaining_time

from utils.union_find import UnionFind, is_corner, is_edge

def evaluate_board(board, player_num, dim):
    uf = UnionFind(dim)
    opponent_num = 1 if player_num == 2 else 2
    
    # Initialize heuristic score
    score = 0

    # Iterate through the board and add stones to Union-Find
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == player_num:
                uf.add_stone((row, col), is_corner((row, col), dim), is_edge((row, col), dim))
                # Check neighboring stones to unionize groups
                neighbors = get_neighbours(dim, (row, col))
                for n in neighbors:
                    if board[n[0]][n[1]] == player_num:
                        uf.union((row, col), n)
            elif board[row][col] == opponent_num:
                uf.add_stone((row, col), is_corner((row, col), dim), is_edge((row, col), dim))
                neighbors = get_neighbours(dim, (row, col))
                for n in neighbors:
                    if board[n[0]][n[1]] == opponent_num:
                        uf.union((row, col), n)
    
    # Check for win conditions for both players
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == player_num:
                result = uf.check_win_condition((row, col))
                if result == "Bridge":
                    score += 1000  # Large reward for bridge
                elif result == "Fork":
                    score += 1000  # Large reward for fork
            elif board[row][col] == opponent_num:
                result = uf.check_win_condition((row, col))
                if result == "Bridge":
                    score -= 1000  # Large penalty if opponent has a bridge
                elif result == "Fork":
                    score -= 1000  # Large penalty if opponent has a fork

    # Add additional heuristic for partial patterns
    for root in uf.parent:
        if board[root[0]][root[1]] == player_num:
            edge_count = bin(uf.edge_bits[root]).count('1')
            corner_count = bin(uf.corner_bits[root]).count('1')
            score += edge_count * 100  # Reward based on proximity to Fork
            score += corner_count * 100  # Reward based on proximity to Bridge
        elif board[root[0]][root[1]] == opponent_num:
            edge_count = bin(uf.edge_bits[root]).count('1')
            corner_count = bin(uf.corner_bits[root]).count('1')
            score -= edge_count * 100  # Penalize based on opponent proximity to Fork
            score -= corner_count * 100  # Penalize based on opponent proximity to Bridge

    return score


# Minimax function with depth limiting
def minimax(board, depth, is_maximizing_player, max_depth, dim, player_num):
    if depth == max_depth or np.all(board != 0):  # Terminal condition
        return evaluate_board(board, player_num, dim)

    valid_moves = get_valid_actions(board)
    
    if is_maximizing_player:
        max_eval = -float('inf')
        for move in valid_moves:
            new_board = board.copy()
            new_board[move] = player_num  # Maximizing player
            eval = minimax(new_board, depth + 1, False, max_depth, dim, player_num)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_board = board.copy()
            new_board[move] = 3 - player_num  # Minimizing player (opponent)
            eval = minimax(new_board, depth + 1, True, max_depth, dim, player_num)
            min_eval = min(min_eval, eval)
        return min_eval


# Function to get the best move using Minimax
def get_best_move(board, max_depth=4,dim=4,player_num=1):
    best_move = None
    best_value = -float('inf')
    
    valid_moves = get_valid_actions(board)
    
    for move in valid_moves:
        new_board = board.copy()
        new_board[move] = 1  # Assuming Player 1 is the AI using Minimax
        move_value = minimax(new_board, 0, False, max_depth,dim,player_num)
        
        if move_value > best_value:
            best_value = move_value
            best_move = move
            
    return best_move