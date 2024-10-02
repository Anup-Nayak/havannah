import numpy as np
from collections import defaultdict
import copy
import time

from helper import *
from debug import *

start_player = 1
dim = 4

def change(player):
    if(player == 1):
        return 2
    else:
        return 1


def heuristic_rollout_policy(possible_moves, board, player_num):
    opponent = 3 - player_num
    corners = get_all_corners(board.shape[0])
    edges = get_all_edges(board.shape[0])
    
    # Prioritize moves based on proximity to edges, corners, and blocking opponent
    best_move = None
    max_score = -float('inf')
    
    for move in possible_moves:
        score = 0
        (x, y) = move
        
        # 1. Prioritize moves near corners and edges
        if move in corners:
            score += 0  # Higher weight for corners
        elif move in edges:
            score += 2  # Higher weight for edge moves
        else:
            score += 1
    
        
        # 3. Prefer moves that connect to existing pieces
        neighbours = get_neighbours(board.shape[0], move)
        for nx, ny in neighbours:
            if board[nx, ny] == player_num:
                score += 4  # Increment score for connecting to own pieces

        # Choose the move with the highest score
        if score > max_score:
            best_move = move
            max_score = score
    
    return best_move if best_move else possible_moves[0]  # Fallback to random if no best move


class MonteCarloTreeSearchNode():
    def __init__(self, state, player_num, parent=None, parent_action=None):
        self.state = state
        self.player_num = player_num
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[0] = 0
        self._results[-1] = 0
        self._untried_actions = self.untried_actions()
        return
    
    
    # Returns the list of untried actions from a given state.
    def untried_actions(self):
        self._untried_actions = self.get_legal_actions()
        return self._untried_actions
    
    
    # Returns the difference of wins - losses
    def q(self):
        return self._results[1] - self._results[-1]
    
    # Returns the number of times each node is visited.
    def n(self):
        return self._number_of_visits
    
    
    # From the present state, next state is generated depending on the action which is carried out. 
    # In this step all the possible child nodes corresponding to generated states are appended to the 
    # children array and the child_node is returned. The states which are possible from the present state 
    # are all generated and the child_node corresponding to this generated state is returned.
    def expand(self):
        
        while(len(self._untried_actions)):
            action = self._untried_actions.pop()
            next_state = self.move(action)
            next_player_number = change(self.player_num)
            child_node = MonteCarloTreeSearchNode(next_state, parent=self, player_num = next_player_number, parent_action=action)
            
            for _ in range(10):
                reward = child_node.rollout()
                child_node.backpropagate(reward)
                
        
            self.children.append(child_node)
        
        return self.best_child() 
    
    
    # This is used to check if the current node is terminal or not. 
    def is_terminal_node(self):
        return self.is_game_over(self.player_num,self.parent_action)
    
    
    # From the current state, entire game is simulated till there is an outcome for the game. 
    # This outcome of the game is returned. For example if it results in a win, the outcome is 1. 
    # Otherwise it is -1 if it results in a loss. And it is 0 if it is a tie. If the entire game is randomly simulated,
    # that is at each turn the move is randomly selected out of set of possible moves, it is called light playout.
    def rollout(self):
        store_state = copy.deepcopy(self.state)
        
        player = self.player_num
        action = self.parent_action
        while not self.is_game_over(player,action):
            
            possible_moves = self.get_legal_actions()
            
            action = self.rollout_policy(possible_moves)
            self.rollout_move(action,player)
            player = change(player)

            
            
        outcome = self.game_result()
        self.state = store_state
        
        return outcome
    
    
    # n this step all the statistics for the nodes are updated. Untill the parent node is reached, 
    # the number of visits for each node is incremented by 1. If the result is 1, 
    # that is it resulted in a win, then the win is incremented by 1. 
    # Otherwise if result is a loss, then loss is incremented by 1.
    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)
    
    # All the actions are poped out of _untried_actions one by one. When it becomes empty, 
    # that is when the size is zero, it is fully expanded.
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    
    # Once fully expanded, this function selects the best child out of the children array. 
    # The first term in the formula corresponds to exploitation and the second term corresponds to exploration.
    def best_child(self, c_param=1.4):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    # change this from random to good moves
    def rollout_policy(self, possible_moves):
        return heuristic_rollout_policy(possible_moves, self.state, self.player_num)
    
    
    def _tree_policy(self):

        current_node = self
        # while not current_node.is_terminal_node():
            
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
        return current_node

    
    
    # This is the best action function which returns the node corresponding to best possible move.
    # The step of expansion, simulation and backpropagation are carried out by the code above.
    def best_action(self):
        c = 5
        start_time = time.time()
        
        simulation_no = 10000
        # print(self.state)
        for i in range(simulation_no):
            if time.time()-start_time >= 15:
                break
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        best_child_node = self.best_child(c_param=1.4)
        
        return best_child_node.parent_action
    
    # mp correct
    def get_legal_actions(self): 
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        
        l = get_valid_actions(self.state,self.player_num)
        return l
     
    # mp correct   
    def is_game_over(self,player,action):
        '''
        Modify according to your game or 
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        
        valid_actions = get_valid_actions(self.state, player)

        if len(valid_actions) == 0:
            return True
        else:
            if(action == None): return False
            win, way = check_win(self.state,action,change(player))
            if(win): return True
            else: return False

    # incorrect
    def game_result(self):
        '''
        Modify according to your game or 
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        player = self.player_num
    
        if(start_player == player):
            (our_player,_) = check_win(self.state,self.parent_action,change(player))
            if(our_player):
                return 1
            else:
                return 0
        else:
            (opp_player,_) = check_win(self.state,self.parent_action,change(player))
            if(opp_player):
                return -1
            else:
                return 0

    # correct
    
    def rollout_move(self,action,player_id):
        (x,y) = action
        self.state[x][y] = player_id
        
        
    def move(self,action):
        '''
        Modify according to your game or 
        needs. Changes the state of your 
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board 
        position is empty. If you place x in
        row 2 column 3, then it would be some 
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns 
        the new state after making a move.
        '''
        
        new_state = copy.deepcopy(self.state)
        
        (x,y) = action
        new_state[x][y] = self.player_num
        
        return new_state

def make_move_6(initial_state,player):
    start_player = player
    dim = initial_state.shape[0]
    root = MonteCarloTreeSearchNode(state = initial_state, player_num = player)
    selected_node = root.best_action()

    return selected_node

