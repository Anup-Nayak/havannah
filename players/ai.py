import time
from collections import defaultdict
import copy
import numpy as np
from helper import *

dimension = 4
my_player = 1
left_time = 0

# from strategies.mcts4 import *
# from strategies.mcts6 import *

def heuristic_rollout_policy_4(possible_moves, board, player_num):
    corners = get_all_corners(board.shape[0])
    edges = get_all_edges(board.shape[0])
    
    best_move = None
    max_score = -float('inf')
    
    for move in possible_moves:
        score = 0
    
        if move in corners:
            score += 3  
        elif move in edges:
            score += 2  
    
        neighbours = get_neighbours(board.shape[0], move)
        for nx, ny in neighbours:
            if board[nx, ny] == player_num:
                score += 4

        if score > max_score:
            best_move = move
            max_score = score
    
    return best_move if best_move else possible_moves[0] 

def heuristic_rollout_policy_6(possible_moves, board, player_num):
    opponent = 3 - player_num
    corners = get_all_corners(board.shape[0])
    edges = get_all_edges(board.shape[0])
    
    best_move = None
    max_score = -float('inf')
    
    for move in possible_moves:
        score = 0
        (x, y) = move
        
        if move in corners:
            score += 0  
        elif move in edges:
            score += 2  
        else:
            score += 1
    
        neighbours = get_neighbours(board.shape[0], move)
        for nx, ny in neighbours:
            if board[nx, ny] == player_num:
                score += 4 

        if score > max_score:
            best_move = move
            max_score = score
    
    return best_move if best_move else possible_moves[0]  

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
    
    def untried_actions(self):
        self._untried_actions = self.get_legal_actions()
        return self._untried_actions
    
    def q(self):
        return self._results[1] - self._results[-1]
    
    def n(self):
        return self._number_of_visits
    
    def expand(self):
        while(len(self._untried_actions)):
            action = self._untried_actions.pop()
            next_state = self.move(action)
            next_player_number = 3-self.player_num
            child_node = MonteCarloTreeSearchNode(next_state, parent=self, player_num = next_player_number, parent_action=action)
            
            for _ in range(10):
                reward = child_node.rollout()
                child_node.backpropagate(reward)
                
        
            self.children.append(child_node)
        
        return self.best_child() 
    
    def is_terminal_node(self):
        return self.is_game_over(self.player_num,self.parent_action)
    
    def rollout(self):
        store_state = copy.deepcopy(self.state)
        
        player = self.player_num
        action = self.parent_action
        while not self.is_game_over(player,action):
            
            possible_moves = self.get_legal_actions()
            
            action = self.rollout_policy(possible_moves)
            self.rollout_move(action,player)
            player = 3-player

            
            
        outcome = self.game_result()
        self.state = store_state
        
        return outcome
    
    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def best_child(self, c_param=1.4):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        return heuristic_rollout_policy_4(possible_moves, self.state, self.player_num)
    
    def _tree_policy(self):
        current_node = self
                    
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
        return current_node

    def best_action(self):
        global left_time
        global dimension
        start_time = time.time()
        unfilled = count_unfilled_positions(self.state)
        filled = count_filled_positions(self.state)
        mid_game = 0.6*(filled+unfilled)
        c = 3
        f1 = 17
        f2 = 25
        if dimension == 4:
            f1 = 25
            f2 = 35
        h_time = f2* (mid_game*mid_game-c*(unfilled-mid_game)*(unfilled-mid_game))/(mid_game*mid_game)
        simulation_no = 100000
        
        for i in range(simulation_no):
            if time.time()-start_time >= min(min(f1,h_time),left_time/3):
                break
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        best_child_node = self.best_child(c_param=1.4)
        
        return best_child_node.parent_action
    
    def get_legal_actions(self): 
        l = get_valid_actions(self.state,self.player_num)
        return l
        
    def is_game_over(self,player,action):
        valid_actions = get_valid_actions(self.state, player)

        if len(valid_actions) == 0:
            return True
        else:
            if(action == None): return False
            win, way = check_win(self.state,action,(3-player))
            if(win): return True
            else: return False

    def game_result(self):
        global my_player
        player = self.player_num
    
        if(my_player == player):
            (our_player,_) = check_win(self.state,self.parent_action,(3-player))
            if(our_player):
                return 1
            else:
                return 0
        else:
            (opp_player,_) = check_win(self.state,self.parent_action,(3-player))
            if(opp_player):
                return -1
            else:
                return 0
  
    def rollout_move(self,action,player_id):
        (x,y) = action
        self.state[x][y] = player_id
        
    def move(self,action):
        new_state = copy.deepcopy(self.state)
        
        (x,y) = action
        new_state[x][y] = self.player_num
        
        return new_state

def make_move_4(initial_state,player):
    root = MonteCarloTreeSearchNode(state = initial_state, player_num = player)
    selected_node = root.best_action()

    return selected_node

def make_move_6(initial_state,player):
    root = MonteCarloTreeSearchNode(state = initial_state, player_num = player)
    selected_node = root.best_action()

    return selected_node

from utils.brute import *

from utils.dim4 import if_dim_is_4
from utils.dim6 import if_dim_is_6

from opening_theory.dim6_ot import strategy
from opening_theory.dim4_ot import check_for_strat_opp

class AIPlayer:

    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        

    def get_move(self, board: np.array) -> Tuple[int, int]:
        global my_player
        global dimension
        dimension = (board.shape[0] + 1) // 2    
        my_player = self.player_number  
        global left_time
        left_time = fetch_remaining_time(self.timer,my_player)
        
        if dimension == 4:
            a = if_dim_is_4(board,self.player_number)
            if a:
                (x,y) = a
                move = (int(x),int(y))
                return move 
        
        if dimension == 4:
            a,b = check_for_strat_opp(board,self.player_number)
            if a:
                (x,y) = b
                move = (int(x),int(y))
                return move
            
        if dimension == 4:
            a = make_move_4(board,self.player_number)
            (x,y) = a
            move = (int(x),int(y))
            return move
            
        if dimension == 6:
            a = if_dim_is_6(board,self.player_number)
            if a:
                (x,y) = a
                move = (int(x),int(y))
                return move 
        
        if dimension == 6:
            if(count_filled_positions(board) <= 40):
                (a,b) = strategy(board,self.player_number)
                if a:
                    (x,y) = b
                    move = (int(x),int(y))
                    return move
                
            a = make_move_6(board,self.player_number)
            (x,y) = a
            move = (int(x),int(y))
            return move
                    
            
           
