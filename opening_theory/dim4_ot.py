from helper import *

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
                        continue
                    elif(board[nbr]==0):
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
