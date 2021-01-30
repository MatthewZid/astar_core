import numpy as np

R = -0.04 # reward
gamma = 0.2 # discount factor
e = 0.0001 # convergence factor

def process_policy(gprev, i, j, action):
    nexti = 0
    nextj = 0
    error_posi = 0
    error_posj = 0

    if action == 1:
        nexti = i
        nextj = j+1
        error_posi = 1
    elif action == -1:
        nexti = i
        nextj = j-1
        error_posi = 1
    elif action == 2:
        nexti = i-1
        nextj = j
        error_posj = 1
    elif action == -2:
        nexti = i+1
        nextj = j
        error_posj = 1

    # check obstacle
    uval_a, uval_b, uval_intended = 0.0, 0.0, 0.0
    if (i+error_posi) > 2 or (j+error_posj) > 3 or ((i+error_posi) == 1 and (j+error_posj) == 1):
        uval_a = gprev[i,j][0]
    else:
        uval_a = gprev[i+error_posi,j+error_posj][0]
    
    if (i-error_posi) < 0 or (j-error_posj) < 0 or ((i-error_posi) == 1 and (j-error_posj) == 1):
        uval_b = gprev[i,j][0]
    else:
        uval_b = gprev[i-error_posi,j-error_posj][0]
    
    if nexti < 0 or nexti > 2 or nextj < 0 or nextj > 3 or (nexti == 1 and nextj == 1):
        uval_intended = gprev[i,j][0]
    else:
        uval_intended = gprev[nexti,nextj][0]
    
    # calculate expected utility
    expected_util = 0.8 * uval_intended + 0.1 * uval_a + 0.1 * uval_b

    return expected_util

def calculate_policy(gprev, grd, i, j, actions, default_policy):
    best_expected_util = np.NINF

    if default_policy:
        best_expected_util = process_policy(gprev, i, j, gprev[i,j][1])
    else:
        for action in actions:
            expected_util = process_policy(gprev, i, j, action)
            if expected_util > best_expected_util:
                best_expected_util = expected_util
                grd[i,j][1] = action
    
    return best_expected_util

def update_util(grd, g, R, e, default_policy=False):    # False: the agent finds the best policy by itself
                                                        # True: the agent uses the provided policy specified in grid creation below
    actions = [-1, 1, 2, -2]
    
    while True:
        d = 0
        gprev = grd.copy()
        for i in range(3):
            for j in range(4):
                if gprev[i,j][1] == 8 or gprev[i,j][1] == 0: continue
                
                expected_util = calculate_policy(gprev,grd,i,j,actions,default_policy)
               
                # find utility value
                grd[i,j][0] = R + gamma * expected_util
                
                if abs(grd[i,j][0] - gprev[i,j][0]) > d:
                    d = abs(gprev[i,j][0] - grd[i,j][0])
                
        if d < e*(1 - gamma)/gamma: break
    
class Node:
    def __init__(self, i, j, g, f, parent = None, children = []):
        self.i = i
        self.j = j
        self.g = g
        self.f = f
        self.parent = parent
        self.children = children

def backtrack(node):
    if node.parent != None: backtrack(node.parent)
    print('Step through {:d},{:d}'.format(node.i, node.j))

"""Set the grid: each cell is of type [utility,policy:up|down|left|right]"""
# left = -1, right = 1, up = 2, down = -2, obstacle = 8, target = 0
# the policy below is the default given policy
grid = np.array([
    [[0.0,1],[0.0,1],[0.0,1],[1.0,0]],
    [[0.0,2],[0.0,8],[0.0,2],[-1.0,0]],
    [[0.0,2],[0.0,-1],[0.0,-1],[0.0,-1]]
])

print("Calculating utilities for gamma = {:.1f}...".format(gamma))

update_util(grid, gamma, R, e, False)

util_view = np.zeros((3,4))
policy = np.empty((3,4), dtype="S2")
for i in range(3):
    for j in range(4):
        util_view[i,j] = grid[i,j][0]
        if grid[i,j][1] == -1: policy[i,j] = '<-'
        elif grid[i,j][1] == 1: policy[i,j] = '->'
        elif grid[i,j][1] == -2: policy[i,j] = 'v'
        elif grid[i,j][1] == 2: policy[i,j] = '^'
        else: policy[i,j] = str(grid[i,j][1])

policy = policy.astype(str)

print("\nFinished!\n")
print("Calculated expected utilities for each state")
print("---------------------------------------------\n")
print(util_view)
print("\nPolicy for each state")
print("------------------------\n")
print(policy)

# Implement A*
visited = []
opened = []
start = Node(2, 0, 0, 0)
goal = None

opened.append(start)

while len(opened) != 0:
    maxf = opened[0].f
    max_pos = 0
    for pos in range(len(opened)):
        if opened[pos].f > maxf:
            maxf = opened[pos].f
            max_pos = pos
    
    current = opened.pop(max_pos)
    # print('Current node: {:d},{:d} with f={:f}'.format(current.i,current.j,current.f))

    if grid[current.i,current.j][1] == 0:
        backtrack(current)
        print('Goal node found!\n')
        print("Position: {:d},{:d}".format(current.i,current.j))
        break

    # find current node's neighbours from grid
    left = current.j - 1
    right = current.j + 1
    up = current.i - 1
    down = current.i + 1

    children = []

    if left >= 0 and grid[current.i,left][1] != 8:
        children.append((current.i,left))
    
    if right < 4 and grid[current.i,right][1] != 8:
        children.append((current.i,right))
    
    if up >= 0 and grid[up,current.j][1] != 8:
        children.append((up,current.j))
    
    if down < 3 and grid[down,current.j][1] != 8:
        children.append((down,current.j))
    
    current.children.clear()

    # check children validity
    for ch in children:
        isvisited = False
        for nd in visited:
            if ch == (nd.i,nd.j):
                isvisited = True
                break
        if isvisited: continue

        cost = 0.0
        h = 0.0
        if grid[ch[0],ch[1]][1] == 0:
            cost = grid[ch[0],ch[1]][0]
            h = 0
        else:
            cost = R
            h = grid[ch[0],ch[1]][0]
        g = current.g + cost
        f = g + h

        isopened = False
        for nd in opened:
            if ch == (nd.i,nd.j) and g < nd.g:
                isopened = True
                break
        if isopened: continue

        # create children nodes
        new_node = Node(ch[0], ch[1], g, f)
        new_node.parent = current
        current.children.append(new_node)
        opened.append(new_node)
    
    # for ch in current.children:
    #     print('Child {:d},{:d} with cost {:f}'.format(ch.i,ch.j,ch.f))
    
    visited.append(current)