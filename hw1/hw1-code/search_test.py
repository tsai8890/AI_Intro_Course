# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

""" The required data structures: priority queue """
import heapq
from re import L

from numpy import dot
class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        # If the two items have the same priority, pick up the latest one
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

""" The auxilary functions for computing distances"""
def manhatten_distance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

def drow(pos1, pos2):
    return abs(pos1[0] - pos2[0])

def dcol(pos1, pos2):
    return abs(pos1[1] - pos2[1])

""" Count the distance from all the dot points to any other point"""
def countAllDistance(maze):
    start_pos = maze.getStart()
    dotStatus = maze.getObjectives()
    start_candidate = [dot for dot in dotStatus] + [start_pos]
    dist_table = dict()

    for from_where in start_candidate:
        visited = dict()
        pri_queue = PriorityQueue()
        pri_queue.push([from_where, 0], 0)

        while not pri_queue.isEmpty():
            curPos, d = pri_queue.pop()
            
            if curPos not in visited:
                dist_table[(from_where, curPos)] = d
                dist_table[(curPos, from_where)] = d
                visited[curPos] = 1
                
                neighbors = maze.getNeighbors(*curPos)
                for nextPos in neighbors:
                    if nextPos not in visited:
                        pri_queue.push([nextPos, d+1], d+1)
    return dist_table

""" The heuristics functions """
def null_heuristics(curState, maze, dist_table=None):
    return 0

def max_manhatten_heuristics(curState, maze, dist_table=None):
    curPos, dotStatus = curState
    max_dist = 0
    for dot in dotStatus:
        dist = manhatten_distance(curPos, dot)
        max_dist = max(max_dist, dist)
    return max_dist

def max_distance_heuristics(curState, maze, dist_table=None):
    assert dist_table != None, 'dist_table為None, 請檢查'
    
    curPos, dotStatus = curState
    max_dist = 0
    for dot in dotStatus:
        dist = dist_table[(curPos, dot)]
        max_dist = max(max_dist, dist)
    return max_dist

def bruteForce_heuristics(curState, maze, dist_table=None):
    min_path = -1
    path, visited = [], {}
    curPos, dotStatus = curState

    def path_combination(curIndex):
        nonlocal min_path, path, visited
        nonlocal dotStatus, dist_table

        if curIndex == len(dotStatus):
            prev = curPos
            path_cost = 0
            for dot in path:
                if dist_table == None or (prev, dot) not in dist_table:
                    path_cost += manhatten_distance(prev, dot)
                else:
                    path_cost += dist_table[(prev, dot)]
                prev = dot
            
            if min_path == -1:
                min_path = path_cost
            else:
                min_path = min(min_path, path_cost)

        else:
            for dot in dotStatus:
                if dot not in visited or visited[dot] == 0:
                    path.append(dot)
                    visited[dot] = 1
                    path_combination(curIndex+1)
                    path.pop()
                    visited[dot] = 0

    path_combination(0)
    return min_path

def two_direction_heuristics(curState, maze, dist_table=None):
    curPos, dotStatus = curState
    curRow, curCol = curPos
    minRow, maxRow = min(dotStatus, key=lambda x: x[0])[0], max(dotStatus, key=lambda x: x[0])[0]
    minCol, maxCol = min(dotStatus, key=lambda x: x[1])[1], max(dotStatus, key=lambda x: x[1])[1]
    total_heuristics = 0

    if minRow < curRow:
        total_heuristics += abs(minRow - curRow)
    
    if maxRow > curRow:
        total_heuristics += abs(maxRow - curRow)

    if minRow < curRow < maxRow:
        total_heuristics += min(abs(maxRow - curRow), abs(curRow - minRow))

    if minCol < curCol:
        total_heuristics += abs(minCol - curCol)
    
    if maxCol > curCol:
        total_heuristics += abs(maxCol - curCol)

    if minCol < curCol < maxCol:
        total_heuristics += min(abs(maxCol - curCol), abs(curCol - minCol))

    return total_heuristics

def partial_brutalForce_heuristics_test(curState, maze, dist_table=None, compute_range=[]):
    assert len(compute_range) >= 2, '需要指定compute_range'
    assert len(curState[1]) > compute_range[-1], \
            str.format('目標數不足')

    curPos, dotStatus = curState[0], tuple([curState[1][i] for i in compute_range])
    tmpState = (curPos, dotStatus)
    return bruteForce_heuristics(tmpState, maze, dist_table)

def partial_brutalForce_heuristics(curState, maze, dist_table=None, brute_threshold=None):
    assert isinstance(brute_threshold, int), '需要指定brute_threshold'
    assert len(curState[1]) >= brute_threshold, \
            str.format('目標數不足%d個,請直接呼叫 brutalForce_heuristics' %(brute_threshold))

    tmpState = (curState[0], curState[1][:brute_threshold])
    return bruteForce_heuristics(tmpState, maze, dist_table)

def test_1(curState, maze, dist_table=None):
    curPos, dotStatus = curState

    if len(dotStatus) <= 5:
        return bruteForce_heuristics(curState, maze, dist_table)

    topDots = [dot for dot in dotStatus if dot[0] <= curPos[0]]
    downDots = [dot for dot in dotStatus if dot[0] >= curPos[0]]
    leftDots = [dot for dot in dotStatus if dot[1] <= curPos[1]]
    rightDots = [dot for dot in dotStatus if dot[1] >= curPos[1]]

    topDots = sorted(topDots, key=lambda x: abs(x[0] - curPos[0]))
    downDots = sorted(downDots, key=lambda x: abs(x[0] - curPos[0]))
    leftDots = sorted(leftDots, key=lambda x: abs(x[1] - curPos[1]))
    rightDots = sorted(rightDots, key=lambda x: abs(x[1] - curPos[1]))

    total_heuristics = 0

    if len(topDots) != 0:
        top_cost = drow(curPos, topDots[-1])
        total_heuristics += top_cost
    
    if len(downDots) != 0:
        down_cost = drow(curPos, downDots[-1])
        total_heuristics += down_cost

    if len(topDots) != 0 and len(downDots) != 0:
        total_heuristics += min(top_cost, down_cost)
    
    if len(leftDots) != 0:
        left_cost = dcol(curPos, leftDots[-1])
        total_heuristics += left_cost
    
    if len(rightDots) != 0:
        right_cost = dcol(curPos, rightDots[-1])
        total_heuristics += right_cost

    if len(leftDots) != 0 and len(rightDots) != 0:
        total_heuristics += min(left_cost, right_cost)

    return total_heuristics

def mix_heuristics(curState, maze, dist_table=None):
    _, dotStatus = curState
    brute_threshold = 5
    partial_brute_num = 3

    if len(dotStatus) <= brute_threshold:
        return bruteForce_heuristics(curState, maze, dist_table)

    elif len(dotStatus) <= brute_threshold+4:
        return max(two_direction_heuristics(curState, maze, dist_table), 
                #    partial_brutalForce_heuristics(curState, maze, dist_table, partial_brute_num),
                   max_distance_heuristics(curState, maze, dist_table))
    else:
        return max(two_direction_heuristics(curState, maze, dist_table), 
                   partial_brutalForce_heuristics(curState, maze, dist_table, partial_brute_num),
                   max_distance_heuristics(curState, maze, dist_table))

""" Greedy Heuristics """
def NumOfGoals_heuristics(curState, maze, dist_table=None):
    _, dotStatus = curState
    return len(dotStatus)

""" Greedy Graph Search """
def DFS_Search(maze, dist_table=None):
    
    def DFS_helper(curPos, maze, path, visited, dotStatus):
        if not hasattr(DFS_helper, 'dotCount'):
            DFS_helper.dotCount = len(dotStatus)

        visited[curPos] = 1
        path.append(curPos)
        if curPos in dotStatus:
            DFS_helper.dotCount -= 1
            if DFS_helper.dotCount == 0:
                return 

        neighbors = maze.getNeighbors(*curPos)
        for nextPos in neighbors:
            if DFS_helper.dotCount == 0:
                return
            
            if nextPos not in visited:
                DFS_helper(nextPos, maze, path, visited, dotStatus)
        path.append(curPos)
        return 
    
    path = []
    visited = dict()
    start_pos = maze.getStart()
    dotStatus = {dot: 1 for dot in maze.getObjectives()}
    neighbors = maze.getNeighbors(*start_pos)

    DFS_helper(start_pos, maze, path, visited, dotStatus)
    return path

""" The prototype of all the graph search algorithms """
def GraphSearch(maze, alpha, beta, heuristics, dist_table=None):
    # State: (curPos, dotStatus)
    # priority queue: 
    #   Item: [State, parent_State, g_n, h_n]
    #   Priority: (f_n, h_n)
    #             - f_n = alpha * g_n + beta * h_n
    #             - 先比f_n，再比h_n(可以break the tie)

    pri_queue = PriorityQueue()

    start_pos = maze.getStart()
    ini_dotStatus = tuple([dot for dot in maze.getObjectives()])
    start_state = (start_pos, ini_dotStatus)

    g_n, h_n = 0, heuristics(start_state, maze, dist_table)
    f_n = alpha*g_n + beta*h_n
    pri_queue.push([tuple(start_state), start_state, g_n, h_n], (f_n, h_n))

    parents, visited = dict(), dict()
    heuristics_record = dict()
    while not pri_queue.isEmpty():
        curState, parentState, g_n, h_n = pri_queue.pop()
        curPos, dotStatus = curState

        if curState not in visited:
            parents[curState] = parentState
            visited[curState] = 1

            if len(dotStatus) == 0:
                path = []
                while True:
                    path.append(curState[0])
                    if parents[curState] == curState:
                        break
                    curState = parents[curState]
                return path[::-1]

            else:
                neighbors = maze.getNeighbors(*curPos)
                for nextPos in neighbors:
                    try:
                        dotPos = dotStatus.index(nextPos)
                        next_dotStatus = dotStatus[:dotPos] + dotStatus[dotPos+1:]
                    except:
                        next_dotStatus = dotStatus

                    nextState = (nextPos, next_dotStatus)
                    
                    if nextState not in visited:
                        next_gn = g_n + 1
                        if nextState in heuristics_record:
                            next_hn = heuristics_record[nextState]
                        else:
                            next_hn = heuristics(nextState, maze, dist_table)
                            heuristics_record[nextState] = next_hn

                        next_fn = alpha*next_gn + beta*next_hn
                        pri_queue.push([nextState, curState, next_gn, next_hn], (next_fn, next_hn))
    return []

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # State: [curPos, dotStatus, parent]
    return GraphSearch(maze, 1, 1, null_heuristics)

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return GraphSearch(maze, 1, 1, max_manhatten_heuristics)
    return GraphSearch(maze, 1, 1+0.00005, max_manhatten_heuristics)

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    dist_table = countAllDistance(maze)
    return GraphSearch(maze, 1, 1, mix_heuristics, dist_table)
    return GraphSearch(maze, 1, 1+0.00005, mix_heuristics, dist_table)

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    dist_table = countAllDistance(maze)
    return GraphSearch(maze, 1, 1, mix_heuristics, dist_table)
    return GraphSearch(maze, 1, 1+0.00005, mix_heuristics, dist_table)

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return GraphSearch(maze, 0, 1, NumOfGoals_heuristics)
