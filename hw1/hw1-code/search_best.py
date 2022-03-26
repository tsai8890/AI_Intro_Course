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

""" The auxilary functions for computing distances"""
def manhatten_distance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

def drow(pos1, pos2):
    return abs(pos1[0] - pos2[0])

def dcol(pos1, pos2):
    return abs(pos1[1] - pos2[1])

""" Use BFS to construct a Graph map """
def getCompleteGraph(maze):
    visited = dict()
    start_pos = maze.getStart()
    pri_queue = PriorityQueue()
    pri_queue.push([start_pos, 0], 0)
    G = dict()

    while not pri_queue.isEmpty():
        curPos, d = pri_queue.pop()
        
        if curPos not in visited:
            visited[curPos] = 1
            neighbors = maze.getNeighbors(*curPos)
            for nextPos in neighbors:
                if curPos not in G:
                    G[curPos] = [[nextPos, 1]]
                else:
                    G[curPos].append([nextPos, 1])

                if nextPos not in visited:
                    pri_queue.push([nextPos, d+1], d+1)
    return G

""" Count the distance from all the dot points to any other point"""
def countAllDistance(maze):
    G = getCompleteGraph(maze)

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

                neighbors = G[curPos]
                for nextPos, _ in neighbors:
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

def Prim_MST_heuristics(curState, maze, dist_table=None):
    curPos, dotStatus = curState
    pri_queue = PriorityQueue()
    dot_candidates = (*dotStatus, curPos)

    for nextPos in dot_candidates:
        if nextPos != curPos:
            if dist_table == None or (curPos, nextPos) not in dist_table:
                cost = manhatten_distance(curPos, nextPos)
            else:
                cost = dist_table[(curPos, nextPos)]
            pri_queue.push([curPos, nextPos, cost], cost)

    points = set()
    points.add(curPos)

    mst_sum = 0
    while len(points) < len(dot_candidates):
        parent, curPos, cost = pri_queue.pop()
        if curPos not in points:
            points.add(curPos)
            mst_sum += cost
            for nextPos in dot_candidates:
                if nextPos not in points and nextPos != curPos:
                    if dist_table == None or (curPos, nextPos) not in dist_table:
                        cost = manhatten_distance(curPos, nextPos)
                    else:
                        cost = dist_table[(curPos, nextPos)]
                    pri_queue.push([curPos, nextPos, cost], cost)
    return mst_sum

def mix_heuristics(curState, maze, dist_table=None):
    _, dotStatus = curState
    brute_threshold = 5

    if len(dotStatus) <= brute_threshold:
        return bruteForce_heuristics(curState, maze, dist_table)
    else:
        return Prim_MST_heuristics(curState, maze, dist_table)

""" Greedy Heuristics """
def NumOfGoals_heuristics(curState, maze, dist_table=None):
    _, dotStatus = curState
    return len(dotStatus)

""" The prototype of all the graph search algorithms """
def GraphSearch(maze, alpha, beta, heuristics, dist_table=None):
    start_pos = maze.getStart()
    ini_dotStatus = tuple([dot for dot in maze.getObjectives()])
    start_state = (start_pos, ini_dotStatus)

    # State: (curPos, dotStatus)
    # priority queue: 
    #   Item: [State, parent_State, g_n, h_n]
    #   Priority: (f_n, h_n)
    #             - f_n = alpha * g_n + beta * h_n
    #             - 先比f_n，再比h_n(可以break the tie)
    pri_queue = PriorityQueue()
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

                # For debugging and check the path consistency
                # print(path[::-1])
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
                        
                        # 非常重要的一步，保證是consistent
                        # 可以先做出admissible的heuristics，接著用下面這一行code，即可保證consistent
                        next_fn = max(alpha*g_n + beta*h_n, alpha*next_gn + beta*next_hn)

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
    return GraphSearch(maze, 1, 1, null_heuristics)

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return GraphSearch(maze, 1, 1, max_manhatten_heuristics)

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    # dist_table = countAllDistance(maze)
    return GraphSearch(maze, 1, 1, mix_heuristics)

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

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return GraphSearch(maze, 0, 1, NumOfGoals_heuristics)
