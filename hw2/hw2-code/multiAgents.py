# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

""" Priority Queue Structure """
import heapq
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

""" Useful Functions """
def manhatten_distance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

def nearest_ghost(curPos, ghosts):
    nearest, min_dist = None, 1000000000
    for ghost in ghosts:
        ghost_pos = ghost.getPosition()
        if manhattanDistance(curPos, ghost_pos) < min_dist:
            min_dist = manhattanDistance(curPos, ghost_pos)
            nearest = ghost
            
    return min_dist, nearest

def Prim_MST_heuristics(curPos, foods, dist_table=None):
    pri_queue = PriorityQueue()
    dot_candidates = (*foods, curPos)

    for nextPos in dot_candidates:
        if nextPos != curPos:
            if dist_table == None or (curPos, nextPos) not in dist_table:
                cost = manhattanDistance(curPos, nextPos)
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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        "*** YOUR CODE HERE ***"
        # Useful information you can extract from a GameState (pacman.py)
        oldPos = currentGameState.getPacmanPosition()
        oldGhostStates = currentGameState.getGhostStates()

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newCapsules = successorGameState.getCapsules()
        newFoodNum = len(newFood.asList())

        # Make the pacman not to stop
        if oldPos == newPos:
            return [-10000000]

        def closer_food(curPos, newPos, foods):
            count = 0
            for food in foods:
                if manhattanDistance(newPos, food) > manhattanDistance(curPos, food):
                    count += 1
            return count 

        newFood_CapsulesDist = Prim_MST_heuristics(newPos, newFood.asList() + newCapsules)
        new_Capsules_dist = Prim_MST_heuristics(newPos, newCapsules) if len(newCapsules) != 0 else 0
        closer_count = closer_food(oldPos, newPos, newFood.asList())
        oldGhostDist, oldGhost = nearest_ghost(oldPos, oldGhostStates)

        # There is a bug when 7->8, 8->9
        if oldGhost.scaredTimer <= 7 and oldGhostDist <= 2:
            newGhostDist, _ = nearest_ghost(newPos, newGhostStates)
            return [newGhostDist - oldGhostDist]

        elif oldGhost.scaredTimer >= 8 and oldGhostDist >= 2:
            newGhostDist, _ = nearest_ghost(newPos, newGhostStates)
            return [oldGhostDist - newGhostDist]

        else:
            return [-newFoodNum, -new_Capsules_dist, -newFood_CapsulesDist, closer_count]


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax_helper(self, gameState, depth, agentIndex):
        if depth == self.depth+1 and agentIndex == 0:
            return self.evaluationFunction(gameState), None

        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None

        best_score, best_action = None, None
        isPacman = (agentIndex == 0)
        legal_moves = gameState.getLegalActions(agentIndex)
        
        for action in legal_moves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex+1 < gameState.getNumAgents():
                result_score, _ = self.minimax_helper(nextState, depth, agentIndex+1)
            else:
                result_score, _ = self.minimax_helper(nextState, depth+1, 0)

            if isPacman:
                if best_score == None or result_score > best_score:
                    best_score, best_action = result_score, action
            else:
                if best_score == None or result_score < best_score:
                    best_score, best_action = result_score, action
                
        return best_score, best_action

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        best_score, best_action = self.minimax_helper(gameState, 1, 0)
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def AlphaBeta_helper(self, gameState, depth, agentIndex, alpha, beta):
        if depth == self.depth+1 and agentIndex == 0:
            return self.evaluationFunction(gameState), None

        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None

        best_score, best_action = None, None
        isPacman = (agentIndex == 0)
        legal_moves = gameState.getLegalActions(agentIndex)
        
        for action in legal_moves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex+1 < gameState.getNumAgents():
                result_score, _ = self.AlphaBeta_helper(nextState, depth, agentIndex+1, alpha, beta)
            else:
                result_score, _ = self.AlphaBeta_helper(nextState, depth+1, 0, alpha, beta)

            if isPacman:
                if best_score == None or result_score > best_score:
                    best_score, best_action = result_score, action
                    
                    # alpha-beta pruning
                    if alpha == None or best_score > alpha:
                        alpha = best_score

                    if beta != None and best_score > beta:
                        break
            else:
                if best_score == None or result_score < best_score:
                    best_score, best_action = result_score, action

                    # alpha-beta pruning
                    if beta == None or best_score < beta:
                        beta = best_score

                    if alpha != None and best_score < alpha:
                        break

        return best_score, best_action

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_score, best_action = self.AlphaBeta_helper(gameState, 1, 0, None, None)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax_helper(self, gameState, depth, agentIndex):
        if depth == self.depth+1 and agentIndex == 0:
            return self.evaluationFunction(gameState), None

        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None

        scoreSum = 0
        best_score, best_action = None, None
        isPacman = (agentIndex == 0)
        legal_moves = gameState.getLegalActions(agentIndex)
        
        for action in legal_moves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex+1 < gameState.getNumAgents():
                result_score, _ = self.expectimax_helper(nextState, depth, agentIndex+1)
            else:
                result_score, _ = self.expectimax_helper(nextState, depth+1, 0)

            if isPacman:
                if best_score == None or result_score > best_score:
                    best_score, best_action = result_score, action
            else:
                scoreSum += result_score
        
        if isPacman:
            return best_score, best_action
        else:
            return scoreSum / len(legal_moves), _

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        best_score, best_action = self.expectimax_helper(gameState, 1, 0)
        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foods = currentGameState.getFood().asList()
    curPos = currentGameState.getPacmanPosition()
    food_pathlen = 0 if len(foods) == 0 else Prim_MST_heuristics(curPos, foods)

    ghosts = currentGameState.getGhostStates()
    Ghost_MinDist, nearest_Ghost = nearest_ghost(curPos, ghosts)
    score = scoreEvaluationFunction(currentGameState)
    
    utility = 0
    if nearest_Ghost.scaredTimer == 0 and Ghost_MinDist <= 2:
        utility = -1000000000
    else:
        utility = 1*(-len(foods)) + 1*(-food_pathlen) + 1*score
    return utility

# Abbreviation
better = betterEvaluationFunction
