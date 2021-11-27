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
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        #the closer to the foods the better state
        foodDistances = [manhattanDistance(newPos, pos) for pos in newFood.asList()]

        if(foodDistances):
            closestFoodPos = min(foodDistances)
            score -= closestFoodPos
        
        #the farther from the ghosts the better state
        ghostPositions = [x.getPosition() for x in newGhostStates]
        for pos in ghostPositions:
            score += manhattanDistance(pos, newPos)*1
        return score


def scoreEvaluationFunction(currentGameState):
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
    
    def getBestAction(self, state):
        val = self.maxValue(state, 0, 0)
        return val[1]
    
    def getValue(self, state, agentId, depth):
        if(state.isWin() or state.isLose() or depth == self.depth):
            return (self.evaluationFunction(state), "Stop")
        elif(agentId == 0):
            return self.maxValue(state, agentId, depth)
        else:
            return self.minValue(state, agentId, depth)

    def maxValue(self, state, agentId, depth):
        action = ""
        value = (-math.inf, action)
        successorTuples = [(state.generateSuccessor(agentId, action), action) for action in state.getLegalActions(agentId)]
        for suc in successorTuples:
            action = suc[1]
            maxVal = max(value[0], self.getValue(suc[0], (agentId+1) % state.getNumAgents(), depth)[0])
            if(maxVal != value[0]):
                value = (maxVal, action)
        return value
    
    def minValue(self, state, agentId, depth):
        action = ""
        value = (math.inf, action)
        successorTuples = [(state.generateSuccessor(agentId, action), action) for action in state.getLegalActions(agentId)]
        for suc in successorTuples:
            action = suc[1]
            nextIndex = (agentId+1) % state.getNumAgents();
            minVal = math.inf
            if(nextIndex != 0):
                minVal = min(value[0], self.getValue(suc[0], nextIndex, depth)[0])
            else:
                minVal = min(value[0], self.getValue(suc[0], nextIndex, depth+1)[0])
                
            if(minVal != value[0]):
                value = (minVal, action)
        return value
    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        "*** YOUR CODE HERE ***"
        
        return self.getBestAction(gameState)
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getBestAction(self, state):
        val = self.maxValue(state, 0, 0, -math.inf, math.inf)
        return val[1]
    
    def getValue(self, state, agentId, depth, alpha, beta):
        if(state.isWin() or state.isLose() or depth == self.depth):
            return (self.evaluationFunction(state), "Stop")
        elif(agentId == 0):
            return self.maxValue(state, agentId, depth, alpha, beta)
        else:
            return self.minValue(state, agentId, depth, alpha, beta)

    def maxValue(self, state, agentId, depth, alpha, beta):
        value = (-math.inf, "")
        for action in state.getLegalActions(agentId):
            successorState = state.generateSuccessor(agentId, action)
            nextAgentId = (agentId+1) % state.getNumAgents()
            maxVal = max(value[0], self.getValue(successorState, nextAgentId, depth, alpha, beta)[0])
            if(maxVal != value[0]):
                value = (maxVal, action)
                
            if value[0] > beta:
                return value
            alpha = max(alpha, value[0])

        return value
    
    def minValue(self, state, agentId, depth, alpha, beta):
        value = (math.inf, "")
        for action in state.getLegalActions(agentId):
            successorState = state.generateSuccessor(agentId, action)
            nextAgentId = (agentId+1) % state.getNumAgents();
            minVal=math.inf
            if(nextAgentId != 0):
                minVal = min(value[0], self.getValue(successorState, nextAgentId, depth, alpha, beta)[0])
            else:
                minVal = min(value[0], self.getValue(successorState, nextAgentId, depth+1, alpha, beta)[0])
                
            if(minVal != value[0]):
                value = (minVal, action)
                    
            if value[0] < alpha:
                return value
            beta = min(beta, value[0])

        return value
    
    def getAction(self, gameState):
               
        return self.getBestAction(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getBestAction(self, state):
        val = self.maxValue(state, 0, 0)
        return val[1]
    
    def getValue(self, state, agentId, depth):
        if(state.isWin() or state.isLose() or depth == self.depth):
            return (self.evaluationFunction(state), "Stop")
        elif(agentId == 0):
            return self.maxValue(state, agentId, depth)
        else:
            return self.expValue(state, agentId, depth)

    def maxValue(self, state, agentId, depth):
        action = ""
        value = (-math.inf, action)
        successorTuples = [(state.generateSuccessor(agentId, action), action) for action in state.getLegalActions(agentId)]
        for suc in successorTuples:
            action = suc[1]
            maxVal = max(value[0], self.getValue(suc[0], (agentId+1) % state.getNumAgents(), depth)[0])
            if(maxVal != value[0]):
                value = (maxVal, action)
        return value
    
    def expValue(self, state, agentId, depth):
        action = ""
        val = 0
        value = (val, action)
        successorTuples = [(state.generateSuccessor(agentId, action), action) for action in state.getLegalActions(agentId)]
        prob = 1/len(successorTuples)
        for suc in successorTuples:
            action = suc[1]
            nextIndex = (agentId+1) % state.getNumAgents();
            if(nextIndex != 0):
                val += self.getValue(suc[0], nextIndex, depth)[0]*prob
            else:
                val += self.getValue(suc[0], nextIndex, depth+1)[0]*prob
        
        randomTuple = random.choice(successorTuples)
        value = (val, randomTuple[1])
        return value

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getBestAction(gameState)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    currentPos = currentGameState.getPacmanPosition()
    currentFoods = currentGameState.getFood().asList()
    foodAmount = len(currentFoods)
    currentCapsules = currentGameState.getCapsules()
    currentGhostStates = currentGameState.getGhostStates()
    currentGhostPositions = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    

    score = currentGameState.getScore()
    
    score -= foodAmount*5
    
    for capsulePos in currentCapsules:
        score -= manhattanDistance(currentPos, capsulePos)
    
    for time in scaredTimes:
        if time > 5:
            for ghostPos in currentGhostPositions:
                score -= manhattanDistance(currentPos, ghostPos)*5
        else:                    
            for ghostPos in currentGhostPositions:
                score += manhattanDistance(currentPos, ghostPos)*2

    return score

# Abbreviation
better = betterEvaluationFunction
