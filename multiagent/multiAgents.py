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
        legalMoves = gameState.getLegalActions()

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        # get lists of food and capsule coordinates
        foodList = newFood.asList()
        capsuleList = successorGameState.getCapsules()

        # initialize score variables
        foodScore = ghostScore = scaredGhostScore = capsuleScore = 0

        # initialize distance arrays
        foodDistances = []
        ghostDistances = []
        scaredGhostDistances = []
        capsuleDistances = []

        # calculate foodScore to add to successor function
        # food score is the inverse of the disstance to the closest food, unless
        # the closest food is at the current position
        if foodList:
            for food in foodList:
                foodDistances.append(manhattanDistance(newPos, food))

            closestFood = min(foodDistances)

            if closestFood == 0:
                foodScore = 100
            else:
                foodScore = 10 / closestFood

        # calculate ghost penalty and scaredGhost score to add to successor function
        # ghost penalty is -1000 if a ghost is within one move of pacman
        # scared ghost score is inverse of the distance to the closest scared ghost
        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0:
                ghostDistances.append(manhattanDistance(newPos, ghostState.getPosition()))

            else:
                scaredGhostDistances.append(manhattanDistance(newPos, ghostState.getPosition()))
            
        if ghostDistances:
            closestGhost = min(ghostDistances)
            if closestGhost <= 1:
                ghostScore = -1000

        if scaredGhostDistances:
            closestScaredGhost = min(scaredGhostDistances)
            # weighted more than the food score to bias pacman toward scared ghosts
            scaredGhostScore = 100 / closestScaredGhost
        
        return successorGameState.getScore() + foodScore + ghostScore + scaredGhostScore

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

        return self.maxValue(gameState, depth = 0, agentIndex = 0)[1]
    
    def terminalTest(self, gameState: GameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == self.depth

            
    def maxValue(self, gameState: GameState, depth, agentIndex):

        # terminal test
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState), None

        # set value to negative infinty
        value = -float('inf')

        # get all legal actions for agent at current state
        legalActions = gameState.getLegalActions(agentIndex)
        actionValues = []

        # loop through all actions to get values and find max value
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            # recursive call to minValue function
            actionValue = self.minValue(successorState, depth, 1)[0]
            actionValues.append(actionValue)
            
            value = max(value, actionValue)

        # find corresponding action for max value
        actionIndices = [index for index in range(len(actionValues)) if actionValues[index] == value]
        actionIndex = random.choice(actionIndices)

        # return max value and corresponding action
        return value, legalActions[actionIndex]
                
    def minValue(self, gameState: GameState, depth, agentIndex):

        # terminal test
        if self.terminalTest(gameState, depth):
                return self.evaluationFunction(gameState), None

        # set value to infinty
        value = float('inf')

        # get all legal actions for agent at current state
        legalActions = gameState.getLegalActions(agentIndex)
        actionValues = []
            
        # take care of multiple min agents, one for each ghost
        numAgents = gameState.getNumAgents()
        if agentIndex < numAgents - 1:

            # loop through all actions to get values and find min value
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # recursive call to minValue function for mulitiple min agents
                actionValue = self.minValue(successorState,depth, agentIndex + 1)[0]
                actionValues.append(actionValue)
                
                value = min(value, actionValue)

            # find corresponding action for min value
            actionIndices = [index for index in range(len(actionValues)) if actionValues[index] == value]
            actionIndex = random.choice(actionIndices)

            # return min value and corresponding action
            return value, legalActions[actionIndex]

        # final min agent
        else:

            # loop through all actions to get values and find min value
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # recursive call to maxValue function
                actionValue = self.maxValue(successorState,depth + 1, 0)[0]
                actionValues.append(actionValue)
                
                value = min(value, actionValue)
                
            # find corresponding action for min value    
            actionIndices = [index for index in range(len(actionValues)) if actionValues[index] == value]
            actionIndex = random.choice(actionIndices)

            # return min value and corresponding action
            return value, legalActions[actionIndex]
        
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # set initial alpha and beta values in call to max value
        return self.maxValue(gameState, depth = 0, agentIndex = 0, alpha = -float('inf'), beta = float('inf'))[1]

    def terminalTest(self, gameState: GameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == self.depth

            
    def maxValue(self, gameState: GameState, depth, agentIndex, alpha, beta):

        # terminal test
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState), None

        # set value to negative infinty
        value = -float('inf')

        # get all legal actions for agent at current state
        legalActions = gameState.getLegalActions(agentIndex)
        actionValues = []

        # loop through all actions to get values and find max value
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            actionValue = self.minValue(successorState, depth, 1, alpha, beta)[0]
            actionValues.append(actionValue)
            
            value = max(value, actionValue)

            # pruning if value > beta
            if value > beta:
                return value, action

            # update alpha value
            alpha = max(alpha, value)

        # find corresponding action for max value  
        actionIndices = [index for index in range(len(actionValues)) if actionValues[index] == value]
        actionIndex = random.choice(actionIndices)

        # return max value and corresponding action
        return value, legalActions[actionIndex]
                
    def minValue(self, gameState: GameState, depth, agentIndex, alpha, beta):

        # terminal test
        if self.terminalTest(gameState, depth):
                return self.evaluationFunction(gameState), None

        # set value to infinty
        value = float('inf')

        # get all legal actions for agent at current state
        legalActions = gameState.getLegalActions(agentIndex)
        actionValues = []
            
        # take care of multiple min agents, one for each ghost
        numAgents = gameState.getNumAgents()
        if agentIndex < numAgents - 1:

            # loop through all actions to get values and find min value
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                actionValue = self.minValue(successorState,depth, agentIndex + 1, alpha, beta)[0]
                actionValues.append(actionValue)
                
                value = min(value, actionValue)

                # pruning if value < beta
                if value < alpha:
                    return value, action

                # update beta value
                beta = min(beta, value)

            # find corresponding action for min value  
            actionIndices = [index for index in range(len(actionValues)) if actionValues[index] == value]
            actionIndex = random.choice(actionIndices)

            # return min value and corresponding action
            return value, legalActions[actionIndex]

        # final min agent
        else:

            # loop through all actions to get values and find min value
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                actionValue = self.maxValue(successorState,depth + 1, 0, alpha, beta)[0]
                actionValues.append(actionValue)
                
                value = min(value, actionValue)

                # pruning if value < beta
                if value < alpha:
                    return value, action
                
                # update beta value
                beta = min(beta, value)

            # find corresponding action for min value     
            actionIndices = [index for index in range(len(actionValues)) if actionValues[index] == value]
            actionIndex = random.choice(actionIndices)

            # return min value and corresponding action
            return value, legalActions[actionIndex]


        
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        return self.maxValue(gameState, depth = 0, agentIndex = 0)[1]

    def terminalTest(self, gameState: GameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == self.depth

            
    def maxValue(self, gameState: GameState, depth, agentIndex):
        
        # terminal test
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState), None

        # set value to negative infinty
        value = -float('inf')

        # get all legal actions for agent at current state
        legalActions = gameState.getLegalActions(agentIndex)
        actionValues = []

        # loop through all actions to get values and find max value
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            actionValue = self.expValue(successorState, depth, 1)[0]
            actionValues.append(actionValue)
            
            value = max(value, actionValue)

        # find corresponding action for max value 
        actionIndices = [index for index in range(len(actionValues)) if actionValues[index] == value]
        actionIndex = random.choice(actionIndices)

        # return max value and corresponding action
        return value, legalActions[actionIndex]
                
    def expValue(self, gameState: GameState, depth, agentIndex):

        # terminal test
        if self.terminalTest(gameState, depth):
                return self.evaluationFunction(gameState), None

        # get all legal actions for agent at current state
        legalActions = gameState.getLegalActions(agentIndex)
        actionValues = []
            
        # take care of multiple exp agents, one for each ghost
        numAgents = gameState.getNumAgents()
        if agentIndex < numAgents - 1:

            # loop through all actions to get values and store them
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                actionValue = self.expValue(successorState,depth, agentIndex + 1)[0]
                actionValues.append(actionValue)

            # calculate average value over all legal actions
            value = sum(actionValues) / len(actionValues)

            # because average value is taken, no specific move is returned
            return value, None

        # last exp agent
        else:

            # loop through all actions to get values and store them
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                actionValue = self.maxValue(successorState,depth + 1, 0)[0]
                actionValues.append(actionValue)
                
            # calculate average value over all legal actions
            value = sum(actionValues) / len(actionValues)
            
            # because average value is taken, no specific move is returned
            return value, None
        
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I first set the evalScore to the .getScore function found in the Pacman.py
    file. This acts as a basline score that will be adjusted by my custom
    evaluation metrics.
    Next I update evalScore for terminal states. If the state is a win or a loss
    these will outweight all other scoring.
    I then get into the heart of my evaluation function. There are penalties for
    both remaining food and remaining capsules. The penalty for remaining
    capsules is larger than that for remaining food in order to bias pacman
    toward the capsules.
    There are also penalties for pacman being close to food and close to scared
    ghosts. These will encourage pacman to eat the closest items first.
    The penalites for scared ghosts are larger than that for food in order to
    bias pacman toward eating scared ghosts when present. This is because eating
    a scared ghost provides a large in game score bonus.
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    foodStates = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsuleStates = currentGameState.getCapsules()
    scaredGhostStates = []

    # intialize evalScore as the value from the .getScore() function
    evalScore = currentGameState.getScore()

    # update evalScore for terminal states
    if currentGameState.isLose():
        evalScore += -100000

    if currentGameState.isWin():
        evalScore += 100000

    # get number of remaining food and capsules
    numFood = len(foodStates)
    numCapsules = len(capsuleStates)

    # penalties for having remaining food and capsules
    evalScore += -100 * numFood
    evalScore += -400 * numCapsules

    # initialize arrays for distances to food and scared ghosts
    foodDistances = []
    scaredGhostDistances = []

    # calculate distance to all remaining food
    for food in foodStates:
        foodDistances.append(manhattanDistance(position, food))
        
    # penalties for pacman being close to food
    for distance in foodDistances:
        if distance < 3:
            evalScore += -2 * distance
        else:
            evalScore += -1 * distance
            
    # calculate distance to any scared ghosts   
    for ghost in ghostStates:
        if ghost.scaredTimer != 0:
            scaredGhostDistances.append(manhattanDistance(position, ghost.getPosition()))

    # penalties for pacman being close to scared ghosts
    # wieghted more than the food penalties so pacman will go for scared ghosts first
    for distance in scaredGhostDistances:
        if distance < 3:
            evalScore += -8 * distance
        else:
            evalScore += -4 * distance


    # return final evalScore value
    return evalScore

    #util.raiseNotDefined()
    

# Abbreviation
better = betterEvaluationFunction
