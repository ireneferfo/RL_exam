import random
import numpy as np

class Strategy:
    '''
    Parent class for all strategies.
    '''
    def get_action(self, iteration):
        pass

    def clone(self): 
        pass

    def update(self, my, their):
        pass


class Cooperate(Strategy):
    '''
    Player that always cooperates.
    '''
    def __init__(self):
        super().__init__()
        self.name = "cooperate"

    def get_action(self, iteration):
        # Cooperate
        return 0 

    def clone(self):
        return Cooperate()


class Defect(Strategy):
    '''
    Player that always defects.
    '''
    def __init__(self):
        super().__init__()
        self.name = "defect"

    def get_action(self, iteration):
        # Defect
        return 1 

    def clone(self):
        return Defect()


class Random(Strategy):
    '''
    Player that plays random actions.
    '''
    def __init__(self):
        super().__init__()
        self.name = "random"

    def get_action(self, iteration):
        return random.choice([0, 1])

    def clone(self):
        return Random()


class TitforTat(Strategy):
    '''
    Player that first cooperates, then copies the opponent's previous action.
    '''
    def __init__(self):
        super().__init__()
        self.name = "TitforTat"
        self.theirPast = None

    def get_action(self, iteration):
        # first action is cooperate 
        return 0 if (iteration == 0) else self.theirPast

    def clone(self):
        return TitforTat()

    def update(self, my, their):
        self.theirPast = their


class QLearning(Strategy):
    '''
    Player that learns the best strategies through the Q-learning algorithm. It needs:
        - alpha (optional): the learning rate, set between 0 and 1. Setting it to 0 means the Q-values are never updated, hence nothing is learned.
                Setting a high value means that learning can occur quickly. Defaults to 0.1.
        - gamma (optional): discount factor, also set between 0 and 1. This models the fact that future rewards are worth less than immediate rewards.
                Defaults to 0.95.
        - epsilon (optional): parameter for the epsilon-greedy policy, set between 0 and 1. It is the probability of taking a random action 
                instead of following the Q-table. Defaults to 0.3.
        - decay (optional): parameter for the decaying epsilon-greedy policy. It is the factor by which 'epsilon' is multiplied at each step, 
                reducing it until reaching a minimum of 0.1. Defaults to 0.999.
    '''
    def __init__(self, alpha = 0.1, gamma =0.95, epsilon = 0.3, decay = 0.999):
        super().__init__()
        self.name = "QLearning"
        self.alpha = alpha
        self.gamma = gamma 
        self.epsilon = epsilon
        self.og_epsilon = epsilon # useful to clone
        self.decay = decay
        self.min_epsilon = 0.1
        self.q_table = np.zeros((2,2,2)) # 4 possible states (2x2 actions), 2 possible actions: 4x2
        # save past actions, initiation is irrelevant to outcome as all entries are 0 in the beginning and first action is chosen randomly
        self.mypast = 0
        self.theirpast = 0
        # flag for last iteration
        self.done = False

# to access q_table: first two indices form the state s (my_action, their_action). Third index is the action a.

    def get_action(self, iteration):
        # decaying epsilon greedy policy
        self.epsilon = max(self.min_epsilon, self.epsilon*self.decay)

        # initialise first state, so first action, randomly
        if (iteration == 0) or (np.random.uniform(0,1) < self.epsilon):
            # choose random action
            action = np.random.choice([0, 1])
        else:
            # select action that maximises Q table for last state
            action = np.argmax(self.q_table[self.mypast][self.theirpast])
        return action

    def update(self, my, their):
        # calculate rewards and update Q table with current actions
        r = self.step(my, their)
        self.q_table = self.single_step_update(my, their, r) 
        # current actions become past actions for next game
        self.mypast = my
        self.theirpast = their
        
    def step(self, my, their):
        # enter payoff matrix for the current state.
        r = self.payoff[my][their][0] # 0 because my payoff is in the first position of the tuple (my_payoff, their_payoff)
        # print('mine: ', my, 'theirs: ', their, 'reward: ', r)
        return r

    def single_step_update(self, my, their, r): 
        if self.done: 
            deltaQ = r + 0 - self.q_table[self.mypast][self.theirpast][self.mypast]
        else:
            # Q-learning update
            deltaQ = r + self.gamma * self.q_table[my][their].max() - self.q_table[self.mypast][self.theirpast][self.mypast]
        self.q_table[self.mypast][self.theirpast][self.mypast] += self.alpha * deltaQ
        return self.q_table

    def clone(self):
        return QLearning(self.alpha, self.gamma, self.og_epsilon, self.decay)

    def print_qtable(self):
        return self.q_table


class Sarsa(Strategy):
    '''
    Player that learns the best strategies through the Sarsa algorithm. It needs:
        - alpha (optional): the learning rate, set between 0 and 1. Setting it to 0 means the Q-values are never updated, hence nothing is learned.
                Setting a high value means that learning can occur quickly. Defaults to 0.1.
        - gamma (optional): discount factor, also set between 0 and 1. This models the fact that future rewards are worth less than immediate rewards.
                Defaults to 0.95.
        - epsilon (optional): parameter for the epsilon-greedy policy, set between 0 and 1. It is the probability of taking a random action 
                instead of following the Q-table. Defaults to 0.3.
        - decay (optional): parameter for the decaying epsilon-greedy policy. It is the factor by which 'epsilon' is multiplied at each step, 
                reducing it until reaching a minimum of 0.1. Defaults to 0.999.
    '''
    def __init__(self, alpha = 0.1, gamma =0.95, epsilon = 0.3, decay = 0.999):
        super().__init__()
        self.name = "Sarsa"
        self.alpha = alpha
        self.gamma = gamma 
        self.epsilon = epsilon
        self.og_epsilon = epsilon # useful to clone
        self.decay = decay
        self.min_epsilon = 0.1
        self.q_table = np.zeros((2,2,2)) # 4 possible states (2x2 actions), 2 possible actions: 4x2
        # save past actions, initiation is irrelevant to outcome as all entries are 0 in the beginning and first action is chosen randomly
        self.mypast = 0
        self.theirpast = 0
        # flag for last iteration
        self.done = False

# to access q_table: first two indices form the state s (my_action, their_action). Third index is the action a.

    def get_action(self, iteration):
        # decaying epsilon greedy policy
        self.epsilon = max(self.min_epsilon, self.epsilon*self.decay)

        # initialise first state, so first action, randomly
        if (iteration == 0) or (np.random.uniform(0,1) < self.epsilon):
            # choose random action
            action = np.random.choice([0, 1])
        else:
            # select action that maximises Q table for last state
            action = np.argmax(self.q_table[self.mypast][self.theirpast])
        return action

    def update(self, my, their):
        # calculate rewards and update Q table with current actions
        r = self.step(my, their)
        self.q_table = self.single_step_update(my, their, r) 
        # current actions become past actions for next game
        self.mypast = my
        self.theirpast = their
        
    def step(self, my, their):
        # enter payoff matrix for the current state.
        r = self.payoff[my][their][0] # 0 because my payoff is in the first position of the tuple (my_payoff, their_payoff)
        # print('mine: ', my, 'theirs: ', their, 'reward: ', r)
        return r

    def single_step_update(self, my, their, r): 
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Employs the experienced action in the new state  <- Q(S_new, A_new).
        """
        if self.done: 
            deltaQ = r + 0 - self.q_table[self.mypast][self.theirpast][self.mypast]
        else:
            # SARSA update
            deltaQ = r + self.gamma * self.q_table[my][their][my] - self.q_table[self.mypast][self.theirpast][self.mypast]
        self.q_table[self.mypast][self.theirpast][self.mypast] += self.alpha * deltaQ
        return self.q_table

    def clone(self):
        return Sarsa(self.alpha, self.gamma, self.og_epsilon, self.decay)

    def print_qtable(self):
        return self.q_table