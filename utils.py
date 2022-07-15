import numpy as np
import math
import matplotlib.pyplot as plt

class Game:
    """
    Class for a single episode of Prisoner's Dilemma. To create a game, it needs:
        - payoff: payoff matrix as a list of lists, one per row (example: [[(3,3),(0,5)],[(5,0),(1,1)]])
        - actions: list of possible actions, encoded numerically from 0
    """
    def __init__(self, payoff, actions):
        self.actions = actions
        self.size = int(math.sqrt(len(payoff)))
        self.scores = np.array(payoff, dtype=[("x", object), ("y", object)])
        self.payoff = payoff 

    def getNash(self):
        """ finds Nash equilibra of the payoff matrix. 
        Definition: Considering the set of possible actions, 
                    if for any pair no individual player can benefit by changing its individual strategy, then that's a Nash equilibrium.

        Returns:
            List: index or indexes of Nash equilibra.
        """
        max_x = np.matrix(self.scores["x"].max(0)).repeat(self.size, axis=0)
        bool_x = self.scores["x"] == max_x
        max_y = (np.matrix(self.scores["y"].max(1)).transpose().repeat(self.size, axis=1))
        bool_y = self.scores["y"] == max_y
        bool_x_y = bool_x & bool_y
        result = np.where(bool_x_y == True)
        listOfCoordinates = list(zip(result[0], result[1]))
        return listOfCoordinates
 

class Meeting:
    """
    Class for multiple episodes of Prisoner's Dilemma. To create a meeting, it needs:
        - game: an object of class Game
        - s1, s2: the strategies of the two players, instances of a Strategy class
        - length (optional): number of games to play in the meeting
    """
    def __init__(self, game, s1, s2, length=1000):
        self.game = game
        self.s1 = s1.clone()
        self.s2 = s2.clone()
        self.length = length
        # cooperation counters
        self.num_cooperation_s1 = 0
        self.num_cooperation_s2 = 0

    def reinit(self):
        """
        Reset meeting scores.
        """
        # cumulative rewards
        self.s1_score = 0
        self.s2_score = 0
        # history of actions
        self.s1_rounds = []
        self.s2_rounds = []

    def run(self):
        """
        Run the meeting.
        """
        # reset meeting scores
        self.reinit()

        # comunicate payoff matrix to players
        self.s1.payoff = self.game.payoff
        self.s2.payoff = self.game.payoff

        # run 'length' games
        for iteration in range(self.length):
            # if it is the last game, set players' 'done' flag to True
            if iteration == self.length -1:
                self.s1.done = True
                self.s2.done = True
            # get each players' action
            c1 = self.s1.get_action(iteration)
            c2 = self.s2.get_action(iteration)
            # if they choose to cooperate, add to the cooperation counter
            if c1 == 0:
                self.num_cooperation_s1 += 1
            if c2 == 0:
                self.num_cooperation_s2 += 1
            # save action in history list
            self.s1_rounds.append(c1)
            self.s2_rounds.append(c2)
            # tell each other their past action
            self.s1.update(c1, c2)
            self.s2.update(c2, c1)
            act = self.game.actions
            # add payoff to cumulative reward
            self.s1_score += self.game.scores["x"][act.index(c1), act.index(c2)]
            self.s2_score += self.game.scores["y"][act.index(c1), act.index(c2)]

    def pretty_print(self, max=50):
        '''
        Print the outcome of the meeting, as the outcome of the first (max 'max') games and the cumulative scores.
        The score is the sum of the scores obtained on each game, according to the payoff matrix. The higher the better.
        '''
        print("{}\t{} ... {} = {}".format(self.s1.name, ' '.join(map(str, self.s1_rounds[:max//2])), ' '.join(map(str, self.s1_rounds[-max//2:])), self.s1_score))
        print("{}\t{} ... {} = {}".format(self.s2.name, ' '.join(map(str, self.s2_rounds[:max//2])), ' '.join(map(str, self.s2_rounds[-max//2:])), self.s2_score))

    def plot_cooperation(self):
        '''
        plot percentage of cooperations
        '''
        # set size
        plt.rcParams["figure.figsize"] = (10,7)

        s1_cooperations_count = [0]
        s2_cooperations_count = [0]
        s1_cooperations_percent = []
        s2_cooperations_percent = []

        for i in range(self.length):
            # count cooperations until time i 
            s1_cooperations_count.append(s1_cooperations_count[-1] + self.s1_rounds[i])
            s2_cooperations_count.append(s2_cooperations_count[-1] + self.s2_rounds[i])
            # make it a percentage over amount of actions taken
            s1_cooperations_percent.append(s1_cooperations_count[-1]/(i+1) * 100)
            s2_cooperations_percent.append(s2_cooperations_count[-1]/(i+1) * 100)

        # plot as lines
        for coop in [s1_cooperations_percent, s2_cooperations_percent]:
            plt.plot(coop)
        # mark where 50% is
        plt.axhline(50, color = 'gray', linestyle = '--', lw = 1)
        plt.xlabel('game');
        plt.ylabel('percentage');
        plt.title("Number of cooperations");
        plt.legend([self.s1.name, self.s2.name]);