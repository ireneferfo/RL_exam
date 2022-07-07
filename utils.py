import numpy as np
import math

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

    # ToDo: map 0 to C, 1 to D
    def pretty_print(self,max=50):
        addon = ''
        if len(self.s1_rounds) > max:
            addon = ' ...'
        print("{}\t{}{} = {}".format(self.s1.name, ' '.join(map(str, self.s1_rounds[:max])), addon, self.s1_score))
        print("{}\t{}{} = {}".format(self.s2.name, ' '.join(map(str, self.s2_rounds[:max])), addon, self.s2_score))