# Iterated Prisoner's Dilemma

The Prisoner's dilemma is an example of a game that shows why two rational individuals might not cooperate, even if it appears in their best interest to do so.  
The most common example cites as follows:  
"Two members of a criminal gang are arrested and imprisoned. Each prisoner is in solitary confinement with no means of speaking to or exchanging messages with the other. The police admit they don't have enough evidence to convict the pair 
on the principal charge. They plan to sentence both to a year in prison on a lesser charge. Simultaneously, the police offer each prisoner a Faustian bargain."

Each player A and B has two actions: cooperate (C) with the other prisoner or defect (D), giving them up to the autorities. Both players decide on their action without knowing the action of the other player.

If two players play the game more than once in succession, they remember previous actions of their opponent and change their strategy accordingly, the game is called Iterated Prisoner's Dilemma (IPD).  
After each game, each player observes both the previous action of their opponent and their own previous action.

## Formalisation
The prisoner's dilemma is a 2-player Matrix game, with only four states and two choices of actions. Since the reward (payoff) for a given action depends also on the actions of the other player, it is an adversarial bandits problem.  
The four states consist of all possible combinations of actions for (A's previous action, B's previous action): $(C,C), (C,D), (D,C), (D,D)$. However, this setting can be simplified in some cases presented later.

At each time $t$, the players simultaneously choose their actions. The rewards depend on both actions, and they can be written as the payoff matrix reported above.  

### Q-Learning 
Our player (agent) learns the value function via Q-learning, a reinforcement learning technique developed in $1989$ that iteratively updates expected cumulative discounted rewards $Q$ given a state $s$ and an action $a$.

The action gets chosen using an $\varepsilon$-greedy policy.  
As the update rule does not depend on the current exploration but on the assumed optimal choice, Q-Learning does not require the current policy to converge towards the optimal policy, therefore doesn't need a decaying $\varepsilon$. 
Q-learning is able to generate an optimal policy even using only uniformly random actions, given sufficient iterations. However, the decay can be set as a parameter when instantiating the player, and can be of help in some situations.

### Opponent's deterministic strategies
Over time various deterministic strategies in the IPD game emerged. The ones that will be considered here are:
* *Always cooperate*: always cooperate, indepentently of period or observed actions. If played against itself, it always receives a reward of 3 in this setting.
* *Always defect*:  This strategy is by definition unexploitable and will always have at least the same average reward as the opposing strategy.
* *Random action*: As the name implies, this strategy plays a random action independent of period and observations. The probability to play a cooperation is set to $0.5$.
* *Tit-for-tat*: This strategy starts with a cooperation and copies the move of the opponent in the following periods.

## Goal
The questions this project tries to answer are: can a Q-Learning agent learn to play:
* vs a deterministic strategy?
* vs another Q-Learning agent?
