import pandas

from multi_armed_bandit.algorithm.epsilon_first import EpsilonFirst
from multi_armed_bandit.algorithm.exp3 import EXP3
from multi_armed_bandit.algorithm.greedy_mix import GreedyMix
from multi_armed_bandit.algorithm.softmax import Softmax
from multi_armed_bandit.algorithm.softmax_decreasing import SoftmaxDecreasing
from multi_armed_bandit.algorithm.softmix import SoftMix
from multi_armed_bandit.algorithm.ucb1 import UCB1
from multi_armed_bandit.algorithm.ucb1_tuned import UCB1Tuned
from multi_armed_bandit.algorithm.ucb2 import UCB2
from multi_armed_bandit.arms.normal_arm import NormalArm
from multi_armed_bandit.algorithm.epsilon_decreasing import EpsilonDecreasing

if __name__ == '__main__':
    arms = [NormalArm(0, 0.3, 1), NormalArm(1, 0.5, 1), NormalArm(2, 1, 1)]
    epsilon = 0.5
    algorithm = UCB2(arms, epsilon)
    results = algorithm.run_simulation(1000)
    # df = pandas.DataFrame(results)
    # df.to_csv("results.csv", index=False)
