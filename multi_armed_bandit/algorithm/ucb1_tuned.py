from typing import List

import numpy

from multi_armed_bandit.algorithm.ucb1 import UCB1
from multi_armed_bandit.arms.abstract_arm import Arm


class UCB1Tuned(UCB1):
    """
    The main feature of UCB1-Tuned is that it is takes into account the variance of each arm and not only empirical
    mean. More specifically, at round t the algorithm picks the i'th arm from max(mu^hat_i + sqrt(ln(t)/n_i *
    min(1/4, V_i(n_i)))). Where V_i(t)=sigma^2_i(t)+sqrt(2ln(t)/n_i(t)). The estimated of variance sigma^2_i(t) can
    be computed as usually by maintaining the empirical sum of squares of the reward, in addition to the empirical mean.
    """

    def __init__(self, arms: List[Arm]) -> None:
        super().__init__(arms)
        self._reward_squares = numpy.zeros(len(self._arms))

    def select_arm(self, iteration_number: int) -> int:
        if 0 in self._counts:
            return int(numpy.argmin(self._counts))
        variances = (self._reward_squares / self._counts) - self._states ** 2
        vi = variances + numpy.sqrt(2 * numpy.log(iteration_number + 1) / self._counts)
        vi = numpy.array([min(0.25, value) for value in vi])
        ucb_values = self._states + numpy.sqrt((2 * numpy.log(iteration_number + 1) / self._counts) * vi)
        return int(numpy.argmax(ucb_values))

    def _after_draw(self, reward: float, chosen_arm_index: int) -> None:
        self._reward_squares[chosen_arm_index] += (reward ** 2)
