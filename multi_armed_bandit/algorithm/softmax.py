import warnings
from typing import List

import numpy

from multi_armed_bandit.algorithm.abstract_algorithm import MABAlgorithm
from multi_armed_bandit.arms.abstract_arm import Arm


class Softmax(MABAlgorithm):
    """
    Also known as Boltzmann Exploration. Each arm is assign with probability that is proportional to its average reward.
    Arms with greater empirical mean are therefore picked with higher probability. The probability to be picked is
    defined: p_i(t+1) = (e^(mu^hat_i(t)/tua)/sum^K_j=1(e^(mu^hat_j(t)/tua))). Where K is number of arms, mu^hat_i(t) is
    the estimated mean reward of the i'th arm, and tua is a temperature parameter controlling the randomness of the
    choice. when tau is close to 0 the policy acts like pure greedy, and as tau tends to infinity, the algorithm picks
    arms uniformly at random.
    """

    def __init__(self, arms: List[Arm], temperature) -> None:
        super().__init__(arms)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature

    def select_arm(self, iteration_number: int) -> int:
        exp = numpy.exp(self._states / self._get_temperature(iteration_number))
        with warnings.catch_warnings(record=True) as w:
            probabilities = exp / numpy.sum(exp, axis=0)
            if w:
                print()
        return int(numpy.argmax(numpy.random.multinomial(1, probabilities)))

    def _get_temperature(self, iteration_number: int):
        return self._temperature

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._temperature == other._temperature and self._arms == other._arms
