from typing import List

import numpy

from multi_armed_bandit.algorithm.abstract_algorithm import MABAlgorithm
from multi_armed_bandit.arms.abstract_arm import Arm


class EpsilonGreedy(MABAlgorithm):
    """
    Epsilon-greedy strategy - The best lever is selected for a proportion 1 - epsilon of the trails, and a lever is
    selected at random (with uniform probability) for a proportion epsilon.
    """

    def __init__(self, arms: List[Arm], epsilon: float) -> None:
        super().__init__(arms)

        if not (0.0 <= epsilon < 1.0):
            raise ValueError("epsilon is not between 0 and 1")

        self._epsilon = epsilon

    def select_arm(self, iteration_number: int) -> int:
        """
        Select the best arm based on the strategy of epsilon proportion of time random and the rest are the
        strongest arm.
        :return: index of chosen arm.
        """
        if numpy.random.random() > self._get_epsilon(iteration_number):
            return int(numpy.argmax(self._states))
        else:
            return numpy.random.choice(len(self._arms))

    def _get_epsilon(self, iteration_number: int) -> float:
        return self._epsilon

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._epsilon == other._epsilon and self._arms == other._arms
