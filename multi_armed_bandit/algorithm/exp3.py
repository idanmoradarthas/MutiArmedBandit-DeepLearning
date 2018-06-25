from typing import List

import numpy

from multi_armed_bandit.algorithm.abstract_algorithm import MABAlgorithm
from multi_armed_bandit.arms.abstract_arm import Arm


class EXP3(MABAlgorithm):
    """
    Exponential weight algorithm. The probability of choosing the lever k ar round t is:
    p_i(t) = (1 - gamma) * (weight_i(t)/ sum^K_j=i(weight_j(t))) + (gamma / K). Where K is the number of arms.
    if lever i has been pulled: weight_i(t + 1) = weight_i(t) * e^((gamma * r_i(t))/(probability_i(t) * K)).
    Where r_i(t) is the drawn reward in round t. Otherwise weight_i(t + 1) = weight_i(t)
    """

    def __init__(self, arms: List[Arm], gamma: float) -> None:
        super().__init__(arms)

        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma must be between 0 and 1")
        self._gamma = gamma
        self._weights = numpy.ones(len(self._arms))
        self._probabilities = numpy.zeros(len(self._arms))

    def select_arm(self, iteration_number: int) -> int:
        self._probabilities = ((1 - self._gamma) * self._weights / numpy.sum(self._weights, axis=0)) + (
                self._gamma / len(self._arms))
        return int(numpy.argmax(numpy.random.multinomial(1, self._probabilities)))

    def _after_draw(self, reward: float, chosen_arm_index: int) -> None:
        probability = self._probabilities[chosen_arm_index]
        growth_factor = numpy.exp(self._gamma * (reward / (probability * len(self._arms))))
        self._weights[chosen_arm_index] = self._weights[chosen_arm_index] * growth_factor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._gamma == other._gamma and self._arms == other._arms
