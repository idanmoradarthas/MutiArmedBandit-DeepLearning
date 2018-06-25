from typing import List

import numpy

from multi_armed_bandit.algorithm.abstract_algorithm import MABAlgorithm
from multi_armed_bandit.arms.abstract_arm import Arm


class UCB2(MABAlgorithm):
    """
    see: https://webdocs.cs.ualberta.ca/~games/go/seminar/notes/2007/slides_ucb.pdf
    """

    def __init__(self, arms: List[Arm], alpha: float) -> None:
        super().__init__(arms)

        if not (0.0 <= alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1")

        self._alpha = alpha
        self._r = numpy.zeros(len(self._arms))
        self._next_iteration_to_calc = 0
        self._current_arm_index = 0

    def select_arm(self, iteration_number: int) -> int:
        if 0 in self._counts:
            chosen_arm = int(numpy.argmin(self._counts))
            self._r[chosen_arm] += 1
            return chosen_arm

        if self._next_iteration_to_calc > iteration_number + 1:
            return self._current_arm_index

        ucb_values = self._states + numpy.sqrt(
            ((1 + self._alpha) * numpy.log(numpy.e * (iteration_number + 1) / self._get_tau())) / 2 * self._get_tau())
        chosen_arm = int(numpy.argmax(ucb_values))
        self._update_next_time_to_calc(chosen_arm)
        return chosen_arm

    def _get_tau(self) -> numpy.ndarray:
        return numpy.ceil((1 + self._alpha) ** self._r)

    def _update_next_time_to_calc(self, arm_index: int) -> None:
        self._current_arm_index = arm_index
        self._next_iteration_to_calc += max(1, self._get_tau_for_single_r(
            self._r[arm_index] + 1) - self._get_tau_for_single_r(self._r[arm_index]))
        self._r[arm_index] += 1

    def _get_tau_for_single_r(self, r: int) -> int:
        return int(numpy.ceil((1 + self._alpha) ** r))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._alpha == other._alpha and self._arms == other._arms
