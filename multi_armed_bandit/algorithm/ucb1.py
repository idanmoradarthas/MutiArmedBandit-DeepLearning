import numpy

from multi_armed_bandit.algorithm.abstract_algorithm import MABAlgorithm


class UCB1(MABAlgorithm):
    """
    Upper Confidence Bounds. Initially, each arm is played once. Afterwards at round t, the algorithm greedily picks
    the i'th arm from max(mu^hat_i(t)+sqrt(2*ln(t)/n_i)). Where mu^hat_i(t) is the estimated mean reward of the i'th
    arm, t is the iteration number and n_i is the number of times the arm was played.
    """

    def select_arm(self, iteration_number: int) -> int:
        if 0 in self._counts:
            return int(numpy.argmin(self._counts))
        ucb_values = self._states + numpy.sqrt(2 * numpy.log(iteration_number + 1) / self._counts)
        return int(numpy.argmax(ucb_values))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self._arms == other._arms
