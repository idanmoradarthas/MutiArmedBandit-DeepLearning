import numpy

from multi_armed_bandit.algorithm.epsilon_greedy import EpsilonGreedy


class EpsilonFirst(EpsilonGreedy):
    """
    Consist of doing the exploration all at once at the beginning. epsilon * T first rounds the levers are pulled
    randomly. (1-epsilon)*T rounds the lever of the highest estimated mean is pulled.
    """

    def select_arm(self, iteration_number: int) -> int:
        if iteration_number < round(self._epsilon * self._iterations):
            return numpy.random.choice(len(self._arms))
        return int(numpy.argmax(self._states))
