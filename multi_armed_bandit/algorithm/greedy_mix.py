import numpy

from multi_armed_bandit.algorithm.epsilon_decreasing import EpsilonDecreasing


class GreedyMix(EpsilonDecreasing):
    """
    GreedyMix slightly differs from the epsilon-decreasing strategy as just presented because it uses a decreasing
    method of log(t) / t instead of 1/t, where t is the iteratation number.
    see: Cesa-Bianchi, Nicolo, and Paul Fischer. "Finite-Time Regret Bounds for the Multiarmed Bandit Problem." In ICML,
    pp. 100-108. 1998.
    """

    def _get_epsilon(self, iteration_number):
        return min(1.0, (self._epsilon * numpy.log(iteration_number + 1)) / (iteration_number + 1))
