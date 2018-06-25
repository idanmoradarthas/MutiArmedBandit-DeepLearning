import numpy

from multi_armed_bandit.algorithm.softmax_decreasing import SoftmaxDecreasing


class SoftMix(SoftmaxDecreasing):
    """
    The SoftMix slightly differs from the decreasing softmax since it uses a temperature decreasing with a log(t)/t
    factor instead of 1/t factor.
    see: Cesa-Bianchi, Nicolo, and Paul Fischer. "Finite-Time Regret Bounds for the Multiarmed Bandit Problem." In ICML,
    pp. 100-108. 1998.
    """

    def _get_temperature(self, iteration_number: int):
        return min(1.0, (self._temperature * numpy.log(iteration_number + 1)) / (iteration_number + 1))
