from multi_armed_bandit.algorithm.softmax import Softmax


class SoftmaxDecreasing(Softmax):
    """
    Modified the same way as epsilon-greedy strategy into decreasing softmax where the temperature decreases with the
    number of rounds played. The decreasing softmax is identical to the softmax but with a temperature tua_t is
    min(1, tau_0/t) where tua_0 is the initial temperature the policy gets and t is the iteration number.
    """

    def _get_temperature(self, iteration_number: int):
        return min(1.0, self._temperature / (iteration_number + 1))
