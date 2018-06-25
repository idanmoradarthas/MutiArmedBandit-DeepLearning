from multi_armed_bandit.algorithm.epsilon_greedy import EpsilonGreedy


class EpsilonDecreasing(EpsilonGreedy):
    """
    The lever with the highest estimated mean is always pulled expect when a random lever is pull instead with
    epsilon_t frequency. epsilon_t is defined as min(1, epsilon_0/t) where epsilon_0 is the initial epsilon the policy
    gets and t is the iteration number.
    """

    def _get_epsilon(self, iteration_number):
        return min(1.0, self._epsilon / (iteration_number + 1))
