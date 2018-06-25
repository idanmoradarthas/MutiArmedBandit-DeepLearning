import numpy
from scipy.stats import bernoulli

from multi_armed_bandit.arms.abstract_arm import Arm


class BernoulliArm(Arm):
    def __init__(self, name: int, p: float) -> None:
        super().__init__(name)
        self._p = p

    def _get_rewards(self, size: int) -> numpy.ndarray:
        """
        Extracts rewards from Bernoulli distribution.
        :param size: number of draws.
        :return: rewards.
        """
        return bernoulli.rvs(self._p, size=size, random_state=42)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._name == other._name and self._probability == other._probability and self._p == other._p
