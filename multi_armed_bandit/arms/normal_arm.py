import numpy
from scipy.stats import norm

from multi_armed_bandit.arms.abstract_arm import Arm


class NormalArm(Arm):
    def __init__(self, name: int, mu: float, sigma: float) -> None:
        super().__init__(name)
        self._mu = mu
        self._sigma = sigma

    def _get_rewards(self, size: int) -> numpy.ndarray:
        """
        Extracts rewards from Normal distribution.
        :param size: number of draws.
        :return: rewards.
        """
        return norm.rvs(self._mu, self._sigma, size=size, random_state=42)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._name == other._name and self._probability == other._probability and self._mu == other._mu \
               and self._sigma == other._sigma
