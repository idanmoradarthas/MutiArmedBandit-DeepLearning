from typing import Union

import numpy


class Arm(object):
    def __init__(self, name: Union[int, float, str]) -> None:
        self._name = name
        self._probability = 0.0

    def get_name(self) -> int:
        """
        Returns the arm name.
        :return: the arm name.
        """
        return self._name

    def set_probability(self, probability) -> None:
        """
        Sets the arm probability.
        :param probability: the arm probability.
        """
        self._probability = probability

    def get_probability(self) -> float:
        """
        Returns the arm probability.
        :return: the arm probability.
        """
        return self._probability

    def draw(self, size: int = 1) -> Union[float, numpy.ndarray]:
        """
        Pull the level of the arm.
        :param size: number of draws.
        :return: the rewards from the draw.
        """
        rewards = self._get_rewards(size)
        if size == 1:
            return rewards[0]
        return rewards

    def _get_rewards(self, size: int) -> numpy.ndarray:
        """
        Extracts rewards per arm distribution.
        :param size: number of draws.
        :return: rewards.
        """
        raise NotImplementedError

    def get_dict(self) -> dict:
        """
        Retrn the arm values as dictionary with name and probability.
        :return:
        """
        return {"name": self.get_name(), "probability": float(self.get_probability())}
