from typing import List

import numpy
import tensorflow

from multi_armed_bandit.arms.abstract_arm import Arm


class MABAlgorithm(object):
    """
    Multi-armed bandit - In probability theory, the multi-armed bandit problem (sometimes called the K- or N-armed
    bandit problem) is a problem in which a fixed limited set of resources must be allocated between competing
    (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially
    known at the time of allocation, and may become better understood as time passes or by allocating resources to the
    choice. taken from https://en.wikipedia.org/wiki/Multi-armed_bandit
    """

    def __init__(self, arms: List[Arm]) -> None:
        self._arms = arms
        self._states = numpy.zeros(len(self._arms))
        self._counts = numpy.zeros(len(self._arms))
        self._iterations = 0

    def select_arm(self, iteration_number: int) -> int:
        """
        The method that returns the index of the Arm that the algorithm selects on the current play.
        :return: index of chosen arm.
        """
        raise NotImplementedError

    def run_simulation(self, iterations) -> List[dict]:
        """
        Run simulation and update the probabilities to pull for each arm.
        :param iterations: number of iterations.
        """
        if iterations < 1:
            raise ValueError("Iterations must be positive")

        self._iterations = iterations

        number_of_arms = len(self._arms)

        results = []
        optimal_strategy_rewards = 0.0
        collected_rewards = 0.0

        rewards = numpy.zeros([number_of_arms, self._iterations])

        tensorflow.reset_default_graph()
        weights = tensorflow.Variable(tensorflow.ones([number_of_arms]))

        reward_holder = tensorflow.placeholder(shape=[1], dtype=tensorflow.float32)
        action_holder = tensorflow.placeholder(shape=[1], dtype=tensorflow.int32)
        responsible_weight = tensorflow.slice(weights, action_holder, [1])
        loss = -(tensorflow.log(responsible_weight) * reward_holder)
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.001)
        update = optimizer.minimize(loss)
        init = tensorflow.global_variables_initializer()
        tensorflow.set_random_seed(42)

        ww = numpy.zeros(number_of_arms)

        with tensorflow.Session() as sess:
            sess.run(init)

            for arm_index in range(0, number_of_arms):
                rewards[arm_index] = self._arms[arm_index].draw(self._iterations)

            self._states = numpy.zeros(number_of_arms)
            self._counts = numpy.zeros(number_of_arms)

            for iteration in range(0, self._iterations):
                chosen_arm_index = self.select_arm(iteration)
                reward = rewards[chosen_arm_index, iteration]

                _, _, ww = sess.run([update, responsible_weight, weights],
                                    feed_dict={reward_holder: [reward], action_holder: [chosen_arm_index]})

                self._counts[chosen_arm_index] += 1
                count = self._counts[chosen_arm_index]

                self._update_current_states(chosen_arm_index, count, reward)

                self._after_draw(reward, chosen_arm_index)

                collected_rewards += reward
                optimal_strategy_rewards += numpy.max(rewards[:, iteration])
                regret = optimal_strategy_rewards - collected_rewards

                results.append(
                    {"iteration": iteration, "chosen_arm": self._arms[chosen_arm_index].get_name(), "regret": regret,
                     "avg_regret": regret / (iteration + 1),
                     "avg_collected_rewards": collected_rewards / (iteration + 1)})

        exp = numpy.exp(ww)
        probabilities = exp / numpy.sum(exp, axis=0)

        for index in range(0, number_of_arms):
            self._arms[index].set_probability(probabilities[index])

        return results

    def _update_current_states(self, chosen_arm_index, count, reward) -> None:
        self._states[chosen_arm_index] = ((count - 1) / count) * self._states[chosen_arm_index] + (
                1 / count) * reward

    def _after_draw(self, reward: float, chosen_arm_index: int) -> None:
        """
        After process method.
        :param reward: the reward drawn.
        :param chosen_arm_index: the chosen arm index.
        """
        pass
