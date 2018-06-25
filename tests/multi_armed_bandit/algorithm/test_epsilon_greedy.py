import pytest

from multi_armed_bandit.algorithm.epsilon_greedy import EpsilonGreedy
from multi_armed_bandit.arms.normal_arm import NormalArm


def test_epsilon_greedy_algorithm():
    arms = [NormalArm(name=0, mu=0.5, sigma=1.0), NormalArm(1, 0.3, 1.0), NormalArm(2, 1.0, 1.0)]
    algorithm = EpsilonGreedy(arms, 0.9)
    algorithm.run_simulation(1000)
    assert pytest.approx(0.31, 0.1) == round(arms[0].get_probability(), 2)
    assert pytest.approx(0.30, 0.1) == round(arms[1].get_probability(), 2)
    assert pytest.approx(0.38, 0.1) == round(arms[2].get_probability(), 2)


def test_epsilon_out_of_range():
    with pytest.raises(ValueError):
        EpsilonGreedy([NormalArm(name=0, mu=0.5, sigma=1.0)], 0.0)

    with pytest.raises(ValueError):
        EpsilonGreedy([NormalArm(name=0, mu=0.5, sigma=1.0)], -5.0)

    with pytest.raises(ValueError):
        EpsilonGreedy([NormalArm(name=0, mu=0.5, sigma=1.0)], 1.1)


def test_one_arm():
    arm = NormalArm(name=0, mu=0.5, sigma=1.0)
    algorithm = EpsilonGreedy([arm], 0.9)
    algorithm.run_simulation(1000)
    assert 1.0 == arm.get_probability()


def test_eq():
    arm = NormalArm(name=0, mu=0.5, sigma=1.0)
    assert EpsilonGreedy([arm], 0.9) == EpsilonGreedy([arm], 0.9)


def test_not_object():
    arm = NormalArm(name=0, mu=0.5, sigma=1.0)
    assert not (1 == EpsilonGreedy([arm], 0.3))


def test_iteration_negative():
    arm = NormalArm(name=0, mu=0.5, sigma=1.0)
    algorithm = EpsilonGreedy([arm], 0.9)

    with pytest.raises(ValueError):
        algorithm.run_simulation(0)

    with pytest.raises(ValueError):
        algorithm.run_simulation(-1)
