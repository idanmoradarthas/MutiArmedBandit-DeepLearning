import pytest

from multi_armed_bandit.arms.normal_arm import NormalArm


def test_draw_normal_arm():
    arm = NormalArm(0, 0, 1)
    assert pytest.approx(0.4967, 0.0001) == round(arm.draw(), 4)


def test_draw_normal_arm_size_2():
    arm = NormalArm(0, 0, 1)
    rewards = arm.draw(2)
    assert pytest.approx(0.4967, 0.0001) == round(rewards[0], 4)
    assert pytest.approx(-0.1382, 0.001) == round(rewards[1], 4)


def test_get_dict():
    arm = NormalArm(0, 0, 1)
    arm.set_probability(0.5)
    assert {"name": 0, "probability": 0.5} == arm.get_dict()


def test_eq():
    assert NormalArm(0, 0, 1) == NormalArm(0, 0, 1)


def test_not_eq():
    assert not NormalArm(0, 0, 1) == NormalArm(1, 0, 1)


def test_not_object():
    assert not 1 == NormalArm(1, 0, 1)
