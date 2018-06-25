from multi_armed_bandit.arms.bernoulli_arm import BernoulliArm


def test_draw_bernoulli_arm():
    arm = BernoulliArm(0, 0.5)
    assert 0 == arm.draw()


def test_draw_bernoulli_arm_size_2():
    arm = BernoulliArm(0, 0.5)
    rewards = arm.draw(2)
    assert 0 == rewards[0]
    assert 1 == rewards[1]


def test_eq():
    assert BernoulliArm(0, 0.5) == BernoulliArm(0, 0.5)


def test_not_eq():
    assert not BernoulliArm(1, 0.5) == BernoulliArm(0, 0.5)


def test_not_object():
    assert not 1 == BernoulliArm(0, 0.5)
