import pytest
from coffee.visitor import Environment


@pytest.fixture
def env():
    return Environment({}, foo=1)


@pytest.fixture
def child(env):
    return Environment(env, bar=2)

@pytest.fixture
def shadow(env):
    return Environment(env, foo=2)


def test_lookup_key(env):
    assert env["foo"] == 1


def test_missing_key(env):
    with pytest.raises(KeyError):
        env["bar"]


def test_child_sees_parent(child):
    assert child["foo"] == 1
    assert child["bar"] == 2


def test_child_shadows_parent(shadow):
    assert shadow["foo"] == 2


def test_repr(child):
    new_env = eval(repr(child))

    assert child["foo"] == 1
    assert child["bar"] == 2
    assert new_env["foo"] == 1
    assert new_env["bar"] == 2


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
