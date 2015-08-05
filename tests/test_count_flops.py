import pytest
from coffee.base import *
from coffee.visitors import EstimateFlops


@pytest.fixture(scope="module")
def v():
    return EstimateFlops()


@pytest.fixture(scope="module",
                params=[Prod, Div, Sum, Sub])
def binop(request):
    a = Symbol("a")
    b = Symbol("b")
    return request.param(a, b)


@pytest.fixture(scope="module",
                params=[Prod, Div, Sum, Sub])
def nested_binop(request, binop):
    c = Symbol("c")
    return request.param(binop, c)


@pytest.fixture(scope="module",
                params=[AVXProd, AVXDiv, AVXSum, AVXSub])
def avxbinop(request):
    a = Symbol("a")
    b = Symbol("b")
    return request.param(a, b)


@pytest.fixture(scope="module",
                params=[AVXProd, AVXDiv, AVXSum, AVXSub])
def nested_avxbinop(request, avxbinop):
    c = Symbol("c")
    return request.param(avxbinop, c)


def test_binop(v, binop):
    assert v.visit(binop) == 1


def test_avxbinop(v, avxbinop):
    assert v.visit(avxbinop) == 4


def test_nested_binop(v, nested_binop):
    assert v.visit(nested_binop) == 2


def test_nested_avxbinop(v, nested_avxbinop):
    assert v.visit(nested_avxbinop) == 8


def test_for(v, binop):
    tree = c_for("i", 10, [binop])
    assert v.visit(tree) == 10


def test_nested_for(v, binop):
    tree = c_for("i", 10, [binop])
    tree = c_for("j", 11, [tree])
    assert v.visit(tree) == 10*11


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
