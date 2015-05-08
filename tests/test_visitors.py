import pytest
from coffee.base import *
from coffee.visitors import *
from collections import Counter


@pytest.mark.parametrize("key",
                         [lambda x: x.symbol,
                          lambda x: x,
                          lambda x: x.symbol == "a"],
                         ids=["symbol_name", "symbol_identity",
                              "symbol_name_is_a"])
@pytest.mark.parametrize("symbols",
                         ["a",
                          "a,a",
                          "a,a,b",
                          "b"])
def test_count_occurences_block(key, symbols):
    v = CountOccurences(key=key)

    symbols = [Symbol(a) for a in symbols.split(",")]
    tree = Block(symbols)

    expect = Counter()
    for sym in symbols:
        expect[key(sym)] += 1

    assert v.visit(tree) == expect


@pytest.mark.parametrize("key",
                         [lambda x: x.symbol,
                          lambda x: x,
                          lambda x: x.symbol == "a"],
                         ids=["symbol_name", "symbol_identity",
                              "symbol_name_is_a"])
@pytest.mark.parametrize("only_rvalues",
                         [False, True],
                         ids=["all_children", "only_rvalues"])
@pytest.mark.parametrize("lvalue",
                         ["a", "b", "c"])
@pytest.mark.parametrize("rvalue",
                         ["a,a",
                          "a,b,c",
                          "c",
                          "b",
                          "d"])
def test_count_occurences_assign(key, only_rvalues,
                                 lvalue, rvalue):
    v = CountOccurences(key=key, only_rvalues=only_rvalues)

    rvalue = [Symbol(a) for a in rvalue.split(",")]

    lvalue = Symbol(lvalue)

    expect = Counter()

    if not only_rvalues:
        expect[key(lvalue)] += 1

    for sym in rvalue:
        expect[key(sym)] += 1

    rvalue = reduce(Prod, rvalue)

    tree = Assign(lvalue, rvalue)

    assert v.visit(tree) == expect


@pytest.mark.parametrize("structure",
                         ([],
                          [[]],
                          [None, []],
                          [None, [[], []]],
                          [None, [[None, [], [[]]]]]))
def test_find_inner_loops(structure):
    v = FindInnerLoops()

    inner_loops = []
    def build_loop(structure):
        ret = []
        for entry in structure:
            if entry is None:
                continue
            else:
                loop = Block([build_loop(entry)])
                ret.append(loop)
        loop = For(Symbol("a"), Symbol("b"), Symbol("c"),
                   Block(ret, open_scope=True))
        if ret == []:
            inner_loops.append(loop)
        return loop

    loop = build_loop(structure)

    expect = sorted(inner_loops)

    loops = v.visit(loop)

    assert sorted(loops) == expect


def test_check_perfect_loop():
    v = CheckPerfectLoop()

    a = Symbol("a")
    b = Symbol("b")
    loop = c_for("i", 10, [Assign(a, b)]).children[0]

    env = dict(in_loop=True, multiple_statements=False)
    assert v.visit(loop, env=env)

    loop2 = c_for("j", 10, [loop]).children[0]

    assert v.visit(loop2, env=env)

    loop3 = c_for("k", 10, [loop2, Assign(b, a)]).children[0]

    assert not v.visit(loop3, env=env)

    loop4 = c_for("k", 10, [Assign(a, b), Assign(b, a)]).children[0]

    assert v.visit(loop4, env=env)


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
