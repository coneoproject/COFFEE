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
    assert v.visit(loop, **env)

    loop2 = c_for("j", 10, [loop]).children[0]

    assert v.visit(loop2, **env)

    loop3 = c_for("k", 10, [loop2, Assign(b, a)]).children[0]

    assert not v.visit(loop3, **env)

    loop4 = c_for("k", 10, [Assign(a, b), Assign(b, a)]).children[0]

    assert v.visit(loop4, **env)


@pytest.fixture
def block_aa():
    a = Symbol("a")
    return Block([a, a])


@pytest.fixture
def fun_aa_in_args():
    a = Symbol("a")
    return FunDecl("void", "foo", [a, a], Block([Assign(Symbol("b"),
                                                        Symbol("c"))]))


@pytest.fixture
def fun_aa_in_body(block_aa):
    return FunDecl("void", "foo", [], block_aa)


@pytest.mark.parametrize("tree",
                         [block_aa(),
                          fun_aa_in_args(),
                          fun_aa_in_body(block_aa())],
                         ids=["block-repeated-aa",
                              "fundecl-repeated-aa-args",
                              "fundecl-repeated-aa-body"])
def test_check_uniqueness(tree):
    v = CheckUniqueness()

    with pytest.raises(RuntimeError):
        v.visit(tree)


@pytest.mark.parametrize("tree",
                         [block_aa(),
                          fun_aa_in_args(),
                          fun_aa_in_body(block_aa())],
                         ids=["block-repeated-aa",
                              "fundecl-repeated-aa-args",
                              "fundecl-repeated-aa-body"])
def test_uniquify(tree):
    v = Uniquify()
    check = CheckUniqueness()

    new_tree = v.visit(tree)

    with pytest.raises(RuntimeError):
        check.visit(tree)

    assert check.visit(new_tree)


def test_symbol_declarations_decl():
    a = Symbol("a")

    tree = Decl("double", a)

    v = SymbolDeclarations()

    ret = v.visit(tree)

    assert set(ret.keys()) == set([a.symbol])


def test_symbol_declarations_block():
    a = Symbol("a")
    b = Symbol("b")

    tree = Block([Decl("int", a),
                  Decl("double", b)])

    v = SymbolDeclarations()

    ret = v.visit(tree)

    assert set(ret.keys()) == set([a.symbol, b.symbol])


def test_symbol_declarations_fundecl_args():
    a = Symbol("a")
    b = Symbol("b")

    body = Block([Assign(b, a)])

    tree = FunDecl("void", "foo", [Decl("double", a), Decl("double", b)],
                   body)

    v = SymbolDeclarations()

    ret = v.visit(tree)
    assert set(ret.keys()) == set([a.symbol, b.symbol])


def test_symbol_declarations_fundecl_body():
    a = Symbol("a")
    b = Symbol("b")

    body = Block([Decl("int", a),
                  Decl("double", b)])

    tree = FunDecl("void", "foo", [],
                   body)

    v = SymbolDeclarations()

    ret = v.visit(tree)
    assert set(ret.keys()) == set([a.symbol, b.symbol])


def test_symbol_declarations_fundecl_both():
    a = Symbol("a")
    b = Symbol("b")

    body = Block([Decl("int", a),
                  Assign(a, b)])

    tree = FunDecl("void", "foo", [Decl("int", b)],
                   body)

    v = SymbolDeclarations()

    ret = v.visit(tree)
    assert set(ret.keys()) == set([a.symbol, b.symbol])


def test_symbol_dependencies_no_nest():
    a = Symbol("a")

    tree = Assign(a, Symbol("1"))

    v = SymbolDependencies()

    ret = v.visit(tree, **SymbolDependencies.default_args)

    assert ret[a] == []


def test_symbol_dependencies_single_loop():

    a = Symbol("a")
    i = Symbol("i")
    tree = c_for(i, 2, [Assign(a, i)])

    v = SymbolDependencies()

    ret = v.visit(tree, **SymbolDependencies.default_args)

    assert ret[a] == [tree.children[0]]


def test_symbol_dependencies_read_single_loop():
    a = Symbol("a", rank=("i", ))
    b = Symbol("b")
    tree = c_for("i", 2, [Assign(b, a)])

    v = SymbolDependencies()
    ret = v.visit(tree, **SymbolDependencies.default_args)

    assert ret[b] == [tree.children[0]]

    assert ret[a] == [tree.children[0]]


def test_symbol_dependencies_double_loop():
    a = Symbol("a", rank=("i", ))
    b = Symbol("b")
    tree = c_for("i", 2, [c_for("j", 1, [IMul(b, a)])])
    v = SymbolDependencies()

    ret = v.visit(tree, **SymbolDependencies.default_args)

    assert ret[b] == [tree.children[0], tree.children[0].body[0]]
    assert ret[a] == [tree.children[0]]


def test_symbol_dependencies_write_then_read_inner_loop():
    a = Symbol("a")
    a2 = Symbol("a")
    b = Symbol("b")
    tree = c_for("i", 2, [c_for("j", 1, [Assign(a, 1)]),
                          c_for("j", 2, [Assign(b, a2)])])

    v = SymbolDependencies()

    ret = v.visit(tree, **SymbolDependencies.default_args)

    assert ret[a2] == [tree.children[0]]
    assert ret[a] == [tree.children[0], tree.children[0].body[0].children[0]]
    assert ret[b] == [tree.children[0], tree.children[0].body[1].children[0]]


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
