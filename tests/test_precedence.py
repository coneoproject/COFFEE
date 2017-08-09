from __future__ import absolute_import, print_function, division

from coffee import base as ast


def test_prod_div():
    tree = ast.Prod("a", ast.Div("1", "b"))

    assert tree.gencode() == "(a) * (((1) / (b)))"


def test_unary_op():
    tree = ast.Not(ast.Or("a", ast.And("b", "c")))

    assert tree.gencode() == "!((a) || (((b) && (c))))"
