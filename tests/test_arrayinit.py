from __future__ import absolute_import, print_function, division

from coffee import base as ast
import numpy as np


def test_funcall_in_arrayinit():
    tree = ast.ArrayInit(np.asarray([ast.FunCall("foo"), ast.Symbol("bar")]))

    assert tree.gencode() == "{foo(), bar}"
