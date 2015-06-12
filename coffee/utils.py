# This file is part of COFFEE
#
# COFFEE is Copyright (c) 2014, Imperial College London.
# Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Utility functions for the transformation of ASTs."""

import resource
import operator
from warnings import warn as warning
from copy import deepcopy as dcopy
from collections import defaultdict

from base import *
from coffee.visitors.inspectors import *


def increase_stack(loop_opts):
    """"Increase the stack size it the total space occupied by the kernel's local
    arrays is too big."""
    # Assume the size of a C type double is 8 bytes
    double_size = 8
    # Assume the stack size is 1.7 MB (2 MB is usually the limit)
    stack_size = 1.7*1024*1024

    size = 0
    for loop_opt in loop_opts:
        decls = loop_opt.decls.values()
        size += sum([reduce(operator.mul, d.sym.rank) for d in decls if d.sym.rank])

    if size*double_size > stack_size:
        # Increase the stack size if the kernel's stack size seems to outreach
        # the space available
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,
                                                       resource.RLIM_INFINITY))
        except resource.error:
            warning("Stack may blow up, and could not increase its size.")
            warning("In case of failure, lower COFFEE's licm level to 1.")


def unroll_factors(loops):
    """Return a dictionary, in which each entry maps an iteration space,
    identified by the iteration variable of a loop, to a suitable unroll factor.
    Heuristically, 1) inner loops are not unrolled to give the backend compiler
    a chance of auto-vectorizing them; 2) loops sizes must be a multiple of the
    unroll factor.

    :param loops: list of for loops for which a suitable unroll factor has to be
                  determined.
    """

    v = inspectors.DetermineUnrollFactors()

    unrolls = [v.visit(l, ret=v.default_retval()) for l in loops]
    ret = unrolls[0]
    ret.update(d for d in unrolls[1:])
    return ret


def postprocess(node):
    """Rearrange the Nodes in the AST rooted in ``node`` to improve the code quality
    when unparsing the tree."""

    class Process:
        start = None
        end = None
        decls = {}
        blockable = []
        _processed = []

        @staticmethod
        def mark(node):
            if Process.start is not None:
                Process._processed.append((node, Process.start, Process.end,
                                           Process.decls, Process.blockable))
            Process.start = None
            Process.end = None
            Process.decls = {}
            Process.blockable = []

    def init_decl(stmt):
        if not isinstance(stmt, Assign):
            return False
        lhs, rhs = stmt.children
        decl = Process.decls.get(lhs.symbol)
        if decl:
            decl.init = rhs
            return True
        return False

    def update(node, parent):
        index = parent.children.index(node)
        if Process.start is None:
            Process.start = index
        Process.end = index
        if not init_decl(node):
            Process.blockable.append(node)

    def make_blocks():
        for node, start, end, _, blockable in reversed(Process._processed):
            node.children[start:end+1] = [Block(blockable, open_scope=False)]

    def _postprocess(node, parent):
        if isinstance(node, FlatBlock) and str(node).isspace():
            update(node, parent)
        elif isinstance(node, (For, If, Switch, FunCall, FunDecl, FlatBlock, LinAlg,
                               Block, Root)):
            Process.mark(parent)
            for n in node.children:
                _postprocess(n, node)
            Process.mark(node)
        elif isinstance(node, Decl):
            if not (node.init and not isinstance(node.init, EmptyStatement)) and \
                    not node.sym.rank:
                Process.decls[node.sym.symbol] = node
            update(node, parent)
        elif isinstance(node, (Assign, Incr, Decr, IMul, IDiv)):
            update(node, parent)

    _postprocess(node, None)
    make_blocks()


def uniquify(exprs):
    """Iterate over ``exprs`` and return a list of expressions in which duplicates
    have been discarded. This function considers two expressions identical if they
    have the same string representation."""
    return dict([(str(e), e) for e in exprs]).values()


#####################################
# Functions to manipulate AST nodes #
#####################################


def ast_replace(node, to_replace, copy=False, mode='all'):
    """Given a dictionary ``to_replace`` s.t. ``{sym: new_sym}``, replace the
    various ``syms`` rooted in ``node`` with ``new_sym``.

    :param copy: if True, a deep copy of the replacing symbol is created.
    :param mode: either ``all``, in which case ``to_replace``'s keys are turned
                 into strings, and all of the occurrences are removed from the
                 AST; or ``symbol``, in which case only (all of) the references
                 to the symbols given in ``to_replace`` are replaced.
    """

    if mode == 'all':
        to_replace = dict(zip([str(s) for s in to_replace.keys()], to_replace.values()))
        __ast_replace = lambda n: to_replace.get(str(n))
    elif mode == 'symbol':
        __ast_replace = lambda n: to_replace.get(n)
    else:
        raise ValueError

    def _ast_replace(node, to_replace, n_replaced):
        replaced = {}
        for i, n in enumerate(node.children):
            replacing = __ast_replace(n)
            if replacing:
                replaced[i] = replacing if not copy else dcopy(replacing)
                n_replaced[str(replacing)] += 1
            else:
                _ast_replace(n, to_replace, n_replaced)
        for i, r in replaced.items():
            node.children[i] = r

    n_replaced = defaultdict(int)
    _ast_replace(node, to_replace, n_replaced)
    return n_replaced


def ast_remove(node, to_remove, mode='all'):
    """Remove the AST node ``to_remove`` from the tree rooted in ``node``.

    :param mode: either ``all``, in which case ``to_remove`` is turned into a
        string (if not a string already) and all of its occurrences are removed
        from the AST; or ``symbol``, in which case only (all of) the references
        to the provided ``to_remove`` node are cut away.
    """

    def _is_removable(n, tr):
        n, tr = (str(n), str(tr)) if mode == 'all' else (n, tr)
        return True if n == tr else False

    def _ast_remove(node, parent, index, tr):
        if _is_removable(node, tr):
            return -1
        if not node.children:
            return index
        _may_remove = [_ast_remove(n, node, i, tr) for i, n in enumerate(node.children)]
        if all([i > -1 for i in _may_remove]):
            # No removals occurred, so just return
            return index
        if all([i == -1 for i in _may_remove]):
            # Removed all of the children, so I'm also going to remove myself
            return -1
        alive = [i for i in _may_remove if i > -1]
        if len(alive) > 1:
            # Some children were removed, but not all of them, so no surgery needed
            return index
        # One child left, need to reattach it as child of my parent
        alive = alive[0]
        parent.children[index] = node.children[alive]
        return index

    if mode not in ['all', 'symbol']:
        raise ValueError

    try:
        if all(_ast_remove(node, None, None, tr) == -1 for tr in to_remove):
            return -1
    except TypeError:
        return _ast_remove(node, None, None, to_remove)


def ast_update_ofs(node, ofs, **kwargs):
    """Change the offsets of the iteration space variables of the symbols rooted
    in ``node``.

    :arg node: root AST node
    :arg ofs: a dictionary ``{'dim': value}``; `dim`'s offset is changed to `value`
    :arg kwargs: optional parameters to drive the transformation:
        * increase: `value` is added to the pre-existing offset, not substituted
    """
    increase = kwargs.get('increase', False)

    symbols = FindInstances(Symbol).visit(node, ret=FindInstances.default_retval())[Symbol]
    for s in symbols:
        new_offset = []
        for r, o in zip(s.rank, s.offset):
            if increase:
                val = ofs.get(r, 0)
                if isinstance(o[1], str) or isinstance(val, str):
                    new_o = "%s + %s" % (o[1], val)
                else:
                    new_o = o[1] + val
            else:
                new_o = ofs.get(r, o[1])
            new_offset.append((o[0], new_o))
        s.offset = tuple(new_offset)

    return node


def ast_update_rank(node, mapper):
    """Change the rank of the symbols rooted in ``node`` as prescribed by
    ``rank``.

    :arg node: Root AST node
    :arg mapper: Describe how to change the rank of a symbol.
    :type mapper: a dictionary. Keys can either be Symbols -- in which case
        values are interpreted as dimensions to be added to the rank -- or
        actual ranks (strings, integers) -- which means rank dimensions are
        replaced; for example, if mapper={'i': 'j'} and node='A[i] = B[i]',
        node will be transformed into 'A[j] = B[j]'
    """

    symbols = FindInstances(Symbol).visit(node, ret=FindInstances.default_retval())[Symbol]
    for s in symbols:
        if mapper.get(s.symbol):
            # Add a dimension
            s.rank = mapper[s.symbol] + s.rank
        else:
            # Try to replace dimensions
            s.rank = tuple([r if r not in mapper else mapper[r] for r in s.rank])

    return node


def ast_update_id(symbol, name, id):
    """Search for string ``name`` in Symbol ``symbol`` and replaces all of the
    occurrences of ``name`` with ``name_id``."""
    if not isinstance(symbol, Symbol):
        return
    new_name = "%s_%s" % (name, str(id))
    if name == symbol.symbol:
        symbol.symbol = new_name
    new_rank = [new_name if name == r else r for r in symbol.rank]
    symbol.rank = tuple(new_rank)


###############################################
# Functions to simplify creation of AST nodes #
###############################################


def ast_make_for(stmts, loop, copy=False):
    """Create a for loop having the same iteration space as  ``loop`` enclosing
    the statements in  ``stmts``. If ``copy == True``, then new instances of
    ``stmts`` are created"""
    wrap = Block(dcopy(stmts) if copy else stmts, open_scope=True)
    new_loop = For(dcopy(loop.init), dcopy(loop.cond), dcopy(loop.incr),
                   wrap, dcopy(loop.pragma))
    return new_loop


def ast_make_expr(op, nodes):
    """Create an ``Expr`` Node of type ``op``, with children given in ``nodes``."""

    def _ast_make_expr(nodes):
        return nodes[0] if len(nodes) == 1 else op(nodes[0], _ast_make_expr(nodes[1:]))

    try:
        return _ast_make_expr(nodes)
    except IndexError:
        return None


def ast_make_alias(node1, node2):
    """Return an object in which the LHS is represented by ``node1`` and the RHS
    by ``node2``, and ``node1`` is an alias for ``node2``; that is, ``node1``
    will point to the same memory region of ``node2``.

    :type node1: either a ``Decl`` or a ``Symbol``. If a ``Decl`` is provided,
                 the init field of the ``Decl`` is used to assign the alias.
    :type node2: either a ``Decl`` or a ``Symbol``. If a ``Decl`` is provided,
                 the symbol is extracted and used for the assignment.
    """
    if not isinstance(node1, (Decl, Symbol)):
        raise RuntimeError("Cannot assign a pointer to %s type" % type(node1))
    if not isinstance(node2, (Decl, Symbol)):
        raise RuntimeError("Cannot assign a pointer to %s type" % type(node1))

    # Handle node2
    if isinstance(node2, Decl):
        node2 = node2.sym
    node2.symbol = node2.symbol.strip('*')
    node2.rank, node2.offset, node2.loop_dep = (), (), ()

    # Handle node1
    if isinstance(node1, Symbol):
        node1.symbol = node1.symbol.strip('*')
        node1.rank, node1.offset, node1.loop_dep = (), (), ()
        return Assign(node1, node2)
    else:
        node1.init = node2
    return node1


###########################################################
# Functions to visit and to query properties of AST nodes #
###########################################################


def visit(node, parent=None, info_items=None):
    """Explore the AST rooted in ``node`` and collect various info, including:

    * Loop nests encountered - a list of tuples, each tuple representing a loop nest
    * Declarations - a dictionary {variable name (str): declaration (AST node)}
    * Symbols (dependencies) - a dictionary {symbol (AST node): [loops] it depends on}
    * Symbols (access mode) - a dictionary {symbol (AST node): access mode (WRITE, ...)}
    * String to Symbols - a dictionary {symbol (str): [(symbol, parent) (AST nodes)]}
    * Expressions - mathematical expressions to optimize (decorated with a pragma)

    :param node: AST root node of the visit
    :param parent: parent node of ``node``
    :param info_items: An optional list of information to gather,
        valid values are::

            - "symbols_dep"
            - "decls"
            - "exprs"
            - "fors"
            - "symbol_refs"
            - "symbols_mode"
    """
    info = {}

    if info_items is None:
        info_items = ['decls', 'symbols_dep', 'symbol_refs',
                      'symbols_mode', 'exprs', 'fors']
    if 'decls' in info_items:
        retval = SymbolDeclarations.default_retval()
        info['decls'] = SymbolDeclarations().visit(node, ret=retval)

    if 'symbols_dep' in info_items:
        deps = SymbolDependencies().visit(node, ret=SymbolDependencies.default_retval(),
                                          **SymbolDependencies.default_args)
        # Prune access mode:
        for k in deps.keys():
            if type(k) is not Symbol:
                del deps[k]
        info['symbols_dep'] = deps

    if 'exprs' in info_items:
        retval = FindCoffeeExpressions.default_retval()
        info['exprs'] = FindCoffeeExpressions().visit(node, parent=parent, ret=retval)

    if 'fors' in info_items:
        retval = FindLoopNests.default_retval()
        info['fors'] = FindLoopNests().visit(node, parent=parent, ret=retval)

    if 'symbol_refs' in info_items:
        retval = SymbolReferences.default_retval()
        info['symbol_refs'] = SymbolReferences().visit(node, parent=parent, ret=retval)

    if 'symbols_mode' in info_items:
        retval = SymbolModes.default_retval()
        info['symbols_mode'] = SymbolModes().visit(node, parent=parent, ret=retval)

    return info


def explore_operator(node):
    """Return a list of the operands composing the operation whose root is
    ``node``."""

    def _explore_operator(node, operator, children):
        for n in node.children:
            if n.__class__ == operator or isinstance(n, Par):
                _explore_operator(n, operator, children)
            else:
                children.append((n, node))

    children = []
    _explore_operator(node, node.__class__, children)
    return children


def inner_loops(node):
    """Find inner loops in the subtree rooted in ``node``."""

    return FindInnerLoops().visit(node)


def is_perfect_loop(loop):
    """Return True if ``loop`` is part of a perfect loop nest, False otherwise."""

    return CheckPerfectLoop().visit(loop)


def count(node, mode='symbol', read_only=False):
    """For each variable ``node``, count how many times it appears as involved
    in some expressions. For example, for the expression: ::

        ``a*(5+c) + b*(a+4)``

    return ::

        ``{a: 2, b: 1, c: 1}``

    :param node: Root of the visited AST
    :param mode: Accepted values are ['symbol', 'symbol_id', 'symbol_str']. This
                 parameter drives the counting and impacts the format of the
                 returned dictionary. In particular, the keys in such dictionary
                 will be:

                * mode == 'symbol': a tuple (symbol name, symbol rank)
                * mode == 'symbol_id': the symbol name only (a string). This \
                                       implies that all symbol occurrences \
                                       accumulate on the same counter, regardless \
                                       of iteration spaces. For example, if \
                                       under ``node`` appear both ``A[0]`` and \
                                       ``A[i][j]``, ``A`` will be counted twice
                * mode == 'symbol_str': a string representation of the symbol

    :param read_only: True if only variables on the right-hand side of a statement
                      should be counted; False if any appearance should be counted.
    """
    modes = ['symbol', 'symbol_id', 'symbol_str']
    if mode == 'symbol':
        key = lambda n: (n.symbol, n.rank)
    elif mode == 'symbol_id':
        key = lambda n: n.symbol
    elif mode == 'symbol_str':
        key = lambda n: str(n)
    else:
        raise RuntimeError("`Count` function got a wrong mode (valid: %s)" % modes)

    v = CountOccurences(key=key, only_rvalues=read_only)
    return v.visit(node, ret=v.default_retval())


def check_type(stmt, decls):
    """Check the types of the ``stmt``'s LHS and RHS. If they match as expected,
    return the type itself. Otherwise, an error is generated, suggesting an issue
    in either the AST itself (i.e., a bug inherent the AST) or, possibly, in the
    optimization process.

    :param stmt: the AST node statement to be checked
    :param decls: a dictionary from symbol identifiers (i.e., strings representing
                  the name of a symbol) to Decl nodes
    """
    v = SymbolReferences()
    lhs_symbol = v.visit(stmt.children[0], parent=stmt, ret=v.default_retval()).keys()[0]
    rhs_symbols = v.visit(stmt.children[1], parent=stmt, ret=v.default_retval()).keys()

    lhs_decl = decls[lhs_symbol]
    rhs_decls = [decls[s] for s in rhs_symbols if s in decls]

    type = lambda d: d.typ.replace('*', '')
    if any([type(lhs_decl) != type(rhs_decl) for rhs_decl in rhs_decls]):
        raise RuntimeError("Non matching types in %s" % str(stmt))

    return type(lhs_decl)


def find_expression(node, e_type, e_dims=None, e_symbol=None):
    """Wrapper of the FindExpression visitor."""
    finder = FindExpression(e_type, e_dims, e_symbol)
    exprs = finder.visit(node, env=FindExpression.default_env)
    if 'in_syms' in exprs:
        exprs.pop('in_syms')
    if 'in_itspace' in exprs:
        exprs.pop('in_itspace')
    return exprs


#######################################################################
# Functions to manipulate iteration spaces in various representations #
#######################################################################


class ItSpace():

    """A collection of routines to manipulate iteration spaces."""

    def __init__(self, mode=0):
        """Initialize an ItSpace object.

        :arg mode: Establish how an interation space is represented.
        :type mode: integer, allowed [0 (default), 1, 2]; respectively, an
            iteration space is represented as:
                * 0: a 2-tuple indicating the bounds of the accessed region
                * 1: a 2-tuple indicating size and offset of the accessed region
                * 2: a For loop object
        """
        assert mode in [0, 1, 2], "Invalid mode for ItSpace()"
        self.mode = mode

    def _convert_to_mode0(self, itspaces):
        if self.mode == 0:
            return itspaces
        elif self.mode == 1:
            return [(ofs, ofs+size) for size, ofs in itspaces]
        elif self.mode == 2:
            return [(l.start, l.end) for l in itspaces]

    def _convert_from_mode0(self, itspaces):
        if self.mode == 0:
            return itspaces
        elif self.mode == 1:
            return [(end-start, start) for start, end in itspaces]
        elif self.mode == 2:
            raise RuntimeError("Cannot convert from mode=0 to mode=2")

    def merge(self, itspaces, within=None):
        """Merge contiguous, possibly overlapping iteration spaces.
        For example (assuming ``self.mode = 0``): ::

            [(1,3), (4,6)] -> ((1,6),)
            [(1,3), (5,6)] -> ((1,3), (5,6))

        :arg within: an integer representing the distance between two iteration
            spaces to be considered adjacent. Defaults to 1.
        """
        itspaces = self._convert_to_mode0(itspaces)
        within = within or 1

        itspaces = sorted(tuple(set(itspaces)))
        merged_itspaces = []
        current_start, current_stop = itspaces[0]
        for start, stop in itspaces:
            if start - within > current_stop:
                merged_itspaces.append((current_start, current_stop))
                current_start, current_stop = start, stop
            else:
                # Ranges adjacent or overlapping: merge.
                current_stop = max(current_stop, stop)
        merged_itspaces.append((current_start, current_stop))

        itspaces = self._convert_from_mode0(merged_itspaces)
        return itspaces

    def intersect(self, itspaces):
        """Compute the intersection of multiple iteration spaces.
        For example (assuming ``self.mode = 0``): ::

            [(1,3)] -> ()
            [(1,3), (4,6)] -> ()
            [(1,3), (2,6)] -> (2,3)
        """
        itspaces = self._convert_to_mode0(itspaces)

        if len(itspaces) in [0, 1]:
            return ()
        itspaces = [set(range(i[0], i[1])) for i in itspaces]
        itspace = set.intersection(*itspaces)
        itspace = sorted(list(itspace)) or [0, -1]
        itspaces = [(itspace[0], itspace[-1]+1)]

        itspace = self._convert_from_mode0(itspaces)[0]
        return itspace

    def to_for(self, itspaces, dims=None, stmts=None):
        """Create ``For`` objects starting from an iteration space."""
        if not dims and self.mode == 2:
            dims = [l.dim for l in itspaces]
        elif not dims:
            dims = ['i%d' % i for i, j in enumerate(itspaces)]

        itspaces = self._convert_to_mode0(itspaces)

        loops = []
        body = Block(stmts or [], open_scope=True)
        for (start, stop), dim in reversed(zip(itspaces, dims)):
            new_for = For(Decl("int", dim, start), Less(dim, stop), Incr(dim, 1), body)
            loops.insert(0, new_for)
            body = Block([new_for], open_scope=True)

        return loops


#############################
# Generic utility functions #
#############################


any_in = lambda a, b: any(i in b for i in a)
flatten = lambda list: [i for l in list for i in l]
bind = lambda a, b: [(a, v) for v in b]
od_find_next = lambda a, b: a.values()[a.keys().index(b)+1]


def is_const_dim(d):
    return isinstance(d, int) or (isinstance(d, str) and d.isdigit())


def insert_at_elem(_list, elem, new_elem, ofs=0):
    ofs = _list.index(elem) + ofs
    new_elem = [new_elem] if not isinstance(new_elem, list) else new_elem
    for e in reversed(new_elem):
        _list.insert(ofs, e)
