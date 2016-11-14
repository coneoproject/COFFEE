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

"""Utility functions for the inspection, transformation, and creation of ASTs."""

from __future__ import absolute_import, print_function, division
from six import iterkeys, iteritems
from six.moves import zip

from copy import deepcopy as dcopy
from collections import defaultdict, OrderedDict, namedtuple

import networkx as nx

from coffee.base import *
from coffee.visitors.inspectors import *


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


def ast_make_expr(op, nodes, balance=True):
    """Create an ``Expr`` Node of type ``op``, with children given in ``nodes``."""

    def _ast_make_expr(nodes):
        return nodes[0] if len(nodes) == 1 else op(nodes[0], _ast_make_expr(nodes[1:]))

    def _ast_make_bal_expr(nodes):
        half = len(nodes) // 2
        return nodes[0] if len(nodes) == 1 else op(_ast_make_bal_expr(nodes[:half]),
                                                   _ast_make_bal_expr(nodes[half:]))

    if len(nodes) == 0:
        return None
    elif balance:
        return _ast_make_bal_expr(nodes)
    else:
        return _ast_make_expr(nodes)


def ast_make_alias(node, alias_name):
    """
    Create an alias of ``node`` (must be of type Decl). The alias symbol is
    given the name ``alias_name``. For example: ::

        (node, alias_name) --> output
        (double * a, b) --> double * b = a
        (double a[1], b) --> double * b = a
        (double a[1][1], b) --> double (*b)[1] = a
    """
    assert isinstance(node, Decl)

    pointers = list(node.pointers)
    if len(node.size) == 1:
        pointers += ['']
    if len(node.size) > 1:
        symbol = SymbolIndirection(alias_name, node.size[1:])
    else:
        symbol = Symbol(alias_name, node.size[1:])

    return Decl(node.typ, symbol, node.lvalue.symbol, qualifiers=node.qual,
                scope=node.scope, pointers=pointers)


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
        for k in list(iterkeys(deps)):
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


def loops_analysis(node, key='default', value='default'):
    """Perform loop dependence analysis in the AST rooted in ``node``. Return
    a dictionary mapping symbols to loops they depend on.

    :arg key: any value in ['default', 'urepr', 'symbol']. With 'urepr' and
        'symbol' different instances of the same Symbol are represented by
        a single entry in the returned dictionary.
    :arg value: any value in ['default', 'dim']. If 'dim' is specified, then
        loop iteration dimensions are used in place of the actual object.
    """

    if key == 'default':
        gen_key = lambda s: s
    elif key == 'urepr':
        gen_key = lambda s: s.urepr
    elif key == 'symbol':
        gen_key = lambda s: s.symbol
    else:
        raise RuntimeError("Illegal key=%s for loop dependence analysis" % key)

    if value == 'default':
        gen_value = lambda d: set(d)
    elif value == 'dim':
        gen_value = lambda d: {l.dim for l in d}
    else:
        raise RuntimeError("Illegal value=%s for loop dependence analysis" % value)

    symbols_dep = visit(node, info_items=['symbols_dep'])['symbols_dep']
    lda = defaultdict(set)
    for s, dep in symbols_dep.items():
        lda[gen_key(s)] |= gen_value(dep)

    return lda


def reachability_analysis(node, decls=None):
    """Perform reachability analysis in the AST rooted in ``node``. Return
    a dictionary mapping symbols to scopes in which they are visible.

    :param decls: an iterator of :class:`Decl`s which are known to be visible
        within ``node``
    """
    symbols_vis, scopes = SymbolVisibility().visit(node)
    for d in decls:
        symbols_vis[d].extend(scopes)
    return symbols_vis


def explore_operator(node):
    """Return a list of the operands composing the operation whose root is
    ``node``."""

    def _explore_operator(node, operator, children):
        for n in node.children:
            if n.__class__ == operator:
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


def in_written(node, key='default'):
    """Return a list of symbols written in ``node``.

    :arg key: any value in ['default', 'urepr', 'symbol']. With 'urepr' and
        'symbol' different instances of the same Symbol are represented by
        a single entry in the returned dictionary.
    """

    if key == 'default':
        gen_key = lambda s: s
    elif key == 'urepr':
        gen_key = lambda s: s.urepr
    elif key == 'symbol':
        gen_key = lambda s: s.symbol
    else:
        raise RuntimeError("Illegal key=%s for loop dependence analysis" % key)

    found = []
    writers = FindInstances(Writer).visit(node, ret=FindInstances.default_retval())
    for type, stmts in writers.items():
        for stmt in stmts:
            found.append(gen_key(stmt.lvalue))

    return found


def count(node, mode='urepr', read_only=False):
    """Count the occurrences of all variables appearing in ``node``. For example,
    for the expression: ::

        ``a*(5+c) + b*(a+4)``

    return ::

        ``{a: 2, b: 1, c: 1}``

    :param node: The root of the AST visited
    :param mode: Set the key in the returned dictionary. Accepted values
        are ['urepr', 'symbol_id'], where:
        * mode == 'urepr': (default) use the symbol representation as key
        * mode == 'symbol_id': use the symbol name as key, thus ignoring
            any iteration space or offset. For example, if both A[0] and A[i]
            appear in ``node``, return {A: 2, ...} (assuming no other
            occurrences of A)
    :param read_only: True if only variables on the right hand side of a statement
                      should be counted; False if any appearance should be counted.
    """
    modes = ['urepr', 'symbol_id']
    if mode == 'urepr':
        key = lambda n: n.urepr
    elif mode == 'symbol_id':
        key = lambda n: n.symbol
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
    lhs_symbol = v.visit(stmt.lvalue, parent=stmt, ret=v.default_retval()).keys()[0]
    rhs_symbols = v.visit(stmt.rvalue, parent=stmt, ret=v.default_retval()).keys()

    lhs_decl = decls[lhs_symbol]
    rhs_decls = [decls[s] for s in rhs_symbols if s in decls]

    type = lambda d: d.typ.replace('*', '')
    if any([type(lhs_decl) != type(rhs_decl) for rhs_decl in rhs_decls]):
        raise RuntimeError("Non matching types in %s" % str(stmt))

    return type(lhs_decl)


def find_expression(node, type=None, dims=None, in_syms=None, out_syms=None):
    """Wrapper of the FindExpression visitor."""
    finder = FindExpression(type, dims, in_syms, out_syms)
    exprs = finder.visit(node, ret=FindExpression.default_retval())
    if 'cleaned' in exprs:
        exprs.pop('cleaned')
    if 'in_syms' in exprs:
        exprs.pop('in_syms')
    if 'out_syms' in exprs:
        exprs.pop('out_syms')
    if 'inner_syms' in exprs:
        exprs.pop('inner_syms')
    if 'in_itspace' in exprs:
        exprs.pop('in_itspace')
    return exprs


#######################################################################
# Functions to manipulate iteration spaces in various representations #
#######################################################################


class ItSpace(object):

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
            return tuple(itspaces)
        elif self.mode == 1:
            return tuple((ofs, ofs+size) for size, ofs in itspaces)
        elif self.mode == 2:
            return tuple((l.start, l.end) for l in itspaces)

    def _convert_from_mode0(self, itspaces):
        if self.mode == 0:
            return itspaces
        elif self.mode == 1:
            return [Region(end-start, start) for start, end in itspaces]
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

        if len(itspaces) == 0:
            return ()
        elif len(itspaces) > 1:
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
        for (start, stop), dim in reversed(list(zip(itspaces, dims))):
            new_for = For(Decl("int", dim, start), Less(dim, stop), Incr(dim, 1), body)
            loops.insert(0, new_for)
            body = Block([new_for], open_scope=True)

        return loops


###############################################################
# Utilities for tracking the global impact of transformations #
###############################################################


class StmtTracker(OrderedDict):

    """Track the location of generic statements in an abstract syntax tree.

    Each key in the dictionary is a string representing a symbol. As such,
    StmtTracker can be used only in SSA scopes. Each entry in the dictionary
    is a 4-tuple containing information about the symbol: ::

        (statement, declaration, closest_for, place)

    whose semantics is, respectively, as follows:

        * The AST node whose ``str(lvalue)`` is used as dictionary key
        * The AST node of the symbol declaration
        * The AST node of the closest loop enclosing the statement
        * The parent of the closest loop
    """

    class StmtInfo(object):
        """Simple container class defining ``StmtTracker`` values."""

        INFO = ['stmt', 'decl', 'loop', 'place']

        def __init__(self, **kwargs):
            for k, v in iteritems(kwargs):
                assert(k in self.__class__.INFO)
                setattr(self, k, v)

    def __init__(self):
        super(StmtTracker, self).__init__()
        self.byvalue = OrderedDict()

    def __setitem__(self, key, value):
        if not isinstance(value, self.StmtInfo):
            if not isinstance(value, tuple):
                raise RuntimeError("StmtTracker accepts tuple or StmtInfo objects")
            assert len(self.StmtInfo.INFO) == len(value)
            value = self.StmtInfo(**dict(zip(self.StmtInfo.INFO, value)))
        self.byvalue[value.stmt.rvalue.urepr] = key
        return OrderedDict.__setitem__(self, key, value)

    def update_stmt(self, sym, **kwargs):
        """Given the symbol ``sym``, it updates information related to it as
        specified in ``kwargs``. If ``sym`` is not present, return ``None``.
        ``kwargs`` is based on the following special keys:

            * "stmt": change the statement
            * "decl": change the declaration
            * "loop": change the closest loop
            * "place": change the parent the closest loop
        """
        if sym not in self:
            return None
        for k, v in iteritems(kwargs):
            assert(k in self.StmtInfo.INFO)
            setattr(self[sym], k, v)

    def update_loop(self, loop_a, loop_b):
        """Replace all occurrences of ``loop_a`` with ``loop_b`` in all entries."""

        for sym, sym_info in self.items():
            if sym_info.loop == loop_a:
                self.update_stmt(sym, **{'loop': loop_b})

    def get_symbol(self, value):
        """Return the key associated to the provided ``value``, or None if not
        present."""
        return self.byvalue.get(value.urepr)

    @property
    def stmt(self, sym):
        return self[sym].stmt if self.get(sym) else None

    @property
    def decl(self, sym):
        return self[sym].decl if self.get(sym) else None

    @property
    def loop(self, sym):
        return self[sym].loop if self.get(sym) else None

    @property
    def place(self, sym):
        return self[sym].place if self.get(sym) else None

    @property
    def all_stmts(self):
        return set((stmt_info.stmt for stmt_info in self.values() if stmt_info.stmt))

    @property
    def all_places(self):
        return set((stmt_info.place for stmt_info in self.values() if stmt_info.place))

    @property
    def all_loops(self):
        return set((stmt_info.loop for stmt_info in self.values() if stmt_info.loop))


class ExpressionGraph(object):

    """Track read-after-write dependencies between symbols."""

    def __init__(self, node):
        """Initialize the ExpressionGraph.

        :param node: root of the AST visited to initialize the ExpressionGraph.
        """
        self.deps = nx.DiGraph()
        writes = FindInstances(Writer).visit(node, ret=FindInstances.default_retval())
        for type, nodes in writes.items():
            for n in nodes:
                if isinstance(n.rvalue, EmptyStatement):
                    continue
                self.add_dependency(n.lvalue, n.rvalue)

    def add_dependency(self, sym, expr):
        """Add dependency between ``sym`` and symbols appearing in ``expr``."""
        retval = FindInstances.default_retval()
        expr_symbols = FindInstances(Symbol).visit(expr, ret=retval)[Symbol]
        for es in expr_symbols:
            self.deps.add_edge(sym.symbol, es.symbol)

    def has_dependency(self):
        """Return True if a read-after-write (raw) or write-after-read (war)
        dependency appears in the graph, False otherwise."""
        if self.deps.edges():
            sources, targets = zip(*self.deps.edges())
            return True if set(sources) & set(targets) else False
        else:
            return False

    def is_read(self, expr, target_sym=None):
        """Return True if any symbols in ``expr`` is read by ``target_sym``,
        False otherwise. If ``target_sym`` is None, Return True if any symbols
        in ``expr`` are read by at least one symbol, False otherwise."""
        retval = FindInstances.default_retval()
        input_syms = FindInstances(Symbol).visit(expr, ret=retval)[Symbol]
        for s in input_syms:
            if s.symbol not in self.deps:
                continue
            elif not target_sym:
                if list(zip(*self.deps.in_edges(s.symbol))):
                    return True
            elif nx.has_path(self.deps, target_sym.symbol, s.symbol):
                return True
        return False

    def is_written(self, expr, target_sym=None):
        """Return True if any symbols in ``expr`` is written by ``target_sym``,
        False otherwise. If ``target_sym`` is None, Return True if any symbols
        in ``expr`` are written by at least one symbol, False otherwise."""
        retval = FindInstances.default_retval()
        input_syms = FindInstances(Symbol).visit(expr, ret=retval)[Symbol]
        for s in input_syms:
            if s.symbol not in self.deps:
                continue
            elif not target_sym:
                if list(zip(*self.deps.out_edges(s.symbol))):
                    return True
            elif nx.has_path(self.deps, s.symbol, target_sym.symbol):
                return True
        return False

    def shares(self, symbols):
        """Return an iterator of tuples, each tuple being a group of symbols
        identifiers sharing the same reads."""
        groups = set()
        for i in [set(self.reads(s)) for s in symbols]:
            group = tuple(j for j in symbols if i.intersection(set(self.reads(j))))
            groups.add(group)
        return list(groups)

    def readers(self, sym):
        """Return the list of symbol identifiers that read from ``sym``."""
        return [i for i, j in self.deps.in_edges(sym)]

    def reads(self, sym):
        """Return the list of symbol identifiers that ``sym`` reads from."""
        return [j for i, j in self.deps.out_edges(sym)]


########################
# Simple support types #
########################


Region = namedtuple('Region', ['size', 'ofs'])


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


def uniquify(exprs):
    """Iterate over ``exprs`` and return a list of expressions in which duplicates
    have been discarded. This function considers two expressions identical if they
    have the same string representation."""
    return OrderedDict([(e.urepr, e) for e in exprs]).values()


def postprocess(node):
    """Rearrange the Nodes in the AST rooted in ``node`` to improve the code quality
    when unparsing the tree."""

    class Process(object):
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

    def init_decl(node):
        lhs, rhs = node.children
        decl = Process.decls.get(lhs.symbol)
        if decl and (not decl.init or isinstance(decl.init, EmptyStatement)):
            decl.init = rhs
            Process.blockable.remove(decl)
            return decl
        else:
            return node

    def update(node, parent):
        index = parent.children.index(node)
        if Process.start is None:
            Process.start = index
        Process.end = index

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
            Process.blockable.append(node)
        elif isinstance(node, AugmentedAssign):
            update(node, parent)
            Process.blockable.append(node)
        elif isinstance(node, Assign):
            update(node, parent)
            Process.blockable.append(init_decl(node))

    _postprocess(node, None)
    make_blocks()
