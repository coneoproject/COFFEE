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

try:
    from collections import OrderedDict
# OrderedDict was added in Python 2.7. Earlier versions can use ordereddict
# from PyPI
except ImportError:
    from ordereddict import OrderedDict
import resource
import operator
from warnings import warn as warning
from copy import deepcopy as dcopy, copy as wcopy
from collections import defaultdict

from base import *


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

    loops_unroll = OrderedDict()

    # First, determine all inner loops
    _inner_loops = [l for l in loops if l in inner_loops(l)]

    # Then, determine possible unroll factors for all loops
    for l in loops:
        if l in _inner_loops:
            loops_unroll[l.dim] = [1]
        else:
            loops_unroll[l.dim] = [i+1 for i in range(l.size) if l.size % (i+1) == 0]

    return loops_unroll


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
        __ast_replace = lambda n: to_replace.get(str(n)) or to_replace.get(str(Par(n)))
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
    """Remove the AST Node ``to_remove`` from the tree rooted in ``node``.

    :param mode: either ``all``, in which case ``to_remove`` is turned into a
                 string (if not a string already) and all of its occurrences are
                 removed from the AST; or ``symbol``, in which case only (all of)
                 the references to the provided ``to_remove`` symbol are cut away.
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
        for tr in to_remove:
            _ast_remove(node, None, -1, tr)
    except TypeError:
        _ast_remove(node, None, -1, to_remove)


def ast_update_ofs(node, ofs):
    """Given a dictionary ``ofs`` s.t. ``{'dim': ofs}``, update the various
    iteration variables in the symbols rooted in ``node``."""
    if isinstance(node, Symbol):
        new_ofs = []
        old_ofs = ((1, 0) for r in node.rank) if not node.offset else node.offset
        for r, o in zip(node.rank, old_ofs):
            new_ofs.append((o[0], ofs[r] if r in ofs else o[1]))
        node.offset = tuple(new_ofs)
    else:
        for n in node.children:
            ast_update_ofs(n, ofs)


def ast_update_rank(node, new_rank):
    """Given a dictionary ``new_rank`` s.t. ``{'sym': new_dim}``, update the
    ranks of the symbols rooted in ``node`` by adding them the dimension
    ``new_dim``."""
    if isinstance(node, FlatBlock):
        return
    elif isinstance(node, Symbol):
        if node.symbol in new_rank:
            node.rank = new_rank[node.symbol] + node.rank
    else:
        for n in node.children:
            ast_update_rank(n, new_rank)


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
    if copy:
        stmts = dcopy(stmts)
    wrap = Block(stmts, open_scope=True)
    new_loop = For(dcopy(loop.init), dcopy(loop.cond), dcopy(loop.incr),
                   wrap, dcopy(loop.pragma))
    return new_loop


def ast_make_expr(op, nodes):
    """Create an ``Expr`` Node of type ``op``, with children given in ``nodes``."""

    def _ast_make_expr(nodes):
        return nodes[0] if len(nodes) == 1 else op(nodes[0], _ast_make_expr(nodes[1:]))

    try:
        expr = _ast_make_expr(nodes)
        return expr if len(nodes) == 1 else Par(expr)
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


def ast_make_copy(arr1, arr2, itspace, op):
    """Create an AST performing a copy from ``arr2`` to ``arr1``.
    Return also an ``ArrayInit`` object indicating how ``arr1`` should be
    initialized prior to the copy."""
    init = ArrayInit("0.0")
    if op == Assign:
        init = EmptyStatement()
    rank = ()
    for i, (start, end) in enumerate(itspace):
        rank += ("i%d" % i,)
    arr1, arr2 = dcopy(arr1), dcopy(arr2)
    body = []
    for a1, a2 in zip(arr1, arr2):
        a1.rank, a2.rank = rank, a2.rank[:-len(rank)] + rank
        body.append(op(a1, a2))
    for i, (start, end) in enumerate(itspace):
        if isinstance(init, ArrayInit):
            init.values = "{%s}" % init.values
        body = c_for(rank[i], end, body, pragma="", init=start)
    return body, init


###########################################################
# Functions to visit and to query properties of AST nodes #
###########################################################


def visit(node, parent=None, search=None, stop_on_search=False):
    """Explore the AST rooted in ``node`` and collect various info, including:

    * Loop nests encountered - a list of tuples, each tuple representing a loop nest
    * Declarations - a dictionary {variable name (str): declaration (AST node)}
    * Symbols (dependencies) - a dictionary {symbol (AST node): [loops] it depends on}
    * Symbols (access mode) - a dictionary {symbol (AST node): access mode (WRITE, ...)}
    * String to Symbols - a dictionary {symbol (str): [(symbol, parent) (AST nodes)]}
    * Expressions - mathematical expressions to optimize (decorated with a pragma)
    * Maximum depth - an integer representing the depth of the most depth loop nest
    * Searched nodes - a dictionary {types of AST node: list of occurrences}

    :param node: AST root node of the visit
    :param parent: parent node of ``node``
    :param search: type(s) of AST nodes to be searched and tracked in the visit
    :param stop_on_search: True if the tree visit should stop going in depth once
                           found a node being searched, False otherwise.
    """

    info = {
        'fors': [],
        'decls': OrderedDict(),
        'symbols_dep': OrderedDict(),
        'symbols_mode': OrderedDict(),
        'symbol_refs': defaultdict(list),
        'exprs': OrderedDict(),
        'max_depth': 0,
        'search': defaultdict(list)
    }

    _inner_loops = inner_loops(node)

    def check_opts(node, parent, fors):
        """Track high-level information."""
        for pragma in node.pragma:
            opts = pragma.split(" ", 2)
            if len(opts) < 3:
                return
            if opts[1] == 'coffee' and opts[2] == 'expression':
                # Found high-level optimisation
                return (parent, fors, node.children[0].rank)

    def inspect(node, parent, **kwargs):
        if search and isinstance(node, search):
            info['search'][type(node)].append(node)
            if stop_on_search:
                return

        if isinstance(node, EmptyStatement):
            pass
        elif isinstance(node, (Block, Root)):
            for n in node.children:
                inspect(n, node)
        elif isinstance(node, FunDecl):
            for n in node.children:
                inspect(n, node)
            for n in node.args:
                inspect(n, node, scope=EXTERNAL)
        elif isinstance(node, For):
            info['cur_nest'].append((node, parent))
            inspect(node.children[0], node)
            inspect(node.init, node)
            inspect(node.cond, node)
            inspect(node.incr, node)
            if node in _inner_loops:
                info['fors'].append(info['cur_nest'])
            info['cur_nest'] = info['cur_nest'][:-1]
        elif isinstance(node, Par):
            inspect(node.child, node)
        elif isinstance(node, Decl):
            node.scope = kwargs.get('scope', LOCAL)
            info['decls'][node.sym.symbol] = node
            inspect(node.sym, node)
        elif isinstance(node, Symbol):
            cur_nest = info['cur_nest']
            access_mode = (kwargs.get('mode', READ), parent.__class__)
            dep = [l for l, _ in cur_nest if l.dim in node.rank]
            if access_mode[0] == WRITE:
                info['symbols_written'][node.symbol] = wcopy(cur_nest)
            if node.symbol in info['symbols_written']:
                dep = tuple(l for l, _ in info['symbols_written'][node.symbol])
            info['symbols_dep'][node] = dep
            info['symbols_mode'][node] = access_mode
            info['symbol_refs'][node.symbol].append((node, parent))
        elif isinstance(node, Expr):
            for child in node.children:
                inspect(child, node)
        elif isinstance(node, FunCall):
            for child in node.children:
                inspect(child, node)
        elif isinstance(node, Perfect):
            expr = check_opts(node, parent, info['cur_nest'])
            if expr:
                info['exprs'][node] = expr
            inspect(node.children[0], node, mode=WRITE)
            for child in node.children[1:]:
                inspect(child, node)
        else:
            pass

    info['cur_nest'] = []
    info['symbols_written'] = {}
    inspect(node, parent)
    info['max_depth'] = max(len(l) for l in info['fors']) if info['fors'] else 0
    info.pop('cur_nest')
    info.pop('symbols_written')
    return info


def inner_loops(node):
    """Find inner loops in the subtree rooted in ``node``."""

    def find_iloops(node, loops):
        if isinstance(node, Perfect):
            return False
        elif isinstance(node, (Block, Root, FunDecl)):
            return any([find_iloops(s, loops) for s in node.children])
        elif isinstance(node, For):
            found = find_iloops(node.children[0], loops)
            if not found:
                loops.append(node)
            return True

    loops = []
    find_iloops(node, loops)
    return loops


def is_perfect_loop(loop):
    """Return True if ``loop`` is part of a perfect loop nest, False otherwise."""

    def check_perfectness(node, found_block=False):
        if isinstance(node, Perfect):
            return True
        elif isinstance(node, For):
            if found_block:
                return False
            return check_perfectness(node.children[0])
        elif isinstance(node, Block):
            stmts = node.children
            if len(stmts) == 1:
                return check_perfectness(stmts[0])
            # Need to check this is the last level of the loop nest, otherwise
            # it can't be a perfect loop nest
            return all([check_perfectness(n, True) for n in stmts])

    if not isinstance(loop, For):
        return False
    return check_perfectness(loop)


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

    def count(node, counter):
        if isinstance(node, Symbol):
            counter[key(node)] += 1
        elif isinstance(node, FlatBlock):
            return
        else:
            to_traverse = node.children
            if isinstance(node, (Assign, Incr, Decr)) and read_only:
                to_traverse = node.children[1:]
            for c in to_traverse:
                count(c, counter)

    counter = defaultdict(int)
    count(node, counter)
    return counter


def check_type(stmt, decls):
    """Check the types of the ``stmt``'s LHS and RHS. If they match as expected,
    return the type itself. Otherwise, an error is generated, suggesting an issue
    in either the AST itself (i.e., a bug inherent the AST) or, possibly, in the
    optimization process.

    :param stmt: the AST node statement to be checked
    :param decls: a dictionary from symbol identifiers (i.e., strings representing
                  the name of a symbol) to Decl nodes
    """
    lhs_symbol = visit(stmt.children[0], stmt)['symbol_refs'].keys()[0]
    rhs_symbols = visit(stmt.children[1], stmt)['symbol_refs'].keys()

    lhs_decl = decls[lhs_symbol]
    rhs_decls = [decls[s] for s in rhs_symbols if s in decls]

    type = lambda d: d.typ.replace('*', '')
    if any([type(lhs_decl) != type(rhs_decl) for rhs_decl in rhs_decls]):
        raise RuntimeError("Non matching types in %s" % str(stmt))

    return type(lhs_decl)


#######################################################################
# Functions to manipulate iteration spaces in various representations #
#######################################################################


def itspace_size_ofs(itspace):
    """Given an ``itspace`` in the form ::

        (('dim', (bound_a, bound_b), ...)),

    return ::

        ((('dim', bound_b - bound_a), ...), (('dim', bound_a), ...))"""
    itspace_info = []
    for var, bounds in itspace:
        itspace_info.append(((var, bounds[1] - bounds[0] + 1), (var, bounds[0])))
    return tuple(zip(*itspace_info))


def itspace_merge(itspaces):
    """Given an iterator of iteration spaces, each iteration space represented
    as a 2-tuple containing the start and end point, return a tuple of iteration
    spaces in which contiguous iteration spaces have been merged. For example:
    ::

        [(1,3), (4,6)] -> ((1,6),)
        [(1,3), (5,6)] -> ((1,3), (5,6))
    """
    itspaces = sorted(tuple(set(itspaces)))
    merged_itspaces = []
    current_start, current_stop = itspaces[0]
    for start, stop in itspaces:
        if start - 1 > current_stop:
            merged_itspaces.append((current_start, current_stop))
            current_start, current_stop = start, stop
        else:
            # Ranges adjacent or overlapping: merge.
            current_stop = max(current_stop, stop)
    merged_itspaces.append((current_start, current_stop))
    return tuple(merged_itspaces)


def itspace_to_for(itspaces, loop_parent):
    """Given an iterator of iteration spaces, each iteration space represented
    as a 2-tuple containing the start and the end point, return a tuple
    ``(loops_info, inner_block)``, in which ``loops_info`` is the tuple of all
    tuples (loop, loop_parent) embedding ``inner_block``."""
    inner_block = Block([], open_scope=True)
    loops, loops_parents = [], [loop_parent]
    loop_body = inner_block
    for i, itspace in enumerate(itspaces):
        start, stop = itspace
        loops.insert(0, For(Decl("int", start, Symbol(0)), Less(start, stop),
                            Incr(start, Symbol(1)), loop_body))
        loop_body = Block([loops[i-1]], open_scope=True)
        loops_parents.append(loop_body)
    # Note that #loops_parents = #loops+1, but by zipping we just cut away the
    # last entry in loops_parents
    loops_info = zip(loops, loops_parents)
    return (tuple(loops_info), inner_block)


def itspace_from_for(loops, mode=0):
    """Given an iterator of for ``loops``, return a tuple that rather contains
    the iteration space of each loop, i.e. given: ::

        [for1, for2, ...]

    If ``mode == 0``, return: ::

        ((start1, bound1, increment1), (start2, bound2, increment2), ...)

    If ``mode > 0``, return: ::

        ((for1_dim, (start1, topiter1)), (for2_dim, (start2, topiter2):, ...)
    """
    if mode == 0:
        return tuple((l.start, l.end, l.increment) for l in loops)
    else:
        return tuple((l.dim, (l.start, l.end - 1)) for l in loops)


def itspace_copy(loop_a, loop_b):
    """Copy the iteration space of ``loop_b`` into ``loop_a``, while preserving
    the body."""
    loop_a.init = dcopy(loop_b.init)
    loop_a.cond = dcopy(loop_b.cond)
    loop_a.incr = dcopy(loop_b.incr)
    loop_a.pragma = dcopy(loop_b.pragma)


#############################
# Generic utility functions #
#############################


any_in = lambda a, b: any(i in b for i in a)
flatten = lambda list: [i for l in list for i in l]
bind = lambda a, b: [(a, v) for v in b]
od_find_next = lambda a, b: a.values()[a.keys().index(b)+1]


def insert_at_elem(_list, elem, new_elem, ofs=0):
    ofs = _list.index(elem) + ofs
    new_elem = [new_elem] if not isinstance(new_elem, list) else new_elem
    for e in reversed(new_elem):
        _list.insert(ofs, e)
