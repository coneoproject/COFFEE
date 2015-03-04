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
from copy import deepcopy as dcopy
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
        if decls:
            size += sum([reduce(operator.mul, d.sym.rank) for d in zip(*decls)[0]
                         if d.sym.rank])

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

    :arg:loops: list of for loops for which a suitable unroll factor has to be
                determined.
    """

    loops_unroll = OrderedDict()

    # First, determine all inner loops
    _inner_loops = [l for l in loops if l in inner_loops(l)]

    # Then, determine possible unroll factors for all loops
    for l in loops:
        if l in _inner_loops:
            loops_unroll[l.itvar] = [1]
        else:
            loops_unroll[l.itvar] = [i+1 for i in range(l.size) if l.size % (i+1) == 0]

    return loops_unroll


#####################################
# Functions to manipulate AST nodes #
#####################################


def ast_replace(node, syms_dict, n_replaced={}, copy=False):
    """Given a dictionary ``syms_dict`` s.t. ``{'syms': to_replace}``, replace the
    various ``syms`` rooted in ``node`` with ``to_replace``. If ``copy`` is True,
    a deep copy of the replacing symbol is created."""

    to_replace = {}
    for i, n in enumerate(node.children):
        replacing = syms_dict.get(str(n)) or syms_dict.get(str(Par(n)))
        if replacing:
            to_replace[i] = replacing if not copy else dcopy(replacing)
            if n_replaced:
                n_replaced[str(replacing)] += 1
        elif not isinstance(n, Symbol):
            # Useless to traverse the tree if the child is a symbol
            ast_replace(n, syms_dict, n_replaced, copy)
    for i, r in to_replace.items():
        node.children[i] = r


def ast_update_ofs(node, ofs):
    """Given a dictionary ``ofs`` s.t. ``{'itvar': ofs}``, update the various
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


def ast_c_for(stmts, loop, copy=False):
    """Create a for loop having the same iteration space as  ``loop`` enclosing
    the statements in  ``stmts``. If ``copy == True``, then new instances of
    ``stmts`` are created"""
    if copy:
        stmts = dcopy(stmts)
    wrap = Block(stmts, open_scope=True)
    new_loop = For(dcopy(loop.init), dcopy(loop.cond), dcopy(loop.incr),
                   wrap, dcopy(loop.pragma))
    return new_loop


def ast_c_sum(symbols):
    """Create a ``Sum`` object starting from a symbols list ``symbols``. If
    the length of ``symbols`` is 1, return ``Symbol(symbols[0])``."""
    if len(symbols) == 1:
        return symbols[0]
    else:
        return Sum(symbols[0], ast_c_sum(symbols[1:]))


def ast_c_make_alias(node1, node2):
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


def visit(node, parent):
    """Explore the AST rooted in ``node`` and collect various info, including:

    * Function declarations - a list of all function declarations encountered
    * Loop nests encountered - a list of tuples, each tuple representing a loop nest
    * Declarations - a dictionary {variable name (str): declaration (ast node)}
    * Symbols - a dictionary {symbol (ast node): iter space (tuple of loop indices)}
    * Expressions - mathematical expressions to optimize (decorated with a pragma)
    * Maximum depth - an integer representing the depth of the most depth loop nest

    :arg node:   AST root node of the visit
    :arg parent: parent node of ``node``
    """

    info = {
        'fun_decls': [],
        'fors': [],
        'decls': {},
        'symbols': {},
        'exprs': {},
        'max_depth': 0
    }
    _inner_loops = inner_loops(node)

    def check_opts(node, parent, fors):
        """Check if node is associated with some pragmas. If that is the case,
        it saves info about the node to speed the transformation process up."""
        if node.pragma:
            opts = node.pragma[0].split(" ", 2)
            if len(opts) < 3:
                return
            if opts[1] == "pyop2":
                if opts[2] == "integration":
                    return
                delim = opts[2].find('(')
                opt_name = opts[2][:delim].replace(" ", "")
                opt_par = opts[2][delim:].replace(" ", "")
                if opt_name == "assembly":
                    # Found high-level optimisation
                    return (parent, fors, (opt_par[1], opt_par[3]))

    def inspect(node, parent, mode=""):
        if isinstance(node, EmptyStatement):
            pass
        elif isinstance(node, (Block, Root)):
            for n in node.children:
                inspect(n, node)
        elif isinstance(node, FunDecl):
            info['fun_decls'].append(node)
            for n in node.children:
                inspect(n, node)
            for n in node.args:
                inspect(n, node)
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
            inspect(node.children[0], node)
        elif isinstance(node, Decl):
            info['decls'][node.sym.symbol] = node
            inspect(node.sym, node)
        elif isinstance(node, Symbol):
            if mode in ['written']:
                info['symbols_written'][node.symbol] = info['cur_nest']
            dep_itspace = node.loop_dep
            if node.symbol in info['symbols_written']:
                dep_loops = info['symbols_written'][node.symbol]
                dep_itspace = tuple(l[0].itvar for l in dep_loops)
            info['symbols'][node] = dep_itspace
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
            inspect(node.children[0], node, "written")
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
        elif isinstance(node, (Block, Root)):
            return any([find_iloops(s, loops) for s in node.children])
        elif isinstance(node, For):
            found = find_iloops(node.children[0], loops)
            if not found:
                loops.append(node)
            return True

    loops = []
    find_iloops(node, loops)
    return loops


def get_fun_decls(node, mode):
    """Search the ``FunDecl`` node rooted in ``node``.

    :param mode: any string in ['kernel', 'all']. If ``kernel`` is passed, then
                 only one ``FunDecl`` is expected in the tree rooted in ``node``
                 (the name "kernel" is to denote that the tree represents a
                 self-contained piece of code in a function); a search is performed
                 and the corresponding node returned. If ``all`` is passed, the
                 whole tree in inspected and all ``FunDecl`` nodes are returned
                 in a list.
    """

    def find_fun_decl(node):
        if isinstance(node, FunDecl):
            return node
        for n in node.children:
            fundecl = find_fun_decl(n)
            if fundecl:
                return fundecl

    allowed = ['kernel', 'all']
    if mode == 'kernel':
        return find_fun_decl(node)
    if mode == 'all':
        return visit(node, None)['fun_decls']
    raise RuntimeError("Only %s modes are allowed by `get_fun_decls`" % allowed)


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


def count_occurrences(node, key=0, read_only=False):
    """For each variable ``node``, count how many times it appears as involved
    in some expressions. For example, for the expression: ::

        ``a*(5+c) + b*(a+4)``

    return ::

        ``{a: 2, b: 1, c: 1}``

    :arg key: This can be any value in [0, 1, 2]. The keys used in the returned
              dictionary can be:

              * ``key == 0``: a tuple (symbol name, symbol rank)
              * ``key == 1``: the symbol name
              * ``key == 2``: a string representation of the symbol
    :arg read_only: True if only variables on the right-hand side of a statement
                    should be counted; False if any appearance should be counted.
    """

    def count(node, counter):
        if isinstance(node, Symbol):
            if key == 0:
                node = (node.symbol, node.rank)
            elif key == 1:
                node = node.symbol
            elif key == 2:
                node = str(node)
            counter[node] += 1
        elif isinstance(node, FlatBlock):
            return
        else:
            to_traverse = node.children
            if isinstance(node, (Assign, Incr, Decr)) and read_only:
                to_traverse = node.children[1:]
            for c in to_traverse:
                count(c, counter)

    if key not in [0, 1, 2]:
        raise RuntimeError("Count_occurrences got a wrong key (valid: 0, 1, 2)")
    counter = defaultdict(int)
    count(node, counter)
    return counter


#######################################################################
# Functions to manipulate iteration spaces in various representations #
#######################################################################


def itspace_size_ofs(itspace):
    """Given an ``itspace`` in the form ::

        (('itvar', (bound_a, bound_b), ...)),

    return ::

        ((('itvar', bound_b - bound_a), ...), (('itvar', bound_a), ...))"""
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
        loops.insert(0, For(Decl("int", start, c_sym(0)), Less(start, stop),
                            Incr(start, c_sym(1)), loop_body))
        loop_body = Block([loops[i-1]], open_scope=True)
        loops_parents.append(loop_body)
    # Note that #loops_parents = #loops+1, but by zipping we just cut away the
    # last entry in loops_parents
    loops_info = zip(loops, loops_parents)
    return (tuple(loops_info), inner_block)


def itspace_from_for(loops, mode):
    """Given an iterator of for ``loops``, return a tuple that rather contains
    the iteration space of each loop, i.e. given: ::

        [for1, for2, ...]

    If ``mode == 0``, return: ::

        ((start1, bound1, increment1), (start2, bound2, increment2), ...)

    If ``mode > 0``, return: ::

        ((for1_itvar, (start1, topiter1)), (for2_itvar, (start2, topiter2):, ...)
    """
    if mode == 0:
        return tuple((l.start, l.end, l.increment) for l in loops)
    else:
        return tuple((l.itvar, (l.start, l.end - 1)) for l in loops)


#############################
# Generic utility functions #
#############################


any_in = lambda a, b: any(i in b for i in a)
flatten = lambda list: [i for l in list for i in l]
bind = lambda a, b: [(a, v) for v in b]

od_find_next = lambda a, b: a.values()[a.keys().index(b)+1]


def set_itspace(loop_a, loop_b):
    """Copy the iteration space of ``loop_b`` into ``loop_a``, while preserving
    the body."""
    loop_a.init = dcopy(loop_b.init)
    loop_a.cond = dcopy(loop_b.cond)
    loop_a.incr = dcopy(loop_b.incr)
    loop_a.pragma = dcopy(loop_b.pragma)


def loops_as_dict(loops):
    loops_itvars = [l.itvar for l in loops]
    return OrderedDict(zip(loops_itvars, loops))
