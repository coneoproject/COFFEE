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
        itspace = l.it_var()
        if l in _inner_loops:
            loops_unroll[itspace] = [1]
        else:
            loops_unroll[itspace] = [i+1 for i in range(l.size()) if l.size() % (i+1) == 0]

    return loops_unroll


################################################################
# Functions to manipulate and to query properties of AST nodes #
################################################################


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


###########################################################
# Functions to visit and to query properties of AST nodes #
###########################################################


def visit(node, parent):
    """Explore the AST rooted in ``node`` and collect various info, including:

    * Loop nests encountered - a list of tuples, each tuple representing a loop nest
    * Declarations - a dictionary {variable name (str): declaration (ast node)}
    * Symbols - a dictionary {symbol (ast node): iter space (tuple of loop indices)}
    * Expressions - mathematical expressions to optimize (decorated with a pragma)
    * Maximum depth - an integer representing the depth of the most depth loop nest

    :arg node:   AST root node of the visit
    :arg parent: parent node of ``node``
    """

    info = {
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
                    return (parent, zip(*fors)[0], (opt_par[1], opt_par[3]))

    def inspect(node, parent, mode=""):
        if isinstance(node, EmptyStatement):
            pass
        elif isinstance(node, (Block, Root)):
            for n in node.children:
                inspect(n, node)
        elif isinstance(node, For):
            info['cur_nest'].append((node, parent))
            inspect(node.children[0], node)
            if node in _inner_loops:
                info['fors'].append(info['cur_nest'])
            info['cur_nest'] = info['cur_nest'][:-1]
        elif isinstance(node, Par):
            inspect(node.children[0], node)
        elif isinstance(node, Decl):
            info['decls'][node.sym.symbol] = node
        elif isinstance(node, Symbol):
            if mode in ['written']:
                info['symbols_written'][node.symbol] = info['cur_nest']
            dep_itspace = node.loop_dep
            if node.symbol in info['symbols_written']:
                dep_loops = info['symbols_written'][node.symbol]
                dep_itspace = tuple(l[0].it_var() for l in dep_loops)
            info['symbols'][node] = dep_itspace
        elif isinstance(node, Expr):
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
    info['max_depth'] = max(len(l) for l in info['fors'])
    info.pop('cur_nest')
    info.pop('symbols_written')
    return info


def inner_loops(node):
    """Find inner loops in the subtree rooted in node."""

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


#######################################################################
# Functions to manipulate iteration spaces in various representations #
#######################################################################


def itspace_size_ofs(itspace):
    """Given an ``itspace`` in the form ::

        (('itvar', (bound_a, bound_b), ...)),

    return ::

        ((('it_var', bound_b - bound_a), ...), (('it_var', bound_a), ...))"""
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
    loops_it_vars = [l.it_var() for l in loops]
    return OrderedDict(zip(loops_it_vars, loops))
