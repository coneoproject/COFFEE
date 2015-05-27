from __future__ import absolute_import
from coffee.visitor import Visitor, Environment
from coffee.base import READ, WRITE, LOCAL, EXTERNAL, Symbol
from collections import defaultdict, OrderedDict, Counter
import itertools

__all__ = ["FindInnerLoops", "CheckPerfectLoop",
           "CountOccurences", "DetermineUnrollFactors",
           "MaxLoopDepth", "FindLoopNests",
           "FindCoffeeExpressions", "SymbolReferences",
           "SymbolDependencies", "SymbolModes",
           "SymbolDeclarations", "FindInstances"]


class FindInnerLoops(Visitor):

    """Find all inner-most loops in an AST.

    Returns a list of the inner-most :class:`.For` loops or an empty
    list if none were found."""

    def visit_object(self, o):
        return []

    def visit_Node(self, o):
        # Concatenate transformed children
        ops, _ = o.operands()
        args = [self.visit(op) for op in ops]
        return list(itertools.chain(*args))

    def visit_For(self, o):
        # Check for loops in children
        children = self.visit(o.children[0])
        if children:
            # Yes, return those
            return children
        # No return ourselves
        return [o]


class CheckPerfectLoop(Visitor):

    """
    Check if a Node is a perfect loop nest.
    """

    def visit_object(self, o, *args, **kwargs):
        # Unhandled, return False to be safe.
        return False

    def visit_Node(self, o, in_loop=False, *args, **kwargs):
        # Assume all nodes are in a perfect loop if they're in a loop.
        return in_loop

    def visit_For(self, o, in_loop=False, multi=False, *args, **kwargs):
        if in_loop and multi:
            return False
        return self.visit(o.children[0], in_loop=True, multi=multi)

    def visit_Block(self, o, in_loop=False, multi=False, *args, **kwargs):
        # Does this block contain multiple statements?
        multi = multi or len(o.children) > 1
        return in_loop and all(self.visit(op, in_loop=in_loop, multi=multi) for op in o.children)


class CountOccurences(Visitor):

    """Count all occurances of :class:`~.Symbol`\s in an AST.

    :arg key: a comparison key for the symbols.
    :arg only_rvalues: optionally only count rvalues in statements.

    Returns a dict mapping symbol keys to number of occurrences.
    """
    def __init__(self, key=lambda x: (x.symbol, x.rank), only_rvalues=False):
        self.key = key
        self.rvalues = only_rvalues
        super(CountOccurences, self).__init__()

    def visit_object(self, o, ret=None, *args, **kwargs):
        # Not a symbol, return identity for summation
        return ret

    def visit_list(self, o, ret=None, *args, **kwargs):
        # Walk list entries (since some operands methods return lists)
        for entry in o:
            ret = self.visit(entry, ret=ret, *args, **kwargs)
        return ret

    def visit_Node(self, o, ret=None, *args, **kwargs):
        ops, _ = o.operands()
        for op in ops:
            ret = self.visit(op, ret=ret, *args, **kwargs)
        return ret

    def visit_Assign(self, o, ret=None, *args, **kwargs):
        if self.rvalues:
            # Only counting rvalues, so don't walk lvalue
            ops = o.children[1:]
        else:
            ops = o.children
        for op in ops:
            ret = self.visit(op, ret=ret, *args, **kwargs)
        return ret

    visit_Incr = visit_Assign

    visit_Decr = visit_Assign

    def visit_Symbol(self, o, ret=None, *args, **kwargs):
        if ret is None:
            ret = Counter()
        ret[self.key(o)] += 1
        return ret


class DetermineUnrollFactors(Visitor):
    """
    Determine unroll factors for all For loops in a tree.

    Returns a dict mapping iteration variable names to possible unroll
    factors for that iteration variable.

    Innermost loops always get an unroll factor of 1, to give the
    backend compiler a chance of auto-vectorizing them.  Outer loops
    are given potential unroll factors that result in no loop
    remainders.
    """
    def visit_object(self, o, ret=None):
        return ret

    def visit_Node(self, o, ret=None):
        ops, _ = o.operands()
        for op in ops:
            ret = self.visit(op, ret=ret)
        return ret

    def visit_For(self, o, ret=None):
        if ret is None:
            ret = {}
        # Check if children contain any loops
        nval = len(ret)
        ret = self.visit(o.children[0], ret=ret)
        if len(ret) == nval:
            # No child for loops
            # Inner loops are not unrollable
            ret[o.dim] = (1, )
            return ret
        # Not an inner loop, determine unroll factors
        ret[o.dim] = tuple(i for i in range(1, o.size+1) if (o.size % i) == 0)
        return ret


class MaxLoopDepth(Visitor):

    """Return the maximum loop depth in the tree."""

    def visit_object(self, o):
        return 0

    def visit_Node(self, o):
        ops, _ = o.operands()
        if len(ops) == 0:
            return 0
        return max(self.visit(op) for op in ops)

    def visit_For(self, o):
        return 1 + max(self.visit(op) for op in o.children)


class FindLoopNests(Visitor):

    """Return a list of lists of loop nests in the tree.

    Each list entry describes a loop nest with the outer-most loop
    first.  Each entry therein is a tuple (loop_node, parent).

    By default the top-level call to visit will record a parent
    of None for the visited Node.  To provide one, pass a keyword
    argument in to the visitor::

    .. code-block::

       v.visit(node, parent=parent)

    """

    def visit_object(self, o, ret=None, *args, **kwargs):
        return ret

    def visit_Node(self, o, ret=None, parent=None, *args, **kwargs):
        ops, _ = o.operands()
        for op in ops:
            # Visit children recording this node as the parent
            ret = self.visit(op, ret=ret, parent=o)
        return ret

    def visit_For(self, o, ret=None, parent=None, *args, **kwargs):
        if ret is None:
            ret = []
        ops, _ = o.operands()
        nval = len(ret)
        for op in ops:
            ret = self.visit(op, ret=ret, parent=o)
        # Cons (node, node_parent) onto front of current loop-nest list
        me = (o, parent)
        if len(ret) == nval:
            # Bottom of the nest, add myself to ret
            ret.append([me])
            return ret
        # Transform new children (inside this loop)
        # [a, b] into [(me, a), (me, b)]
        for a in ret[nval:]:
            a.insert(0, me)
        return ret


class FindCoffeeExpressions(Visitor):

    """
    Search the tree for :class:`~.Writer` statements annotated with
    :data:`"#pragma coffee expression"`.  Return a dict mapping the
    annotated node to a tuple of (node_parent, containing_loop_nest,
    index_access).

    By default the top-level call to visit will record a node_parent
    of None for the visited Node.  To provide one, pass a keyword
    argument in to the visitor::

    .. code-block::

       v.visit(node, parent=parent)

    """

    def visit_object(self, o, ret=None, *args, **kwargs):
        return ret

    def visit_Node(self, o, ret=None, *args, **kwargs):
        ops, _ = o.operands()
        for op in ops:
            ret = self.visit(op, ret=ret, parent=o)
        return ret

    def visit_Writer(self, o, ret=None, parent=None, *args, **kwargs):
        if ret is None:
            ret = OrderedDict()
        for p in o.pragma:
            opts = p.split(" ", 2)
            # Don't care if we don't have three values
            if len(opts) < 3:
                continue
            if opts[1] == "coffee" and opts[2] == "expression":
                # (parent, loop-nest, rank)
                ret[o] = (parent, None, o.children[0].rank)
                return ret
        return ret

    def visit_For(self, o, ret=None, parent=None, *args, **kwargs):
        if ret is None:
            ret = OrderedDict()
        nval = len(ret)

        ops, _ = o.operands()
        for op in ops:
            ret = self.visit(op, ret=ret, parent=o)

        # Nothing inside this for loop was annotated (we didn't see a
        # Writer node with #pragma coffee expression)
        if len(ret) == nval:
            return ret
        me = (o, parent)
        # Add nest structure to new items
        keys = ret.keys()[nval:]
        for k in keys:
            p, nest, rank = ret[k]
            if nest is None:
                # Statement is directly underneath this loop, so the
                # loop nest structure is just the current loop
                nest = [me]
            else:
                # Inside a nested set of loops, so prepend current
                # loop info to nest structure
                nest = [me] + nest
            ret[k] = p, nest, rank
        return ret


class SymbolReferences(Visitor):

    """
    Visit the tree and return a dict mapping symbol names to tuples of
    (node, node_parent) that reference the symbol with that name.
    The node is the Symbol node with said name, the node_parent is the
    parent of that node.

    By default the top-level call to visit will record a node_parent
    of None for the visited Node.  To provide one, pass a keyword
    argument in to the visitor::

    .. code-block::

       v.visit(node, parent=parent)

    """

    def visit_Symbol(self, o, ret=None, parent=None):
        if ret is None:
            ret = defaultdict(list)

        # Map name to (node, parent) tuple
        ret[o.symbol].append((o, parent))
        return ret

    def visit_object(self, o, ret=None, *args, **kwargs):
        # Identity
        return ret

    def visit_list(self, o, ret=None, *args, **kwargs):
        for entry in o:
            ret = self.visit(entry, ret=ret, *args, **kwargs)
        return ret

    def visit_Node(self, o, ret=None, *args, **kwargs):
        ops, _ = o.operands()
        for op in ops:
            ret = self.visit(op, ret=ret, parent=o)
        return ret


class SymbolDependencies(Visitor):

    """
    Visit the tree and return a dict collecting symbol dependencies.

    The returned dict contains maps from nodes to a (possibly
    empty) loop list the symbol depends on.
    """

    default_env = dict(loop_nest=[], write=False)

    def visit_Symbol(self, o, env):
        write = env["write"]
        nest = env["loop_nest"]
        if write:
            # Remember that this symbol /name/ was written,
            # as well as the full current loop nest for the
            # symbol itself
            return {o: [l for l in nest],
                    o.symbol: True}
        # Not being written, only care if the loop indices
        # of the current nest access the symbol
        return {o: [l for l in nest if l.dim in o.rank]}

    def visit_object(self, o, env):
        return {}

    def visit_Node(self, o, env):
        ops, _ = o.operands()
        args = [self.visit(op, env=env) for op in ops]
        ret = OrderedDict()
        # Merge values from children
        for a in args:
            ret.update(a)
        return ret

    visit_EmptyStatement = visit_object

    def visit_Decl(self, o, env):
        # Declaration init could have symbol access
        args = [self.visit(op, env=env) for op in [o.sym, o.init]]
        ret = OrderedDict()
        for a in args:
            ret.update(a)
        return ret

    visit_FunCall = visit_Node

    def visit_Invert(self, o, env):
        return self.visit(o.children[0], env=env)

    def visit_Writer(self, o, env):
        ret = OrderedDict()
        write_env = Environment(env, write=True)
        lvalue = self.visit(o.children[0], env=write_env)
        ret.update(lvalue)
        for r in o.children[1:]:
            d = self.visit(r, env=env)
            ret.update(d)
        return ret

    def visit_For(self, o, env):
        loop_nest = env["loop_nest"]
        new_env = Environment(env, loop_nest=loop_nest + [o])
        # Don't care about symbol access in increments, only children
        args = [self.visit(op, env=new_env) for op in o.children]
        ret = OrderedDict()
        for a in args:
            ret.update(a)

        # Dependencies for variables that are written in one nest
        # and read in a subsequent one need to respect this.
        nest = new_env["loop_nest"]
        for k, v in ret.iteritems():
            if type(k) is not Symbol:
                continue
            if k.symbol in ret:
                # Symbol name was written in some nest
                # The dependency for this symbol is therefore
                # whatever nest came from visiting the children
                # plus the current nest at this point in the tree,
                # suitably uniquified.
                new_v = [l for l in nest]
                new_v.extend([l for l in v if l not in new_v])
                ret[k] = new_v

        return ret


class SymbolModes(Visitor):

    """
    Visit the tree and return a dict mapping Symbols to tuples of
    (access mode, parent class).

    :class:`~.Symbol`\s are accessed as READ-only unless they appear
    as lvalues in a :class:`~.Writer` statement.

    By default the top-level call to visit will record a parent class
    of NoneType for Symbols without a parent.  To pass in a parent by
    hand, provide a keyword argument to the visitor::

    .. code-block::

       v.visit(symbol, parent=parent)

    """

    def visit_object(self, o, ret=None, *args, **kwargs):
        return ret

    def visit_Node(self, o, ret=None, *args, **kwargs):
        ops, _ = o.operands()
        for op in ops:
            # WARNING, if the same Symbol object appears multiple
            # times, the "last" access wins, rather than WRITE winning.
            # This assumes all nodes in the tree are unique instances
            ret = self.visit(op, ret=ret, parent=o)
        return ret

    def visit_Symbol(self, o, ret=None, mode=READ, parent=None, *args, **kwargs):
        if ret is None:
            ret = OrderedDict()
        ret[o] = (mode, parent.__class__)
        return ret

    # Don't do anything with declarations.  If you want lvalues to get
    # a WRITE unless uninitialised, then custom visitor must be
    # written.
    visit_Decl = visit_object

    def visit_Writer(self, o, ret=None, *args, **kwargs):
        # lvalues have access mode WRITE
        ret = self.visit(o.children[0], ret=ret, parent=o, mode=WRITE)
        # All others have access mode READ
        for op in o.children[1:]:
            ret = self.visit(op, ret=ret, parent=o)
        return ret

    visit_Invert = visit_Writer


class SymbolDeclarations(Visitor):

    """Return a dict mapping symbol names to a tuple of the declaring
    node.  The node is annotated in place with information about
    whether it is a LOCAL declaration or EXTERNAL (via a function
    argument).
    """

    def __init__(self):
        super(SymbolDeclarations, self).__init__()

    def visit_object(self, o, ret=None, *args, **kwargs):
        return ret

    def visit_Node(self, o, ret=None, *args, **kwargs):
        ops, _ = o.operands()
        for op in ops:
            ret = self.visit(op, ret=ret, *args, **kwargs)
        return ret

    def visit_FunDecl(self, o, ret=None, *args, **kwargs):
        for op in o.args:
            ret = self.visit(op, ret=ret, scope=EXTERNAL)
        for op in o.children:
            ret = self.visit(op, ret=ret, *args, **kwargs)
        return ret

    def visit_Decl(self, o, ret=None, scope=LOCAL, *args, **kwargs):
        if ret is None:
            ret = OrderedDict()
        o.scope = scope
        ret[o.sym.symbol] = o
        return ret


class FindInstances(Visitor):

    """
    Visit the tree and return a dict mapping types to a list of
    instances of that type in the tree.

    :arg types: list of types or single type to search for in the tree.
    :arg stop_when_found: optional, don't traverse the children of matching types.
    :arg with_parent: optional, track also the parent of the matching type.
    """

    def __init__(self, types, stop_when_found=False, with_parent=False):
        self.types = types
        self.stop_when_found = stop_when_found
        self.with_parent = with_parent
        super(FindInstances, self).__init__()

    def visit_object(self, o, ret=None, *args, **kwargs):
        return ret

    def visit_list(self, o, ret=None, *args, **kwargs):
        for entry in o:
            ret = self.visit(entry, ret=ret, *args, **kwargs)
        return ret

    def visit_Node(self, o, ret=None, parent=None, *args, **kwargs):
        if ret is None:
            ret = defaultdict(list)
        if isinstance(o, self.types):
            found = (o, parent) if self.with_parent else o
            ret[type(o)].append(found)
            # Don't traverse children if stop-on-found
            if self.stop_when_found:
                return ret
        # Not found, or traversing children anyway
        ops, _ = o.operands()
        for op in ops:
            ret = self.visit(op, ret=ret, parent=o)
        return ret
