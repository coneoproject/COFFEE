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

    def visit_object(self, o, env):
        return []

    def visit_Node(self, o, env, *args, **kwargs):
        # Concatenate transformed children
        return list(itertools.chain(*args))

    def visit_For(self, o, env):
        # Check for loops in children
        children = self.visit(o.children[0], env=env)
        if children:
            # Yes, return those
            return children
        # No return ourselves
        return [o]


class CheckPerfectLoop(Visitor):

    """
    Check if a Node is a perfect loop nest.
    """

    def visit_object(self, o, env):
        # Unhandled, return False to be safe.
        return False

    def visit_Perfect(self, o, env):
        # Perfect nodes are in a perfect loop if they're in a loop.
        try:
            return env["in_loop"]
        except KeyError:
            return False

    def visit_For(self, o, env):
        if env["in_loop"] and env["multiple_statements"]:
            return False
        new_env = Environment(env, in_loop=True)
        return self.visit(o.children[0], env=new_env)

    def visit_Block(self, o, env):
        # Does this block contain multiple statements?
        multi = env["multiple_statements"] or len(o.children) > 1
        new_env = Environment(env, multiple_statements=multi)
        return env["in_loop"] and all(self.visit(op, env=new_env) for op in o.children)


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

    def visit_object(self, o, env):
        # Not a symbol, return identity for summation
        return {}

    def visit_list(self, o, env):
        # Walk list entries (since some operands methods return lists)
        ret = Counter()
        for entry in o:
            ret.update(self.visit(entry, env=env))
        return ret

    def visit_Node(self, o, env, *args, **kwargs):
        # Walk the transformed children and sum up.
        ret = Counter()
        for d in args:
            ret.update(d)
        return ret

    def visit_Assign(self, o, env):
        ret = Counter()
        if self.rvalues:
            # Only counting rvalues, so don't walk lvalue
            ops = o.children[1:]
        else:
            ops = o.children
        args = [self.visit(op, env=env) for op in ops]
        for d in args:
            ret.update(d)
        return ret

    visit_Incr = visit_Assign

    visit_Decr = visit_Assign

    def visit_Symbol(self, o, env):
        return {self.key(o): 1}


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
    def visit_object(self, o, env):
        return {}

    def visit_Node(self, o, env, *args, **kwargs):
        ret = {}
        for a in args:
            ret.update(a)
        return ret

    def visit_For(self, o, env):
        ret = self.visit(o.children[0], env=env)
        if len(ret) is 0:
            # No child for loops
            # Inner loops are not unrollable
            ret[o.dim] = (1, )
            return ret
        # Not an inner loop, determine unroll factors
        ret[o.dim] = tuple(i for i in range(1, o.size+1) if (o.size % i) == 0)
        return ret


class MaxLoopDepth(Visitor):

    """Return the maximum loop depth in the tree."""

    def visit_object(self, o, env):
        return 0

    def visit_Node(self, o, env, *args, **kwargs):
        if len(args) == 0:
            return 0
        return max(args)

    def visit_For(self, o, env, *args, **kwargs):
        return 1 + max(args)


class FindLoopNests(Visitor):

    """Return a list of lists of loop nests in the tree.

    Each list entry describes a loop nest with the outer-most loop
    first.  Each entry therein is a tuple (loop_node, node_parent).

    By default the top-level call to visit will record a node_parent
    of None for the visited Node.  To provide one, pass an
    environment in to the visitor::

    .. code-block::

       v.visit(node, {"node_parent": parent})

    """

    def visit_object(self, o, env):
        return []

    def visit_Node(self, o, env):
        ops, _ = o.operands()
        new_env = Environment(env, node_parent=o)
        # Visit children recording this node as the parent
        return list(itertools.chain(*[self.visit(op, env=new_env) for op in ops]))

    def visit_For(self, o, env):
        # Transform operands using generic visit_Node
        args = self.visit_Node(o, env)
        # Cons (node, node_parent) onto front of current loop-nest list
        try:
            parent = env["node_parent"]
        except KeyError:
            parent = None
        me = (o, parent)
        if len(args) == 0:
            # Bottom of the nest, just return self
            return ([me], )
        # Transform child [a, b] into [(me, a), (me, b)]
        return [list(itertools.chain((me, ), a)) for a in args]


class FindCoffeeExpressions(Visitor):

    """
    Search the tree for :class:`~.Perfect` statements annotated with
    :data:`"#pragma coffee expression"`.  Return a dict mapping the
    annotated node to a tuple of (node_parent, containing_loop_nest,
    index_access).

    By default the top-level call to visit will record a node_parent
    of None for the visited Node.  To provide one, pass an
    environment in to the visitor::

    .. code-block::

       v.visit(node, {"node_parent": parent})

    """

    def visit_object(self, o, env):
        return {}

    def visit_Node(self, o, env):
        ops, _ = o.operands()
        new_env = Environment(env, node_parent=o)
        args = [self.visit(op, env=new_env) for op in ops]
        ret = OrderedDict()
        # Merge values from children
        for a in args:
            ret.update(a)
        return ret

    def visit_Perfect(self, o, env):
        for p in o.pragma:
            opts = p.split(" ", 2)
            # Don't care if we don't have three values
            if len(opts) < 3:
                continue
            if opts[1] == "coffee" and opts[2] == "expression":
                # (parent, loop-nest, rank)
                try:
                    parent = env["node_parent"]
                except KeyError:
                    parent = None
                return {o: (parent, None, o.children[0].rank)}
        return {}

    def visit_For(self, o, env):
        args = self.visit_Node(o, env)
        # Nothing inside this for loop was annotated (we didn't see a
        # Perfect node with #pragma coffee expression)
        if len(args) == 0:
            return {}
        ret = OrderedDict()
        try:
            parent = env["node_parent"]
        except KeyError:
            parent = None
        me = (o, parent)
        for k, v in args.iteritems():
            p, nest, rank = v
            if nest is None:
                # Statement is directly underneath this loop, so the
                # loop nest structure is just the current loop
                nest = (me, )
            else:
                # Inside a nested set of loops, so prepend current
                # loop info to nest structure
                nest = list(itertools.chain((me, ), nest))
            # Merge with updated nest info
            ret[k] = (p, nest, rank)
        return ret


class SymbolReferences(Visitor):

    """
    Visit the tree and return a dict mapping symbol names to tuples of
    (node, node_parent) that reference the symbol with that name.
    The node is the Symbol node with said name, the node_parent is the
    parent of that node.

    By default the top-level call to visit will record a node_parent
    of None for the visited Node.  To provide one, pass an
    environment in to the visitor::

    .. code-block::

       v.visit(node, {"node_parent": parent})

    """

    def visit_Symbol(self, o, env):
        # Map name to (node, parent) tuple
        try:
            parent = env["node_parent"]
        except KeyError:
            parent = None
        return {o.symbol: [(o, parent)]}

    def visit_object(self, o, env):
        # Identity
        return {}

    def visit_list(self, o, env):
        ret = defaultdict(list)
        for entry in o:
            a = self.visit(entry, env=env)
            for k, v in a.iteritems():
                ret[k].extend(v)
        return ret

    def visit_Node(self, o, env):
        ops, _ = o.operands()
        new_env = Environment(env, node_parent=o)
        args = [self.visit(op, env=new_env) for op in ops]
        ret = defaultdict(list)
        # Merge dicts
        for a in args:
            for k, v in a.iteritems():
                ret[k].extend(v)
        return ret


class SymbolDependencies(Visitor):

    """
    Visit the tree and return a dict collecting symbol dependencies.

    The returned dict contains both maps from nodes to a (possibly
    empty) loop list the symbol depends on, and maps symbol names to
    WRITE access.
    """

    def visit_Symbol(self, o, env):
        # Empty loop nest for starters
        return {o: []}

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

    # Don't treat funcalls like perfect, but rather nodes
    visit_FunCall = visit_Node

    def visit_Perfect(self, o, env):
        ret = OrderedDict()
        lvalue = self.visit(o.children[0], env=env)
        for k, v in lvalue.iteritems():
            ret[k] = v
            # lvalue name is WRITTEN
            ret[k.symbol] = WRITE
        for r in o.children[1:]:
            d = self.visit(r, env=env)
            ret.update(d)
        return ret

    def visit_For(self, o, env):
        # Don't care about symbol access in increments, only children
        args = [self.visit(op, env=env) for op in o.children]
        ret = OrderedDict()
        # First merge args
        for a in args:
            ret.update(a)
        # Now fill in symbol loop dependencies.  If the symbol name is
        # WRITTEN anywhere, then it's dependent on all loops in the
        # current nest, otherwise only the ones that access it.
        for k, v in ret.iteritems():
            if type(k) is not Symbol:
                # name, don't care about it here.
                continue
            mode = ret.get(k.symbol, READ)
            if (mode == WRITE) or (o.dim in k.rank):
                ret[k] = [o] + v
        return ret


class SymbolModes(Visitor):

    """
    Visit the tree and return a dict mapping Symbols to tuples of
    (access mode, parent class).

    :class:`~.Symbol`\s are accessed as READ-only unless they appear
    as lvalues in a :class:`~.Perfect` statement.

    By default the top-level call to visit will record a parent class
    of NoneType for Symbols without a parent.  To pass in a parent by
    hand, provide an environment to the visitor.

    .. code-block::

       v.visit(symbol, {"node_parent": parent})

    """
    def visit_object(self, o, env):
        return {}

    def visit_Node(self, o, env):
        new_env = Environment(env, node_parent=o)
        ret = OrderedDict()
        ops, _ = o.operands()
        for op in ops:
            # WARNING, if the same Symbol object appears multiple
            # times, the "last" access wins, rather than WRITE winning.
            # This assumes all nodes in the tree are unique instances
            ret.update(self.visit(op, env=new_env))
        return ret

    def visit_Symbol(self, o, env):
        try:
            mode = env["access_mode"]
        except KeyError:
            mode = READ
        try:
            parent = env["node_parent"]
        except KeyError:
            parent = None
        return {o: (mode, parent.__class__)}

    # Don't do anything with declarations.  If you want lvalues to get
    # a WRITE unless uninitialised, then custom visitor must be
    # written.
    visit_Decl = visit_object

    def visit_Perfect(self, o, env):
        new_env = Environment(env, node_parent=o)
        write_env = Environment(new_env, access_mode=WRITE)
        ret = OrderedDict()
        # lvalues have access mode WRITE
        ret.update(self.visit(o.children[0], env=write_env))
        # All others have access mode READ
        for op in o.children[1:]:
            ret.update(self.visit(op, env=new_env))
        return ret


class SymbolDeclarations(Visitor):

    """Return a dict mapping symbol names to a tuple of the declaring
    node.  The node is annotated in place with information about
    whether it is a LOCAL declaration or EXTERNAL (via a function
    argument).
    """
    def visit_object(self, o, env):
        return {}

    def visit_Node(self, o, env, *args, **kwargs):
        ret = OrderedDict()
        for a in args:
            # WARNING, last sight of symbol wins.
            ret.update(a)
        return ret

    def visit_FunDecl(self, o, env):
        new_env = Environment(env, scope=EXTERNAL)
        ret = OrderedDict()
        for op in o.args:
            ret.update(self.visit(op, env=new_env))
        return ret

    def visit_Decl(self, o, env):
        # FIXME: be side-effect free
        try:
            o.scope = env["scope"]
        except KeyError:
            o.scope = LOCAL
        return {o.sym.symbol: o}


class FindInstances(Visitor):

    """
    Visit the tree and return a dict mapping types to a list of
    instances of that type in the tree.

    :arg types: list of types or single type to search for in the
          tree.
    :arg stop_when_found: optional, don't traverse the children of
          matching types.
    """

    def __init__(self, types, stop_when_found=False):
        self.types = types
        self.stop_when_found = stop_when_found
        super(FindInstances, self).__init__()

    def visit_object(self, o, env):
        return {}

    def visit_list(self, o, env):
        ret = defaultdict(list)
        for entry in o:
            a = self.visit(entry, env=env)
            for k, v in a.iteritems():
                ret[k].extend(v)
        return ret

    def visit_Node(self, o, env):
        ret = defaultdict(list)
        if isinstance(o, self.types):
            ret[type(o)].append(o)
            # Don't traverse children if stop-on-found
            if self.stop_when_found:
                return ret
        # Not found, or traversing children anyway
        ops, _ = o.operands()
        for op in ops:
            a = self.visit(op, env=env)
            for k, v in a.iteritems():
                ret[k].extend(v)
        return ret