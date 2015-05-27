from __future__ import absolute_import
import inspect

__all__ = ["Visitor"]


class Environment(object):

    """
    An environment containing state.

    This is effectively a dictionary that looks up unknown keys in a
    parent.  It only implements :data:`__getitem__`, and
    :data:`__setitem__` and exposes the
    dict directly as :attr:`mapping`.

    Used to implement "stacked" environments in :class:`Visitor`\s.

    :arg parent: The parent environment (pass an empty dict as an
         empty environment).
    :kwargs kwargs: values to set in the environment.
    """

    def __init__(self, parent, **kwargs):
        super(Environment, self).__init__()
        self.parent = parent
        self.mapping = dict(**kwargs)

    def __getitem__(self, key):
        """Look up an item in this :class:`Environment`.

        :arg key: The item to look up.
        """
        try:
            return self.mapping[key]
        except KeyError:
            return self.parent[key]

    def __repr__(self):
        mappings = ", ".join("%s=%r" % (k, v) for (k, v) in self.mapping.iteritems())
        return "Environment(parent=%r, %s)" % (self.parent, mappings)

    def __str__(self):
        dicts = []
        # Walk up stack, gathering mappings.
        while True:
            try:
                dicts.append(self.mapping)
            except AttributeError:
                # Hit top of stack
                dicts.append(self)
                break
            self = self.parent
        # Build environment from root down to self for printing
        vals = {}
        while True:
            try:
                vals.update(dicts.pop())
            except IndexError:
                break
        return "Environment: %s" % vals


class Visitor(object):

    """
    A generic visitor for a COFFEE AST.

    To define handlers, subclasses should define :data:`visit_Foo`
    methods for each class :data:`Foo` they want to handle.

    If a specific method for a class :data:`Foo` is not found, the MRO
    of the class is walked in order until a matching method is found.

    The method signature is:

    .. code-block::

       def visit_Foo(self, o, [*args, **kwargs]):
           pass

    The handler is responsible for visiting the children (if any) of
    the node :data:`o`.  :data:`*args` and :data:`**kwargs` may be
    used to pass information up and down the call stack.  You can also
    pass named keyword arguments, e.g.:

    .. code-block::

       def visit_Foo(self, o, parent=None, *args, **kwargs):
           pass
    """
    def __init__(self):
        handlers = {}
        # visit methods are spelt visit_Foo.
        prefix = "visit_"
        # Inspect the methods on this instance to find out which
        # handlers are defined.
        for (name, meth) in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith(prefix):
                continue
            # Check the argument specification
            # Valid options are:
            #    visit_Foo(self, o, [*args, **kwargs])
            argspec = inspect.getargspec(meth)
            if len(argspec.args) < 2:
                raise RuntimeError("Visit method signature must be visit_Foo(self, o, [*args, **kwargs])")
            handlers[name[len(prefix):]] = meth
        self._handlers = handlers

    """
    :attr:`default_args`. A dict of default keyword arguments for the
    visitor.

    These are not used by default in :meth:`visit`, however,
    a caller may pass them explicitly to :meth:`visit` by accessing
    :attr:`default_args`.  For example::

    .. code-block::

       v = FooVisitor()
       v.visit(node, **v.default_args)
    """
    default_args = {}

    def lookup_method(self, instance):
        """Look up a handler method for a visitee.

        :arg instance: The instance to look up a method for.
        """
        cls = instance.__class__
        try:
            # Do we have a method handler defined for this type name
            return self._handlers[cls.__name__]
        except KeyError:
            # No, walk the MRO.
            for klass in cls.mro()[1:]:
                entry = self._handlers.get(klass.__name__)
                if entry:
                    # Save it on this type name for faster lookup next time
                    self._handlers[cls.__name__] = entry
                    return entry
        raise RuntimeError("No handler found for class %s", cls.__name__)

    def visit(self, o, *args, **kwargs):
        """Apply this :class:`Visitor` to an AST.

        :arg o: The :class:`Node` to visit.
        :arg args: Optional arguments to pass to the visit methods.
        :arg kwargs: Optional keyword arguments to pass to the visit methods.
        """
        meth = self.lookup_method(o)
        return meth(o, *args, **kwargs)

    def reuse(self, o, *args, **kwargs):
        """A visit method to reuse a node, ignoring children."""
        return o

    def maybe_reconstruct(self, o, *args, **kwargs):
        """A visit method that reconstructs nodes if their children
        have changed."""
        ops, okwargs = o.operands()
        new_ops = [self.visit(op, *args, **kwargs) for op in ops]
        if all(a is b for a, b in zip(ops, new_ops)):
            return o
        return o.reconstruct(*new_ops, **okwargs)

    def always_reconstruct(self, o, *args, **kwargs):
        """A visit method that always reconstructs nodes."""
        ops, okwargs = o.operands()
        new_ops = [self.visit(op, *args, **kwargs) for op in ops]
        return o.reconstruct(*new_ops, **okwargs)
