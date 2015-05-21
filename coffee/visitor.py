from __future__ import absolute_import
import inspect

__all__ = ["Environment", "Visitor"]


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

    Pre- and post-order traversal is implemented in a single
    :meth:`visit` method, with the argument list for a method handler
    indicating if it expects transformed children or not.

    The two method signatures are:

    .. code-block::

       def visit_Foo(self, o, env):
           pass

    for pre-order traversal, in which case the handler should take
    care to traverse the children if necessary.  And:

    .. code-block::

       def visit_Foo(self, o, env, *args, **kwargs):
           pass

    for post-order traversal, in which case handlers will be called on
    the children first and passed in through :data:`*args`.
    :data:`**kwargs` will contain any keyword arguments that were used
    in the construction of the instance, see
    :meth:`~.Node.reconstruct` and :meth:`~.Node.operands` for more details.

    In both cases, the instance itself is passed in as :data:`o` and
    the visitor also receives a :class:`Environment` object in
    :data:`env` which it may inspect if wished.
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
            #    visit_Foo(self, o, env)
            # or
            #    visit_Foo(self, o, env, *args, **kwargs)
            #
            # The latter indicates that the visit handler expects
            # the children to have already been visited, the
            # former indicates that the method will handle it
            # itself.
            argspec = inspect.getargspec(meth)
            if len(argspec.args) != 3:
                raise RuntimeError("Visit method signature must be visit_Foo(self, o, env, [*args, **kwargs])")
            children_first = argspec.varargs is not None
            if children_first and argspec.keywords is None:
                raise RuntimeError("Post-order visitor must be visit_Foo(self, o, env, *args, **kwargs)")
            handlers[name[len(prefix):]] = (meth, children_first)
        self._handlers = handlers

    """
    :attr:`default_env`. Provide the visitor with a default environment.
    This environment is not used by default in :meth:`visit`, however,
    a caller may obtain it to pass to :meth:`visit` by accessing
    :attr:`default_env`.  For example::

    .. code-block::

       v = FooVisitor()
       v.visit(node, env=v.default_env)
    """
    default_env = None

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

    def visit(self, o, env=None):
        """Apply this :class:`Visitor` to an AST.

        :arg o: The :class:`Node` to visit.
        :arg env: An optional :class:`Environment` to pass to the
             visitor.
        """
        meth, children_first = self.lookup_method(o)

        if env is None:
            # Default to empty environment
            env = {}

        if children_first:
            # Visit children then call handler on us
            ops, kwargs = o.operands()
            return meth(o, env, *[self.visit(op, env=env) for op in ops], **kwargs)
        else:
            # Handler deals with children
            return meth(o, env)

    def reuse(self, o, env):
        """A visit method to reuse a node, ignoring children."""
        return o

    def maybe_reconstruct(self, o, env, *args, **kwargs):
        """A visit method that reconstructs nodes if their children
        have changed."""
        oargs, okwargs = o.operands()
        if all(a is b for a, b in zip(oargs, args)) and \
           okwargs == kwargs:
            return o
        return o.reconstruct(*args, **kwargs)

    def always_reconstruct(self, o, env, *args, **kwargs):
        """A visit method that always reconstructs nodes."""
        return o.reconstruct(*args, **kwargs)
