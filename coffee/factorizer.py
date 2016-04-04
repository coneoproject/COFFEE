# This file is part of COFFEE
#
# COFFEE is Copyright (c) 2016, Imperial College London.
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

import operator

from base import *
from utils import *


class Term():
    """A Term represents a product between 'operands' and 'factors'. In a
    product /a*(b+c)/, /a/ is the 'operand', while /b/ and /c/ are the 'factors'.
    The symbol /+/ is the 'op' of the Term.
    """

    def __init__(self, operands, factors=None, op=None):
        self.operands = operands
        self.factors = factors or []
        self.op = op

    @property
    def operands_ast(self):
        return ast_make_expr(Prod, self.operands)

    @property
    def factors_ast(self):
        return ast_make_expr(self.op, self.factors)

    @property
    def generate_ast(self):
        if len(self.factors) == 0:
            return self.operands_ast
        elif len(self.operands) == 0:
            return self.factors_ast
        elif len(self.factors) == 1 and \
                all(isinstance(i, Symbol) and i.symbol == 1.0 for i in self.factors):
            return self.operands_ast
        else:
            return Prod(self.operands_ast, self.factors_ast)

    def add_operands(self, operands):
        for o in operands:
            if o not in self.operands:
                self.operands.append(o)

    def remove_operands(self, operands):
        for o in operands:
            if o in self.operands:
                self.operands.remove(o)

    def add_factors(self, factors):
        for f in factors:
            if f not in self.factors:
                self.factors.append(f)

    def remove_factors(self, factors):
        for f in factors:
            if f in self.factors:
                self.factors.remove(f)

    @staticmethod
    def process(symbols, should_factorize, op=None):
        operands = [s for s in symbols if should_factorize(s)]
        factors = [s for s in symbols if not should_factorize(s)]
        return Term(operands, factors, op)


class Factorizer():

    def __init__(self, stmt):
        self.stmt = stmt

    def _simplify_sum(self, terms):
        unique_terms = OrderedDict()
        for t in terms:
            unique_terms.setdefault(str(t.generate_ast), list()).append(t)

        for t_repr, t_list in unique_terms.items():
            occurrences = len(t_list)
            unique_terms[t_repr] = t_list[0]
            if occurrences > 1:
                unique_terms[t_repr].add_factors([Symbol(occurrences)])

        terms[:] = unique_terms.values()

    def _heuristic_collection(self, terms):
        if not self.heuristic or any(t.operands for t in terms):
            return
        tracker = OrderedDict()
        for t in terms:
            symbols = [s for s in t.factors if isinstance(s, Symbol)]
            for s in symbols:
                tracker.setdefault(s.urepr, []).append(t)
        reverse_tracker = OrderedDict()
        for s, ts in tracker.items():
            reverse_tracker.setdefault(tuple(ts), []).append(s)
        # 1) At least one symbol appearing in all terms: use that as operands ...
        operands = [(ts, s) for ts, s in reverse_tracker.items() if ts == tuple(terms)]
        # 2) ... Or simply pick operands greedily
        if not operands:
            handled = set()
            for ts, s in reverse_tracker.items():
                if len(ts) > 1 and all(t not in handled for t in ts):
                    operands.append((ts, s))
                    handled |= set(ts)
        for ts, s in operands:
            for t in ts:
                new_operands = [i for i in t.factors if
                                isinstance(i, Symbol) and i.urepr in s]
                t.remove_factors(new_operands)
                t.add_operands(new_operands)

    def _premultiply_symbols(self, symbols):
        floats = [s for s in symbols if isinstance(s.symbol, (int, float))]
        if len(floats) > 1:
            other_symbols = [s for s in symbols if s not in floats]
            prem = reduce(operator.mul, [s.symbol for s in floats], 1.0)
            prem = [Symbol(prem)] if prem not in [1, 1.0] else []
            return prem + other_symbols
        else:
            return symbols

    def _filter(self, factorizable_term):
        o = factorizable_term.operands_ast
        grp = self.adhoc.get(o.urepr, []) if isinstance(o, Symbol) else []
        if not grp:
            return False
        for f in factorizable_term.factors:
            retval = FindInstances.default_retval()
            symbols = FindInstances(Symbol).visit(f, ret=retval)[Symbol]
            if any(s.urepr in grp for s in symbols):
                return False
        return True

    def _factorize(self, node, parent):
        if isinstance(node, Symbol):
            return Term.process([node], self.should_factorize)

        elif isinstance(node, Par):
            return self._factorize(node.child, node)

        elif isinstance(node, (FunCall, Div)):
            # Try to factorize /within/ the children, but then return saying
            # "I'm not factorizable any further"
            for n in node.children:
                self._factorize(n, node)
            return Term([], [node])

        elif isinstance(node, Prod):
            children = explore_operator(node)
            symbols = [n for n, _ in children if isinstance(n, Symbol)]
            other_nodes = [(n, p) for n, p in children if n not in symbols]
            symbols = self._premultiply_symbols(symbols)
            factorized = Term.process(symbols, self.should_factorize, Prod)
            terms = [self._factorize(n, p) for n, p in other_nodes]
            for t in terms:
                factorized.add_operands(t.operands)
                factorized.add_factors(t.factors)
            return factorized

        # The fundamental case is when /node/ is a Sum (or Sub, equivalently).
        # Here, we try to factorize the terms composing the operation
        elif isinstance(node, (Sum, Sub)):
            children = explore_operator(node)
            # First try to factorize within /node/'s children
            terms = [self._factorize(n, p) for n, p in children]
            # Check if it's possible to aggregate operations
            # Example: replace (a*b)+(a*b) with 2*(a*b)
            self._simplify_sum(terms)
            # No global factorization rule is used, so just try to maximize
            # factorization within /this/ Sum/Sub
            self._heuristic_collection(terms)
            # Finally try to factorize some of the operands composing the operation
            factorized = OrderedDict()
            for t in terms:
                operand = [t.operands_ast] if t.operands else []
                factor = [t.factors_ast] if t.factors else [Symbol(1.0)]
                factorizable_term = Term(operand, factor, node.__class__)
                if self._filter(factorizable_term):
                    # Skip
                    factorized[t] = t
                else:
                    # Do factorize
                    _t = factorized.setdefault(str(t.operands_ast), factorizable_term)
                    _t.add_factors(factor)
            factorized = [t.generate_ast for t in factorized.values()]
            factorized = ast_make_expr(Sum, factorized)
            parent.children[parent.children.index(node)] = factorized
            return Term([], [factorized])

        else:
            return Term([], [node])

    def factorize(self, should_factorize, **kwargs):
        expressions = kwargs.get('subexprs', [(self.stmt.rvalue, self.stmt)])
        adhoc = kwargs.get('adhoc', {})

        self.should_factorize = should_factorize
        self.adhoc = adhoc if any(v for v in adhoc.values()) else {}
        self.heuristic = kwargs.get('heuristic', False)

        for node, parent in expressions:
            self._factorize(node, parent)
