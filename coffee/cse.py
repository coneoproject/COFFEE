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

from __future__ import absolute_import, print_function, division
from six.moves import zip

import operator

from .base import *
from .utils import *
from coffee.visitors import EstimateFlops
from .expression import MetaExpr
from .logger import log, COST_MODEL
from functools import reduce


class Temporary(object):

    """A Temporary stores useful information for a statement (e.g., an Assign
    or an AugmentedAssig) that computes a temporary variable; that is, a variable
    that is read in more than one place."""

    def __init__(self, node, main_loop, nest, linear_reads_costs=None):
        self.level = -1
        self.pushed = False
        self.readby = []

        self.node = node
        self.main_loop = main_loop
        self.nest = nest
        self.linear_reads_costs = linear_reads_costs or OrderedDict()
        self.flops = EstimateFlops().visit(node)

    @property
    def name(self):
        return self.symbol.symbol if self.symbol else None

    @property
    def rank(self):
        return self.symbol.rank if self.symbol else None

    @property
    def linearity_degree(self):
        return len(self.main_linear_loops)

    @property
    def symbol(self):
        if isinstance(self.node, Writer):
            return self.node.lvalue
        elif isinstance(self.node, Symbol):
            return self.node
        else:
            return None

    @property
    def expr(self):
        if isinstance(self.node, Writer):
            return self.node.rvalue
        else:
            return None

    @property
    def urepr(self):
        return self.symbol.urepr

    @property
    def reads(self):
        return FindInstances(Symbol).visit(self.expr)[Symbol] if self.expr else []

    @property
    def linear_reads(self):
        return self.linear_reads_costs.keys() if self.linear_reads_costs else []

    @property
    def loops(self):
        return list(zip(*self.nest))[0]

    @property
    def main_linear_loops(self):
        return [l for l in self.main_loops if l.is_linear]

    @property
    def main_linear_nest(self):
        return [(l, p) for l, p in self.main_nest if l in self.linear_loops]

    @property
    def main_loops(self):
        index = self.loops.index(self.main_loop)
        return [l for l in self.loops[:index + 1]]

    @property
    def main_nest(self):
        return [(l, p) for l, p in self.nest if l in self.main_loops]

    @property
    def flops_projection(self):
        # #muls + #sums
        nmuls = len(self.linear_reads)
        return (nmuls) + (nmuls - 1)

    @property
    def is_ssa(self):
        return self.symbol not in self.readby

    @property
    def is_static_init(self):
        return isinstance(self.expr, ArrayInit)

    @property
    def is_increment(self):
        return isinstance(self.node, Incr)

    @property
    def reductions(self):
        return [l for l in self.main_loops if l.dim not in self.rank]

    @property
    def nreductions(self):
        return len(self.reductions)

    def niters(self, mode='all', handle=None):
        assert mode in ['all', 'outer', 'nonlinear', 'in', 'out']
        handle = handle or []
        limit = self.loops.index(self.main_loop)
        loops = self.loops[:limit + 1]
        if mode == 'all':
            sizes = [l.size for l in loops]
        elif mode == 'outer':
            sizes = [l.size for l in loops if l is not self.main_loop]
        elif mode == 'nonlinear':
            sizes = [l.size for l in loops if not l.is_linear]
        elif mode == 'in':
            sizes = [l.size for l in loops if l.dim in handle]
        else:
            sizes = [l.size for l in loops if l.dim not in handle]
        return reduce(operator.mul, sizes, 1)

    def depends(self, others):
        """Return True if ``self`` reads a temporary or is read by a temporary
        that appears in the iterator ``others``, False otherwise."""
        dependencies = self.linear_reads + self.reads
        for t in others:
            if any(s.urepr == t.urepr for s in dependencies):
                return True
        return False

    def reconstruct(self):
        temporary = Temporary(self.node, self.main_loop, self.nest,
                              OrderedDict(self.linear_reads_costs))
        temporary.level = self.level
        temporary.readby = list(self.readby)
        return temporary

    def __str__(self):
        return "%s: level=%d, flops/iter=%d, linear_reads=[%s], isread=[%s]" % \
            (self.symbol, self.level, self.flops,
             ", ".join([str(i) for i in self.linear_reads]),
             ", ".join([str(i) for i in self.readby]))


class CSEUnpicker(object):

    """Analyze loops in which some temporary variables are computed and, applying
    a cost model, decides whether to leave a temporary intact or inline it for
    creating factorization and code motion opportunities.

    The cost model exploits one particular property of loops, namely linearity in
    symbols (further information concerning loop linearity is available in the module
    ``expression.py``)."""

    def __init__(self, exprs, header, hoisted, decls):
        self.exprs = exprs
        self.header = header
        self.hoisted = hoisted
        self.decls = decls

    @property
    def type(self):
        return self.exprs.values()[0].type

    @property
    def linear_dims(self):
        return self.exprs.values()[0].linear_dims

    def _push_temporaries(self, temporaries, trace, global_trace, ra):

        def is_pushable(temporary, temporaries):
            # To be pushable ...
            if not temporary.is_ssa:
                # ... must be written only once
                return False
            if not temporary.readby:
                # ... must actually be read by some other temporaries (the output
                # variables are not)
                return False
            if temporary.is_static_init:
                # ... its rvalue must not be an array initializer
                return False
            if temporary.depends(temporaries):
                # ... it cannot depend on other temporaries in the same level
                return False
            pushed_in = [global_trace.get(rb.urepr) for rb in temporary.readby]
            pushed_in = set(rb.main_loop.children[0] for rb in pushed_in if rb)
            reads = [s for s in temporary.reads if not s.is_number]
            for s in reads:
                # ... all the read temporaries must be accessible in the loops in which
                # they will be pushed
                if s.urepr in global_trace and global_trace[s.urepr].pushed:
                    continue
                if any(l not in ra[self.decls[s.symbol]] for l in pushed_in):
                    return False
            return True

        to_replace, modified_temporaries = {}, OrderedDict()
        for t in temporaries:
            # Track temporaries to be pushed from /level-1/ into the later /level/s
            if not is_pushable(t, temporaries):
                continue
            to_replace[t.symbol] = t.expr or t.symbol
            for rb in t.readby:
                modified_temporaries[rb.urepr] = trace.get(rb.urepr,
                                                           global_trace[rb.urepr])
            # The temporary is going to be pushed, so we can remove it as long as
            # it is not needed somewhere else
            if t.node in t.main_loop.body and\
                    all(rb.urepr in global_trace for rb in t.readby):
                global_trace[t.urepr].pushed = True
                t.main_loop.body.remove(t.node)
                self.decls.pop(t.name, None)

        # Transform the AST (note: node replacement must happen in the order
        # in which the temporaries have been encountered)
        modified_temporaries = sorted(modified_temporaries.values(),
                                      key=lambda t: global_trace.keys().index(t.urepr))
        for t in modified_temporaries:
            ast_replace(t.node, to_replace, copy=True)
        replaced = [t.urepr for t in to_replace.keys()]

        # Update the temporaries
        for t in modified_temporaries:
            for r, c in t.linear_reads_costs.items():
                if r.urepr in replaced:
                    t.linear_reads_costs.pop(r)
                    r_linear_reads_costs = global_trace[r.urepr].linear_reads_costs
                    for p, p_c in r_linear_reads_costs.items() or [(r, 0)]:
                        t.linear_reads_costs[p] = c + p_c

    def _transform_temporaries(self, temporaries):
        from .rewriter import ExpressionRewriter

        # Never attempt to transform the main expression
        temporaries = [t for t in temporaries if t.node not in self.exprs]

        lda = loops_analysis(self.header, key='symbol', value='dim')

        # Expand + Factorize
        rewriters = OrderedDict()
        for t in temporaries:
            expr_info = MetaExpr(self.type, t.main_loop.block, t.main_nest)
            ew = ExpressionRewriter(t.node, expr_info, self.decls, self.header,
                                    self.hoisted)
            ew.replacediv()
            ew.expand(mode='all', lda=lda)
            ew.reassociate(lambda i: all(r != t.main_loop.dim for r in lda[i.symbol]))
            ew.factorize(mode='adhoc', adhoc={i.urepr: [] for i in t.linear_reads}, lda=lda)
            rewriters[t] = ew

        lda = loops_analysis(self.header, value='dim')

        # Code motion
        for t, ew in rewriters.items():
            ew.licm(mode='only_outlinear', lda=lda, global_cse=True)
            if t.linearity_degree > 1:
                ew.licm(mode='only_linear', lda=lda)

    def _analyze_expr(self, expr, loop, lda):
        finder = FindInstances(Symbol)
        reads = finder.visit(expr, ret=FindInstances.default_retval())[Symbol]
        reads = [s for s in reads if s.symbol in self.decls]
        syms = [s for s in reads if any(d in loop.dim for d in lda[s])]

        linear_reads_costs = OrderedDict()

        def wrapper(node, found=0):
            if isinstance(node, Symbol):
                if node in syms:
                    linear_reads_costs.setdefault(node, 0)
                    linear_reads_costs[node] += found
                return
            elif isinstance(node, (EmptyStatement, ArrayInit)):
                return
            elif isinstance(node, (Prod, Div)):
                found += 1
            operands = list(zip(*explore_operator(node)))[0]
            for o in operands:
                wrapper(o, found)
        wrapper(expr)

        return reads, linear_reads_costs

    def _analyze_loop(self, loop, nest, lda, global_trace):
        linear_dims = [l.dim for l, _ in nest if l.is_linear]

        trace = OrderedDict()
        for node in loop.body:
            if not isinstance(node, Writer):
                not_ssa = [trace[w] for w in in_written(node, key='urepr') if w in trace]
                for t in not_ssa:
                    t.readby.append(t.symbol)
                continue
            reads, linear_reads_costs = self._analyze_expr(node.rvalue, loop, lda)
            affected = [s for s in reads if any(i in linear_dims for i in lda[s])]
            for s in affected:
                if s.urepr in global_trace:
                    temporary = global_trace[s.urepr]
                    temporary.readby.append(node.lvalue)
                    temporary = temporary.reconstruct()
                    temporary.level = -1
                    trace[s.urepr] = temporary
                else:
                    temporary = trace.setdefault(s.urepr, Temporary(s, loop, nest))
                    temporary.readby.append(node.lvalue)
            new_temporary = Temporary(node, loop, nest, linear_reads_costs)
            new_temporary.level = max([trace[s.urepr].level for s
                                       in new_temporary.linear_reads] or [-2]) + 1
            trace[node.lvalue.urepr] = new_temporary

        return trace

    def _group_by_level(self, trace):
        levels = defaultdict(list)

        for temporary in trace.values():
            levels[temporary.level].append(temporary)
        return levels

    def _cost_cse(self, levels, bounds=None):
        if bounds is not None:
            lb, up = bounds[0], bounds[1] + 1
            levels = {i: levels[i] for i in range(lb, up)}
        cost = 0
        for level, temporaries in levels.items():
            cost += sum(t.flops*t.niters('all') for t in temporaries)
        return cost

    def _cost_fact(self, trace, levels, lda, bounds):
        # Check parameters
        assert len(bounds) == 2 and bounds[1] >= bounds[0]
        assert bounds[0] in levels.keys() and bounds[1] in levels.keys()

        # Determine current costs of individual loop regions
        input_cost = self._cost_cse(levels, (min(levels.keys()), max(levels.keys())))
        uptolevel_cost, post_cse_cost = input_cost, input_cost
        level_inloop_cost, total_outloop_cost = 0, 0

        # We are going to modify a copy of the temporaries dict
        new_trace = OrderedDict()
        for s, t in trace.items():
            new_trace[s] = t.reconstruct()

        # Cost induced by the untransformed temporaries
        pre_cse_cost = self._cost_cse(levels, (min(levels.keys()), bounds[0]))

        best = (bounds[0], bounds[0], uptolevel_cost)
        fact_levels = {k: v for k, v in levels.items() if k > bounds[0] and k <= bounds[1]}
        for level, temporaries in sorted(fact_levels.items(), key=lambda i_j: i_j[0]):
            level_inloop_cost = 0
            for t in temporaries:
                # Compute the cost induced by /t/ in the outer loops after fact+licm
                t_outloop_cost, linear_reads = 0, []
                for read, cost in t.linear_reads_costs.items():
                    traced = new_trace.get(read.urepr)
                    if traced and traced.level >= bounds[0]:
                        handle = traced.linear_reads or [read]
                        if cost:
                            for i in handle:
                                # One prod in the closest linear loop
                                t_outloop_cost += t.niters('out', lda[i])
                                # The rest falls outside of the linear loops
                                t_outloop_cost += (cost - 1)*t.niters('nonlinear')
                    else:
                        handle = [read]
                    linear_reads.extend(handle)
                factors = {as_urepr(i): i for i in linear_reads}.values()
                # Take into account the increased number of sums (due to fact)
                hoist_region = set.union(*[lda[i] for i in factors])
                niters = t.niters('out', hoist_region)
                t_outloop_cost += (len(linear_reads) - len(factors))*niters
                total_outloop_cost += t_outloop_cost

                # Compute the cost induced by /t/ in the main loop after fact+licm
                # We end up creating n prods and n -1 sums
                t_inloop_cost = 2*len(factors) - 1
                level_inloop_cost += t_inloop_cost*t.niters('all')

                # Take into account any hoistable reductions
                if t.is_increment:
                    for i in factors:
                        handle = [l.dim for l in t.reductions if l.dim not in i.rank]
                        level_inloop_cost -= t.niters('all') - t.niters('out', handle)

                # Keep the trace up-to-date
                linear_reads_costs = {i: 1 for i in factors}
                new_trace[t.urepr].linear_reads_costs = linear_reads_costs

            # Some temporaries within levels < /level/ might also appear in
            # subsequent loops or levels beyond /level/, so they still contribute
            # to the operation count
            for t in list(flatten([levels[j] for j in range(level)])):
                if any(rb.urepr not in new_trace for rb in t.readby) or \
                        any(new_trace[rb.urepr].level > level for rb in t.readby):
                    # Note: condition 1) is basically saying "if I'm read by
                    # a temporary that is not in this loop's trace, then I must
                    # be read in some other loops".
                    level_inloop_cost += \
                        new_trace[t.urepr].flops_projection*t.niters('all')

            post_cse_cost = self._cost_cse(fact_levels, (level + 1, bounds[1]))

            # Compute the total cost
            total_inloop_cost = pre_cse_cost + level_inloop_cost + post_cse_cost
            uptolevel_cost = total_outloop_cost + total_inloop_cost

            # Update the best alternative
            if uptolevel_cost < best[2]:
                best = (bounds[0], level, uptolevel_cost)

            log('[CSE]: unpicking between [%d, %d]:' % (bounds[0], level), COST_MODEL)
            log('       flops: %d -> %d (hoist=%d, preCSE=%d, fact=%d, postCSE=%d)' %
                (input_cost, uptolevel_cost, total_outloop_cost, pre_cse_cost,
                 level_inloop_cost, post_cse_cost), COST_MODEL)

        return best

    def unpick(self):
        # Collect all necessary info
        external_decls = [d for d in self.decls.values() if d.scope == EXTERNAL]
        fors = visit(self.header, info_items=['fors'])['fors']
        lda = loops_analysis(self.header, value='dim')

        # Collect all loops to be analyzed
        nests = OrderedDict()
        for nest in fors:
            for loop, parent in nest:
                if loop.is_linear:
                    nests[loop] = nest

        # Analyze loops
        global_trace = OrderedDict()
        mapper = OrderedDict()
        for loop, nest in nests.items():
            trace = self._analyze_loop(loop, nest, lda, global_trace)
            if trace:
                mapper[loop] = trace
                global_trace.update(trace)

        for loop, trace in mapper.items():
            # Compute the best cost alternative
            levels = self._group_by_level(trace)
            min_level, max_level = min(levels.keys()), max(levels.keys())
            current_cost = self._cost_cse(levels, (min_level, max_level))
            global_best = (min_level, min_level, current_cost)
            for i in sorted(levels.keys()):
                local_best = self._cost_fact(trace, levels, lda, (i, max_level))
                if local_best[2] < global_best[2]:
                    global_best = local_best

            log("-- Best: [%d, %d] (cost=%d) --" % global_best, COST_MODEL)

            # Transform the loop
            for i in range(global_best[0] + 1, global_best[1] + 1):
                ra = reachability_analysis(self.header, external_decls)
                self._push_temporaries(levels[i-1], trace, global_trace, ra)
                self._transform_temporaries(levels[i])

        # Clean up
        for transformed_loop, nest in reversed(nests.items()):
            for loop, parent in nest:
                if loop == transformed_loop and not loop.body:
                    parent.children.remove(loop)
