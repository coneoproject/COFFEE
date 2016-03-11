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

from sys import maxint

from base import *
from utils import *
from coffee.visitors import EstimateFlops
from expression import MetaExpr


class Temporary():

    def __init__(self, node, loop, reads_costs=None):
        self.level = -1
        self.node = node
        self.loop = loop
        self.reads_costs = reads_costs or {}
        self.is_read = []
        self.cost = EstimateFlops().visit(node)

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
        return self.reads_costs.keys() if self.reads_costs else []

    @property
    def project(self):
        return len(self.reads)

    @property
    def dependencies(self):
        return [s.urepr for s in self.reads]

    def dependson(self, other):
        return other.urepr in self.dependencies

    def reconstruct(self):
        temporary = Temporary(self.node, self.loop, dict(self.reads_costs))
        temporary.level = self.level
        temporary.is_read = list(self.is_read)
        return temporary

    def __str__(self):
        return "%s: level=%d, cost=%d, reads=[%s], isread=[%s]" % \
            (self.symbol, self.level, self.cost,
             ", ".join([str(i) for i in self.reads]),
             ", ".join([str(i) for i in self.is_read]))


class CSEUnpicker():

    def __init__(self, stmt, expr_info, header, hoisted, decls, expr_graph):
        self.stmt = stmt
        self.expr_info = expr_info
        self.header = header
        self.hoisted = hoisted
        self.decls = decls
        self.expr_graph = expr_graph

    def _push_temporaries(self, trace, levels, loop, cur_level, global_trace):
        assert [i in levels.keys() for i in [cur_level-1, cur_level]]

        # Remove temporaries being pushed
        for t in levels[cur_level-1]:
            if t.node in t.loop.body and all(ir.urepr in trace for ir in t.is_read):
                t.loop.body.remove(t.node)

        # Track temporaries to be pushed from /level-1/ into the later /level/s
        to_replace, modified_temporaries = {}, OrderedDict()
        for t in levels[cur_level-1]:
            to_replace[t.symbol] = t.expr or t.symbol
            for ir in t.is_read:
                modified_temporaries[ir.urepr] = trace.get(ir.urepr,
                                                           global_trace[ir.urepr])

        # Update the temporaries
        replaced = [t.urepr for t in to_replace.keys()]
        for t in modified_temporaries.values():
            ast_replace(t.node, to_replace, copy=True)
            for r, c in t.reads_costs.items():
                if r.urepr in replaced:
                    t.reads_costs.pop(r)
                    for p, p_c in global_trace[r.urepr].reads_costs.items() or [(r, 0)]:
                        t.reads_costs[p] = c + p_c

    def _transform_temporaries(self, temporaries, loop, nest):
        from rewriter import ExpressionRewriter
        lda = ldanalysis(self.header, key='symbol', value='dim')

        # Expand + Factorize
        rewriters = OrderedDict()
        for t in temporaries:
            expr_info = MetaExpr(self.expr_info.type, loop.children[0], nest, (loop.dim,))
            ew = ExpressionRewriter(t.node, expr_info, self.decls, self.header,
                                    self.hoisted, self.expr_graph)
            ew.replacediv()
            ew.expand(mode='all', not_aggregate=True, lda=lda)
            ew.factorize(mode='adhoc', adhoc={i.urepr: [] for i in t.reads}, lda=lda)
            ew.factorize(mode='heuristic')
            rewriters[t] = ew

        lda = ldanalysis(self.header, value='dim')

        # Code motion
        for t, ew in rewriters.items():
            ew.licm(mode='only_outdomain', lda=lda, global_cse=True)

    def _analyze_expr(self, expr, loop, lda):
        finder = FindInstances(Symbol)
        syms = finder.visit(expr, ret=FindInstances.default_retval())[Symbol]
        syms = [s for s in syms
                if any(l in self.expr_info.domain_dims for l in lda[s])]

        syms_costs = defaultdict(int)

        def wrapper(node, found=0):
            if isinstance(node, Symbol):
                if node in syms:
                    syms_costs[node] += found
                return
            elif isinstance(node, (Prod, Div)):
                found += 1
            operands = zip(*explore_operator(node))[0]
            for o in operands:
                wrapper(o, found)
        wrapper(expr)

        return syms_costs

    def _analyze_loop(self, loop, lda, global_trace):
        trace = OrderedDict()

        for stmt in loop.body:
            if not isinstance(stmt, Writer):
                continue
            syms_costs = self._analyze_expr(stmt.rvalue, loop, lda)
            for s in syms_costs.keys():
                if s.urepr in global_trace:
                    temporary = global_trace[s.urepr]
                    temporary.is_read.append(stmt.lvalue)
                    temporary = temporary.reconstruct()
                    temporary.level = -1
                    trace[s.urepr] = temporary
                else:
                    temporary = trace.setdefault(s.urepr, Temporary(s, loop))
                    temporary.is_read.append(stmt.lvalue)
            new_temporary = Temporary(stmt, loop, syms_costs)
            new_temporary.level = max([trace[s.urepr].level for s in new_temporary.reads]) + 1
            trace[stmt.lvalue.urepr] = new_temporary

        return trace

    def _group_by_level(self, trace):
        levels = defaultdict(list)

        for temporary in trace.values():
            levels[temporary.level].append(temporary)
        return levels

    def _cost_cse(self, loop, levels, keys=None):
        if keys is not None:
            levels = {k: levels[k] for k in keys}
        cost = 0
        for level, temporaries in levels.items():
            cost += sum(t.cost for t in temporaries)
        return cost*loop.size

    def _cost_fact(self, trace, levels, loop, bounds):
        # Check parameters
        bounds = bounds or (min(levels.keys()), max(levels.keys()))
        assert len(bounds) == 2 and bounds[1] >= bounds[0]
        assert [i in levels.keys() for i in bounds]
        fact_levels = OrderedDict([(k, v) for k, v in levels.items()
                                   if k > bounds[0] and k <= bounds[1]])

        # Cost of levels that won't be factorized
        cse_cost = self._cost_cse(loop, levels, range(min(levels.keys()), bounds[0] + 1))

        # We are going to modify a copy of the temporaries dict
        new_trace = OrderedDict()
        for s, t in trace.items():
            new_trace[s] = t.reconstruct()

        best = (bounds[0], bounds[0], maxint)
        total_outloop_cost = 0
        for level, temporaries in sorted(fact_levels.items(), key=lambda (i, j): i):
            level_inloop_cost = 0
            for t in temporaries:

                # Calculate the operation count for /t/ if we applied expansion + fact
                reads = []
                for read, cost in t.reads_costs.items():
                    reads.extend(new_trace[read.urepr].reads or [read.urepr])
                    # The number of operations induced outside /loop/ (after hoisting)
                    total_outloop_cost += new_trace[read.urepr].project*cost

                # Factorization will kill duplicates and increase the number of sums
                # in the outer loop
                fact_syms = {s.urepr if isinstance(s, Symbol) else s for s in reads}
                total_outloop_cost += len(reads) - len(fact_syms)

                # The operation count, after factorization, within /loop/, induced by /t/
                # Note: if n=len(fact_syms), then we'll have n prods, n-1 sums
                level_inloop_cost += 2*len(fact_syms) - 1

                # Update the trace because we want to track the cost after "pushing" the
                # temporaries on which /t/ depends into /t/ itself
                new_trace[t.urepr].reads_costs = {s: 1 for s in fact_syms}

            # Some temporaries at levels < /i/ may also appear in:
            # 1) subsequent loops
            # 2) levels beyond /i/
            for t in list(flatten([levels[j] for j in range(level)])):
                if any(ir.urepr not in new_trace for ir in t.is_read) or \
                        any(new_trace[ir.urepr].level > level for ir in t.is_read):
                    # Note: condition 1) is basically saying "if I'm read from
                    # a temporary that is not in this loop's trace, then I must
                    # be read in some other loops".
                    level_inloop_cost += t.cost

            # Total cost = cost_after_fact_up_to_level + cost_inloop_cse
            #            = cost_hoisted_subexprs + cost_inloop_fact + cost_inloop_cse
            uptolevel_cost = cse_cost + total_outloop_cost + loop.size*level_inloop_cost
            uptolevel_cost += self._cost_cse(loop, fact_levels, range(level + 1, bounds[1] + 1))

            # Update the best alternative
            if uptolevel_cost < best[2]:
                best = (bounds[0], level, uptolevel_cost)

            cse = self._cost_cse(loop, fact_levels, range(level + 1, bounds[1] + 1))
            print "Cost after pushing up to level", level, ":", uptolevel_cost, \
                "(", cse_cost, "+", total_outloop_cost, "+", loop.size*level_inloop_cost, "+", cse, ")"
        print "************************"

        return best

    def unpick(self):
        fors = visit(self.header, info_items=['fors'])['fors']
        lda = ldanalysis(self.header, value='dim')

        # Collect all loops to be analyzed
        nests = OrderedDict()
        for nest in fors:
            for loop, parent in nest:
                if loop.is_linear:
                    nests[loop] = nest

        # Analysis of loops
        global_trace = OrderedDict()
        mapper = OrderedDict()
        for loop in nests.keys():
            trace = self._analyze_loop(loop, lda, global_trace)
            if trace:
                mapper[loop] = trace
                global_trace.update(trace)

        for loop, trace in mapper.items():
            # Do not attempt to transform the main loop nest
            nest = nests[loop]
            if self.expr_info.loops_info == nest:
                continue

            # Compute the best cost alternative
            levels = self._group_by_level(trace)
            min_level, max_level = min(levels.keys()), max(levels.keys())
            global_best = (min_level, max_level, maxint)
            for i in sorted(levels.keys()):
                local_best = self._cost_fact(trace, levels, loop, (i, max_level))
                if local_best[2] < global_best[2]:
                    global_best = local_best

            # Transform the loop
            for i in range(global_best[0] + 1, global_best[1] + 1):
                self._push_temporaries(trace, levels, loop, i, global_trace)
                self._transform_temporaries(levels[i], loop, nest)

        # Clean up
        for transformed_loop, nest in reversed(nests.items()):
            for loop, parent in nest:
                if loop == transformed_loop and not loop.body:
                    parent.children.remove(loop)
