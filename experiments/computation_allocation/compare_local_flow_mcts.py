"""Compare local branch flow with alternating-player PUCT on identical trees."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from test_local_flow import tree, search


def mcts(data, strength, exploration, budget):
    truth, signal = data
    children, priors = {}, {}
    visits, totals = {}, {}
    evaluated = set()

    def expand(i):
        if 2*i+2 >= len(truth):
            return
        cs = [2*i+1, 2*i+2]
        children[i] = cs
        logits = [strength*signal[c] for c in cs]
        weights = [math.exp(x-max(logits)) for x in logits]
        for c, w in zip(cs, weights):
            # Same positive prior floor as flow, normalized for PUCT.
            priors[c] = (.01 + w/sum(weights))/1.02

    expand(0)
    for _ in range(budget):
        i, path = 0, [0]
        while i in children:
            sign = 1 if ((i+1).bit_length()-1) % 2 == 0 else -1
            def priority(c):
                count = visits.get(c, 0)
                q = totals.get(c, 0.)/count if count else 0.
                return sign*q + exploration*priors[c]*math.sqrt(max(1, visits.get(i, 0)))/(1+count), -c
            i = max(children[i], key=priority)
            path.append(i)
        evaluated.add(i)
        value = truth[i] if 2*i+2 >= len(truth) else 0.
        expand(i)
        for node in path:
            visits[node] = visits.get(node, 0)+1
            totals[node] = totals.get(node, 0.)+value
    chosen = max((1, 2), key=lambda c: (visits.get(c, 0), -c))
    return max(truth[1:3])-truth[chosen], len(evaluated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train = [tree(i) for i in range(100)]
    budgets = (16, 32, 48)
    strengths = (0., .5, 1., 2., 4., 8.)
    flow_losses = {s: statistics.fmean(search(t, s, True, b) for t in train for b in budgets) for s in strengths}
    flow_strength = min(flow_losses, key=flow_losses.get)
    grid = []
    for s in strengths:
        for c in (.25, .5, 1., 2., 4.):
            loss = statistics.fmean(mcts(t, s, c, b)[0] for t in train for b in budgets)
            grid.append(dict(strength=s, exploration=c, train_regret=loss))
    tuned = min(grid, key=lambda r: r['train_regret'])
    shared = min((r for r in grid if r['strength'] == flow_strength), key=lambda r: r['train_regret'])
    uniform = min((r for r in grid if r['strength'] == 0), key=lambda r: r['train_regret'])
    rows = []
    for misleading in (False, True):
        tests = [tree(10000+i, misleading) for i in range(300)]
        for budget in (*budgets, 62):
            baseline = [search(t, flow_strength, True, budget) for t in tests]
            for method, params in [('flow', None), ('mcts-same-policy', shared), ('mcts-tuned-policy', tuned), ('mcts-uniform', uniform)]:
                start = time.perf_counter()
                if params is None:
                    results = [(search(t, flow_strength, True, budget), budget) for t in tests]
                else:
                    results = [mcts(t, params['strength'], params['exploration'], budget) for t in tests]
                elapsed = time.perf_counter()-start
                regrets = [r for r, _ in results]
                diffs = [r-f for r, f in zip(regrets, baseline)]
                delta = statistics.fmean(diffs)
                margin = 1.96*statistics.stdev(diffs)/math.sqrt(len(diffs))
                row = dict(misleading=misleading, budget=budget, method=method,
                    mean_regret=statistics.fmean(regrets), optimal_fraction=sum(r < 1e-12 for r in regrets)/len(regrets),
                    mean_unique_evaluated=statistics.fmean(n for _, n in results), python_ms_per_tree=elapsed*1000/len(tests),
                    regret_minus_flow=delta, paired_normal_95_ci=[delta-margin, delta+margin])
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(
        protocol='Depth-5 binary alternating max/min trees. Neutral internal evaluator; terminal rewards only. 100 training and 300 disjoint test seeds. Same synthetic local signal. PUCT mean backup and most-visited root move. Each simulation charges one frontier/terminal evaluation, including repeated terminal visits; unique coverage also reported. Equal evaluation requests, not equal wall time. Python timings are illustrative.',
        caveat='Policy signal is generated from hidden minimax plus noise: informative-prior mechanism test, not neural policy learning or chess strength. Root prior is deliberately reversed only at test time. No rollout beyond newly expanded frontier.',
        flow_strength=flow_strength, flow_training_losses=flow_losses,
        mcts_shared=shared, mcts_tuned=tuned, mcts_uniform=uniform, mcts_training_grid=grid, results=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
