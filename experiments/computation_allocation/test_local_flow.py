"""Seeded synthetic falsification test; no chess or production-search changes.

The policy sees a noisy synthetic local signal, NOT exact minimax labels.
That signal is generated from the hidden value, so this tests a mechanism under
an informative-prior assumption, not whether a chess policy can learn it.
Policy strength is fitted by black-box search reward on separate training trees.
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


def tree(seed, misleading=False, depth=5):
    rng = random.Random(seed)
    n = 2 ** (depth + 1) - 1
    values = [0.] * n
    for i in range(2**depth - 1, n):
        values[i] = rng.uniform(-1, 1)
    for i in reversed(range(2**depth - 1)):
        level = (i + 1).bit_length() - 1
        values[i] = (max if level % 2 == 0 else min)(values[2*i+1:2*i+3])
    signal = [0.] * n
    for i in range(1, n):
        parent = (i-1)//2
        sign = 1 if ((parent+1).bit_length()-1) % 2 == 0 else -1
        signal[i] = sign * values[i] + rng.gauss(0, .45)
    if misleading:
        best = 1 if values[1] >= values[2] else 2
        signal[best], signal[3-best] = -3., 3.
    # All internal frontier estimates are neutral: reward is delayed to leaves.
    return values, signal


def search(data, strength, adaptive, budget, endpoint=False):
    truth, signal = data
    n = len(truth)
    children = {}
    conduct = {}
    estimate = {0: 0.}
    credit = {}
    evaluations = 0
    floor = .01

    def expand(i):
        if 2*i+2 >= n:
            return
        cs = [2*i+1, 2*i+2]
        children[i] = cs
        logits = [strength*signal[c] for c in cs]
        exps = [math.exp(x-max(logits)) for x in logits]
        for c, x in zip(cs, exps):
            conduct[c] = floor + x/sum(exps)
            estimate[c] = 0.
            credit[c] = 0.

    def backup(i):
        while i:
            i = (i-1)//2
            sign = ((i+1).bit_length()-1) % 2
            estimate[i] = (min if sign else max)(estimate[c] for c in children[i])

    expand(0)
    while evaluations < budget and credit:
        effective = {}
        def solve(i):
            if i in credit:
                effective[i] = math.inf
            else:
                effective[i] = sum(branch(c) for c in children.get(i, []))
            return effective[i]
        def branch(c):
            g = solve(c)
            return conduct[c] if math.isinf(g) else conduct[c]*g/(conduct[c]+g)
        solve(0)
        flow = {}
        def route(i, current):
            if i in credit:
                flow[i] = current
                credit[i] += current
                return
            cs = children.get(i, [])
            gs = [conduct[c] if math.isinf(effective[c]) else conduct[c]*effective[c]/(conduct[c]+effective[c]) for c in cs]
            total = sum(gs)
            if total:
                for c, g in zip(cs, gs):
                    route(c, current*g/total)
        route(0, 1.)
        batch = sorted(credit, key=lambda i: (-credit[i], i))[:min(4, budget-evaluations)]
        deposits = {}
        for i in batch:
            old_root = estimate[0]
            old = estimate[i]
            terminal = 2*i+2 >= n
            estimate[i] = truth[i] if terminal else 0.
            backup(i)
            utility = min(1., .05 + .35*abs(estimate[i]-old) + .35*abs(estimate[0]-old_root) + .25*terminal)
            path = []
            c = i
            while c:
                path.append(c)
                c = (c-1)//2
            for c in (path[:1] if endpoint else path):
                deposits[c] = deposits.get(c, 0.) + flow.get(i, 0.)*utility/len(path)
        if adaptive:
            for c in conduct:
                conduct[c] = max(floor, .97*conduct[c] + .75*deposits.get(c, 0.))
        for i in batch:
            del credit[i]
            expand(i)
        evaluations += len(batch)
    chosen = max((1, 2), key=lambda c: (estimate[c], -c))
    return max(truth[1:3])-truth[chosen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train = [tree(i) for i in range(100)]
    candidates = (0., .5, 1., 2., 4., 8.)
    losses = {s: statistics.fmean(search(t, s, True, b) for t in train for b in (16, 32, 48)) for s in candidates}
    strength = min(losses, key=losses.get)
    rows = []
    for misleading in (False, True):
        tests = [tree(10000+i, misleading) for i in range(300)]
        for budget in (16, 32, 48, 62):
            for name, s, adaptive, endpoint in (
                ('uniform-static', 0., False, False),
                ('uniform-adaptive', 0., True, False),
                ('policy-static', strength, False, False),
                ('policy-adaptive', strength, True, False),
                ('policy-endpoint', strength, True, True),
            ):
                regrets = [search(t, s, adaptive, budget, endpoint) for t in tests]
                row = dict(misleading=misleading, budget=budget, method=name, mean_regret=statistics.fmean(regrets), optimal_fraction=sum(r < 1e-12 for r in regrets)/len(regrets))
                rows.append(row)
                print(json.dumps(row), flush=True)
                if budget == 62:
                    assert max(regrets) < 1e-12, 'full enumeration must recover exact minimax'
    result = dict(training_trees=100, held_out_trees=300, strength=strength, training_losses=losses, caveat=__doc__, results=rows)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
