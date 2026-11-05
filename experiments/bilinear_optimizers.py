"""Apply the actual four optimizers to min_x max_y x*y under equal oracle budgets."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ganopt.optim import METHODS


STEP_SIZES = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
LABELS = ('Adam', 'ExtraAdam', 'Optimistic Adam', 'Lookahead-Minmax')


def trajectory(method, step_size, budget):
    point = torch.ones(2, dtype=torch.float64)
    kwargs = {'lr': step_size, 'betas': (0.5, 0.9), 'eps': 1e-8}
    if method == 'lookahead_minmax':
        kwargs.update(k=5, alpha=0.5)
    optimizer = METHODS[method]([point], **kwargs)
    calls = 0
    rows = []

    def record(update):
        x, y = point.tolist()
        rows.append((calls, update, x, y, float(torch.linalg.vector_norm(point))))

    def oracle():
        nonlocal calls
        calls += 1
        x, y = point.detach().unbind()
        return [torch.stack((y, -x))], {}

    record(0)
    for update in range(1, budget // optimizer.oracle_cost + 1):
        optimizer.step(oracle)
        record(update)
    if calls != budget:
        raise RuntimeError(f'{method}: expected {budget} evaluations, observed {calls}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.budget <= 0 or args.budget % 10:
        parser.error('--budget must be a positive multiple of 10 (ExtraAdam and Lookahead boundaries)')
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, constrained_layout=True)

    with (args.output / 'trajectory.csv').open('w', newline='') as traces, \
         (args.output / 'endpoints.csv').open('w', newline='') as endpoints:
        trace_writer = csv.writer(traces)
        end_writer = csv.writer(endpoints)
        fields = ['method', 'step_size', 'oracle_calls', 'updates', 'x', 'y', 'distance']
        trace_writer.writerow(fields)
        end_writer.writerow(fields)
        for eta, axis in zip(STEP_SIZES, axes.flat, strict=True):
            for method, label in zip(METHODS, LABELS, strict=True):
                rows = trajectory(method, eta, args.budget)
                trace_writer.writerows((method, eta, *row) for row in rows)
                end_writer.writerow((method, eta, *rows[-1]))
                values = np.asarray(rows)
                axis.semilogy(values[:, 0], np.maximum(values[:, 4], 1e-16), label=label)
            axis.set(title=f'Step size {eta:g}', xlabel='Joint gradient evaluations')
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel('Distance to (0, 0)')
    axes[1, 0].set_ylabel('Distance to (0, 0)')
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('Bilinear game: adaptive optimizers from the same point')
    figure.savefig(args.output / 'distance_curves.png', dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    main()
