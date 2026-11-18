"""Resume the 12 CIFAR-10 128px tuning evaluations and select learning rates."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from experiments import evaluate_cifar10_main as evaluator
from experiments.cifar128_parallel import GPU, completed, training_configs


def make_jobs(root):
    from ganopt.data import split_indices
    from ganopt.runtime import METRIC_PROTOCOL, source_hash

    base = evaluator.read_json('configs/cifar10_128/tuning.json')
    _, validation = split_indices(50000, base['split_seed'], 5000)
    count = 2000
    reference = hashlib.sha256(validation[:count].astype('<i8').tobytes()).hexdigest()
    source = source_hash()
    jobs = []
    for name, config in training_configs(base):
        run = root / name
        calls = config['oracle_budget']
        checkpoint = run / 'checkpoint.pt'
        if not completed(run, calls) or not checkpoint.is_file():
            raise ValueError(f'Training checkpoint is missing or incomplete: {run}')
        jobs.append(dict(method=name, seed=config['seed'], calls=calls, split='validation',
                         count=count, config=config, checkpoint=str(checkpoint),
                         folder=str(run / 'evaluation'), source_hash=source,
                         metric_protocol=METRIC_PROTOCOL, reference_indices_sha256=reference))
    return jobs


def check_dependencies():
    import importlib.metadata
    import numpy as np
    import scipy
    from torch_fidelity.metric_fid import fid_statistics_to_metric

    if scipy.__version__ != '1.15.3' or importlib.metadata.version('torch-fidelity') != '0.3.0':
        raise RuntimeError('Expected SciPy 1.15.3 and torch-fidelity 0.3.0 in the AutoDL venv')
    sample = dict(mu=np.zeros(2), sigma=np.eye(2))
    fid = fid_statistics_to_metric(sample, sample, False)['frechet_inception_distance']
    if not np.isfinite(fid) or abs(fid) > 1e-8:
        raise RuntimeError('FID dependency check failed')
    weights = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'
    if evaluator.digest(weights) != '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2':
        raise RuntimeError(f'Inception weights checksum mismatch: {weights}')


def save_scores(root, jobs):
    from scripts.freeze_protocol import check_metric_evidence

    rows = []
    for job in jobs:
        if not evaluator.metric_valid(job):
            raise ValueError(f'Missing or invalid tuning evaluation: {job["folder"]}')
        record = evaluator.read_json(Path(job['folder']) / 'metrics.json')
        config = job['config']
        check_metric_evidence(record, config, job['calls'])
        metrics = record['metrics']
        rows.append(dict(method=config['algorithm'], lr=config['lr'],
                         FID=metrics['frechet_inception_distance'],
                         KID=metrics['kernel_inception_distance_mean']))
    rows.sort(key=lambda row: (row['method'], row['FID'], row['KID'], row['lr']))
    selected = {}
    for row in rows:
        row['selected'] = row['method'] not in selected
        selected.setdefault(row['method'], row['lr'])
    path = root / 'tuning_scores.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'lr', 'FID', 'KID', 'selected'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    print(path.read_text(), end='')
    print('SELECTED_LEARNING_RATES:', json.dumps(selected, sort_keys=True))
    print(f'Scores: {path}')
    print(f'TUNING_EVALUATION_COMPLETED: {len(rows)}/{len(jobs)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing 128px tuning directory')
    parser.add_argument('--workers', type=int, default=4, help='Upper bound on concurrent FID/KID jobs')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.root.resolve()
    jobs = make_jobs(root)
    pending = [job for job in jobs if not evaluator.metric_valid(job)]
    finished = len(jobs) - len(pending)
    print(f'PROGRESS: {finished}/{len(jobs)}', flush=True)

    def on_done(job, seconds):
        nonlocal finished
        finished += 1
        print(f'PROGRESS: {finished}/{len(jobs)} {job["method"]} {seconds:.1f}s', flush=True)

    if pending:
        with GPU(None).exclusive():
            check_dependencies()
            resources = evaluator.preflight(root, pending, args.workers)
            print('RESOURCES:', json.dumps(resources), flush=True)
            evaluator.run_group(pending, resources['maximum_workers'], resources['threads'],
                                Path.cwd(), root, on_done)
    save_scores(root, jobs)


if __name__ == '__main__':
    main()
