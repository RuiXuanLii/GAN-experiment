"""Audit trusted local checkpoints and collect round3 synthetic diagnostics.

No training is performed here. torch.load(weights_only=False) is intentionally
limited to checkpoints created locally by this project's trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ganopt.data import BatchStream
from ganopt.metrics import synthetic_metrics
from ganopt.models import make_models
from ganopt.runtime import source_hash, stable_hash
from experiments.run_round3 import (ARCHIVED_REFERENCE, EVALUATION, METHODS,
    PRIMARY_CONDITION, REPORTING_SEEDS, TAIL_CHECKPOINTS, expected_counts,
    initial_parameter_hash, load_manifest, portable_path, protocol_code_hashes,
    resolve_path, validate_reporting_config)

METRICS = ('covered_modes', 'quality_fraction', 'mean_nearest_distance', 'assignment_entropy')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(run, config, metadata):
    return {'case': run.name, 'method': config['algorithm'], 'seed': config['seed'],
            'lr': config['lr'], 'stage': config['stage'], 'condition': PRIMARY_CONDITION,
            'training_schedule': config.get('training_schedule', 'synchronous'),
            'critic_steps': config.get('critic_steps', 1),
            'gp_lambda': config['gp_lambda'], 'hidden': config['hidden'],
            'critic_activation': config.get('critic_activation', 'softplus'),
            'batch_size': config['batch_size'], 'eval_samples': config['eval_samples'],
            'evaluation_seed': EVALUATION['seed'], 'config_hash': stable_hash(config),
            'source_hash': metadata['source_hash'],
            'initial_parameter_hash': metadata['initial_parameter_hash']}


def _check_state(state, config, metadata, calls):
    if (state['config'] != config or state['config_hash'] != stable_hash(config)
            or state['source_hash'] != metadata['source_hash']):
        raise ValueError('Checkpoint config/source identity mismatch')
    if state['oracle_calls'] != calls:
        raise ValueError('Checkpoint filename/field-call mismatch')
    counts = expected_counts(config, calls)
    if state['updates'] != counts['parameter_updates']:
        raise ValueError('Checkpoint actual parameter update count mismatch')
    if state['data']['batches'] != counts['data_batches']:
        raise ValueError('Checkpoint actual data batch count mismatch')
    if state['data']['split_hash'] != metadata['split_hash']:
        raise ValueError('Checkpoint data identity mismatch')
    optimizer = state['optimizer']
    if (optimizer['class'] != METHODS[config['algorithm']].__name__
            or optimizer['lr'] != config['lr'] or tuple(optimizer['betas']) != tuple(config['betas'])
            or optimizer['eps'] != config['eps']
            or any(s['n'] != calls for s in optimizer['states'])):
        raise ValueError('Checkpoint optimizer identity or actual moment clock mismatch')
    if config['algorithm'] == 'lookahead_minmax' and state['optimizer']['inner_steps'] != state['updates']:
        raise ValueError('Lookahead actual inner-step count mismatch')
    return counts


def _samples(state, config, noise):
    with torch.random.fork_rng(devices=[]):
        generator, _ = make_models(config)
    generator.load_state_dict(state['generator'])
    generator.eval()
    with torch.no_grad():
        return generator(noise).numpy()


def mode_counts(samples):
    angles = np.arange(8)*2*np.pi/8
    centers = 2*np.stack([np.cos(angles), np.sin(angles)], axis=1)
    distances = np.linalg.norm(samples[:, None]-centers[None], axis=2)
    assignments = distances.argmin(axis=1)
    good = distances.min(axis=1) < .15
    assigned = np.bincount(assignments, minlength=8)
    qualified = np.bincount(assignments[good], minlength=8)
    return {**{f'assigned_mode_{i}': int(n) for i, n in enumerate(assigned)},
            **{f'good_mode_{i}': int(n) for i, n in enumerate(qualified)}}


def _validate_logs(run, config, calls, completed):
    path = run/'metrics.csv'
    if not path.exists():
        if calls >= config['log_every'] or completed:
            raise ValueError('Missing actual training metrics.csv')
        return
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    seen = []
    for row in rows:
        count = int(row['oracle_calls'])
        if count > calls:
            raise ValueError('Training log extends beyond authoritative checkpoint')
        counts = expected_counts(config, count)
        for key in ['player_gradient_calls', 'parameter_updates', 'data_examples_drawn', 'sample_evaluations']:
            if int(row[key]) != counts[key]:
                raise ValueError(f'Training log actual {key} mismatch')
        seen.append(count)
    expected = list(range(config['log_every'], calls+1, config['log_every']))
    # The trainer logs a nonperiodic endpoint for deliberately paused fixture runs.
    if completed and calls not in expected:
        expected.append(calls)
    if completed and seen != expected:
        raise ValueError('Missing, duplicate, or unordered training log checkpoints')
    if seen != sorted(set(seen)):
        raise ValueError('Duplicate or unordered training log checkpoints')


def evaluate_run(run, config, expected_source=None, allow_failed=False):
    """Return checkpoint evidence; no incomplete or stale run is silently skipped."""
    torch.set_num_threads(1)
    run = Path(run)
    expected_source = source_hash() if expected_source is None else expected_source
    saved_config = json.loads((run/'config.json').read_text())
    metadata = json.loads((run/'metadata.json').read_text())
    result = json.loads((run/'result.json').read_text())
    if saved_config != config or metadata['config'] != config or metadata['config_hash'] != stable_hash(config):
        raise ValueError(f'Run configuration identity mismatch: {run}')
    if metadata['source_hash'] != expected_source:
        raise ValueError(f'Run source identity mismatch: {run}')
    if metadata['initial_parameter_hash'] != initial_parameter_hash(config):
        raise ValueError(f'Run initial parameters mismatch: {run}')
    completed = result.get('status') == 'completed'
    failed = result.get('status') == 'failed'
    if not completed and not (allow_failed and failed and result.get('error')):
        raise ValueError(f'Run must be completed (or explicitly failed tuning evidence): {run}')
    final = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False)
    calls = result['oracle_calls'] if completed else result['last_valid_oracle_calls']
    if completed and calls != config['oracle_budget']:
        raise ValueError('Completed run did not reach the common field budget')
    if calls > config['oracle_budget']:
        raise ValueError('Run exceeded the common field budget')
    final_counts = _check_state(final, config, metadata, calls)
    _validate_logs(run, config, calls, completed)
    scheduled = {int(p.stem.removeprefix('calls_')): p
                 for p in (run/'checkpoints').glob('calls_*.pt')}
    expected_calls = set(range(config['checkpoint_every'], calls+1, config['checkpoint_every']))
    if set(scheduled) != expected_calls:
        raise ValueError('Missing or unexpected actual saved checkpoints')
    noise = torch.randn(config['eval_samples'], config['z_dim'],
                        generator=torch.Generator().manual_seed(EVALUATION['seed']))
    identity = _identity(run, config, metadata)
    rows, hashes = [], {'checkpoint.pt': sha256(run/'checkpoint.pt')}
    endpoint_samples = None
    for count, path in sorted(scheduled.items()):
        state = torch.load(path, map_location='cpu', weights_only=False)
        counts = _check_state(state, config, metadata, count)
        samples = _samples(state, config, noise)
        digest = sha256(path)
        hashes[str(path.relative_to(run))] = digest
        rows.append(dict(identity, status=result['status'], **counts,
                         checkpoint_sha256=digest, **synthetic_metrics(samples), **mode_counts(samples)))
        if count == calls:
            endpoint_samples = samples
            for model in ['generator', 'critic']:
                if any(not torch.equal(state[model][key], final[model][key]) for key in state[model]):
                    raise ValueError('Final checkpoint differs from saved endpoint parameters')
    summary = dict(identity, status=result['status'], **final_counts,
                   checkpoint_sha256=hashes['checkpoint.pt'], error=result.get('error', ''),
                   training_seconds=final['training_seconds'], wall_seconds=final['wall_seconds'])
    if completed:
        final_samples = _samples(final, config, noise)
        saved = np.load(run/'samples.npy', allow_pickle=False)
        if not np.array_equal(final_samples, saved):
            raise ValueError('Endpoint samples.npy consistency mismatch')
        if endpoint_samples is not None and not np.array_equal(endpoint_samples, final_samples):
            raise ValueError('Saved checkpoint endpoint samples mismatch')
        metrics = synthetic_metrics(final_samples)
        if any(result.get(key) != value for key, value in metrics.items()):
            raise ValueError('Endpoint synthetic metric/result consistency mismatch')
        if result['parameter_updates'] != final_counts['parameter_updates']:
            raise ValueError('Result actual parameter update count mismatch')
        if not math.isfinite(float(result['training_seconds'])) or result['training_seconds'] <= 0:
            raise ValueError('Invalid actual training time')
        summary.update(metrics)
        summary.update(mode_counts(final_samples))
        for key, value in metrics.items():
            summary[f'final_{key}'] = value
            summary[f'last5_mean_{key}'] = float(np.mean([r[key] for r in rows[-TAIL_CHECKPOINTS:]]))
        summary['tail_checkpoint_count'] = min(TAIL_CHECKPOINTS, len(rows))
        summary['tail_checkpoint_calls'] = ';'.join(str(r['oracle_calls']) for r in rows[-TAIL_CHECKPOINTS:])
        return {'summary': summary, 'checkpoints': rows, 'samples': final_samples,
                'checkpoint_hashes': hashes}
    # A terminal failure is retained, never assigned replacement metric values.
    summary.update({key: None for key in METRICS})
    summary.update({prefix+key: None for prefix in ['final_', 'last5_mean_'] for key in METRICS})
    return {'summary': summary, 'checkpoints': rows, 'samples': None, 'checkpoint_hashes': hashes}


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty scientific evidence table')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _method_summary(summaries):
    result = []
    for method in METHODS:
        selected = [row for row in summaries if row['method'] == method]
        if {row['seed'] for row in selected} != set(REPORTING_SEEDS) or len(selected) != 3:
            raise ValueError('Reporting aggregate requires exactly three fresh seeds per method')
        row = {'method': method, 'condition': PRIMARY_CONDITION, 'n': 3,
               'seeds': ';'.join(str(s) for s in REPORTING_SEEDS), 'lr': selected[0]['lr'],
               'oracle_calls': selected[0]['oracle_calls'], 'std_definition': 'sample standard deviation, ddof=1'}
        for prefix in ['final_', 'last5_mean_']:
            for metric in METRICS:
                values = [r[prefix+metric] for r in selected]
                row[prefix+metric+'_mean'] = float(np.mean(values))
                row[prefix+metric+'_std'] = float(np.std(values, ddof=1))
        result.append(row)
    return result


def _plots(output, phase, rows, endpoints):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for endpoint in endpoints:
        series = [r for r in rows if r['case'] == endpoint['case']]
        if not series:
            continue
        label = f"{endpoint['method']} lr={endpoint['lr']:g} s={endpoint['seed']} ({endpoint['status']})"
        for axis, metric in zip(axes, METRICS[:3]):
            axis.plot([r['oracle_calls'] for r in series], [r[metric] for r in series],
                      label=label, marker='.', markersize=3, linewidth=1)
    for axis, title in zip(axes, ['Modes covered / 8', 'Fraction within radius 0.15', 'Mean nearest distance']):
        axis.set(xlabel='Joint synchronous field calls', ylabel=title)
        axis.grid(alpha=.2)
    axes[0].set_ylim(-.2, 8.2)
    axes[0].legend(fontsize=5)
    figure.suptitle(f'{phase.capitalize()}: {PRIMARY_CONDITION}')
    figure.savefig(output/f'{phase}_progress.png', dpi=160)
    plt.close(figure)


def _scatter(output, phase, sample_sets, reference):
    n = len(sample_sets)+1
    columns = min(4, n)
    figure, axes = plt.subplots((n+columns-1)//columns, columns,
        figsize=(columns*4, ((n+columns-1)//columns)*4), squeeze=False, layout='constrained')
    angles = np.arange(8)*2*np.pi/8
    centers = 2*np.stack([np.cos(angles), np.sin(angles)], axis=1)
    for axis, (title, samples) in zip(axes.flat, [('Real target control', reference)]+sample_sets):
        axis.scatter(reference[:, 0], reference[:, 1], s=2, color='gray', alpha=.2)
        axis.scatter(samples[:, 0], samples[:, 1], s=2, alpha=.4)
        axis.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', s=25)
        # Every generated point is included; divergent endpoints expand the axes.
        joined = np.vstack([reference, samples, centers])
        low, high = joined.min(0)-.3, joined.max(0)+.3
        axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), title=title, aspect='equal')
        axis.tick_params(labelsize=7)
    for axis in list(axes.flat)[n:]:
        axis.set_visible(False)
    figure.suptitle(f'{phase.capitalize()} endpoints: all 2048 generated points per run')
    figure.savefig(output/f'{phase}_samples.png', dpi=160)
    plt.close(figure)


def _copy_run(run, destination, samples):
    destination.mkdir(parents=True, exist_ok=True)
    names = ['config.json', 'metadata.json', 'result.json', 'metrics.csv']
    copied = []
    for name in names:
        path = run/name
        if not path.exists():
            if name == 'metrics.csv' and samples is None:
                continue
            raise FileNotFoundError(path)
        data = path.read_bytes()
        (destination/name).write_bytes(data.replace(b'\r\n', b'\n') if name.endswith('.csv') else data)
        copied.append(destination/name)
    if samples is not None:
        shutil.copyfile(run/'samples.npy', destination/'samples.npy')
        copied.append(destination/'samples.npy')
    elif (destination/'samples.npy').exists():
        (destination/'samples.npy').unlink()
    return copied


def collect_phase(root='runs/round3', phase='reporting', output='results/round3'):
    if phase not in ['tuning', 'reporting']:
        raise ValueError('phase must be tuning or reporting')
    root, output = resolve_path(root), resolve_path(output)
    manifest = load_manifest(root, phase)
    study_hash = sha256(root/'tuning_jobs.json')
    freeze_hash = stable_hash(json.loads((root/'protocol_freeze.json').read_text())) if (root/'protocol_freeze.json').exists() else None
    # Verify ownership before overwriting any shared freeze or phase evidence.
    for name in ['tuning_active.json', 'reporting_active.json']:
        if (output/name).exists():
            previous = json.loads((output/name).read_text())
            if (previous.get('study_hash') != study_hash
                    or previous.get('source_hash') != source_hash()
                    or previous.get('protocol_code_hashes') != protocol_code_hashes()
                    or (previous.get('freeze_hash') is not None and previous['freeze_hash'] != freeze_hash)):
                raise ValueError('Output belongs to a different study/source/freeze identity')
    evaluated, summaries, checkpoints, sample_sets = [], [], [], []
    for job in manifest['jobs']:
        config = json.loads(resolve_path(job['config']).read_text())
        run = resolve_path(job['output'])
        value = evaluate_run(run, config, source_hash(), allow_failed=phase == 'tuning')
        if value['summary']['status'] == 'completed' and value['summary']['tail_checkpoint_count'] != TAIL_CHECKPOINTS:
            raise ValueError('Five actual saved checkpoints required for scientific reporting')
        evaluated.append((run, value))
        summaries.append(value['summary'])
        checkpoints.extend(value['checkpoints'])
        if value['samples'] is not None:
            sample_sets.append((run.name, value['samples']))
    # Seed controls initialization; all methods/candidates must share it within a seed.
    for seed in sorted({r['seed'] for r in summaries}):
        if len({r['initial_parameter_hash'] for r in summaries if r['seed'] == seed}) != 1:
            raise ValueError('Common-seed initial parameter hashes differ')
    aggregates = _method_summary(summaries) if phase == 'reporting' else None
    output.mkdir(parents=True, exist_ok=True)
    active = []
    def register(path):
        active.append(str(path.relative_to(output)))
    for run, value in evaluated:
        destination = output/'runs'/phase/run.name
        for path in _copy_run(run, destination, value['samples']):
            register(path)
    for name, values in [(f'{phase}_summary.csv', summaries), (f'{phase}_checkpoints.csv', checkpoints)]:
        write_csv(output/name, values)
        register(output/name)
    if aggregates is not None:
        write_csv(output/'reporting_method_summary.csv', aggregates)
        register(output/'reporting_method_summary.csv')
    config = json.loads(resolve_path(manifest['jobs'][0]['config']).read_text())
    stream = BatchStream(dict(config, seed=123456, device='cpu'))
    reference = np.concatenate([stream.next()[0].numpy() for _ in range(32)])[:2048]
    control = synthetic_metrics(reference)
    if control['covered_modes'] != 8 or control['quality_fraction'] <= .97:
        raise ValueError('Real target control failed unchanged synthetic diagnostics')
    (output/'target_control.json').write_text(json.dumps({'seed': 123456, 'samples': 2048, **control}, indent=2)+'\n')
    register(output/'target_control.json')
    _plots(output, phase, checkpoints, summaries)
    _scatter(output, phase, sample_sets, reference)
    for name in [f'{phase}_progress.png', f'{phase}_samples.png']:
        register(output/name)
    # Copy preparation/freeze evidence and all prepared configs for reproducibility.
    for name in ['tuning_jobs.json', 'reporting_jobs.json', 'protocol_freeze.json']:
        if (root/name).exists():
            shutil.copyfile(root/name, output/name)
            register(output/name)
    for job in manifest['jobs']:
        source = resolve_path(job['config'])
        destination = output/'configs'/phase/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        register(destination)
    record = {'phase': phase, 'condition': PRIMARY_CONDITION, 'archived_reference': ARCHIVED_REFERENCE,
              'source_hash': source_hash(), 'protocol_code_hashes': protocol_code_hashes(),
              'study_hash': study_hash, 'freeze_hash': freeze_hash,
              'evaluation': EVALUATION, 'runs': [row['case'] for row in summaries],
              'files': sorted(set(active))}
    (output/f'{phase}_active.json').write_text(json.dumps(record, indent=2)+'\n')
    # Include only files named by the current phase inventories, never stale folders.
    files = set()
    for name in ['tuning_active.json', 'reporting_active.json']:
        path = output/name
        if path.exists():
            phase_record = json.loads(path.read_text())
            if phase_record['source_hash'] != source_hash():
                raise ValueError('Existing active phase evidence has a different source hash')
            files.update(phase_record['files'])
            files.add(name)
    hashes = {name: sha256(output/name) for name in sorted(files)}
    (output/'artifact_hashes.json').write_text(json.dumps({'sha256': hashes,
        'source_hash': source_hash(), 'protocol_code_hashes': protocol_code_hashes()}, indent=2)+'\n')
    return {'phase': phase, 'runs': len(summaries), 'failed_runs': sum(r['status'] == 'failed' for r in summaries),
            'output': portable_path(output), 'method_summary': aggregates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/round3'))
    parser.add_argument('--phase', choices=['tuning', 'reporting'], required=True)
    parser.add_argument('--output', type=Path, default=Path('results/round3'))
    args = parser.parse_args()
    print(json.dumps(collect_phase(args.root, args.phase, args.output), indent=2))


if __name__ == '__main__':
    main()
