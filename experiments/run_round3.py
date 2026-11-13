"""Prepare, freeze, and run the canonical synchronous eight-Gaussian study.

Run from any working directory with ``python -m experiments.run_round3`` after
putting the repository on PYTHONPATH. Live checkpoints stay in runs/round3.
The freeze consumes only locally produced, trusted tuning checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from ganopt.models import make_models
from ganopt.optim import METHODS
from ganopt.runtime import source_hash, stable_hash, validate

REPO = Path(__file__).resolve().parents[1]
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
TUNING_SEED = 9001
REPORTING_SEEDS = (11, 22, 33)
TAIL_CHECKPOINTS = 5
EVALUATION = {'seed': 123456, 'samples': 2048,
              'metrics': 'ganopt.metrics.synthetic_metrics unchanged'}
PRIMARY_CONDITION = 'canonical joint synchronous 1:1, all four METHODS'
ARCHIVED_REFERENCE = 'round2 Adam 5:1 alternating reference: separate conditions, not primary'
SELECTION_RULE = ('last5 saved checkpoints: mean covered_modes descending, '
                  'mean quality_fraction descending, mean mean_nearest_distance ascending, '
                  'smaller learning rate tie-break; tuning seed 9001 only')


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def portable_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def protocol_code_hashes():
    return {name: hashlib.sha256((REPO / 'experiments' / name).read_bytes()).hexdigest()
            for name in ['run_round3.py', 'collect_round3.py']}


def common_config(phase):
    if phase not in ('tuning', 'reporting'):
        raise ValueError('phase must be tuning or reporting')
    return {'protocol': 'v1.0', 'stage': f'round3_{phase}', 'dataset': 'synthetic',
            'split_seed': 2026, 'batch_size': 64, 'z_dim': 8, 'hidden': 128,
            'base_channels': 32, 'betas': [.5, .9], 'eps': 1e-8, 'gp_lambda': .1,
            'oracle_budget': 120000, 'log_every': 1200, 'checkpoint_every': 6000,
            'extra_batch': 'independent', 'lookahead_k': 5, 'lookahead_alpha': .5,
            'device': 'cpu', 'threads': 1, 'data_root': 'data', 'download': False,
            'eval_samples': 2048, 'keep_checkpoints': True, 'critic_activation': 'relu',
            'training_schedule': 'synchronous', 'critic_steps': 1,
            'round3_source_hash': source_hash()}


def make_config(method, seed, lr, phase):
    if method not in METHODS:
        raise ValueError('Unknown canonical method')
    config = dict(common_config(phase), algorithm=method, seed=seed, lr=lr)
    validate(config)
    return config


def expected_counts(config, calls):
    cost = METHODS[config['algorithm']].oracle_cost
    cycle = cost * (config['lookahead_k'] if config['algorithm'] == 'lookahead_minmax' else 1)
    if not isinstance(calls, int) or calls < 0 or calls % cycle:
        raise ValueError('Field calls must align to a complete method cycle')
    if config['extra_batch'] != 'independent':
        raise ValueError('Round3 requires independent data batches')
    return {'oracle_calls': calls, 'player_gradient_calls': 2*calls,
            'parameter_updates': calls//cost, 'data_batches': calls,
            'data_examples_drawn': calls*config['batch_size'],
            'sample_evaluations': calls*config['batch_size']}


def initial_parameter_hash(config):
    # Model construction uses only Torch RNG. Preserve the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        generator, critic = make_models(config)
    params = list(generator.parameters()) + list(critic.parameters())
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params)).hexdigest()


def write_new_json(path, value, identical_ok=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if identical_ok and path.exists() and json.loads(path.read_text()) == value:
        return
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True)+'\n')


def _jobs(root, phase, configs):
    jobs = []
    for config in configs:
        chash = stable_hash(config)
        job_id = f"{config['algorithm']}-s{config['seed']}-lr{config['lr']:g}-{chash[:8]}"
        config_path = root / 'configs' / phase / (job_id+'.json')
        write_new_json(config_path, config, identical_ok=True)
        jobs.append({'job_id': job_id, 'config': portable_path(config_path),
                     'output': portable_path(root / 'runs' / phase / job_id),
                     'config_hash': chash, 'source_hash': source_hash(),
                     'expected_initial_parameter_hash': initial_parameter_hash(config)})
    return {'schema_version': 1, 'phase': phase, 'condition': PRIMARY_CONDITION,
            'source_hash': source_hash(), 'protocol_code_hashes': protocol_code_hashes(),
            'evaluation': EVALUATION, 'jobs': jobs}


def prepare(root='runs/round3'):
    root = resolve_path(root)
    if (root/'tuning_jobs.json').exists():
        return load_manifest(root, 'tuning')
    configs = [make_config(method, TUNING_SEED, lr, 'tuning')
               for method in METHODS for lr in LEARNING_RATES]
    manifest = _jobs(root, 'tuning', configs)
    write_new_json(root / 'tuning_jobs.json', manifest, identical_ok=True)
    return manifest


def validate_tuning_config(config):
    if (config.get('seed') != TUNING_SEED or config.get('lr') not in LEARNING_RATES
            or config.get('algorithm') not in METHODS):
        raise ValueError('Invalid tuning seed, learning rate, or method')
    if config != make_config(config['algorithm'], TUNING_SEED, config['lr'], 'tuning'):
        raise ValueError('Invalid tuning common configuration or source')


def _validate_freeze(frozen):
    if (not frozen.get('frozen') or frozen.get('schema_version') != 1
            or frozen.get('source_hash') != source_hash()
            or frozen.get('common') != common_config('reporting')
            or frozen.get('protocol_code_hashes') != protocol_code_hashes()
            or frozen.get('evaluation') != EVALUATION
            or frozen.get('condition') != PRIMARY_CONDITION
            or frozen.get('selection_rule') != SELECTION_RULE
            or frozen.get('tuning_seed') != TUNING_SEED
            or frozen.get('learning_rate_candidates') != list(LEARNING_RATES)
            or frozen.get('reporting_seeds') != list(REPORTING_SEEDS)):
        raise ValueError('Round3 freeze/common/source/protocol mismatch')
    rates = frozen.get('lr_by_algorithm', {})
    if set(rates) != set(METHODS) or any(lr not in LEARNING_RATES for lr in rates.values()):
        raise ValueError('Round3 freeze learning rate mismatch')


def reporting_configs(frozen, freeze_path):
    _validate_freeze(frozen)
    return [dict(make_config(method, seed, frozen['lr_by_algorithm'][method], 'reporting'),
                 freeze_file=portable_path(freeze_path), round3_freeze_hash=stable_hash(frozen))
            for method in METHODS for seed in REPORTING_SEEDS]


def validate_reporting_config(config):
    if not config.get('freeze_file'):
        raise ValueError('Reporting configuration requires protocol freeze')
    frozen = json.loads(resolve_path(config['freeze_file']).read_text())
    _validate_freeze(frozen)
    if config.get('round3_freeze_hash') != stable_hash(frozen):
        raise ValueError('Reporting freeze identity mismatch')
    expected = reporting_configs(frozen, resolve_path(config['freeze_file']))
    if config not in expected:
        raise ValueError('Frozen reporting configuration mismatch')
    return frozen


def load_manifest(root, phase):
    root = resolve_path(root)
    manifest = json.loads((root / f'{phase}_jobs.json').read_text())
    if (manifest.get('phase') != phase or manifest.get('source_hash') != source_hash()
            or manifest.get('protocol_code_hashes') != protocol_code_hashes()
            or manifest.get('condition') != PRIMARY_CONDITION
            or manifest.get('evaluation') != EVALUATION):
        raise ValueError('Jobs manifest source/protocol mismatch')
    if len(manifest.get('jobs', [])) != 12:
        raise ValueError('Exactly 12 candidate/reporting jobs required')
    seen = set()
    for job in manifest['jobs']:
        config = json.loads(resolve_path(job['config']).read_text())
        if phase == 'tuning':
            validate_tuning_config(config)
            identity = (config['algorithm'], config['lr'])
        else:
            validate_reporting_config(config)
            identity = (config['algorithm'], config['seed'])
        if identity in seen:
            raise ValueError('Duplicate candidate/reporting job')
        seen.add(identity)
        if (job['config_hash'] != stable_hash(config) or job['source_hash'] != source_hash()
                or job['expected_initial_parameter_hash'] != initial_parameter_hash(config)):
            raise ValueError('Job config/source/initialization identity mismatch')
    return manifest


def select_learning_rates(records):
    if len(records) != 12:
        raise ValueError('Exactly 12 tuning candidates required')
    by_method = {method: [] for method in METHODS}
    seen = set()
    for record in records:
        config = record['config']
        validate_tuning_config(config)
        identity = (config['algorithm'], config['lr'])
        if identity in seen:
            raise ValueError('Duplicate tuning candidate')
        seen.add(identity)
        status = record.get('status')
        if status == 'failed':
            if not record.get('error'):
                raise ValueError('Failed tuning candidate needs explicit failure evidence')
            if not record['error'].startswith('FloatingPointError('):
                raise ValueError('Resolve infrastructure failures before tuning selection; only numerical failures may eliminate a candidate')
            continue
        if status != 'completed':
            raise ValueError('Every tuning candidate must have explicit terminal status')
        values = [record[f'last5_mean_{key}'] for key in
                  ['covered_modes', 'quality_fraction', 'mean_nearest_distance']]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Nonfinite tuning metric evidence')
        by_method[config['algorithm']].append((-values[0], -values[1], values[2], config['lr']))
    rates = {}
    for method, candidates in by_method.items():
        if not candidates:
            raise ValueError(f'{method}: at least one completed tuning candidate required')
        rates[method] = min(candidates)[3]
    return rates


def freeze(root='runs/round3'):
    root = resolve_path(root)
    path = root / 'protocol_freeze.json'
    if path.exists():
        raise FileExistsError('Protocol freeze is immutable; use a new study root')
    manifest = load_manifest(root, 'tuning')
    from experiments.collect_round3 import evaluate_run
    records, evidence = [], []
    for job in manifest['jobs']:
        config = json.loads(resolve_path(job['config']).read_text())
        evaluated = evaluate_run(resolve_path(job['output']), config, source_hash(), allow_failed=True)
        records.append(dict(evaluated['summary'], config=config))
        evidence.append({'job_id': job['job_id'], 'summary': evaluated['summary'],
                         'checkpoint_sha256': evaluated['checkpoint_hashes']})
    hashes = {r['initial_parameter_hash'] for r in records}
    if len(hashes) != 1:
        raise ValueError('All tuning candidates must share initial parameters')
    frozen = {'schema_version': 1, 'frozen': True, 'source_hash': source_hash(),
              'common': common_config('reporting'), 'lr_by_algorithm': select_learning_rates(records),
              'reporting_seeds': list(REPORTING_SEEDS), 'tuning_seed': TUNING_SEED,
              'learning_rate_candidates': list(LEARNING_RATES), 'selection_rule': SELECTION_RULE,
              'evaluation': EVALUATION, 'condition': PRIMARY_CONDITION,
              'archived_reference': ARCHIVED_REFERENCE, 'protocol_code_hashes': protocol_code_hashes(),
              'tuning_initial_parameter_hash': next(iter(hashes)), 'tuning_evidence': evidence,
              'failed_tuning_candidates': [r for r in records if r['status'] == 'failed'],
              'tuning_manifest_sha256': hashlib.sha256((root/'tuning_jobs.json').read_bytes()).hexdigest()}
    configs = reporting_configs(frozen, path)
    reporting_manifest = _jobs(root, 'reporting', configs)
    # Exclusive creation prevents accidental replacement of a prior scientific decision.
    write_new_json(root/'reporting_jobs.json', reporting_manifest, identical_ok=True)
    write_new_json(path, frozen)
    return frozen


def run_one(config_path, output=None, resume=False, stop_after=None):
    config_path = resolve_path(config_path)
    config = json.loads(config_path.read_text())
    if config.get('stage') == 'round3_tuning':
        validate_tuning_config(config)
    elif config.get('stage') == 'round3_reporting':
        validate_reporting_config(config)
    else:
        raise ValueError('run-one accepts only prepared round3 configurations')
    if stop_after is not None and stop_after % config['checkpoint_every']:
        raise ValueError('Round3 pauses must align to a saved checkpoint interval')
    phase = config['stage'].removeprefix('round3_')
    root = config_path.parent.parent.parent
    manifest = load_manifest(root, phase)
    jobs = [j for j in manifest['jobs'] if resolve_path(j['config']).resolve() == config_path.resolve()]
    if len(jobs) != 1:
        raise ValueError('Configuration is not an identified prepared job')
    expected_output = resolve_path(jobs[0]['output'])
    if output is not None and resolve_path(output).resolve() != expected_output.resolve():
        raise ValueError('Output must match the prepared job identity')
    from ganopt.train import train
    return train(config, expected_output, resume=resume, stop_after=stop_after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['prepare', 'freeze', 'collect']:
        item = sub.add_parser(command)
        item.add_argument('--root', default='runs/round3', type=Path)
        if command == 'collect':
            item.add_argument('--phase', choices=['tuning', 'reporting'], required=True)
            item.add_argument('--output', default='results/round3', type=Path)
    item = sub.add_parser('run-one')
    item.add_argument('--config', required=True, type=Path)
    item.add_argument('--output', type=Path)
    item.add_argument('--resume', action='store_true')
    item.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(args.root)
    elif args.command == 'freeze':
        value = freeze(args.root)
    elif args.command == 'run-one':
        value = str(run_one(args.config, args.output, args.resume, args.stop_after))
    else:
        from experiments.collect_round3 import collect_phase
        value = collect_phase(args.root, args.phase, args.output)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
