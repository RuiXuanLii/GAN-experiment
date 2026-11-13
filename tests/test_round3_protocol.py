"""Protocol guards; deliberately short fixture runs are not scientific results."""
import copy
import importlib
import json
from pathlib import Path

import pytest


def modules():
    # Missing behavior should fail as an assertion, not as a collection error.
    assert importlib.util.find_spec('experiments.run_round3'), 'round3 orchestration is missing'
    assert importlib.util.find_spec('experiments.collect_round3'), 'round3 collection is missing'
    return (importlib.import_module('experiments.run_round3'),
            importlib.import_module('experiments.collect_round3'))


def selection_records(protocol):
    records = []
    for method in protocol.METHODS:
        for lr in protocol.LEARNING_RATES:
            records.append({'config': protocol.make_config(method, 9001, lr, 'tuning'),
                'status': 'completed', 'last5_mean_covered_modes': 4., 'last5_mean_quality_fraction': .5,
                'last5_mean_mean_nearest_distance': .2})
    return records


def test_lr_selection_uses_only_tuning_candidates_and_last_five_means():
    protocol, _ = modules()
    records = selection_records(protocol)
    # Earlier/final maxima are decoys: selection must use only last-five means.
    records[0].update(final_covered_modes=8, best_covered_modes=8)
    records[1]['last5_mean_covered_modes'] = 5.
    records[2].update(last5_mean_covered_modes=5., last5_mean_quality_fraction=.6)
    selected = protocol.select_learning_rates(records)
    assert selected['adam'] == 1e-3
    assert all(selected[m] == 1e-4 for m in protocol.METHODS if m != 'adam')
    records[0]['config']['seed'] = 11
    with pytest.raises(ValueError, match='tuning'):
        protocol.select_learning_rates(records)
    records = selection_records(protocol)
    records[0]['config']['lr'] = 9e-4
    with pytest.raises(ValueError, match='tuning'):
        protocol.select_learning_rates(records)
    with pytest.raises(ValueError, match='12|candidate'):
        protocol.select_learning_rates(selection_records(protocol)[:-1])
    records = selection_records(protocol)
    records[0].update(status='failed', error='OSError(fixture infrastructure failure)')
    with pytest.raises(ValueError, match='numerical|infrastructure'):
        protocol.select_learning_rates(records)
    records[0].update(status='failed', error='FloatingPointError(fixture failure)')
    assert protocol.select_learning_rates(records)['adam'] == 3e-4
    for record in records[:3]:
        record.update(status='failed', error='FloatingPointError(fixture failure)')
    with pytest.raises(ValueError, match='completed'):
        protocol.select_learning_rates(records)


def test_prepare_enforces_common_field_budget_and_real_cost_counts(tmp_path):
    protocol, _ = modules()
    manifest = protocol.prepare(tmp_path / 'study')
    assert len(manifest['jobs']) == 12
    initial_hashes = set()
    for job in manifest['jobs']:
        c = json.loads(protocol.resolve_path(job['config']).read_text())
        assert c['oracle_budget'] == 120000
        assert c['log_every'] == 1200 and c['checkpoint_every'] == 6000
        assert c['hidden'] == 128 and c['critic_activation'] == 'relu'
        assert c['gp_lambda'] == .1 and c['extra_batch'] == 'independent'
        initial_hashes.add(job['expected_initial_parameter_hash'])
        counts = protocol.expected_counts(c, 120000)
        assert counts['data_batches'] == 120000
        assert counts['data_examples_drawn'] == 120000 * 64
        assert counts['parameter_updates'] == (60000 if c['algorithm'] == 'extraadam' else 120000)
        assert counts['player_gradient_calls'] == 240000
        with pytest.raises(ValueError, match='cycle'):
            protocol.expected_counts(dict(c, algorithm='lookahead_minmax'), 120001)
    assert len(initial_hashes) == 1


def frozen_fixture(protocol, path):
    freeze = {'schema_version': 1, 'frozen': True,
        'source_hash': protocol.source_hash(), 'common': protocol.common_config('reporting'),
        'lr_by_algorithm': {m: 1e-4 for m in protocol.METHODS},
        'reporting_seeds': [11, 22, 33], 'tuning_seed': 9001,
        'learning_rate_candidates': list(protocol.LEARNING_RATES),
        'selection_rule': protocol.SELECTION_RULE,
        'evaluation': protocol.EVALUATION,
        'condition': protocol.PRIMARY_CONDITION,
        'protocol_code_hashes': protocol.protocol_code_hashes()}
    path.write_text(json.dumps(freeze))
    return freeze


def test_frozen_reporting_config_rejects_lr_budget_source_and_freeze_mutation(tmp_path):
    protocol, _ = modules()
    path = tmp_path / 'protocol_freeze.json'
    freeze = frozen_fixture(protocol, path)
    config = protocol.reporting_configs(freeze, path)[0]
    assert protocol.validate_reporting_config(config) == freeze
    for key, value in [('lr', 3e-4), ('oracle_budget', 60000), ('seed', 9001),
                       ('extra_batch', 'same'), ('round3_source_hash', 'wrong')]:
        changed = dict(config, **{key: value})
        with pytest.raises(ValueError):
            protocol.validate_reporting_config(changed)
    changed_freeze = copy.deepcopy(freeze)
    changed_freeze['lr_by_algorithm']['adam'] = 2e-4
    path.write_text(json.dumps(changed_freeze))
    with pytest.raises(ValueError, match='freeze'):
        protocol.validate_reporting_config(config)
    with pytest.raises(FileExistsError):
        protocol.freeze(tmp_path)


@pytest.mark.parametrize('method', ['adam', 'extraadam', 'optimistic_adam', 'lookahead_minmax'])
def test_checkpoint_collection_checks_actual_counts_and_endpoint_samples(tmp_path, method):
    protocol, collector = modules()
    import numpy as np
    import torch
    from ganopt.train import train
    c = protocol.make_config(method, 9001, 1e-4, 'tuning')
    c.update(stage='smoke', hidden=8, batch_size=4, z_dim=2, eval_samples=32,
             oracle_budget=20, log_every=10, checkpoint_every=10)
    run = train(c, tmp_path / method)
    evidence = collector.evaluate_run(run, c, protocol.source_hash())
    assert [r['oracle_calls'] for r in evidence['checkpoints']] == [10, 20]
    assert evidence['summary']['data_batches'] == 20
    assert evidence['summary']['parameter_updates'] == (10 if method == 'extraadam' else 20)
    for row in evidence['checkpoints']:
        assert sum(row[f'assigned_mode_{i}'] for i in range(8)) == 32
        assert sum(row[f'good_mode_{i}'] for i in range(8)) <= 32
        assert sum(row[f'good_mode_{i}'] >= .01*32 for i in range(8)) == row['covered_modes']
    saved = np.load(run / 'samples.npy')
    np.save(run / 'samples.npy', saved + 1)
    with pytest.raises(ValueError, match='samples'):
        collector.evaluate_run(run, c, protocol.source_hash())
    np.save(run / 'samples.npy', saved)
    checkpoint = torch.load(run / 'checkpoint.pt', weights_only=False)
    checkpoint['data']['batches'] -= 1
    torch.save(checkpoint, run / 'checkpoint.pt')
    with pytest.raises(ValueError, match='batch'):
        collector.evaluate_run(run, c, protocol.source_hash())
    checkpoint['data']['batches'] += 1
    checkpoint['optimizer']['states'][0]['n'] -= 1
    torch.save(checkpoint, run/'checkpoint.pt')
    with pytest.raises(ValueError, match='optimizer|Optimizer'):
        collector.evaluate_run(run, c, protocol.source_hash())


def test_prepare_rejects_mutated_existing_job_configuration(tmp_path):
    protocol, _ = modules()
    root = tmp_path/'study'
    manifest = protocol.prepare(root)
    path = protocol.resolve_path(manifest['jobs'][0]['config'])
    changed = json.loads(path.read_text())
    changed['gp_lambda'] = 9.
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='configuration|identity'):
        protocol.prepare(root)


def test_short_fixture_prepare_freeze_report_retains_failures_and_active_only_files(tmp_path, monkeypatch):
    protocol, collector = modules()
    original_common = protocol.common_config
    def short_common(phase):
        c = original_common(phase)
        c.update(hidden=8, batch_size=4, z_dim=2, eval_samples=32,
                 oracle_budget=100, log_every=20, checkpoint_every=20)
        return c
    monkeypatch.setattr(protocol, 'common_config', short_common)
    root = tmp_path/'fixture-study'
    tuning = protocol.prepare(root)
    with pytest.raises(FileNotFoundError):
        protocol.freeze(root)
    # Exercise real locally produced checkpoints, with one explicitly terminal
    # failed candidate. These reduced runs exist only inside pytest's tmp_path.
    from ganopt.train import train
    for index, job in enumerate(tuning['jobs']):
        c = json.loads(protocol.resolve_path(job['config']).read_text())
        run = train(c, protocol.resolve_path(job['output']))
        if index == 0:
            result = json.loads((run/'result.json').read_text())
            (run/'result.json').write_text(json.dumps({'status': 'failed',
                'last_valid_oracle_calls': result['oracle_calls'],
                'error': 'FloatingPointError(short fixture terminal failure)'}))
    frozen = protocol.freeze(root)
    assert len(frozen['failed_tuning_candidates']) == 1
    assert frozen['lr_by_algorithm']['adam'] != protocol.LEARNING_RATES[0]
    reporting = protocol.load_manifest(root, 'reporting')
    for job in reporting['jobs']:
        protocol.run_one(job['config'])
    output = tmp_path/'fixture-output'
    collector.collect_phase(root, 'tuning', output)
    # A stale file must not be swept into the active artifact manifest.
    destination = output/'runs'/'tuning'/tuning['jobs'][0]['job_id']
    (destination/'obsolete.json').write_text('{}')
    collector.collect_phase(root, 'tuning', output)
    collector.collect_phase(root, 'reporting', output)
    hashes = json.loads((output/'artifact_hashes.json').read_text())['sha256']
    assert not any(name.endswith('obsolete.json') for name in hashes)
    assert 'protocol_freeze.json' in hashes
    assert all(name in hashes for name in ['tuning_jobs.json', 'reporting_jobs.json',
        'tuning_summary.csv', 'tuning_checkpoints.csv', 'reporting_summary.csv',
        'reporting_checkpoints.csv', 'reporting_method_summary.csv'])
    for name in hashes:
        assert collector.sha256(output/name) == hashes[name]
        if name.endswith('.csv'):
            assert b'\r' not in (output/name).read_bytes()
    with (output/'reporting_method_summary.csv').open() as handle:
        import csv
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and all(row['n'] == '3' for row in rows)
    assert all(row['std_definition'] == 'sample standard deviation, ddof=1' for row in rows)
    another = tmp_path/'another-fixture-study'
    protocol.prepare(another)
    with pytest.raises(ValueError, match='study'):
        collector.collect_phase(another, 'tuning', output)


def test_run_one_rejects_off_checkpoint_pause_before_starting(tmp_path, monkeypatch):
    protocol, _ = modules()
    original_common = protocol.common_config
    def short_common(phase):
        c = original_common(phase)
        c.update(hidden=8, batch_size=4, z_dim=2, eval_samples=32,
                 oracle_budget=20, log_every=10, checkpoint_every=10)
        return c
    monkeypatch.setattr(protocol, 'common_config', short_common)
    manifest = protocol.prepare(tmp_path/'fixture')
    job = manifest['jobs'][0]
    with pytest.raises(ValueError, match='checkpoint'):
        protocol.run_one(job['config'], stop_after=5)
    assert not protocol.resolve_path(job['output']).exists()
