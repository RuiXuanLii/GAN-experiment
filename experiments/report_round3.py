"""Validate collected round3 evidence, then render a compact Chinese report.

This script consumes archived CSV/JSON/NumPy evidence only. It neither trains
models nor loads live optimizer checkpoints. Seed 11 is the fixed visualization
seed for all four primary methods, chosen before the reporting runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ganopt.data import BatchStream
from ganopt.metrics import synthetic_metrics
from ganopt.runtime import source_hash, stable_hash
from experiments.collect_round3 import METRICS, mode_counts
from experiments.run_round3 import (EVALUATION, METHODS, PRIMARY_CONDITION,
    REPORTING_SEEDS, TAIL_CHECKPOINTS, expected_counts, initial_parameter_hash,
    portable_path, protocol_code_hashes, reporting_configs, resolve_path,
    select_learning_rates, validate_tuning_config)

VISUALIZATION_SEED = 11
LABELS = {'adam': 'Adam', 'extraadam': 'ExtraAdam',
          'optimistic_adam': 'Optimistic Adam', 'lookahead_minmax': 'Lookahead-Minmax'}
REPORT_FILES = ('README.md', 'final_validation.json', 'comparison.png')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty evidence CSV: {path}')
    return rows


def number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Missing/non-numeric evidence: {label}') from exc
    if not math.isfinite(result):
        raise ValueError(f'Nonfinite evidence: {label}')
    return result


def equal_number(actual, expected, label):
    if not math.isclose(number(actual, label), float(expected), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f'Evidence mismatch: {label}')


def check_fields(row, values, label):
    for key, expected in values.items():
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            equal_number(row.get(key), expected, f'{label}.{key}')
        elif expected is None:
            if row.get(key) not in (None, ''):
                raise ValueError(f'Evidence mismatch: {label}.{key}')
        elif row.get(key) != expected:
            raise ValueError(f'Evidence mismatch: {label}.{key}')


def recompute_reporting(summaries, checkpoints, reported_methods, budget=120000, spacing=6000):
    """Independently reaggregate every run and all three-seed method statistics."""
    expected_pairs = {(m, s) for m in METHODS for s in REPORTING_SEEDS}
    pairs = [(r['method'], int(r['seed'])) for r in summaries]
    if len(summaries) != 12 or set(pairs) != expected_pairs or len(set(pairs)) != 12:
        raise ValueError('Reporting requires exactly 12 distinct method/fresh-seed results')
    cases = [r['case'] for r in summaries]
    if len(set(cases)) != 12 or any(r['case'] not in cases for r in checkpoints):
        raise ValueError('Unknown or duplicate reporting checkpoint case')
    expected_calls = list(range(spacing, budget+1, spacing))
    if len(expected_calls) < TAIL_CHECKPOINTS:
        raise ValueError('At least five actual checkpoints required')
    per_seed = []
    for summary in summaries:
        case = summary['case']
        series = sorted([r for r in checkpoints if r['case'] == case], key=lambda r: int(r['oracle_calls']))
        if [int(r['oracle_calls']) for r in series] != expected_calls:
            raise ValueError(f'Missing/duplicate reporting checkpoints: {case}')
        if summary['status'] != 'completed' or int(summary['oracle_calls']) != budget:
            raise ValueError(f'Incomplete reporting result: {case}')
        tail = series[-TAIL_CHECKPOINTS:]
        if int(summary['tail_checkpoint_count']) != TAIL_CHECKPOINTS or summary['tail_checkpoint_calls'] != ';'.join(str(r['oracle_calls']) for r in tail):
            raise ValueError(f'Wrong last5 checkpoint identities: {case}')
        result = {'case': case, 'method': summary['method'], 'seed': int(summary['seed']),
                  'lr': number(summary['lr'], 'lr'), 'oracle_calls': budget,
                  'last5_checkpoint_calls': [int(r['oracle_calls']) for r in tail]}
        for row in series:
            if row['method'] != summary['method'] or int(row['seed']) != int(summary['seed']):
                raise ValueError(f'Checkpoint method/seed pairing mismatch: {case}')
            values = {metric: number(row[metric], f'{case}.{metric}') for metric in METRICS}
            if (not 0 <= values['covered_modes'] <= 8 or not values['covered_modes'].is_integer()
                    or not 0 <= values['quality_fraction'] <= 1
                    or values['mean_nearest_distance'] < 0
                    or not -1e-12 <= values['assignment_entropy'] <= math.log(8)+1e-12):
                raise ValueError(f'Out-of-range synthetic checkpoint diagnostics: {case}')
        for metric in METRICS:
            final = number(series[-1][metric], metric)
            tail_mean = float(np.mean([number(r[metric], metric) for r in tail]))
            equal_number(summary[metric], final, f'{case}.endpoint.{metric}')
            equal_number(summary['final_'+metric], final, f'{case}.final_{metric}')
            equal_number(summary['last5_mean_'+metric], tail_mean, f'{case}.last5_mean_{metric}')
            result['final_'+metric] = final
            result['last5_mean_'+metric] = tail_mean
        qualities = [number(r['quality_fraction'], 'tail quality') for r in tail]
        result.update(within_seed_last5_quality_sd=float(np.std(qualities, ddof=1)),
            minimum_last5_coverage=min(int(r['covered_modes']) for r in tail),
            minimum_last5_quality=min(qualities), maximum_last5_quality=max(qualities))
        per_seed.append(result)
    reported = {r['method']: r for r in reported_methods}
    if len(reported_methods) != 4 or set(reported) != set(METHODS):
        raise ValueError('Exactly four collector method aggregates required')
    methods = []
    for method in METHODS:
        rows = [r for r in per_seed if r['method'] == method]
        if len({r['lr'] for r in rows}) != 1:
            raise ValueError('Reporting seeds use different learning rates')
        aggregate = {'method': method, 'n': 3, 'lr': rows[0]['lr'], 'oracle_calls': budget}
        check_fields(reported[method], {'n': 3, 'lr': rows[0]['lr'], 'oracle_calls': budget}, 'collector aggregate')
        for prefix in ['final_', 'last5_mean_']:
            for metric in METRICS:
                values = [r[prefix+metric] for r in rows]
                for suffix, value in [('_mean', np.mean(values)), ('_std', np.std(values, ddof=1))]:
                    key = prefix+metric+suffix
                    equal_number(reported[method][key], value, f'collector aggregate.{method}.{key}')
                    aggregate[key] = float(value)
        aggregate.update(
            within_seed_last5_quality_sd_mean=float(np.mean([r['within_seed_last5_quality_sd'] for r in rows])),
            within_seed_last5_quality_sd_by_seed={str(r['seed']): r['within_seed_last5_quality_sd'] for r in rows},
            minimum_coverage_across_15_tail_checkpoints=min(r['minimum_last5_coverage'] for r in rows),
            minimum_quality_across_15_tail_checkpoints=min(r['minimum_last5_quality'] for r in rows))
        methods.append(aggregate)
    return {'per_seed': per_seed, 'methods': methods}


def verify_inventory(root):
    inventory = read_json(root/'artifact_hashes.json')
    hashes = inventory['sha256']
    if inventory['source_hash'] != source_hash() or inventory['protocol_code_hashes'] != protocol_code_hashes():
        raise ValueError('Artifact inventory source/protocol mismatch')
    for name, digest in hashes.items():
        path = root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(path) != digest:
            raise ValueError(f'Artifact SHA256 mismatch: {name}')
    active_files = set()
    study_hash = sha256(root/'tuning_jobs.json')
    freeze_hash = stable_hash(read_json(root/'protocol_freeze.json'))
    for phase in ['tuning', 'reporting']:
        name = f'{phase}_active.json'
        active = read_json(root/name)
        if (active['source_hash'] != source_hash() or active['protocol_code_hashes'] != protocol_code_hashes()
                or active['study_hash'] != study_hash
                or active.get('freeze_hash') not in (None, freeze_hash)
                or (phase == 'reporting' and active.get('freeze_hash') != freeze_hash)):
            raise ValueError('Collected source/study/freeze pairing mismatch')
        if len(active['runs']) != 12 or len(set(active['runs'])) != 12:
            raise ValueError('Every phase must preserve all 12 candidate/result records')
        active_files.update(active['files'])
        active_files.add(name)
    if not active_files.issubset(hashes):
        raise ValueError('Active evidence is missing from the SHA256 inventory')
    required = {'tuning_jobs.json', 'reporting_jobs.json', 'protocol_freeze.json',
        'tuning_summary.csv', 'tuning_checkpoints.csv', 'reporting_summary.csv',
        'reporting_checkpoints.csv', 'reporting_method_summary.csv', 'target_control.json',
        'tuning_active.json', 'reporting_active.json'}
    if not required.issubset(active_files):
        raise ValueError('Required consumed input is missing from the active inventory')
    for phase in ['tuning', 'reporting']:
        for job in read_json(root/f'{phase}_jobs.json')['jobs']:
            folder = f"runs/{phase}/{job['job_id']}"
            consumed = {f'{folder}/{name}' for name in ['config.json', 'metadata.json', 'result.json']}
            consumed.add(f"configs/{phase}/{Path(job['config']).name}")
            if not consumed.issubset(active_files):
                raise ValueError('Required per-run consumed input is missing from the active inventory')
            if read_json(root/folder/'result.json')['status'] == 'completed' and f'{folder}/samples.npy' not in active_files:
                raise ValueError('Required endpoint samples are missing from the active inventory')
    return active_files, len(hashes)


def _audit_archived_runs(root, phase, summaries, checkpoints, frozen):
    manifest = read_json(root/f'{phase}_jobs.json')
    if (manifest['source_hash'] != source_hash() or manifest['protocol_code_hashes'] != protocol_code_hashes()
            or manifest['phase'] != phase or manifest['condition'] != PRIMARY_CONDITION
            or manifest['evaluation'] != EVALUATION or len(manifest['jobs']) != 12):
        raise ValueError('Archived jobs manifest mismatch')
    by_case = {r['case']: r for r in summaries}
    if len(summaries) != 12 or len(by_case) != 12 or set(by_case) != {j['job_id'] for j in manifest['jobs']}:
        raise ValueError('Manifest must preserve all 12 run results')
    if any(row['case'] not in by_case for row in checkpoints):
        raise ValueError(f'Unknown {phase} checkpoint case')
    active = read_json(root/f'{phase}_active.json')
    if set(active['runs']) != set(by_case):
        raise ValueError('Active run identities differ from results')
    configs, sample_sets = {}, {}
    expected_reporting = reporting_configs(frozen, root/'protocol_freeze.json')
    expected_reporting = [{k: v for k, v in c.items() if k != 'freeze_file'} for c in expected_reporting]
    for job in manifest['jobs']:
        case = job['job_id']
        folder = root/'runs'/phase/case
        config = read_json(folder/'config.json')
        metadata, result = read_json(folder/'metadata.json'), read_json(folder/'result.json')
        if read_json(root/'configs'/phase/Path(job['config']).name) != config:
            raise ValueError('Archived prepared and executed configs differ')
        if phase == 'tuning':
            validate_tuning_config(config)
        elif {k: v for k, v in config.items() if k != 'freeze_file'} not in expected_reporting:
            raise ValueError('Reporting config does not match this frozen protocol')
        chash = stable_hash(config)
        init_hash = initial_parameter_hash(config)
        if (job['config_hash'] != chash or metadata['config_hash'] != chash or metadata['config'] != config
                or job['source_hash'] != source_hash() or metadata['source_hash'] != source_hash()
                or job['expected_initial_parameter_hash'] != init_hash or metadata['initial_parameter_hash'] != init_hash):
            raise ValueError('Source/config/initialization evidence mismatch')
        summary = by_case[case]
        check_fields(summary, {'method': config['algorithm'], 'seed': config['seed'], 'lr': config['lr'],
            'stage': config['stage'], 'config_hash': chash, 'source_hash': source_hash(),
            'initial_parameter_hash': init_hash, 'condition': PRIMARY_CONDITION,
            'training_schedule': 'synchronous', 'critic_steps': 1, 'eval_samples': config['eval_samples'],
            'evaluation_seed': EVALUATION['seed']}, f'{phase}.{case}')
        if result['status'] != summary['status']:
            raise ValueError('Run result and summary status mismatch')
        calls = result['oracle_calls'] if result['status'] == 'completed' else result['last_valid_oracle_calls']
        check_fields(summary, expected_counts(config, calls), f'{case}.actual counts')
        series = sorted([r for r in checkpoints if r['case'] == case], key=lambda r: int(r['oracle_calls']))
        expected_calls = list(range(config['checkpoint_every'], calls+1, config['checkpoint_every']))
        if [int(row['oracle_calls']) for row in series] != expected_calls:
            raise ValueError(f'Missing/duplicate/out-of-range {phase} checkpoints: {case}')
        for row in series:
            check_fields(row, {'method': config['algorithm'], 'seed': config['seed'], 'lr': config['lr'],
                'config_hash': chash, 'source_hash': source_hash(), 'initial_parameter_hash': init_hash}, f'{case}.checkpoint identity')
            check_fields(row, expected_counts(config, int(row['oracle_calls'])), f'{case}.checkpoint counts')
            assigned = [number(row[f'assigned_mode_{i}'], 'mode count') for i in range(8)]
            good = [number(row[f'good_mode_{i}'], 'qualified mode count') for i in range(8)]
            if any(v < 0 or not v.is_integer() for v in assigned+good) or any(g > a for g, a in zip(good, assigned)):
                raise ValueError('Invalid actual per-mode counts')
            equal_number(sum(assigned), config['eval_samples'], 'assigned point total')
            equal_number(sum(good)/config['eval_samples'], row['quality_fraction'], 'qualified fraction')
            equal_number(sum(v >= .01*config['eval_samples'] for v in good), row['covered_modes'], 'mode coverage count')
            proportions = np.array(assigned)/config['eval_samples']
            positive = proportions[proportions > 0]
            equal_number(-(positive*np.log(positive)).sum(), row['assignment_entropy'], 'assignment entropy')
        if result['status'] == 'completed':
            tail = series[-TAIL_CHECKPOINTS:]
            if len(tail) != TAIL_CHECKPOINTS:
                raise ValueError(f'Five completed {phase} checkpoints required: {case}')
            check_fields(summary, {'tail_checkpoint_count': TAIL_CHECKPOINTS,
                'tail_checkpoint_calls': ';'.join(str(row['oracle_calls']) for row in tail)}, f'{phase}.last5 identities')
            for metric in METRICS:
                values = [number(row[metric], f'{phase}.checkpoint.{metric}') for row in series]
                equal_number(summary['final_'+metric], values[-1], f'{phase}.final.{metric}')
                equal_number(summary['last5_mean_'+metric], np.mean(values[-TAIL_CHECKPOINTS:]), f'{phase}.last5.{metric}')
            samples = np.load(folder/'samples.npy', allow_pickle=False)
            if samples.shape != (config['eval_samples'], 2) or not np.isfinite(samples).all():
                raise ValueError('Invalid endpoint sample array')
            diagnostics = {**synthetic_metrics(samples), **mode_counts(samples)}
            check_fields(summary, diagnostics, f'{case}.independent endpoint metrics')
            check_fields(result, synthetic_metrics(samples), f'{case}.endpoint result')
            if result['oracle_calls'] != config['oracle_budget']:
                raise ValueError('Completed run has wrong field budget')
            equal_number(result['parameter_updates'], expected_counts(config, calls)['parameter_updates'], 'result accepted updates')
            sample_sets[case] = samples
        elif phase != 'tuning' or not result.get('error', '').startswith('FloatingPointError('):
            raise ValueError('Reporting failure or unresolved infrastructure failure')
        configs[case] = config
    for seed in {c['seed'] for c in configs.values()}:
        identities = {by_case[case]['initial_parameter_hash'] for case, c in configs.items() if c['seed'] == seed}
        if len(identities) != 1:
            raise ValueError('Common-seed initialization pairing mismatch')
    return configs, sample_sets


def _archived_reference(root):
    rows = read_csv(root/'summary.csv')
    selected = [r for r in rows if r['case'] in {
        f'adam_alternating_d5_h128_s{s}_120k' for s in [9001, 9002, 9003]}]
    if len(selected) != 3 or {int(r['seed']) for r in selected} != {9001, 9002, 9003}:
        raise ValueError('Archived round2 reference must contain its three original seeds')
    for row in selected:
        check_fields(row, {'method': 'adam', 'training_schedule': 'alternating', 'critic_steps': 5,
            'oracle_calls': 120000, 'g_updates': 20000, 'd_updates': 100000}, 'archived Adam5:1')
    quality = [number(r['quality_fraction'], 'archived quality') for r in selected]
    return {'condition': 'archived round2 Adam alternating D:G=5:1; separate conditions',
        'seeds': [9001, 9002, 9003], 'quality_mean': float(np.mean(quality)),
        'quality_sample_sd': float(np.std(quality, ddof=1)),
        'final_coverage': [int(r['covered_modes']) for r in selected],
        'source_summary_sha256': sha256(root/'summary.csv')}


def _figure(root, summaries, checkpoints, samples, reference):
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    centers = 2*np.stack([np.cos(np.arange(8)*np.pi/4), np.sin(np.arange(8)*np.pi/4)], axis=1)
    panels = [('Real target control', reference)]
    for method in METHODS:
        row = next(r for r in summaries if r['method'] == method and int(r['seed']) == VISUALIZATION_SEED)
        panels.append((f"{LABELS[method]} | seed {VISUALIZATION_SEED}\nQ={100*float(row['quality_fraction']):.1f}%, C={int(row['covered_modes'])}/8", samples[row['case']]))
    for axis, (label, points) in zip(axes.flat, panels):
        axis.scatter(reference[:, 0], reference[:, 1], s=2, color='#aaaaaa', alpha=.2, rasterized=True)
        axis.scatter(points[:, 0], points[:, 1], s=2, color='#267ab4', alpha=.4, rasterized=True)
        axis.scatter(centers[:, 0], centers[:, 1], marker='x', s=22, color='#bf3434')
        # Full data limits include every generated point, including divergent ones.
        joined = np.vstack([reference, points, centers])
        low, high = joined.min(axis=0)-.3, joined.max(axis=0)+.3
        axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), title=label, aspect='equal')
        axis.tick_params(labelsize=8)
    axis = axes.flat[5]
    for method in METHODS:
        cases = [r['case'] for r in summaries if r['method'] == method]
        series = [sorted([r for r in checkpoints if r['case'] == case], key=lambda r: int(r['oracle_calls'])) for case in cases]
        calls = np.array([int(r['oracle_calls']) for r in series[0]])/1000
        values = 100*np.array([[float(r['quality_fraction']) for r in rows] for rows in series])
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        line, = axis.plot(calls, mean, label=LABELS[method], linewidth=1.5)
        axis.fill_between(calls, np.maximum(0, mean-sd), np.minimum(100, mean+sd),
                          color=line.get_color(), alpha=.12, linewidth=0)
    axis.set(xlabel='Joint field calls (thousands)', ylabel='Quality fraction (%)',
             ylim=(0, 100), title='Three fresh seeds: mean ± sample SD')
    axis.grid(alpha=.2)
    axis.legend(fontsize=7)
    figure.savefig(root/'comparison.png', dpi=160)
    plt.close(figure)


def _readme(root, stats, reference, environment):
    rows = stats['methods']
    def pm(row, key, scale=1, precision=2):
        return f"{scale*row[key+'_mean']:.{precision}f}±{scale*row[key+'_std']:.{precision}f}"
    final_table = '\n'.join(
        f"| {LABELS[r['method']]} | {r['lr']:g} | {pm(r, 'final_covered_modes')} | {pm(r, 'final_quality_fraction', 100)} | {pm(r, 'final_mean_nearest_distance', precision=4)} | {pm(r, 'last5_mean_quality_fraction', 100)} |" for r in rows)
    stability_table = '\n'.join(
        f"| {LABELS[r['method']]} | "+' / '.join(f"{100*r['within_seed_last5_quality_sd_by_seed'][str(s)]:.2f}" for s in REPORTING_SEEDS)+f" | {r['minimum_coverage_across_15_tail_checkpoints']}/8 | {100*r['minimum_quality_across_15_tail_checkpoints']:.2f}% |" for r in rows)
    text = f'''# 第三轮：四优化器同步比较

四法均使用联合同步1:1、宽度128、ReLU、GP=0.1及相同数据流；是继承的WGAN-GP变体，不是原论文基准复现。每条轨迹120,000次联合场调用；每次包含G、D偏导。ExtraAdam每次接受更新用两次调用，故接受60,000步，其余120,000步。调参仅用seed9001及三档学习率，依末5检查点覆盖、合格率、距离顺序选择后冻结，再运行新种子11/22/33。

| 方法 | 冻结LR | 终点覆盖/8 | 终点合格率% | 终点平均距离 | 末5平均合格率% |
|---|---:|---:|---:|---:|---:|
{final_table}

±为三种子间样本标准差。末5为96k至120k、间隔6k；合格半径严格小于0.15，每团至少21/2048个合格点算覆盖。下表波动标准差沿各种子末5个检查点计算，不能单凭波动小认定训练稳定；最低覆盖取三种子合计15个末段检查点。选中学习率是三档候选内的结果，且均位于候选上界，不代表充分搜索后的最优值。

| 方法 | 末5合格率波动SD：11 / 22 / 33（百分点） | 15点最低覆盖 | 15点最低合格率 |
|---|---:|---:|---:|
{stability_table}

![固定seed11终点、真实对照及三种子均值曲线](comparison.png)

四法均展示预先固定的seed11终点，包含全部2048点；坐标按全数据扩展。真实对照复用收集器的BatchStream。曲线为三种子均值±样本SD，阴影按百分比范围0–100%截取。完整12项结果均保留，未用最佳检查点替代终点。覆盖和合格率不能充分评价团内分布形状。这是三种子的探索性比较，不保证普遍收敛或统计显著性；并行CPU耗时不作速度排名。

第二轮Adam交替5:1参考合格率为{100*reference['quality_mean']:.2f}%±{100*reference['quality_sample_sd']:.2f}个百分点，属于不同训练条件，不纳入本轮均值或排名；见[原结果](../round2/README.md)和[原CSV](../round2/summary.csv)。

[设计](../../docs/round3-study-design.md)、[冻结协议](protocol_freeze.json)、[调参](tuning_summary.csv)、[12项结果](reporting_summary.csv)、[全部检查点](reporting_checkpoints.csv)、[逐次文件](runs/reporting/)、[独立核验](final_validation.json)、[文件哈希](artifact_hashes.json)。

从仓库根目录在空`runs/round3`下复现；中断后对对应`run-one`加`--resume`：

```bash
python -m experiments.run_round3 prepare
for c in runs/round3/configs/tuning/*.json; do python -m experiments.run_round3 run-one --config "$c"; done
python -m experiments.run_round3 freeze
for c in runs/round3/configs/reporting/*.json; do python -m experiments.run_round3 run-one --config "$c"; done
python -m experiments.run_round3 collect --phase tuning
python -m experiments.run_round3 collect --phase reporting
python -m experiments.report_round3 --results results/round3
```

本次报告环境为{environment['system']}，CUDA检测为`{environment['cuda_available']}`；当前Linux不能写入本机D盘，未完成CIFAR-10训练或FID/KID。请在Windows运行以下下载、校验命令，目标固定为`D:\\ESTR3108\\project\\CIFAR-10\\dataset`；详见[本机步骤](../../docs/cifar10-windows-zh.md)及[配置](../../configs/windows/cifar10_pilot.json)。

```powershell
py -3 scripts/prepare_cifar10.py --root "D:\\ESTR3108\\project\\CIFAR-10\\dataset"
py -3 scripts/prepare_cifar10.py --verify-only
```
'''
    (root/'README.md').write_text(text)


def report(results='results/round3', round2='results/round2'):
    root, round2 = resolve_path(results), resolve_path(round2)
    active_files, verified_count = verify_inventory(root)
    frozen = read_json(root/'protocol_freeze.json')
    # reporting_configs checks all frozen common/source/protocol conditions.
    reporting_configs(frozen, root/'protocol_freeze.json')
    if frozen['tuning_manifest_sha256'] != sha256(root/'tuning_jobs.json'):
        raise ValueError('Freeze and tuning-manifest identity mismatch')
    summaries, checkpoints = read_csv(root/'reporting_summary.csv'), read_csv(root/'reporting_checkpoints.csv')
    tuning, tuning_checkpoints = read_csv(root/'tuning_summary.csv'), read_csv(root/'tuning_checkpoints.csv')
    configs, samples = _audit_archived_runs(root, 'reporting', summaries, checkpoints, frozen)
    tuning_configs, _ = _audit_archived_runs(root, 'tuning', tuning, tuning_checkpoints, frozen)
    candidates = [dict(row, config=tuning_configs[row['case']]) for row in tuning]
    for candidate in candidates:
        if candidate['status'] == 'completed':
            for metric in METRICS:
                key = 'last5_mean_'+metric
                candidate[key] = number(candidate[key], key)
    if select_learning_rates(candidates) != frozen['lr_by_algorithm']:
        raise ValueError('Frozen LRs differ from tuning-only selection')
    frozen_evidence = {r['job_id']: r for r in frozen['tuning_evidence']}
    if len(frozen['tuning_evidence']) != 12 or set(frozen_evidence) != {r['case'] for r in tuning}:
        raise ValueError('Freeze does not preserve all 12 tuning candidates')
    for row in tuning:
        check_fields(row, frozen_evidence[row['case']]['summary'], 'frozen tuning summary')
    stats = recompute_reporting(summaries, checkpoints, read_csv(root/'reporting_method_summary.csv'),
        budget=frozen['common']['oracle_budget'], spacing=frozen['common']['checkpoint_every'])
    first_config = configs[summaries[0]['case']]
    stream = BatchStream(dict(first_config, seed=123456, device='cpu'))
    real = np.concatenate([stream.next()[0].numpy() for _ in range(32)])[:2048]
    control = synthetic_metrics(real)
    check_fields(read_json(root/'target_control.json'), control, 'BatchStream target control')
    archived = _archived_reference(round2)
    environment = {'system': platform.system(), 'cuda_available': torch.cuda.is_available(),
        'windows_target': r'D:\ESTR3108\project\CIFAR-10\dataset',
        'windows_download_performed_here': False, 'image_training_or_fid_kid_performed': False}
    validation = {'schema_version': 1, 'passed': True, 'condition': PRIMARY_CONDITION,
        'source_hash': source_hash(), 'protocol_code_hashes': protocol_code_hashes(),
        'report_script_sha256': sha256(Path(__file__)), 'freeze_hash': stable_hash(frozen),
        'study_hash': sha256(root/'tuning_jobs.json'), 'visualization_seed': VISUALIZATION_SEED,
        'reporting_seeds': list(REPORTING_SEEDS), 'all_12_reporting_results_preserved': True,
        'all_12_tuning_candidates_preserved': True, 'verified_input_sha256_files': verified_count,
        'selected_lr': frozen['lr_by_algorithm'], 'tail_checkpoint_count_per_seed': 5,
        'sample_sd_ddof': 1, 'tail_checkpoint_rows_reaggregated': 60,
        'historical_checkpoint_validation': 'CSV identities, actual counts, per-mode counts, and aggregation; live checkpoints not reloaded',
        'endpoint_validation': 'all 12 samples.npy re-evaluated with unchanged synthetic_metrics and mode counts',
        'tuning_statuses': {r['case']: r['status'] for r in tuning},
        'target_control': control, 'archived_round2_separate_reference': archived,
        'environment': environment, **stats}
    # All evidence gates pass before the first official report output is written.
    _figure(root, summaries, checkpoints, samples, real)
    _readme(root, stats, archived, environment)
    (root/'final_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    hashes = {name: sha256(root/name) for name in sorted(active_files | set(REPORT_FILES))}
    (root/'artifact_hashes.json').write_text(json.dumps({'sha256': hashes,
        'source_hash': source_hash(), 'protocol_code_hashes': protocol_code_hashes(),
        'report_script_sha256': sha256(Path(__file__))}, indent=2)+'\n')
    return {'passed': True, 'reporting_runs': 12, 'visualization_seed': VISUALIZATION_SEED,
            'output': portable_path(root), 'files': list(REPORT_FILES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/round3'))
    parser.add_argument('--round2', type=Path, default=Path('results/round2'))
    args = parser.parse_args()
    print(json.dumps(report(args.results, args.round2), indent=2))


if __name__ == '__main__':
    main()
