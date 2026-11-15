"""Resume frozen CIFAR-10 evaluation, screening actual 1/2/4-job throughput.

Run from the project root. This file deliberately stays outside ganopt/scripts.
Only trusted, locally produced training checkpoints are loaded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

METHODS = ('adam', 'extraadam', 'optimistic_adam', 'lookahead_minmax')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_jobs(root):
    root = Path(root).resolve()
    if not (root / 'TRAINING_COMPLETED').exists() or (root / 'training.exitcode').read_text().strip() != '0':
        raise ValueError('The training suite must finish successfully first')
    freeze = read_json(root / 'freeze.json')
    if not freeze.get('frozen'):
        raise ValueError('Protocol is not frozen')
    if sorted(freeze['reporting_seeds']) != [11, 22, 33]:
        raise ValueError('Expected reporting seeds 11, 22, 33 exactly once')
    expected = {(m, s) for m in METHODS for s in freeze['reporting_seeds']}
    found = {}
    for folder in sorted(root.glob('main-cifar10-*')):
        if not folder.is_dir():
            continue
        config = read_json(folder / 'config.json')
        key = config['algorithm'], config['seed']
        if key not in expected or key in found:
            raise ValueError(f'Unexpected or duplicate formal run: {folder}')
        if config.get('stage') != 'main' or config.get('dataset') != 'cifar10':
            raise ValueError(f'Not a main CIFAR-10 run: {folder}')
        if any(config.get(k) != v for k, v in freeze['common'].items()):
            raise ValueError(f'Frozen configuration mismatch: {folder}')
        if config['lr'] != freeze['lr_by_algorithm'][key[0]]:
            raise ValueError(f'Frozen learning rate mismatch: {folder}')
        if read_json(folder / 'metadata.json')['source_hash'] != freeze['source_hash']:
            raise ValueError(f'Frozen source mismatch: {folder}')
        result = read_json(folder / 'result.json')
        if result['status'] != 'completed' or result['oracle_calls'] != config['oracle_budget']:
            raise ValueError(f'Incomplete training run: {folder}')
        if read_json(config['freeze_file']) != freeze:
            raise ValueError(f'Run points at a different freeze: {folder}')
        found[key] = folder, config
    if set(found) != expected:
        raise ValueError(f'Missing formal runs: {expected - set(found)}')
    jobs = []
    for seed in freeze['reporting_seeds']:
        for method in METHODS:
            folder, config = found[method, seed]
            budget = config['oracle_budget']
            if budget != 20000 or config['checkpoint_every'] != 2000:
                raise ValueError('This driver requires the frozen 20,000-call/2,000-interval suite')
            points = [('validation', n, 5000) for n in range(2000, budget + 1, 2000)]
            points += [('test', budget, 10000)]
            for split, calls, count in points:
                checkpoint = folder / 'checkpoints' / f'calls_{calls:08d}.pt' if split == 'validation' else folder / 'checkpoint.pt'
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                output = folder / 'evaluation_validation' / f'calls_{calls:08d}' if split == 'validation' else folder / 'evaluation_test'
                jobs.append(dict(checkpoint=str(checkpoint), folder=str(output), config=config,
                                 method=method, seed=seed, calls=calls, split=split, count=count,
                                 source_hash=freeze['source_hash'], metric_protocol=freeze['metric_protocol']))
    if len(jobs) != 132:
        raise ValueError('Expected exactly 132 formal evaluation points')
    return jobs


def metric_valid(job):
    folder = Path(job['folder'])
    try:
        manifest = read_json(folder / 'manifest.json')
        result = read_json(folder / 'metrics.json')
        if not isinstance(manifest, dict) or not isinstance(result, dict) or not isinstance(result.get('metrics'), dict):
            return False
        expected = dict(config=job['config'], dataset='cifar10', source_hash=job['source_hash'],
                        oracle_calls=job['calls'], reference_split=job['split'], count=job['count'],
                        evaluation_seed=123456, metric_protocol=job['metric_protocol'],
                        checkpoint_sha256=digest(job['checkpoint']),
                        real_indices_sha256=job['reference_indices_sha256'])
        if any(manifest.get(k) != v for k, v in expected.items()):
            return False
        if result.get('status') != 'completed' or result.get('manifest') != manifest:
            return False
        for key, value in dict(source_hash=job['source_hash'], metric_protocol=job['metric_protocol'],
                               torch_fidelity_version='0.3.0', feature_extractor='inception-v3-compat').items():
            if result.get(key) != value:
                return False
        metrics = result['metrics']
        duration = float(result['evaluation_seconds'])
        if not math.isfinite(duration) or duration < 0:
            return False
        names = ('frechet_inception_distance', 'kernel_inception_distance_mean', 'kernel_inception_distance_std')
        if not all(math.isfinite(float(metrics[n])) for n in names):
            return False
        if metrics[names[0]] < 0 or metrics[names[2]] < 0:
            return False
        for kind in ('real', 'fake'):
            with os.scandir(folder / kind) as entries:
                if sum(e.is_file() and e.name.endswith('.png') for e in entries) != job['count']:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def preserve_partial(folder):
    folder = Path(folder)
    if not folder.exists():
        return None
    backup = folder.with_name(folder.name + '.partial-' + uuid.uuid4().hex[:12])
    folder.rename(backup)
    return backup


def choose_workers(records):
    best = sorted(records, key=lambda r: r['workers'])[0]
    for record in sorted(records, key=lambda r: r['workers'])[1:]:
        if record['jobs_per_second'] >= 1.10 * best['jobs_per_second']:
            best = record
    return best['workers']


def cpu_budget():
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    for path in ('/sys/fs/cgroup/cpu.max',):
        if Path(path).exists():
            quota, period = Path(path).read_text().split()
            if quota != 'max':
                count = min(count, max(1, int(quota) // int(period)))
    quota_path = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    if quota_path.exists():
        quota = int(quota_path.read_text())
        if quota > 0:
            count = min(count, max(1, quota // int(quota_path.with_name('cpu.cfs_period_us').read_text())))
    return count


def memory_headroom(host_available, limit, used):
    return max(0, min(host_available, limit - used))


def preflight(root, pending, requested_workers=None):
    if not pending:
        return dict(maximum_workers=0, threads=1, already_complete=True)
    disk = shutil.disk_usage(root)
    required_files = sum(2 * j['count'] for j in pending)
    fs = os.statvfs(root)
    required_bytes = required_files * max(4096, fs.f_frsize) + 1024**3
    if disk.free < required_bytes or (fs.f_files and fs.f_favail < required_files + 10000):
        raise RuntimeError(f'Insufficient free disk/inodes for retained PNGs: need about {required_bytes/1024**3:.1f} GiB and {required_files} files')
    info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free,memory.total',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(info) != 1:
        raise RuntimeError('This driver expects one visible physical GPU')
    gpu, free_mb, total_mb = [s.strip() for s in info[0].split(',')]
    benchmark_file = root / 'evaluation-benchmark.json'
    benchmark = read_json(benchmark_file) if benchmark_file.exists() else {}
    per_worker_mb = 6144.
    if benchmark.get('gpu') == gpu and benchmark.get('torch_peak_reserved_GiB') is not None:
        per_worker_mb = max(2048., (float(benchmark['torch_peak_reserved_GiB']) + .75) * 1024)
    # torch-fidelity 0.3.0 starts four data-loader processes per GPU job.
    usable_mb = max(0., float(free_mb) - .10 * float(total_mb))
    capacity = requested_workers if requested_workers is not None else min(4, max(1, cpu_budget() // 5))
    maximum = min(capacity, int(usable_mb // per_worker_mb))
    meminfo = dict((line.split(':')[0], int(line.split()[1])) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    available_bytes = meminfo.get('MemAvailable', 0) * 1024
    for limit_path, used_path in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                                  ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        if Path(limit_path).exists() and Path(used_path).exists():
            limit = Path(limit_path).read_text().strip()
            if limit != 'max':
                available_bytes = memory_headroom(available_bytes, int(limit), int(Path(used_path).read_text()))
    maximum = min(maximum, int(available_bytes // (3 * 1024**3)))
    if maximum < 1:
        raise RuntimeError('Insufficient free GPU or host memory for one evaluation worker')
    if requested_workers is None:
        maximum = max(n for n in (1, 2, 4) if n <= maximum)
    return dict(gpu=gpu, gpu_free_MiB=float(free_mb), gpu_total_MiB=float(total_mb),
                cpu_budget=cpu_budget(), threads=1 if requested_workers is not None else max(1, min(4, cpu_budget() // maximum - 4)),
                maximum_workers=maximum, loader_workers_per_process=4,
                host_available_GiB=available_bytes/1024**3,
                requested_workers=requested_workers, gpu_budget_fraction=.90,
                estimated_per_worker_MiB=per_worker_mb,
                free_disk_GiB=disk.free/1024**3, free_inodes=fs.f_favail)


def run_point(job, threads):
    folder = Path(job['folder'])
    folder.parent.mkdir(parents=True, exist_ok=True)
    lock = folder.with_name(folder.name + '.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return evaluate_point(job, threads)
    finally:
        lock.close()


def evaluate_point(job, threads):
    folder = Path(job['folder'])
    if metric_valid(job):
        return
    import torch
    from ganopt.runtime import source_hash
    from scripts.export_images import export
    from scripts.evaluate_images import evaluate
    torch.set_num_threads(threads)
    state = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
    if state['config'] != job['config'] or state['oracle_calls'] != job['calls'] or state['source_hash'] != job['source_hash']:
        raise ValueError('Checkpoint content disagrees with planned point')
    del state
    if source_hash() != job['source_hash']:
        raise ValueError('Frozen training source has changed')
    backup = preserve_partial(folder)
    if backup:
        print(f'PRESERVED_PARTIAL: {backup}', flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    export(job['checkpoint'], folder, job['config']['data_root'], split=job['split'],
           count=job['count'], seed=123456, device='cuda')
    torch.cuda.synchronize()
    exported = time.perf_counter()
    result = evaluate(folder, cuda=True)
    torch.cuda.synchronize()
    write_json(folder / 'execution.json', dict(export_seconds=exported-started,
               evaluation_seconds=time.perf_counter()-exported, total_seconds=time.perf_counter()-started,
               gpu=torch.cuda.get_device_name(0), threads=threads,
               torch_peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3))
    if not metric_valid(job):
        raise ValueError('New evaluation failed provenance/count/metric verification')
    print(json.dumps(result['metrics']), flush=True)


def run_group(jobs, workers, threads, project, root, on_done):
    def launch(job):
        name = f"{job['method']}-s{job['seed']}-{job['split']}-{job['calls']:08d}"
        log = root / 'evaluation_logs' / (name + '.log')
        log.parent.mkdir(exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        command = [sys.executable, str(Path(__file__).resolve()), '--project', str(project),
                   '--root', str(root), '--threads', str(threads), '--job', json.dumps(job)]
        with log.open('a') as stream:
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode or not metric_valid(job):
            raise RuntimeError(f'Evaluation failed; inspect {log}')
        return job, time.perf_counter() - started
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(launch, job) for job in jobs]
    try:
        for future in as_completed(futures):
            job, seconds = future.result()
            on_done(job, seconds)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return time.perf_counter() - started


def run_suite(root, project, requested_workers=None):
    import scipy
    from ganopt.runtime import source_hash, METRIC_PROTOCOL
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    import numpy as np
    if scipy.__version__ != '1.15.3':
        raise RuntimeError('SciPy 1.15.3 is required for this recorded environment')
    sample = dict(mu=np.zeros(2), sigma=np.eye(2))
    if abs(fid_statistics_to_metric(sample, sample, False)['frechet_inception_distance']) > 1e-8:
        raise RuntimeError('FID dependency check failed')
    freeze = read_json(root / 'freeze.json')
    if source_hash() != freeze['source_hash'] or METRIC_PROTOCOL != freeze['metric_protocol']:
        raise ValueError('Frozen source or image-metric protocol mismatch')
    weights = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'
    if digest(weights) != '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2':
        raise RuntimeError('Verified Inception weights must be cached before evaluation')
    jobs = make_jobs(root)
    from ganopt.data import split_indices
    _, validation_indices = split_indices(50000, freeze['common']['split_seed'], 5000)
    for job in jobs:
        indices = validation_indices if job['split'] == 'validation' else np.arange(10000)
        job['reference_indices_sha256'] = hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
    done = {j['folder'] for j in jobs if metric_valid(j)}
    pending = [j for j in jobs if j['folder'] not in done]
    resources = preflight(root, pending, requested_workers)
    write_json(root / 'evaluation-resources.json', resources)
    print('RESOURCES:', json.dumps(resources), flush=True)
    print(f'PROGRESS: {len(done)}/{len(jobs)}; validation={sum(j["split"]=="validation" for j in pending)} test={sum(j["split"]=="test" for j in pending)} pending', flush=True)
    started = time.perf_counter()

    def on_done(job, seconds):
        done.add(job['folder'])
        write_json(root / 'evaluation-progress.json', dict(completed=len(done), total=len(jobs),
                   last=job['folder'], last_point_wall_seconds=seconds, session_seconds=time.perf_counter()-started,
                   updated=datetime.now().astimezone().isoformat()))
        print(f'PROGRESS: {len(done)}/{len(jobs)} {job["method"]} seed={job["seed"]} {job["split"]} calls={job["calls"]} wall={seconds:.1f}s', flush=True)

    records = []
    for workers in (1, 2, 4):
        if requested_workers is not None:
            break
        if workers > resources['maximum_workers']:
            continue
        cohort = [j for j in jobs if j['split'] == 'validation' and j['folder'] not in done][:workers]
        if len(cohort) < workers:
            break
        print(f'BENCHMARK_START: workers={workers}, fresh_points={len(cohort)}', flush=True)
        seconds = run_group(cohort, workers, resources['threads'], project, root, on_done)
        record = dict(workers=workers, points=len(cohort), seconds=seconds, jobs_per_second=len(cohort)/seconds)
        records.append(record)
        print('BENCHMARK_RESULT:', json.dumps(record), flush=True)
    workers = resources['maximum_workers'] if requested_workers is not None and pending else (choose_workers(records) if records else 1)
    write_json(root / 'evaluation-parallelism.json', dict(resources=resources, screen=records, selected_workers=workers))
    print(f'SELECTED_WORKERS: {workers}; threads_per_worker={resources["threads"]}', flush=True)
    remaining = [j for j in jobs if j['folder'] not in done]
    if remaining:
        run_group(remaining, workers, resources['threads'], project, root, on_done)
    rows = []
    for job in jobs:
        if not metric_valid(job):
            raise ValueError(f'Missing or invalid final point: {job["folder"]}')
        result = read_json(Path(job['folder']) / 'metrics.json')
        rows.append(dict(method=job['method'], seed=job['seed'], split=job['split'],
                         oracle_calls=job['calls'], count=job['count'], folder=job['folder'],
                         evaluation_seconds=result['evaluation_seconds'], **result['metrics']))
    write_json(root / 'image_metrics.json', rows)
    with (root / 'image_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / 'EVALUATION_COMPLETED').touch()
    print(f'ALL_EVALUATIONS_COMPLETED: {len(rows)}/132', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--root', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, help='Use this concurrency directly, bounded by 90%% GPU memory and host-memory headroom; skip throughput screening')
    parser.add_argument('--job', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error('--workers must be positive')
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)
    root = (args.root or Path('runs/latest_cifar10_main.txt').read_text().strip())
    root = Path(root).resolve()
    if args.job:
        run_point(json.loads(args.job), args.threads)
        return
    lock = (root / 'evaluation-suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'EVALUATION_COMPLETED').unlink(missing_ok=True)
    status = 1
    try:
        run_suite(root, project, args.workers)
        status = 0
    finally:
        (root / 'evaluation.exitcode').write_text(str(status) + '\n')


if __name__ == '__main__':
    main()
