"""Benchmark and schedule CIFAR-10 128px training jobs on one NVIDIA GPU."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time


MODEL_KEYS = ("dataset", "image_size", "batch_size", "base_channels", "z_dim")
MEMORY_LIMIT = 0.90
TARGET_UTILIZATION = 80


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class GPU:
    def __init__(self, selector):
        rows = self.query("gpu", "uuid,name,memory.total", selector).splitlines()
        if len(rows) != 1:
            raise ValueError("Select one GPU with --gpu INDEX")
        self.uuid, self.name, total = (entry.strip() for entry in rows[0].split(","))
        self.memory_limit = float(total) * MEMORY_LIMIT
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.uuid)
        self.env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    @staticmethod
    def query(kind, fields, selector):
        command = ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"]
        if selector is not None:
            command.extend(["-i", selector])
        return subprocess.check_output(command, text=True, timeout=10).strip()

    def sample(self):
        return tuple(map(int, self.query("gpu", "utilization.gpu,memory.used", self.uuid).split(",")))

    @contextmanager
    def exclusive(self):
        with Path(f"/tmp/ganopt-{self.uuid}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pids = self.query("compute-apps", "pid", self.uuid)
            if pids and pids != "No running processes found":
                raise RuntimeError(f"GPU is already running compute jobs: {pids}")
            yield


def start(config_file, folder, log, gpu, resume=False):
    command = [sys.executable, "-u", "-m", "ganopt.train",
               "--config", str(config_file), "--output", str(folder)]
    if resume:
        command.append("--resume")
    with log.open("a") as output:
        return subprocess.Popen(command, env=gpu.env, stdout=output, stderr=subprocess.STDOUT)


def stop(children):
    active = [child for child in children if child.poll() is None]
    for child in active:
        child.terminate()
    for child in active:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def completed(folder, calls):
    result = folder / "result.json"
    if not result.is_file():
        return False
    data = read(result)
    return data.get("status") == "completed" and data.get("oracle_calls") == calls


def benchmark_group(base, root, gpu, workers, calls, baseline_memory):
    directory = root / f"workers_{workers}"
    directory.mkdir()
    jobs = []
    for worker in range(workers):
        config = dict(base, seed=base["seed"] + worker, algorithm="adam", stage="pilot",
                      oracle_budget=calls, log_every=calls, checkpoint_every=calls,
                      keep_checkpoints=False, eval_samples=1, download=False)
        config_file = directory / f"config_{worker}.json"
        write(config_file, config)
        jobs.append((config_file, directory / f"worker_{worker}", directory / f"worker_{worker}.log"))

    children = []
    peak = baseline_memory
    samples = []
    shared_since = None
    began = time.monotonic()
    error = None
    try:
        for config_file, folder, log in jobs:
            children.append(start(config_file, folder, log, gpu))
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (0, None) for code in codes):
                raise RuntimeError(f"Worker exit codes {codes}. See {directory}/*.log")
            if all(code == 0 for code in codes):
                break
            busy, memory = gpu.sample()
            peak = max(peak, memory)
            now = time.monotonic()
            # The trainer writes an initial checkpoint and logs its only row before final image export.
            shared = all((folder / "checkpoint.pt").exists() and not (folder / "metrics.jsonl").exists()
                         for _, folder, _ in jobs)
            if shared:
                if shared_since is None:
                    shared_since = now
            else:
                shared_since = None
            training_window = shared_since is not None and now - shared_since >= 1.0
            samples.append(dict(seconds=now - began, utilization=busy, memory_mib=memory,
                                training_window=training_window))
            if memory >= gpu.memory_limit:
                raise RuntimeError("Reached the 90% device memory limit during the benchmark")
            time.sleep(1)
        if not all(completed(folder, calls) for _, folder, _ in jobs):
            raise RuntimeError(f"Incomplete benchmark results in {directory}")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        error = str(exc)
    finally:
        stop(children)

    elapsed = time.monotonic() - began
    steady = [sample["utilization"] for sample in samples if sample["training_window"]]
    write(directory / "gpu_samples.json", samples)
    return dict(workers=workers, error=error, wall_seconds=elapsed, peak_mib=peak,
                baseline_mib=baseline_memory, samples=len(steady),
                mean_gpu_percent=mean(steady) if len(steady) >= 3 else None,
                calls_per_second=workers * calls / elapsed if error is None else None)


def benchmark(args, base, gpu):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"Benchmark output must be empty: {args.output}")
    groups = []
    for workers in sorted(set(args.workers)):
        _, memory = gpu.sample()
        projected = memory
        if groups:
            previous = groups[-1]
            added = previous["peak_mib"] - previous["baseline_mib"]
            projected += added * workers / previous["workers"]
        if projected >= gpu.memory_limit:
            print(f"Skipped {workers} workers due to projected device memory above 90%", flush=True)
            break
        print(f"Benchmarking {workers} workers on {gpu.name}", flush=True)
        group = benchmark_group(base, args.output, gpu, workers, args.calls, memory)
        groups.append(group)
        print(json.dumps(group), flush=True)
        if group["error"]:
            break

    valid = [group for group in groups if group["error"] is None]
    reached = [group for group in valid if (group["mean_gpu_percent"] or 0) >= TARGET_UTILIZATION]
    best = max(reached or valid, key=lambda group: group["calls_per_second"], default=None)
    summary = dict(gpu_uuid=gpu.uuid, model={key: base[key] for key in MODEL_KEYS},
                   target_gpu_percent=TARGET_UTILIZATION, target_met=bool(reached),
                   selected_workers=best["workers"] if best else None, groups=groups)
    summary_path = args.output / "benchmark_summary.json"
    write(summary_path, summary)
    print(f"selected_workers={summary['selected_workers']} target_met={bool(reached)}", flush=True)
    print(summary_path, flush=True)
    if best is None:
        raise RuntimeError("No benchmark group completed")


def training_configs(base):
    from ganopt.optim import METHODS
    from ganopt.runtime import stable_hash

    if base["stage"] == "tuning":
        if base["seed"] != 9001:
            raise ValueError("Tuning requires seed 9001")
        for method in METHODS:
            for lr in (5e-5, 1e-4, 2e-4):
                config = dict(base, algorithm=method, lr=lr)
                yield f"tuning-{method}-lr{lr:g}-{stable_hash(config)[:8]}", config
    elif base["stage"] == "main":
        freeze = read(Path(base["freeze_file"]))
        for seed in freeze["reporting_seeds"]:
            for method in METHODS:
                config = dict(base, seed=seed, algorithm=method, lr=freeze["lr_by_algorithm"][method])
                yield f"main-cifar10-{method}-s{seed}-{stable_hash(config)[:8]}", config
    else:
        raise ValueError("Training configuration must use the tuning or main stage")


def train(args, base, gpu):
    from ganopt.runtime import source_hash, stable_hash

    record = read(args.benchmark)
    if record["gpu_uuid"] != gpu.uuid or record["model"] != {key: base[key] for key in MODEL_KEYS}:
        raise ValueError("Benchmark does not match this GPU and model configuration")
    workers = record["selected_workers"]
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("No valid worker count in benchmark summary")
    source = source_hash()
    # The controller does not initialize CUDA; GPU validation runs in each training child.
    jobs = []
    for name, config in training_configs(base):
        folder = args.output / name
        if (folder / "metadata.json").exists():
            metadata = read(folder / "metadata.json")
            if metadata["source_hash"] != source or metadata["config_hash"] != stable_hash(config):
                raise ValueError(f"Existing run does not match source and config: {folder}")
        if completed(folder, config["oracle_budget"]):
            print(f"SKIP {name} completed", flush=True)
            continue
        resume = (folder / "checkpoint.pt").is_file()
        if folder.exists() and any(folder.iterdir()) and not resume:
            raise ValueError(f"Run has files but no checkpoint: {folder}")
        jobs.append((folder, config, resume))
    if not jobs:
        return
    control = args.output / f"_parallel_{time.time_ns()}"
    control.mkdir(parents=True)
    active = {}
    telemetry = control / "gpu_samples.jsonl"
    print(f"workers={workers} remaining_runs={len(jobs)} logs={control}", flush=True)
    try:
        while jobs or active:
            for child, (folder, calls) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                del active[child]
                if code != 0 or not completed(folder, calls):
                    raise RuntimeError(f"Training failed for {folder.name}. See {control / (folder.name + '.log')}")
                print(f"DONE {folder.name}", flush=True)
            if jobs or active:
                busy, memory = gpu.sample()
                with telemetry.open("a") as samples:
                    samples.write(json.dumps(dict(time=time.time(), active_workers=len(active),
                                                  gpu_percent=busy, memory_mib=memory)) + "\n")
            if jobs and len(active) < workers:
                if memory < gpu.memory_limit:
                    folder, config, resume = jobs.pop(0)
                    config_file = control / f"{folder.name}.json"
                    write(config_file, config)
                    child = start(config_file, folder, config_file.with_suffix(".log"), gpu, resume)
                    active[child] = (folder, config["oracle_budget"])
                    print(f"START {folder.name} resume={resume}", flush=True)
                elif not active:
                    raise RuntimeError("GPU memory is already at or above 90%")
            if jobs or active:
                time.sleep(1)
    finally:
        stop(active)
    print("PARALLEL_TRAINING_COMPLETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", help="NVIDIA index or UUID. Optional on a one-GPU instance.")
    modes = parser.add_subparsers(dest="mode", required=True)
    bench = modes.add_parser("benchmark")
    bench.add_argument("--config", type=Path, default=Path("configs/cifar10_128/pilot.json"))
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--calls", type=int, default=600)
    bench.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8, 12])
    run = modes.add_parser("train")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    try:
        base = read(args.config)
        if (base["dataset"], base["image_size"], base["batch_size"]) != ("cifar10", 128, 8):
            raise ValueError("Expected CIFAR-10 at resolution 128 and batch size 8")
        if args.mode == "benchmark" and (args.calls < 1 or min(args.workers) < 1):
            raise ValueError("Calls and worker counts must be positive")
        gpu = GPU(args.gpu)
        with gpu.exclusive():
            if args.mode == "benchmark":
                benchmark(args, base, gpu)
            else:
                train(args, base, gpu)
    except (OSError, RuntimeError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
