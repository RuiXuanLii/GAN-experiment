# CIFAR-10 formal evaluation

Evaluate the existing frozen 12-run suite without changing training code or configuration. There are 120 validation points (5,000 reference/generated images at calls 2,000 through 20,000) and 12 final test points (10,000 reference/generated images at calls 20,000). Validation and test results must remain separate. Existing valid Adam seed-11 final validation results count toward the 132 points.

The standalone driver lives outside `ganopt/` and `scripts/`, so it does not change their frozen source hash. Its parent process validates the run inventory, completion evidence, manifests, checkpoint checksums, and cached metric protocol. Each new point invokes the existing exporter and evaluator in a separate process. Incomplete exports are preserved under unique backup names before retry. A global lock prevents duplicate drivers; point locks prevent overlapping writers.

Select concurrency from actual throughput of fresh 5,000-image validation jobs at 1, 2, and 4 workers, subject to CPU and memory screening. All benchmark points contribute to the final results. Launching a fresh process per point is used for both the screen and the remaining queue. Prefer fewer workers unless a larger count improves measured throughput by at least 10%. This is an operational screen, not a formal GPU scaling claim. CPU/BLAS thread limits are fixed during the screen and subsequent queue. Evaluation throughput is separate from original isolated training-time comparisons.

The library starts four CPU data-loader processes per evaluation. The conservative cap allows at least five CPU slots per GPU job; a 16-core allocation therefore screens 1 and 2 workers and skips 4. GPU memory headroom and host/container memory headroom can reduce this further. A single-process peak is only a memory screening input, not proof of a speedup.

The user subsequently requested direct launch from the measured memory footprint without further throughput screening. Use `--workers 8` for that execution: one CPU compute thread per job; reserve 0.75 GiB beyond the measured 1.834 GiB Torch peak per process; cap total estimated use at 90% of device memory, subtracting current non-worker use. Host/container free memory may reduce concurrency further. This estimates capacity rather than guaranteeing measured occupancy or speedup. The underlying library still starts four CPU loader processes per job. Omit `--workers` to use the optional throughput screen instead.

Retain the canonical PNGs, manifests, metrics, logs, and checkpoints. Check free space and inode availability before starting. Cancel queued work on failure, retain completed work, and return nonzero. Resume revalidates all canonical points and skips only valid results. Write `EVALUATION_COMPLETED` only after all 132 expected results validate.

## Implementation plan

1. Add stdlib tests for inventory completeness, validation/test job counts, cache provenance/count/finite-value validation, negative KID acceptance, conservative worker selection, and interrupted-output preservation. Observe failures before implementation.
2. Implement `experiments/evaluate_cifar10_main.py`, a standalone parent/worker CLI. Keep GPU imports inside runtime paths so orchestration tests run without Torch.
3. Run the orchestration tests and syntax checks. Have an independent reviewer inspect provenance, resumption, scheduling, and completion handling. Full GPU integration remains to be run on the user's AutoDL instance.
4. Publish only the new driver, its tests, and this document through a GitHub commit with the existing branch head as a lease. Download the pinned driver on AutoDL, verify its SHA-256, launch under nohup, and inspect the progress log.

## Running and checking

From the existing repository root, use the existing virtual environment and set `TORCH_HOME=/root/autodl-tmp/ESTR3114/torch-cache`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `MPLBACKEND=Agg`. Run `python experiments/evaluate_cifar10_main.py --root PATH_TO_FORMAL_RUN_DIRECTORY`; `--project` defaults to the current working directory. The standalone file may instead be downloaded into the run directory and invoked there with the repository as working directory. No `git pull` or training-source edits are needed.

`evaluation-suite.log` is the suggested top-level redirected log. Each point has its own log under `evaluation_logs/`. Progress is printed after each point and also written to `evaluation-progress.json`. The screen and selected concurrency are saved to `evaluation-parallelism.json`, and the resource snapshot to `evaluation-resources.json`. Successful completion requires all of: `ALL_EVALUATIONS_COMPLETED: 132/132`, `evaluation.exitcode` equal to `0`, and the `EVALUATION_COMPLETED` marker. Final data are `image_metrics.csv` and `image_metrics.json`. Inspect failures in the indicated point log; restarting preserves partial exports and reuses only fully verified results. Backups are retained and consume additional space.

## Local validation limits

The orchestration and provenance tests use stdlib fixtures and do not require CUDA or Torch. The current editing environment has neither Torch nor pytest; GPU export/extraction and the repository's existing 101 Torch-dependent tests cannot be rerun here. Existing training/evaluation source files are unchanged. GPU integration of the new scheduling path must be observed on AutoDL before claiming the full queue succeeds.
