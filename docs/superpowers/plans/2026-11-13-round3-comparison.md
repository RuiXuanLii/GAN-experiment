# Four-optimizer synthetic comparison implementation plan

> **For agentic workers:** Execute the approved study in this session; independent Windows preparation and method review are delegated. Use verified existing numerical primitives rather than inventing optimizer variants.

**Goal:** Complete equal-budget tuning and three-new-seed reporting for four canonical joint optimizers; prepare Windows CIFAR download at the user's exact path.

**Architecture:** Keep `ganopt.train` and optimizer definitions unchanged. New experiment orchestration freezes actual tuning evidence and calls the existing trainer. A separate standard-library downloader prepares the Windows data directory only when run locally.

**Tech stack:** Python, PyTorch2.5.1CPU, NumPy, Matplotlib, pytest; Windows downloader uses standard library.

**Spec:** `docs/round3-study-design.md`

## Global constraints

Use canonical joint1:1, not naive per-player5:1 wrappers. Common architecture h128/ReLU/GP0.1/batch64/z8, beta0.5/0.9. Both phases120k full fields, logs1200, checkpoints6000. Tune only9001 over1e-4/3e-4/1e-3, select末5 means, freeze before reporting11/22/33. CIFAR Windows path exact; no download to Linux scratch.

## Review focus

- Extra two calls versus one accepted step; both moments advance.
- Same-seed weights and data stream identities across algorithms.
- Freeze cannot accept partial or mismatched evidence, or reporting parameter mutation.
- Reporting seeds cannot influence tuning selection.
- Collection uses saved actual checkpoints and preserves failures/divergent samples.

## Tasks

- [x] Implement `experiments/run_round3.py`: prepare tuning configs/manifests, call existing train, verify evidence and freeze, create reporting configs. Verify with meaningful freeze/selection/budget tests.
- [x] Implement `experiments/collect_round3.py`: evaluate checkpoints, validate final samples/counts, archive exact run artifacts, plot primary comparison and separate schedule reference. Verify from actual outputs; no best-checkpoint replacement.
- [x] Implement Windows-only-default CIFAR preparation, checksum verification, exact-path config and Chinese instructions; test tiny local fixtures only.
- [x] Review math/protocol and new code independently; run full unit suite; lock core/scripts source before training.
- [x] Run all12 tuning candidates; verify counts, source and init hashes; freeze selections.
- [x] Run all12 reporting jobs from frozen selections; collect final and末5 statistics, generate and inspect plots.
- [x] Write concise Chinese results/limitations and local Windows download command; test/hash, commit, push to existing authorized repo and verify remote files.

## Execution notes

Existing session authorization covers implementation, reversible branch work, experiments and repository updates. No new algorithm or external data download is authorized merely to fill results. Windows filesystem access is unavailable in this runtime; report that accurately and deliver the runnable local command.
