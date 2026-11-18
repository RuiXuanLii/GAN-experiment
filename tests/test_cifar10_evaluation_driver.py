import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DRIVER = Path(__file__).resolve().parents[1] / 'experiments/evaluate_cifar10_main.py'
if DRIVER.exists():
    spec = importlib.util.spec_from_file_location('evaluation_driver', DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
else:
    driver = None


class EvaluationDriverTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(driver, 'Evaluation orchestration is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.freeze = {'frozen': True, 'source_hash': 'frozen-source',
                       'common': {'oracle_budget': 20000, 'dataset': 'cifar10'},
                       'reporting_seeds': [11, 22, 33],
                       'lr_by_algorithm': {m: .0001 for m in driver.METHODS},
                       'metric_protocol': {'torch_fidelity_version': '0.3.0',
                                           'feature_extractor': 'inception-v3-compat',
                                           'evaluation_seed': 123456,
                                           'kid_subset_size': 1000, 'kid_subsets': 50}}
        self.write(self.root / 'freeze.json', self.freeze)
        (self.root / 'TRAINING_COMPLETED').touch()
        (self.root / 'training.exitcode').write_text('0\n')
        for method in driver.METHODS:
            for seed in [11, 22, 33]:
                folder = self.root / f'main-cifar10-{method}-s{seed}-fixture'
                config = {'stage': 'main', 'algorithm': method, 'seed': seed,
                          'dataset': 'cifar10', 'oracle_budget': 20000,
                          'checkpoint_every': 2000, 'lr': .0001,
                          'freeze_file': str(self.root / 'freeze.json')}
                self.write(folder / 'config.json', config)
                self.write(folder / 'metadata.json', {'source_hash': 'frozen-source'})
                self.write(folder / 'result.json', {'status': 'completed', 'oracle_calls': 20000})
                (folder / 'checkpoint.pt').write_bytes(b'final-checkpoint')
                (folder / 'checkpoints').mkdir()
                for calls in range(2000, 20001, 2000):
                    (folder / 'checkpoints' / f'calls_{calls:08d}.pt').write_bytes(str(calls).encode())

    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def configure_suite(self, *, budget=20000, checkpoint_every=2000, image_size=32,
                        freeze_interval=False):
        self.freeze['common']['oracle_budget'] = budget
        if image_size != 32:
            self.freeze['common']['image_size'] = image_size
        if freeze_interval:
            self.freeze['common']['checkpoint_every'] = checkpoint_every
        self.write(self.root / 'freeze.json', self.freeze)
        for folder in self.root.glob('main-cifar10-*'):
            config = json.loads((folder / 'config.json').read_text())
            config.update(oracle_budget=budget, checkpoint_every=checkpoint_every)
            if image_size != 32:
                config['image_size'] = image_size
            self.write(folder / 'config.json', config)
            self.write(folder / 'result.json', {'status': 'completed', 'oracle_calls': budget})
            for calls in range(checkpoint_every, budget + 1, checkpoint_every):
                (folder / 'checkpoints' / f'calls_{calls:08d}.pt').write_bytes(str(calls).encode())

    def completed_job(self, image_size=32):
        if image_size != 32:
            self.configure_suite(image_size=image_size)
        job = driver.make_jobs(self.root)[0]
        job['reference_indices_sha256'] = 'a' * 64
        folder = Path(job['folder'])
        manifest = {'config': job['config'], 'dataset': 'cifar10',
                    'source_hash': job['source_hash'], 'oracle_calls': job['calls'],
                    'reference_split': job['split'], 'count': job['count'],
                    'evaluation_seed': 123456, 'metric_protocol': job['metric_protocol'],
                    'real_indices_sha256': job['reference_indices_sha256'],
                    'checkpoint_sha256': hashlib.sha256(Path(job['checkpoint']).read_bytes()).hexdigest()}
        if image_size == 64:
            manifest.update(image_size=64, real_resize='pil_bilinear')
        result = {'status': 'completed', 'manifest': manifest, 'evaluation_seconds': 10.,
                  'source_hash': job['source_hash'], 'metric_protocol': job['metric_protocol'],
                  'torch_fidelity_version': '0.3.0', 'feature_extractor': 'inception-v3-compat',
                  'metrics': {'frechet_inception_distance': 200.,
                              'kernel_inception_distance_mean': -.0001,
                              'kernel_inception_distance_std': .001}}
        self.write(folder / 'manifest.json', manifest)
        self.write(folder / 'metrics.json', result)
        for kind in ['real', 'fake']:
            (folder / kind).mkdir()
            for i in range(job['count']):
                (folder / kind / f'{i:06d}.png').touch()
        return job, manifest, result

    def test_exact_inventory_and_test_endpoints(self):
        jobs = driver.make_jobs(self.root)
        self.assertEqual(len(jobs), 132)
        self.assertEqual(sum(j['split'] == 'validation' for j in jobs), 120)
        tests = [j for j in jobs if j['split'] == 'test']
        self.assertEqual(len(tests), 12)
        self.assertTrue(all(j['calls'] == 20000 and j['count'] == 10000 for j in tests))
        self.assertEqual(len({j['folder'] for j in jobs}), 132)

    def test_64_suite_uses_frozen_budget_and_shared_checkpoint_interval(self):
        self.configure_suite(budget=6000, checkpoint_every=3000, image_size=64,
                             freeze_interval=True)
        jobs = driver.make_jobs(self.root)
        self.assertEqual(len(jobs), 36)
        self.assertEqual({j['calls'] for j in jobs if j['split'] == 'validation'}, {3000, 6000})
        self.assertEqual(sum(j['split'] == 'test' and j['calls'] == 6000 for j in jobs), 12)
        self.assertTrue(all(j['config']['image_size'] == 64 for j in jobs))

    def test_legacy_freeze_without_interval_uses_consistent_run_interval(self):
        self.configure_suite(budget=6000, checkpoint_every=3000)
        self.assertEqual(len(driver.make_jobs(self.root)), 36)
        path = next(self.root.glob('main-cifar10-*/config.json'))
        config = json.loads(path.read_text())
        config['checkpoint_every'] = 2000
        self.write(path, config)
        with self.assertRaises(ValueError):
            driver.make_jobs(self.root)

    def test_budget_must_be_positive_and_divisible_by_interval(self):
        for budget in (0, 5000):
            with self.subTest(budget=budget):
                self.configure_suite(budget=budget, checkpoint_every=2000)
                with self.assertRaises(ValueError):
                    driver.make_jobs(self.root)

    def test_64_cache_binds_resolution_and_reference_resize(self):
        job, manifest, result = self.completed_job(image_size=64)
        self.assertTrue(driver.metric_valid(job))
        for key, wrong in [('image_size', 32), ('real_resize', 'pil_nearest')]:
            with self.subTest(key=key):
                correct = manifest.pop(key)
                self.write(Path(job['folder']) / 'manifest.json', manifest)
                self.write(Path(job['folder']) / 'metrics.json', result)
                self.assertFalse(driver.metric_valid(job))
                manifest[key] = wrong
                self.write(Path(job['folder']) / 'manifest.json', manifest)
                self.write(Path(job['folder']) / 'metrics.json', result)
                self.assertFalse(driver.metric_valid(job))
                manifest[key] = correct

    def test_64_disk_preflight_accounts_for_four_png_blocks(self):
        self.configure_suite(image_size=64)
        pending = driver.make_jobs(self.root)[:1]
        free = 1024**3 + 10000 * 8192
        filesystem = SimpleNamespace(f_frsize=4096, f_files=10000000, f_favail=10000000)
        with patch.object(driver.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)), \
             patch.object(driver.os, 'statvfs', return_value=filesystem), \
             patch.object(driver.subprocess, 'check_output',
                          side_effect=AssertionError('Undersized disk passed preflight')):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient free disk'):
                driver.preflight(self.root, pending)

    def test_benchmark_memory_estimate_is_bound_to_resolution(self):
        self.assertTrue(hasattr(driver, 'worker_memory_estimate'))
        legacy = {'gpu': 'test-gpu', 'torch_peak_reserved_GiB': 2.}
        self.assertEqual(driver.worker_memory_estimate(legacy, 'test-gpu', 32), 2816.)
        self.assertEqual(driver.worker_memory_estimate(legacy, 'test-gpu', 64), 6144.)
        sized = dict(legacy, image_size=64)
        self.assertEqual(driver.worker_memory_estimate(sized, 'test-gpu', 64), 2816.)
        self.assertEqual(driver.worker_memory_estimate(sized, 'test-gpu', 32), 6144.)
        nested = dict(legacy, manifest={'config': {'image_size': 64}, 'image_size': 64})
        self.assertEqual(driver.worker_memory_estimate(nested, 'test-gpu', 64), 2816.)

    def test_missing_run_is_not_silently_ignored(self):
        path = next(self.root.glob('main-cifar10-*/result.json'))
        path.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            driver.make_jobs(self.root)

    def test_source_mismatch_is_rejected(self):
        self.write(next(self.root.glob('main-cifar10-*/metadata.json')), {'source_hash': 'other'})
        with self.assertRaises(ValueError):
            driver.make_jobs(self.root)

    def test_valid_cache_including_negative_kid_is_accepted(self):
        job, _, _ = self.completed_job()
        self.assertTrue(driver.metric_valid(job))

    def test_cache_with_wrong_checkpoint_is_rejected(self):
        job, manifest, result = self.completed_job()
        manifest['checkpoint_sha256'] = 'bad'
        result['manifest'] = manifest
        self.write(Path(job['folder']) / 'manifest.json', manifest)
        self.write(Path(job['folder']) / 'metrics.json', result)
        self.assertFalse(driver.metric_valid(job))

    def test_disagreeing_embedded_manifest_is_rejected(self):
        job, _, result = self.completed_job()
        result['manifest']['count'] = 2000
        self.write(Path(job['folder']) / 'metrics.json', result)
        self.assertFalse(driver.metric_valid(job))

    def test_missing_png_and_nonfinite_metrics_are_rejected(self):
        job, _, result = self.completed_job()
        image = Path(job['folder']) / 'fake/000000.png'
        image.unlink()
        self.assertFalse(driver.metric_valid(job))
        image.touch()
        result['metrics']['frechet_inception_distance'] = float('nan')
        self.write(Path(job['folder']) / 'metrics.json', result)
        self.assertFalse(driver.metric_valid(job))

    def test_partial_output_is_preserved_before_retry(self):
        job = driver.make_jobs(self.root)[0]
        folder = Path(job['folder'])
        self.write(folder / 'manifest.json', {'incomplete': True})
        backup = driver.preserve_partial(folder)
        self.assertFalse(folder.exists())
        self.assertTrue((backup / 'manifest.json').exists())

    def test_worker_choice_requires_clear_throughput_gain(self):
        records = [{'workers': 1, 'jobs_per_second': 1.},
                   {'workers': 2, 'jobs_per_second': 1.05},
                   {'workers': 4, 'jobs_per_second': 1.6}]
        self.assertEqual(driver.choose_workers(records), 4)
        self.assertEqual(driver.choose_workers(records[:2]), 1)

    def test_reference_sample_hash_is_verified(self):
        job, manifest, result = self.completed_job()
        manifest['real_indices_sha256'] = 'b' * 64
        result['manifest'] = manifest
        self.write(Path(job['folder']) / 'manifest.json', manifest)
        self.write(Path(job['folder']) / 'metrics.json', result)
        self.assertFalse(driver.metric_valid(job))

    def test_cached_evaluation_requires_a_valid_duration(self):
        job, _, result = self.completed_job()
        result.pop('evaluation_seconds')
        self.write(Path(job['folder']) / 'metrics.json', result)
        self.assertFalse(driver.metric_valid(job))

    def test_container_memory_headroom_bounds_host_memory(self):
        self.assertTrue(hasattr(driver, 'memory_headroom'))
        self.assertEqual(driver.memory_headroom(90, 8, 6), 2)
        self.assertEqual(driver.memory_headroom(4, 8, 1), 4)
        self.assertEqual(driver.memory_headroom(90, 8, 9), 0)

    def test_valid_point_can_be_resumed_without_importing_torch(self):
        job, _, _ = self.completed_job()
        driver.run_point(job, threads=1)

    def test_wrong_shape_cache_is_treated_as_incomplete(self):
        job, _, _ = self.completed_job()
        self.write(Path(job['folder']) / 'manifest.json', None)
        self.assertFalse(driver.metric_valid(job))

    def test_duplicate_reporting_seeds_are_rejected(self):
        self.freeze['reporting_seeds'] = [11, 11, 22]
        self.write(self.root / 'freeze.json', self.freeze)
        for folder in self.root.glob('main-cifar10-*-s33-*'):
            shutil.rmtree(folder)
        with self.assertRaises(ValueError):
            driver.make_jobs(self.root)

    def test_completed_only_preflight_needs_no_gpu_query(self):
        self.assertEqual(driver.preflight(self.root, [])['maximum_workers'], 0)


if __name__ == '__main__':
    unittest.main()
