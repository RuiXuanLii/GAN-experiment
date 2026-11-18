import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import datasets

from ganopt.models import make_models
from ganopt.optim import METHODS
from ganopt.runtime import source_hash, validate_frozen_config
from ganopt.train import train
from scripts.evaluate_images import evaluate
from scripts.export_images import export


def test_wide_gan_has_the_expected_size_and_parameter_count():
    config = dict(dataset='cifar10', z_dim=64, base_channels=32)
    torch.set_num_threads(1)
    old_g, old_d = make_models(config)
    new_g, new_d = make_models(dict(config, image_size=64, base_channels=64))
    count = lambda *models: sum(p.numel() for m in models for p in m.parameters())
    with torch.no_grad():
        assert old_g(torch.zeros(2, 64)).shape == (2, 3, 32, 32)
        new_images = new_g(torch.zeros(2, 64))
        assert new_images.shape == (2, 3, 64, 64)
        assert new_d(new_images).shape == (2, 1)
    assert count(old_g, old_d) == 466436
    assert count(new_g, new_d) == 1402820


@pytest.fixture
def fixture_cifar(monkeypatch):
    """A nonconstant 32x32 RGB fixture, with enough indices to exercise held-out data."""
    pixels = np.arange(32 * 32 * 3, dtype=np.int32).reshape(32, 32, 3)
    original = Image.fromarray((pixels % 256).astype(np.uint8))

    class FixtureCifar:
        def __init__(self, root, train=True, download=False):
            assert train and not download

        def __len__(self):
            return 5010

        def __getitem__(self, index):
            return original.copy(), 0

    monkeypatch.setattr(datasets, 'CIFAR10', FixtureCifar)
    return original


@pytest.mark.parametrize('method', list(METHODS))
def test_wide_training_and_checkpoint_smoke(tmp_path, fixture_cifar, method):
    config = json.loads(Path('configs/cifar10_64/pilot.json').read_text())
    config.update(stage='image_smoke', device='cpu', algorithm=method, batch_size=2,
                  oracle_budget=10, log_every=10, checkpoint_every=10, eval_samples=2)
    run = train(config, tmp_path / 'run')
    samples = np.load(run / 'samples.npy', allow_pickle=False)
    result = json.loads((run / 'result.json').read_text())
    assert samples.shape == (2, 3, 64, 64) and np.isfinite(samples).all()
    assert result['status'] == 'completed' and result['oracle_calls'] == 10
    assert result['parameter_updates'] == (5 if method == 'extraadam' else 10)


def test_export_uses_matching_bilinear_references_and_validates_size(tmp_path, fixture_cifar):
    config = json.loads(Path('configs/cifar10_64/pilot.json').read_text())
    config.update(base_channels=4)
    generator, _ = make_models(config)
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(dict(config=config, source_hash=source_hash(),
                    oracle_calls=config['oracle_budget'],
                    generator=generator.state_dict()), checkpoint)
    out = export(checkpoint, tmp_path / 'images', tmp_path / 'data', count=2000)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['image_size'] == 64 and manifest['real_resize'] == 'pil_bilinear'
    with Image.open(out / 'real/000000.png') as real:
        assert np.array_equal(np.asarray(real), np.asarray(fixture_cifar.resize((64, 64), Image.Resampling.BILINEAR)))
    with Image.open(out / 'fake/000000.png') as fake:
        assert fake.size == (64, 64)
    fixture_cifar.save(out / 'fake/000000.png')
    with pytest.raises(ValueError, match='64×64 PNG'):
        evaluate(out)


def test_legacy_freeze_blocks_switching_to_64(tmp_path):
    freeze_file = tmp_path / 'freeze.json'
    freeze_file.write_text(json.dumps(dict(frozen=True, source_hash=source_hash(),
                                         common={}, lr_by_algorithm={'adam': .0001},
                                         reporting_seeds=[11])))
    config = dict(freeze_file=str(freeze_file), algorithm='adam', lr=.0001, seed=11)
    validate_frozen_config(config)
    with pytest.raises(ValueError, match='image_size'):
        validate_frozen_config(dict(config, image_size=64))
