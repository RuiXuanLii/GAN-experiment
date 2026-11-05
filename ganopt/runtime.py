"""Configuration validation, provenance, RNG and atomic checkpoint utilities."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import numpy as np
import torch
from .optim import METHODS


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_hash():
    root=Path(__file__).resolve().parents[1]; h=hashlib.sha256()
    for path in sorted(list((root/'ganopt').rglob('*.py'))+list((root/'scripts').rglob('*.py'))):
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def environment(device='cpu'):
    versions={}
    for name in ['torch','torchvision','numpy','matplotlib','pytest','torch-fidelity']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':versions,
            'cuda_available':torch.cuda.is_available(),'torch_cuda_build':torch.version.cuda,
            'device_name':torch.cuda.get_device_name(torch.device(device)) if torch.cuda.is_available() and str(device).startswith('cuda') else None,
            'cuda_total_memory_mb':torch.cuda.get_device_properties(torch.device(device)).total_memory/1024**2 if torch.cuda.is_available() and str(device).startswith('cuda') else None}


def process_peak_rss_mb():
    """Process-wide peak on Unix; current RSS fallback on Windows (runner peaks it)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2)
    except ImportError:
        import psutil
        return psutil.Process().memory_info().rss/1024**2


def seed_everything(seed,threads=1):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(state,path):
    path=Path(path); temporary=path.with_suffix('.tmp')
    torch.save(state,temporary); os.replace(temporary,path)


def validate_frozen_config(config):
    """Pure evidence gate, usable by CPU exporters without requiring a GPU."""
    if not config.get('freeze_file'): raise ValueError('Main experiment requires freeze_file after pilot')
    freeze=json.loads(Path(config['freeze_file']).read_text())
    if not freeze.get('frozen') or freeze['source_hash']!=source_hash():
        raise ValueError('Protocol freeze/source mismatch')
    for key,value in freeze['common'].items():
        if config.get(key)!=value: raise ValueError(f'Frozen configuration mismatch: {key}')
    if config['lr']!=freeze['lr_by_algorithm'][config['algorithm']]:
        raise ValueError('Frozen learning rate mismatch')
    if config['seed'] not in freeze['reporting_seeds']: raise ValueError('Frozen reporting seed mismatch')
    return freeze


METRIC_PROTOCOL={'torch_fidelity_version':'0.3.0','feature_extractor':'inception-v3-compat',
                 'evaluation_seed':123456,'kid_subset_size':1000,'kid_subsets':50}


def validate_image_export(config,split,count,seed,calls):
    """Prevent test-data access before the fixed, frozen primary endpoint."""
    if seed!=METRIC_PROTOCOL['evaluation_seed']: raise ValueError('Fixed evaluation seed required')
    stage=config['stage']
    if split=='test':
        if stage!='main': raise ValueError('Test set reserved for frozen main experiments')
        freeze=validate_frozen_config(config)
        if calls!=freeze['common']['oracle_budget']: raise ValueError('Test export requires final endpoint')
        if count!=10000: raise ValueError('Final test evaluation requires 10,000 images')
    elif stage in ['pilot','tuning']:
        if count!=2000 or calls!=config['oracle_budget']:
            raise ValueError('Pilot/tuning requires 2,000-image validation at final endpoint')
    elif stage=='main':
        freeze=validate_frozen_config(config)
        if count!=5000 or calls<=0 or calls%config['checkpoint_every']:
            raise ValueError('Main curve requires 5,000 validation images at completed checkpoints')
    else: raise ValueError('Image metrics require pilot, tuning or frozen main stage')
    if stage=='main' and freeze['metric_protocol']!=METRIC_PROTOCOL:
        raise ValueError('Frozen metric protocol mismatch')


def validate(config):
    if config['algorithm'] not in METHODS: raise ValueError('Unknown algorithm')
    if config['dataset'] not in ['synthetic','cifar10','mnist']: raise ValueError('Unknown dataset')
    if config.get('extra_batch','independent') not in ['independent','same']:
        raise ValueError('Unknown extra_batch setting')
    cost=METHODS[config['algorithm']].oracle_cost
    alignment=cost*(config['lookahead_k'] if config['algorithm']=='lookahead_minmax' else 1)
    for key in ['batch_size','z_dim','threads','oracle_budget','log_every','checkpoint_every','eval_samples']:
        if not isinstance(config[key],int) or config[key]<1: raise ValueError(f'{key} must be positive integer')
    for key in ['oracle_budget','log_every','checkpoint_every']:
        if config[key]%alignment: raise ValueError(f'{key} must align to complete method cycles')
    if config['gp_lambda']<0: raise ValueError('gp_lambda must be nonnegative')
    if config.get('stage')=='main':
        validate_frozen_config(config)
    device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no accessible CUDA device')
    if config['dataset']!='synthetic' and config.get('stage') in ['pilot','tuning','main'] and device.type!='cuda':
        raise ValueError('Image pilot/tuning/main require CUDA; use image_smoke for CPU shape test')
    return alignment
