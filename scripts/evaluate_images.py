"""FID/KID from exported images and an actual pretrained Inception extractor.

No random-weight or pixel-feature fallback. Missing packages/weights fail visibly.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import math
import torch
from ganopt.runtime import source_hash,validate_image_export,METRIC_PROTOCOL


def evaluate(folder,cuda=False):
    folder=Path(folder); manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['source_hash']!=source_hash(): raise ValueError('Export source mismatch')
    validate_image_export(manifest['config'],manifest['reference_split'],manifest['count'],manifest['evaluation_seed'],manifest['oracle_calls'])
    if manifest.get('metric_protocol')!=METRIC_PROTOCOL: raise ValueError('Image metric protocol mismatch')
    for sub in ['real','fake']:
        if len(list((folder/sub).glob('*.png')))!=manifest['count']:
            raise ValueError('Image count disagrees with manifest')
    if cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    import torch_fidelity
    if importlib.metadata.version('torch-fidelity')!=METRIC_PROTOCOL['torch_fidelity_version']:
        raise ValueError('Install the frozen torch-fidelity version from requirements-image-metrics.txt')
    started=time.perf_counter()
    metrics=torch_fidelity.calculate_metrics(input1=str(folder/'fake'),input2=str(folder/'real'),
        cuda=cuda,fid=True,kid=True,isc=False,kid_subset_size=1000,kid_subsets=50,
        rng_seed=manifest['evaluation_seed'],verbose=True)
    if any(not math.isfinite(float(v)) for v in metrics.values()): raise FloatingPointError('Nonfinite image metric')
    result={'status':'completed','metrics':metrics,'evaluation_seconds':time.perf_counter()-started,
            'torch_fidelity_version':importlib.metadata.version('torch-fidelity'),
            'feature_extractor':METRIC_PROTOCOL['feature_extractor'],
            'metric_protocol':METRIC_PROTOCOL,
            'source_hash':source_hash(),'manifest':manifest,
            'mnist_domain_warning':manifest['dataset']=='mnist'}
    (folder/'metrics.json').write_text(json.dumps(result,indent=2)); return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True)
    p.add_argument('--cuda',action='store_true');a=p.parse_args();print(json.dumps(evaluate(a.folder,a.cuda),indent=2))


if __name__=='__main__': main()
