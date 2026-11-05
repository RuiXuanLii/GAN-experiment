"""Equal per-method three-rate tuning runs; image metrics are separate explicit jobs."""
import argparse
import copy
import json
from pathlib import Path
from ganopt.optim import METHODS
from ganopt.runtime import stable_hash
from ganopt.train import train


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/cifar10_tuning.json')
    p.add_argument('--output',default='experiments/tuning');p.add_argument('--resume',action='store_true')
    a=p.parse_args();base=json.loads(Path(a.config).read_text())
    if base['stage']!='tuning' or base['seed']!=9001: raise ValueError('Use reserved tuning seed 9001')
    failures=[]
    for method in METHODS:
        for lr in [5e-5,1e-4,2e-4]:
            c=copy.deepcopy(base);c.update(algorithm=method,lr=lr)
            path=Path(a.output)/f"tuning-{method}-lr{lr:g}-{stable_hash(c)[:8]}"
            try: train(c,path,resume=a.resume);print(path)
            except Exception as e: failures.append({'run':str(path),'error':repr(e)});print(path,'failed',repr(e))
    Path(a.output).mkdir(parents=True,exist_ok=True)
    (Path(a.output)/'failures.json').write_text(json.dumps(failures,indent=2))
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
