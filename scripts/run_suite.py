"""Run a shared configuration with all four methods, without hidden overrides."""
import argparse
import copy
import csv
import json
from pathlib import Path
from ganopt.optim import METHODS
from ganopt.runtime import stable_hash
from ganopt.train import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--output',default='experiments')
    p.add_argument('--seeds',type=int,nargs='+'); p.add_argument('--resume',action='store_true')
    p.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    args=p.parse_args(); base=json.loads(Path(args.config).read_text()); root=Path(args.output)
    seeds=args.seeds or [base['seed']]
    if base.get('stage')=='main':
        frozen=json.loads(Path(base['freeze_file']).read_text())
        seeds=args.seeds or frozen['reporting_seeds']
        if set(seeds)!=set(frozen['reporting_seeds']): raise ValueError('Main suite must include all frozen reporting seeds')
    rows=[]
    for seed in seeds:
        for method in args.methods:
            config=copy.deepcopy(base); config.update(seed=seed,algorithm=method)
            if config.get('stage')=='main' and config.get('freeze_file'):
                freeze=json.loads(Path(config['freeze_file']).read_text())
                config['lr']=freeze['lr_by_algorithm'][method]
            run_id=f"{config['stage']}-{config['dataset']}-{method}-s{seed}-{stable_hash(config)[:8]}"
            path=root/run_id
            try:
                train(config,path,resume=args.resume)
                result=json.loads((path/'result.json').read_text())
                rows.append({'experiment_id':run_id,'algorithm':method,'seed':seed,
                             'status':result['status'],'oracle_calls':result.get('oracle_calls'),
                             'training_seconds':result.get('training_seconds'),
                             'covered_modes':result.get('covered_modes'),
                             'quality_fraction':result.get('quality_fraction'),'error':None})
                print(run_id,result['status'])
            except Exception as exc:
                rows.append({'experiment_id':run_id,'algorithm':method,'seed':seed,'status':'failed',
                             'oracle_calls':None,'training_seconds':None,'covered_modes':None,
                             'quality_fraction':None,'error':repr(exc)})
                print(run_id,'failed:',repr(exc))
    root.mkdir(parents=True,exist_ok=True)
    (root/'suite.json').write_text(json.dumps(rows,indent=2))
    with (root/'suite.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if any(r['status']=='failed' for r in rows): raise SystemExit(1)


if __name__=='__main__': main()
