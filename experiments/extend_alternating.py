"""Continue a completed diagnostic under a larger budget in a NEW lineage-linked run.

Only oracle_budget changes. Parent files remain untouched. A known driver update
that trims crash-stale CSV rows is recorded explicitly; uninterrupted numerical
updates are unchanged (verified against fresh uninterrupted training).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from ganopt.runtime import atomic_save, source_hash, stable_hash
from experiments.alternating_reference import train_alternating


def extend(parent, output, budget):
    parent=Path(parent);output=Path(output)
    checkpoint=parent/'checkpoint.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    c=state['config']
    if not isinstance(budget,int) or budget<=c['oracle_budget'] or budget%(c['critic_steps']+1):
        raise ValueError('Extended budget must increase and align to a full cycle')
    result=json.loads((parent/'result.json').read_text())
    if result['status']!='completed' or state['oracle_calls']!=c['oracle_budget']:
        raise ValueError('Parent must be a completed full-budget run')
    if state['source_hash']!=source_hash():raise ValueError('Core source changed; extension refused')
    root=Path(__file__).resolve().parents[1]
    current_script=root/'experiments/alternating_reference.py'
    driver_hash=hashlib.sha256(current_script.read_bytes()).hexdigest()
    snapshot=root/'evidence/round2/alternating_reference_at_run.py'
    allowed={driver_hash}
    if snapshot.exists():allowed.add(hashlib.sha256(snapshot.read_bytes()).hexdigest())
    if state['script_hash'] not in allowed:raise ValueError('Unrecognized parent diagnostic driver')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Extension output exists')
    previous_driver=state['script_hash']
    updated=dict(c,oracle_budget=budget)
    lineage={'parent_run':str(parent),'parent_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'parent_oracle_calls':state['oracle_calls'],'parent_g_updates':state['g_updates'],'parent_d_updates':state['d_updates'],
        'parent_config_hash':state['config_hash'],'config_changes':{'oracle_budget':{'from':c['oracle_budget'],'to':budget}},
        'parent_driver_sha256':previous_driver,'continuation_driver_sha256':driver_hash,
        'driver_migration':'Known crash-resume CSV consistency hardening only; numerical transitions unchanged.'}
    state.update(config=updated,config_hash=stable_hash(updated),script_hash=driver_hash)
    output.mkdir(parents=True,exist_ok=True)
    atomic_save(state,output/'checkpoint.pt')
    if (parent/'metrics.csv').exists():shutil.copy2(parent/'metrics.csv',output/'metrics.csv')
    (output/'lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
    return train_alternating(updated,output,resume=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget',type=int,required=True)
    args=p.parse_args();extend(args.parent,args.output,args.budget)
