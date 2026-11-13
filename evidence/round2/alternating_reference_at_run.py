"""Adam-only alternating schedule diagnostic; not the common optimizer benchmark.

Each call evaluates the full existing game field and applies only one player's
gradient. Discarded gradients remain counted. Separate Adam clocks advance only
when that player's gradient is applied. D steps precede each G step.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from ganopt.data import BatchStream
from ganopt.game import field
from ganopt.metrics import synthetic_metrics
from ganopt.models import make_models
from ganopt.optim import Adam
from ganopt.runtime import atomic_save, environment, seed_everything, source_hash, stable_hash


def train_alternating(config, output, stop_after=None, resume=False):
    c=config;output=Path(output)
    if c['algorithm']!='adam' or c['dataset']!='synthetic' or c['device']!='cpu':
        raise ValueError('Reference supports CPU synthetic Adam only')
    if c.get('training_schedule')!='alternating':raise ValueError('Expected alternating schedule')
    k=c['critic_steps']
    if not isinstance(k,int) or k<1:raise ValueError('critic_steps must be a positive integer')
    cycle=k+1;target=c['oracle_budget'] if stop_after is None else stop_after
    for value in [c['oracle_budget'],c['checkpoint_every'],c['log_every'],target]:
        if not isinstance(value,int) or value<1 or value%cycle:raise ValueError('Budgets and logging must align to complete alternating cycles')
    if target>c['oracle_budget']:raise ValueError('Target exceeds budget')
    if output.exists() and any(output.iterdir()) and not resume:raise FileExistsError('Run exists')
    output.mkdir(parents=True,exist_ok=True)
    seed_everything(c['seed'],c['threads']);G,D=make_models(c)
    gp=list(G.parameters());dp=list(D.parameters());ng=len(gp)
    kwargs={'lr':c['lr'],'betas':tuple(c['betas']),'eps':c['eps']}
    go=Adam(gp,**kwargs);do=Adam(dp,**kwargs);stream=BatchStream(c)
    initial_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in gp+dp)).hexdigest()
    shash=source_hash();script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();chash=stable_hash(c)
    calls=gu=du=0;seconds=0.
    if resume:
        s=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False)
        if (s['config_hash'],s['source_hash'],s['script_hash'])!=(chash,shash,script_hash):raise ValueError('Resume provenance mismatch')
        G.load_state_dict(s['generator']);D.load_state_dict(s['critic']);go.load_state_dict(s['g_optimizer']);do.load_state_dict(s['d_optimizer']);stream.load_state_dict(s['data'])
        calls,gu,du,seconds=s['oracle_calls'],s['g_updates'],s['d_updates'],s['training_seconds']
    if target<calls:raise ValueError('Cannot resume to earlier budget')
    metadata={'experiment_id':output.name,'source_hash':shash,'experiment_script_sha256':script_hash,
        'config_hash':chash,'initial_parameter_hash':initial_hash,'split_hash':stream.split_hash,
        'config':c,'environment':environment('cpu'),
        'counting':'Full joint field evaluated each call; only active block applied; both partial gradients counted.',
        'evaluation_status':'Adam-only schedule diagnostic, not four-optimizer benchmark'}
    (output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    noise=torch.randn(c['eval_samples'],c['z_dim'],generator=torch.Generator().manual_seed(123456))
    def evaluate():
        with torch.no_grad():samples=G(noise).numpy()
        result={'status':'completed' if calls==c['oracle_budget'] else 'paused',
            'oracle_calls':calls,'player_gradient_calls':2*calls,'parameter_updates':gu+du,
            'g_updates':gu,'d_updates':du,'training_seconds':seconds,
            'diagnostic_only':True,'image_metrics_executed':False,**synthetic_metrics(samples)}
        np.save(output/'samples.npy',samples)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        (output/f'evaluation_{calls}.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    def save():
        atomic_save({'generator':G.state_dict(),'critic':D.state_dict(),
            'g_optimizer':go.state_dict(),'d_optimizer':do.state_dict(),
            'data':stream.state_dict(),'config':c,'config_hash':chash,'source_hash':shash,'script_hash':script_hash,
            'oracle_calls':calls,'g_updates':gu,'d_updates':du,'training_seconds':seconds},output/'checkpoint.pt')
        if calls and calls%c['checkpoint_every']==0:
            (output/'checkpoints').mkdir(exist_ok=True)
            shutil.copy2(output/'checkpoint.pt',output/'checkpoints'/f'calls_{calls:08d}.pt')
    save()
    while calls<target:
        started=time.perf_counter()
        for _ in range(k):
            def d_closure():
                nonlocal calls
                grads,metrics=field(G,D,*stream.next(),gp_lambda=c['gp_lambda']);calls+=1
                return grads[ng:],metrics
            metrics=do.step(d_closure);du+=1
        def g_closure():
            nonlocal calls
            grads,metrics=field(G,D,*stream.next(),gp_lambda=c['gp_lambda']);calls+=1
            return grads[:ng],metrics
        metrics=go.step(g_closure);gu+=1;seconds+=time.perf_counter()-started
        if calls%c['log_every']==0 or calls==target:
            row={'oracle_calls':calls,'player_gradient_calls':2*calls,'g_updates':gu,'d_updates':du,
                'training_seconds':seconds,'data_examples_drawn':stream.batches*c['batch_size'],**metrics}
            exists=(output/'metrics.csv').exists()
            with (output/'metrics.csv').open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if not exists:writer.writeheader()
                writer.writerow(row)
        if calls%c['checkpoint_every']==0 or calls==target:
            save();result=evaluate();print(output.name,json.dumps(result),flush=True)
    save();evaluate()
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();train_alternating(json.loads(args.config.read_text()),args.output,resume=args.resume)
