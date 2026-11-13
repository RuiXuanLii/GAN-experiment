"""Freeze a failed run's G and measure whether extra D training repairs critic ranking.

This diagnostic does not train G or compare optimizers. Every field evaluation
still computes both partial gradients; only the D block is applied.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from ganopt.data import BatchStream
from ganopt.game import field
from ganopt.models import make_models
from ganopt.optim import Adam
from ganopt.runtime import seed_everything, source_hash


def parameter_hash(model):
    return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()


def probe(checkpoint, output, steps):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    c=dict(state['config'],device='cpu')
    seed_everything(c['seed'],1)
    G,D=make_models(c);G.load_state_dict(state['generator']);D.load_state_dict(state['critic'])
    ghash=parameter_hash(G);gcount=len(list(G.parameters()))
    opt=Adam(D.parameters(),lr=c['lr'],betas=tuple(c['betas']),eps=c['eps'])
    saved=dict(state['optimizer']);saved['states']=saved['states'][gcount:]
    opt.load_state_dict(saved)
    stream=BatchStream(c);stream.load_state_dict(state['data'])
    # Fixed independent samples, never included in the additional D updates.
    heldout=BatchStream(dict(c,seed=c['seed']+100000))
    real=[];fake=[]
    with torch.no_grad():
        for _ in range(32):
            x,z,_=heldout.next();real.append(x);fake.append(G(z))
    real=torch.cat(real);fake=torch.cat(fake)
    def gap():
        with torch.no_grad():return float((D(real).mean()-D(fake).mean()).item())
    rows=[{'extra_d_updates':0,'heldout_critic_gap':gap()}]
    for step in range(1,steps+1):
        def closure():
            grads,stats=field(G,D,*stream.next(),gp_lambda=c['gp_lambda'])
            return grads[gcount:],stats
        opt.step(closure)
        if step%250==0 or step==steps:
            rows.append({'extra_d_updates':step,'heldout_critic_gap':gap()})
    assert parameter_hash(G)==ghash,'Probe unexpectedly changed G'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'progress.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'source_checkpoint':str(checkpoint),'source_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_training_hash':state['source_hash'],'probe_source_hash':source_hash(),
        'probe_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config':c,'generator_unchanged':True,'generator_hash':ghash,
        'additional_joint_field_calls':steps,'additional_g_updates':0,'additional_d_updates':steps,
        'heldout_sample_count':len(real),'heldout_stream_seed':c['seed']+100000,
        'heldout_gap_before':rows[0]['heldout_critic_gap'],'heldout_gap_after':rows[-1]['heldout_critic_gap']}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    args=p.parse_args()
    if args.steps<1:p.error('--steps must be positive')
    probe(args.checkpoint,args.output,args.steps)
