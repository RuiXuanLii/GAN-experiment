"""One shared, synchronous WGAN-GP loop, with exact CPU checkpoint resumption."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import shutil
import torch
from .data import BatchStream
from .game import field
from .models import make_models
from .metrics import synthetic_metrics
from .optim import METHODS
from .runtime import (atomic_save,environment,process_peak_rss_mb,rng_state,restore_rng,seed_everything,
                      source_hash,stable_hash,validate)


def train(config: dict, output: Path|str, stop_after: int|None=None, resume: bool=False) -> Path:
    """Run to oracle_budget (or a complete-cycle stop_after) and return run folder.

    Only trusted, locally produced checkpoints should be loaded: torch.load with
    weights_only=False restores Python/NumPy RNG state as well as tensors.
    """
    align=validate(config); output=Path(output)
    target=config['oracle_budget'] if stop_after is None else stop_after
    if target<1 or target>config['oracle_budget'] or target%align:
        raise ValueError('stop_after must be within budget and align to method cycle')
    chash=stable_hash(config); shash=source_hash()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('Run output exists; use --resume or a new directory')
    if resume and not (output/'checkpoint.pt').exists(): raise FileNotFoundError('No checkpoint to resume')
    output.mkdir(parents=True,exist_ok=True)
    seed_everything(config['seed'],config['threads']); device=torch.device(config['device'])
    G,D=make_models(config); G.to(device); D.to(device)
    params=list(G.parameters())+list(D.parameters())
    opt_kwargs={'lr':config['lr'],'betas':tuple(config['betas']),'eps':config['eps']}
    if config['algorithm']=='lookahead_minmax':
        opt_kwargs.update(k=config['lookahead_k'],alpha=config['lookahead_alpha'])
    opt=METHODS[config['algorithm']](params,**opt_kwargs)
    stream=BatchStream(config)
    init_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in params)).hexdigest()
    experiment_id=f"{config['stage']}-{config['dataset']}-{config['algorithm']}-s{config['seed']}-{chash[:8]}"
    calls=updates=0; training_seconds=wall_offset=0.
    if resume:
        state=torch.load(output/'checkpoint.pt',map_location=device,weights_only=False)
        if state['config_hash']!=chash: raise ValueError('Resume configuration mismatch')
        if state['source_hash']!=shash: raise ValueError('Resume source mismatch')
        G.load_state_dict(state['generator']); D.load_state_dict(state['critic'])
        opt.load_state_dict(state['optimizer']); stream.load_state_dict(state['data'])
        restore_rng(state['rng']); calls=state['oracle_calls']; updates=state['updates']
        training_seconds=state['training_seconds']; wall_offset=state['wall_seconds']
        # Logs may have reached disk before the last atomic checkpoint. Trim to it.
        if (output/'metrics.jsonl').exists():
            lines=(output/'metrics.jsonl').read_text().splitlines(); rows=[]
            for index,line in enumerate(lines):
                try: rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if index!=len(lines)-1: raise ValueError('Corrupt interior experiment log')
                    # Last append may have been interrupted; checkpoint remains authoritative.
            rows=[r for r in rows if r['oracle_calls']<=calls]
            (output/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            if rows:
                with (output/'metrics.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            elif (output/'metrics.csv').exists():
                (output/'metrics.csv').unlink()
    if target<calls: raise ValueError('Cannot resume to an earlier oracle budget')
    metadata={'experiment_id':experiment_id,'config_hash':chash,'source_hash':shash,
              'initial_parameter_hash':init_hash,'split_hash':stream.split_hash,
              'parameter_count':sum(p.numel() for p in params),'environment':environment(device),
              'evaluation_status':'synthetic diagnostics only' if config['dataset']=='synthetic' else 'image FID/KID not run in training',
              'config':config}
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    started=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    def synchronize():
        if device.type=='cuda': torch.cuda.synchronize(device)
    def save():
        atomic_save({'generator':G.state_dict(),'critic':D.state_dict(),'optimizer':opt.state_dict(),
            'config':config,'config_hash':chash,'source_hash':shash,'rng':rng_state(),
            'data':stream.state_dict(),'oracle_calls':calls,'updates':updates,
            'training_seconds':training_seconds,'wall_seconds':wall_offset+time.perf_counter()-started},
            output/'checkpoint.pt')
        if config.get('keep_checkpoints',False) and calls>0 and calls%config['checkpoint_every']==0:
            (output/'checkpoints').mkdir(exist_ok=True)
            shutil.copy2(output/'checkpoint.pt',output/'checkpoints'/f'calls_{calls:08d}.pt')
    def write_row(metrics,delta):
        row={'experiment_id':experiment_id,'oracle_calls':calls,'player_gradient_calls':2*calls,
             'parameter_updates':updates,'sample_evaluations':calls*config['batch_size'],
             'data_examples_drawn':stream.batches*config['batch_size'],
             'training_seconds':training_seconds,'wall_seconds':wall_offset+time.perf_counter()-started,
             'cpu_peak_rss_mb':process_peak_rss_mb(),
             'cuda_peak_allocated_mb':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.,
             'cuda_peak_reserved_mb':torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else 0.,
             'parameter_delta_norm':delta,**metrics}
        with (output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        exists=(output/'metrics.csv').exists()
        with (output/'metrics.csv').open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if not exists: writer.writeheader()
            writer.writerow(row)
    save()
    try:
        while calls<target:
            before=[p.detach().clone() for p in params]
            before_stream=stream.state_dict(); before_calls=calls
            cached_batch=None
            def closure():
                nonlocal calls,cached_batch
                if config['algorithm']=='extraadam' and config['extra_batch']=='same':
                    if cached_batch is None: cached_batch=stream.next()
                    batch=cached_batch
                else: batch=stream.next()
                calls+=1
                return field(G,D,*batch,gp_lambda=config['gp_lambda'])
            synchronize(); t=time.perf_counter()
            try: metrics=opt.step(closure)
            except Exception:
                stream.load_state_dict(before_stream); calls=before_calls; save(); raise
            synchronize(); training_seconds+=time.perf_counter()-t; updates+=1
            delta=torch.sqrt(sum((p.detach()-v).square().sum() for p,v in zip(params,before))).item()
            if calls%config['log_every']==0 or calls==target:
                write_row(metrics,delta)
            if calls%config['checkpoint_every']==0 or calls==target: save()
        # Independent evaluation RNG does not consume the training/data streams.
        eval_rng=torch.Generator(device='cpu').manual_seed(123456)
        noise=torch.randn(config['eval_samples'],config['z_dim'],generator=eval_rng).to(device)
        G.eval()
        with torch.no_grad(): samples=G(noise).cpu()
        G.train()
        import numpy as np
        np.save(output/'samples.npy',samples.numpy())
        result={'status':'completed' if calls==config['oracle_budget'] else 'paused',
                'oracle_calls':calls,'parameter_updates':updates,'training_seconds':training_seconds,
                'wall_seconds':wall_offset+time.perf_counter()-started,
                'diagnostic_only':config['stage'] in ['smoke','image_smoke','pilot'],
                'image_metrics_executed':False}
        if config['dataset']=='synthetic': result.update(synthetic_metrics(samples.numpy()))
        (output/'result.json').write_text(json.dumps(result,indent=2)); save()
    except Exception as exc:
        (output/'result.json').write_text(json.dumps({'status':'failed','last_valid_oracle_calls':calls,
            'error':repr(exc),'image_metrics_executed':False},indent=2))
        raise
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--output')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--stop-after',type=int)
    args=parser.parse_args(); config=json.loads(Path(args.config).read_text())
    folder=args.output or str(Path('experiments')/f"{config['stage']}-{config['dataset']}-{config['algorithm']}-s{config['seed']}-{stable_hash(config)[:8]}")
    print(train(config,folder,stop_after=args.stop_after,resume=args.resume))


if __name__=='__main__': main()
