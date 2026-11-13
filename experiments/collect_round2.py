"""Evaluate trusted local round-two checkpoints without consuming training RNG."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ganopt.data import BatchStream
from ganopt.metrics import synthetic_metrics
from ganopt.models import make_models


def collect(root, output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    rows=[]; endpoints=[]; all_samples=[]; active_artifacts=[]
    angles=np.arange(8)*2*np.pi/8
    centers=2*np.stack([np.cos(angles),np.sin(angles)],axis=1)
    for record in sorted(root.glob('*/result.json')):
        run=record.parent
        config=json.loads((run/'config.json').read_text())
        result=json.loads(record.read_text())
        if result['status']!='completed':
            continue
        metadata=json.loads((run/'metadata.json').read_text())
        paths={int(p.stem.split('_')[-1]):p for p in (run/'checkpoints').glob('calls_*.pt')}
        paths[result['oracle_calls']]=run/'checkpoint.pt'
        noise=torch.randn(config['eval_samples'],config['z_dim'],generator=torch.Generator().manual_seed(123456))
        series=[]
        for calls,checkpoint in sorted(paths.items()):
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert calls==state['oracle_calls']
            assert state['config']==config
            assert state['source_hash']==metadata['source_hash']
            G,_=make_models(config);G.load_state_dict(state['generator']);G.eval()
            with torch.no_grad(): samples=G(noise).numpy()
            metrics=synthetic_metrics(samples)
            distances=np.linalg.norm(samples[:,None]-centers[None],axis=2)
            assignments=distances.argmin(1); good=distances.min(1)<.15
            counts=np.bincount(assignments[good],minlength=8)
            row={'case':run.name,'method':config['algorithm'],'seed':config['seed'],
                 'lr':config['lr'],'gp_lambda':config['gp_lambda'],
                 'critic_activation':config.get('critic_activation','softplus'),
                 'hidden':config.get('hidden',32),
                 'training_schedule':config.get('training_schedule','synchronous'),
                 'critic_steps':config.get('critic_steps',1),
                 'oracle_calls':calls,
                 'g_updates':state.get('g_updates',state.get('updates')),
                 'd_updates':state.get('d_updates',state.get('updates')),**metrics,
                 **{f'good_mode_{i}':int(v) for i,v in enumerate(counts)}}
            rows.append(row);series.append(row)
        last=series[-1]
        assert {k:last[k] for k in metrics}=={k:result[k] for k in metrics}
        assert np.array_equal(samples,np.load(run/'samples.npy'))
        endpoints.append({**last,'parameter_updates':result['parameter_updates'],
            'training_seconds':result['training_seconds'],
            'initial_parameter_hash':metadata['initial_parameter_hash'],
            'source_hash':metadata['source_hash']})
        destination=output/'runs'/run.name;destination.mkdir(parents=True,exist_ok=True)
        names=['config.json','metadata.json','result.json','metrics.csv']+[p.name for p in sorted(run.glob('evaluation_*.json'))]
        if (run/'lineage.json').exists():names.append('lineage.json')
        for name in names:
            artifact=destination/name
            data=(run/name).read_bytes()
            artifact.write_bytes(data.replace(b'\r\n',b'\n') if name.endswith('.csv') else data)
            active_artifacts.append(artifact)
        # Small generated sample files make scatter plots and counts independently inspectable.
        np.save(destination/'samples.npy',samples)
        active_artifacts.append(destination/'samples.npy')
        all_samples.append((run.name,samples))
    if not endpoints:
        raise ValueError('No completed runs')
    for name,data in [('checkpoints.csv',rows),('summary.csv',endpoints)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    cfg=json.loads(next(root.glob('*/config.json')).read_text())
    reference_stream=BatchStream(dict(cfg,device='cpu'))
    reference=np.concatenate([reference_stream.next()[0].numpy() for _ in range(32)])[:2048]
    reference_metrics=synthetic_metrics(reference)
    assert reference_metrics['covered_modes']==8 and reference_metrics['quality_fraction']>.97
    (output/'target_control.json').write_text(json.dumps(reference_metrics,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for case in [r['case'] for r in endpoints]:
        series=[r for r in rows if r['case']==case]
        for ax,key in zip(axes,['covered_modes','quality_fraction','mean_nearest_distance']):
            ax.plot([r['oracle_calls'] for r in series],[r[key] for r in series],label=case,marker='.',markersize=3)
    for ax,title in zip(axes,['Modes covered / 8','Fraction within radius 0.15','Mean distance to nearest center']):
        ax.set(xlabel='Joint gradient evaluations',ylabel=title);ax.grid(alpha=.2)
    axes[0].set_ylim(-.2,8.2);axes[0].legend(fontsize=6)
    fig.savefig(output/'progress.png',dpi=150);plt.close(fig)
    n=len(all_samples)+1;cols=min(4,n);fig,axes=plt.subplots((n+cols-1)//cols,cols,figsize=(cols*4,((n+cols-1)//cols)*4),squeeze=False,layout='constrained')
    for ax,(title,samples) in zip(axes.flat,[('Real target',reference)]+all_samples):
        ax.scatter(reference[:,0],reference[:,1],s=2,c='gray',alpha=.2,label='real target')
        ax.scatter(samples[:,0],samples[:,1],s=2,alpha=.35,label='generated' if title!='Real target' else 'real target')
        ax.scatter(centers[:,0],centers[:,1],c='red',marker='x',s=25)
        # Include all samples, including those that diverge outside [-3,3].
        joined=np.vstack([samples,reference]);low=joined.min(0)-.3;high=joined.max(0)+.3
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),title=title,aspect='equal')
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:ax.set_visible(False)
    fig.savefig(output/'samples.png',dpi=150);plt.close(fig)
    manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(active_artifacts)}
    manifest['collector_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'active_runs.json').write_text(json.dumps([r['case'] for r in endpoints],indent=2)+'\n')
    print(json.dumps(endpoints,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();collect(args.runs,args.output)
