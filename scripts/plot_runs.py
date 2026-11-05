"""Reusable diagnostic curves, generated sample scatter/grid, actual run tables."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',required=True)
    p.add_argument('--output',default='figures'); args=p.parse_args()
    root=Path(args.runs); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    paths=sorted(x.parent for x in root.rglob('result.json') if (x.parent/'metrics.jsonl').exists())
    if not paths: raise SystemExit('No executed run records found')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for run in paths:
        c=json.loads((run/'config.json').read_text()); rows=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
        x=[r['oracle_calls'] for r in rows]; label=f"{c['algorithm']} s{c['seed']}"
        for ax,key in zip(axes.flat,['g_loss','d_loss','gp','g_grad_norm']):
            ax.plot(x,[r[key] for r in rows],label=label,alpha=.8)
            ax.set(xlabel='Joint gradient-oracle evaluations',ylabel=key); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Executed WGAN-GP smoke diagnostics (not image-quality ranking)')
    fig.savefig(out/'training_diagnostics.png',dpi=160); plt.close(fig)
    selected=paths[:4]; fig,axes=plt.subplots(1,len(selected),figsize=(4*len(selected),4),squeeze=False,constrained_layout=True)
    for ax,run in zip(axes.flat,selected):
        c=json.loads((run/'config.json').read_text()); samples=np.load(run/'samples.npy')
        if samples.ndim==2:
            a=np.arange(8)*2*np.pi/8; centers=2*np.stack([np.cos(a),np.sin(a)],axis=1)
            ax.scatter(samples[:,0],samples[:,1],s=3,alpha=.3,color='#3660b8')
            ax.scatter(centers[:,0],centers[:,1],s=40,c='#d34444',marker='x',label='True centers')
            ax.set(xlim=(-3,3),ylim=(-3,3),aspect='equal')
        else:
            from torchvision.utils import make_grid
            import torch
            grid=make_grid(torch.from_numpy(samples[:16]),nrow=4,normalize=True,value_range=(-1,1))
            ax.imshow(grid.permute(1,2,0).numpy()); ax.axis('off')
        ax.set_title(c['algorithm'],fontsize=10)
    fig.suptitle('Short-run generated samples; diagnostic evidence only')
    fig.savefig(out/'generated_samples.png',dpi=160); plt.close(fig)
    print(out)


if __name__=='__main__': main()
