"""Collect and validate the eight-Gaussian runs from round one.

Run this with locally generated, trusted checkpoints only.
"""
import argparse
from pathlib import Path
import csv
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ganopt.data import BatchStream
from ganopt.metrics import synthetic_metrics
from ganopt.models import make_models
from ganopt.optim import METHODS
from ganopt.runtime import source_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root, out = args.runs, args.output
labels = ['Adam', 'ExtraAdam', 'Optimistic Adam', 'Lookahead-Minmax']
metrics_dir = out/'synthetic_runs'
all_rows = []
metadata = []
for record in sorted(root.rglob('result.json')):
    run = record.parent
    c = json.loads((run/'config.json').read_text())
    m = json.loads((run/'metadata.json').read_text())
    r = json.loads(record.read_text())
    assert r['status']=='completed' and c['seed']==9001
    assert r['oracle_calls']==c['oracle_budget']
    assert r['parameter_updates'] * METHODS[c['algorithm']].oracle_cost == r['oracle_calls']
    assert m['source_hash']==source_hash()
    assert synthetic_metrics(np.load(run/'samples.npy')) == {key:r[key] for key in ('covered_modes','quality_fraction','mean_nearest_distance','assignment_entropy')}
    metadata.append(m)
    all_rows.append({'experiment_id':m['experiment_id'],'method':c['algorithm'],'lr':c['lr'],'seed':c['seed'],
                     **{key:r[key] for key in ('oracle_calls','parameter_updates','training_seconds','covered_modes','quality_fraction','mean_nearest_distance','assignment_entropy')}})
    dest = metrics_dir / run.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('config.json','metadata.json','result.json','metrics.csv','metrics.jsonl','samples.npy'):
        shutil.copy2(run/name,dest/name)
assert len(all_rows)==16
assert len({m['initial_parameter_hash'] for m in metadata})==1
assert len({m['split_hash'] for m in metadata})==1
for calls, lr in ((3000,1e-4),(3000,3e-4),(3000,1e-3),(10000,1e-3)):
    assert {r['method'] for r in all_rows if r['oracle_calls']==calls and r['lr']==lr}==set(METHODS)
with (out/'synthetic_summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)

progress=[]
fig, axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
metrics = ('covered_modes','quality_fraction','mean_nearest_distance')
for method,label in zip(METHODS,labels,strict=True):
    run = next(root.glob(f'extended/pilot-synthetic-{method}-*'))
    c = json.loads((run/'config.json').read_text())
    eval_rng = torch.Generator(device='cpu').manual_seed(123456)
    noise = torch.randn(c['eval_samples'],c['z_dim'],generator=eval_rng)
    series=[]
    for ckpt in sorted((run/'checkpoints').glob('*.pt')):
        state = torch.load(ckpt,map_location='cpu',weights_only=False)
        generator,_ = make_models(c)
        generator.load_state_dict(state['generator']); generator.eval()
        with torch.no_grad():
            scores = synthetic_metrics(generator(noise).numpy())
        series.append({'method':method,'oracle_calls':state['oracle_calls'],**scores})
    assert len(series)==10 and series[-1]['oracle_calls']==10000
    progress.extend(series)
    for ax,key in zip(axes,metrics,strict=True):
        scale=100 if key=='quality_fraction' else 1
        ax.plot([r['oracle_calls'] for r in series],[r[key]*scale for r in series],label=label,marker='o',markersize=2)
for ax,label in zip(axes,('Modes covered (out of 8)','Samples within radius 0.15 (%)','Mean distance to nearest center'),strict=True):
    ax.set(xlabel='Joint gradient evaluations',ylabel=label); ax.grid(alpha=.25)
axes[0].set(ylim=(-.2,8.2)); axes[0].legend(fontsize=7,loc='upper right')
fig.suptitle('Eight-Gaussian pilot: one seed, learning rate 0.001')
fig.savefig(out/'synthetic_progress.png',dpi=150);plt.close(fig)
with (out/'synthetic_checkpoints.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(progress[0]));writer.writeheader();writer.writerows(progress)
config=metadata[0]['config']
stream=BatchStream(config)
reference=np.concatenate([stream.next()[0].numpy() for _ in range(config['eval_samples']//config['batch_size'])])
reference_metrics=synthetic_metrics(reference)
assert reference_metrics['covered_modes']==8 and reference_metrics['quality_fraction']>.97
(out/'synthetic_target_control.json').write_text(json.dumps({'samples':len(reference),**reference_metrics},indent=2)+'\n')
print('Synthetic runs:',len(all_rows),'all equal initial parameters, budgets verified')
print('Reference:',reference_metrics)
for method in METHODS:
    rows=[r for r in progress if r['method']==method]
    print(method,'best covered',max(r['covered_modes'] for r in rows),'best quality fraction',max(r['quality_fraction'] for r in rows),'last',rows[-1])
