"""Create the main-run freeze only from four completed GPU pilots and tuning metrics."""
import argparse
import json
import math
import hashlib
from pathlib import Path
from ganopt.optim import METHODS
from ganopt.runtime import source_hash,METRIC_PROTOCOL
from ganopt.data import split_indices

COMMON_KEYS=['protocol','dataset','split_seed','batch_size','z_dim','base_channels','betas',
             'eps','gp_lambda','extra_batch','lookahead_k','lookahead_alpha','device','threads']


def check_metric_evidence(record,config,calls):
    """Reject incomparable or nonfinite records rather than assigning fake scores."""
    manifest=record.get('manifest',{})
    if record.get('status')!='completed' or record.get('source_hash')!=source_hash():
        raise ValueError('Incomplete/stale metric evidence')
    if manifest.get('source_hash')!=source_hash() or manifest.get('config')!=config:
        raise ValueError('Metric checkpoint/config evidence mismatch')
    if record.get('metric_protocol')!=METRIC_PROTOCOL or manifest.get('metric_protocol')!=METRIC_PROTOCOL:
        raise ValueError('Metric protocol mismatch')
    if record.get('torch_fidelity_version')!=METRIC_PROTOCOL['torch_fidelity_version'] or record.get('feature_extractor')!=METRIC_PROTOCOL['feature_extractor']:
        raise ValueError('Metric feature implementation/version mismatch')
    if manifest.get('evaluation_seed')!=123456 or manifest.get('reference_split')!='validation' or manifest.get('count')!=2000 or manifest.get('oracle_calls')!=calls:
        raise ValueError('Metric seed/split/count/endpoint mismatch')
    size,valsize=(50000,5000) if config['dataset']=='cifar10' else (60000,6000)
    _,validation=split_indices(size,config['split_seed'],valsize)
    expected=hashlib.sha256(validation[:2000].astype('<i8').tobytes()).hexdigest()
    if manifest.get('real_indices_sha256')!=expected: raise ValueError('Reference identity mismatch')
    if not math.isfinite(record['evaluation_seconds']) or record['evaluation_seconds']<=0:
        raise ValueError('Invalid metric timing')
    metrics=record['metrics']
    for key in ['frechet_inception_distance','kernel_inception_distance_mean']:
        if key not in metrics or not math.isfinite(float(metrics[key])): raise ValueError('Nonfinite/missing metric')
    if metrics['frechet_inception_distance']<0: raise ValueError('Negative FID evidence')


def freeze(pilot_root,tuning_root,output,gpu_hours,budget=20000):
    if not math.isfinite(gpu_hours) or gpu_hours<=0: raise ValueError('gpu_hours must be finite and positive')
    if budget<1 or budget%10: raise ValueError('Main budget must be a positive multiple of 10')
    common=None; pilots={}; candidates={m:[] for m in METHODS}; records=[];failed=[]
    for path in Path(pilot_root).rglob('result.json'):
        run=path.parent; c=json.loads((run/'config.json').read_text()); r=json.loads(path.read_text())
        if c['stage']!='pilot': continue
        meta=json.loads((run/'metadata.json').read_text())
        if r['status']!='completed' or meta['source_hash']!=source_hash() or not c['device'].startswith('cuda'):
            raise ValueError(f'Pilot incomplete or source/device mismatch: {run}')
        if c['dataset'] not in ['cifar10','mnist'] or c['seed']!=9001 or c['lr']!=1e-4 or c['oracle_budget']!=200 or r['oracle_calls']!=200:
            raise ValueError('Pilot must use shared image data, seed9001, default LR and 200 calls')
        if not math.isfinite(r['training_seconds']) or r['training_seconds']<=0:
            raise ValueError('Invalid pilot timing')
        current={k:c[k] for k in COMMON_KEYS}
        if common is not None and current!=common: raise ValueError('Pilot shared configurations differ')
        common=current
        rows=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
        peak=max(row['cuda_peak_reserved_mb'] for row in rows)
        capacity=meta['environment']['cuda_total_memory_mb']
        if not capacity or not math.isfinite(capacity) or capacity<=0 or not math.isfinite(peak) or peak<=0 or peak>=.8*capacity:
            raise ValueError('Pilot exceeds memory gate or has invalid memory evidence')
        if c['algorithm'] in pilots: raise ValueError('Select exactly one pilot per method')
        pilots[c['algorithm']]={'seconds_per_call':r['training_seconds']/r['oracle_calls'],
                               'peak_reserved_mb':peak,'run':str(run)}
        metric_path=run/'evaluation/metrics.json'
        if not metric_path.exists(): raise ValueError('Pilot image evaluation is missing')
        metric=json.loads(metric_path.read_text());check_metric_evidence(metric,c,200);records.append(metric)
    if set(pilots)!=set(METHODS): raise ValueError('Four completed GPU pilots required')
    attempted={m:set() for m in METHODS}
    for result_path in Path(tuning_root).rglob('result.json'):
        run=result_path.parent;c=json.loads((run/'config.json').read_text());result=json.loads(result_path.read_text())
        if c['stage']!='tuning': continue
        meta=json.loads((run/'metadata.json').read_text())
        if meta['source_hash']!=source_hash() or {k:c[k] for k in COMMON_KEYS}!=common:
            raise ValueError('Tuning source/common configuration mismatch')
        if c['seed']!=9001 or c['oracle_budget']!=2000 or c['lr'] not in [5e-5,1e-4,2e-4]:
            raise ValueError('Tuning seed/rate/budget mismatch')
        method=c['algorithm']
        if c['lr'] in attempted[method]: raise ValueError('Duplicate tuning candidate')
        attempted[method].add(c['lr'])
        if result['status']=='failed':
            if not result.get('error','').startswith('FloatingPointError('):
                raise ValueError('Resolve nonnumerical tuning execution failures before freezing')
            failed.append({'algorithm':method,'lr':c['lr'],'run':str(run),'error':result['error']});continue
        if result['status']!='completed' or result['oracle_calls']!=2000:
            raise ValueError('Tuning run has not completed the fixed budget')
        metrics_path=run/'evaluation/metrics.json'
        if not metrics_path.exists(): raise ValueError('Successful tuning candidate is missing image metrics')
        r=json.loads(metrics_path.read_text());check_metric_evidence(r,c,2000)
        candidates[c['algorithm']].append((r['metrics']['frechet_inception_distance'],
            r['metrics']['kernel_inception_distance_mean'],c['lr'],str(metrics_path)))
        records.append(r)
    learning_rates={}
    for method,values in candidates.items():
        if attempted[method]!={5e-5,1e-4,2e-4} or not values:
            raise ValueError(f'{method}: all rates must be attempted, and at least one must complete real evaluation')
        learning_rates[method]=sorted(values)[0][2]
    # Deliberately conservative screening heuristic, not measured main runtime.
    train_hours=1.25*sum(p['seconds_per_call'] for p in pilots.values())*budget*3/3600
    eval_seconds=max(r['evaluation_seconds'] for r in records)
    eval_hours=1.25*(eval_seconds*2.5*math.ceil(budget/2000)*4*3+eval_seconds*5*4*3)/3600
    if train_hours+eval_hours>gpu_hours:
        raise ValueError(f'Estimated main training+evaluation {train_hours+eval_hours:.2f}h exceeds {gpu_hours}h; reduce all-method budget and justify')
    common.update(oracle_budget=budget)
    result={'frozen':True,'protocol':'v1.0','source_hash':source_hash(),'common':common,
            'lr_by_algorithm':learning_rates,'reporting_seeds':[11,22,33],
            'metric_protocol':METRIC_PROTOCOL,'failed_tuning_candidates':failed,
            'pilots':pilots,'tuning_metrics':[v[3] for values in candidates.values() for v in values],
            'resource_estimate':{'training_hours_with_margin':train_hours,'evaluation_hours_heuristic':eval_hours,
                'available_main_gpu_hours':gpu_hours,'note':'Pilot-derived estimates; evaluation scaling is a screening heuristic, not an exact model'}}
    Path(output).write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',required=True);p.add_argument('--tuning',required=True)
    p.add_argument('--output',default='freeze.json');p.add_argument('--gpu-hours',type=float,required=True)
    p.add_argument('--budget',type=int,default=20000);a=p.parse_args();print(json.dumps(freeze(a.pilot,a.tuning,a.output,a.gpu_hours,a.budget),indent=2))


if __name__=='__main__': main()
