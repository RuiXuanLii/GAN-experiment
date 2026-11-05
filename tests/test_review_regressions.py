"""Reproductions of independently reviewed defects; fixtures are NOT GPU results."""
import copy
import csv
import json
from pathlib import Path
import pytest
import torch
from ganopt.optim import Adam,ExtraAdam,OptimisticAdam,LookaheadMinmax
from ganopt.runtime import source_hash
from ganopt.train import train
from scripts.export_images import export
from scripts.freeze_protocol import freeze
from tests.test_training import cfg


@pytest.mark.parametrize('cls',[Adam,ExtraAdam,OptimisticAdam,LookaheadMinmax])
def test_finite_gradient_moment_overflow_is_rejected_transactionally(cls):
    p=torch.nn.Parameter(torch.ones(1)); opt=cls([p])
    with pytest.raises(FloatingPointError): opt.step(lambda: ([torch.tensor([1e20])],{}))
    assert p.item()==1 and opt.state_dict()['states'][0]['n']==0


def test_extra_correction_moment_overflow_restores_both_transitions():
    p=torch.nn.Parameter(torch.ones(1)); opt=ExtraAdam([p]); calls=0
    def closure():
        nonlocal calls
        calls+=1
        return [torch.ones(1) if calls==1 else torch.tensor([1e20])],{}
    with pytest.raises(FloatingPointError):opt.step(closure)
    assert p.item()==1 and opt.state_dict()['states'][0]['n']==0


def test_torn_trailing_log_recovers_with_intact_checkpoint(tmp_path):
    c=cfg();run=train(c,tmp_path/'run',stop_after=10)
    with (run/'metrics.jsonl').open('a') as f:f.write('{"oracle_calls":20')
    train(c,run,resume=True)
    assert [json.loads(l)['oracle_calls'] for l in (run/'metrics.jsonl').read_text().splitlines()]==[10,20]


def test_checkpoint_zero_recovery_removes_all_stale_csv_rows(tmp_path):
    c=cfg();run=train(c,tmp_path/'run'); original=torch.load(run/'checkpoint.pt',weights_only=False)
    # Recreate an initial valid checkpoint from the original initialized models.
    from ganopt.models import make_models
    from ganopt.data import BatchStream
    from ganopt.runtime import seed_everything,rng_state
    seed_everything(c['seed'],c['threads']);G,D=make_models(c);stream=BatchStream(c)
    original.update(generator=G.state_dict(),critic=D.state_dict(),optimizer=Adam(list(G.parameters())+list(D.parameters()),lr=c['lr']).state_dict(),data=stream.state_dict(),rng=rng_state(),oracle_calls=0,updates=0,training_seconds=0.,wall_seconds=0.)
    torch.save(original,run/'checkpoint.pt');train(c,run,resume=True)
    csv_calls=[int(r['oracle_calls']) for r in csv.DictReader((run/'metrics.csv').open())]
    assert csv_calls==[10,20]


def test_test_export_rejects_intermediate_main_checkpoint_before_data_access(tmp_path):
    c=cfg();c.update(dataset='cifar10',stage='main',oracle_budget=20000)
    path=tmp_path/'invalid.pt';torch.save({'config':c,'source_hash':source_hash(),'oracle_calls':0},path)
    with pytest.raises(ValueError,match='freeze|endpoint|10,000'):
        export(path,tmp_path/'images',tmp_path/'data',split='test',count=2000)


@pytest.mark.parametrize('hours',[float('nan'),float('inf'),0.,-1.])
def test_freeze_requires_finite_positive_actual_budget(tmp_path,hours):
    with pytest.raises(ValueError,match='gpu_hours'):
        freeze(tmp_path,tmp_path,tmp_path/'freeze.json',gpu_hours=hours)


def test_invalid_history_load_preserves_existing_state():
    p=torch.nn.Parameter(torch.ones(1));opt=OptimisticAdam([p]);opt.step(lambda:([torch.ones(1)],{}))
    saved=opt.state_dict();bad=copy.deepcopy(saved);bad['states'][0]['n']=9;bad['previous'][0]=torch.ones(3)
    with pytest.raises(ValueError):opt.load_state_dict(bad)
    assert opt.state_dict()['states'][0]['n']==1


def protocol_fixture(root):
    """Minimal mock evidence to test gates; never represents measured GPU data."""
    from ganopt.runtime import METRIC_PROTOCOL
    from ganopt.data import split_indices
    from ganopt.optim import METHODS
    import hashlib,numpy as np
    base=json.loads((Path(__file__).parents[1]/'configs/cifar10_pilot.json').read_text())
    _,val=split_indices(50000,2026,5000)
    refhash=hashlib.sha256(val[:2000].astype('<i8').tobytes()).hexdigest()
    def metric(c,calls):
        return {'status':'completed','source_hash':source_hash(),'evaluation_seconds':.2,
            'torch_fidelity_version':'0.3.0','feature_extractor':'inception-v3-compat','metric_protocol':METRIC_PROTOCOL,
            'metrics':{'frechet_inception_distance':1.,'kernel_inception_distance_mean':.01},
            'manifest':{'config':c,'source_hash':source_hash(),'dataset':'cifar10','reference_split':'validation',
                'count':2000,'evaluation_seed':123456,'oracle_calls':calls,'metric_protocol':METRIC_PROTOCOL,
                'real_indices_sha256':refhash}}
    for method in METHODS:
        p=root/'pilot'/method;p.mkdir(parents=True);c=dict(base,algorithm=method)
        (p/'config.json').write_text(json.dumps(c));(p/'result.json').write_text(json.dumps({'status':'completed','oracle_calls':200,'training_seconds':1.}))
        (p/'metadata.json').write_text(json.dumps({'source_hash':source_hash(),'environment':{'cuda_total_memory_mb':1000}}))
        (p/'metrics.jsonl').write_text(json.dumps({'cuda_peak_reserved_mb':100})+'\n')
        (p/'evaluation').mkdir();(p/'evaluation/metrics.json').write_text(json.dumps(metric(c,200)))
        for lr in [5e-5,1e-4,2e-4]:
            t=root/'tuning'/f'{method}-{lr}';t.mkdir(parents=True);c=dict(base,algorithm=method,stage='tuning',lr=lr,oracle_budget=2000)
            (t/'config.json').write_text(json.dumps(c));(t/'result.json').write_text(json.dumps({'status':'completed','oracle_calls':2000,'training_seconds':10.}))
            (t/'metadata.json').write_text(json.dumps({'source_hash':source_hash()}))
            (t/'evaluation').mkdir();(t/'evaluation/metrics.json').write_text(json.dumps(metric(c,2000)))
    return root/'pilot',root/'tuning'


@pytest.mark.parametrize('defect',['nan','mixed_version','mixed_seed','wrong_pilot','missing_pilot_evaluation'])
def test_freeze_rejects_invalid_or_incomparable_metric_evidence(tmp_path,defect):
    pilot,tuning=protocol_fixture(tmp_path)
    mp=next(tuning.rglob('metrics.json'));r=json.loads(mp.read_text())
    if defect=='nan':r['metrics']['frechet_inception_distance']=float('nan')
    if defect=='mixed_version':r['torch_fidelity_version']='0.4.0'
    if defect=='mixed_seed':r['manifest']['evaluation_seed']=55
    if defect in ['nan','mixed_version','mixed_seed']:mp.write_text(json.dumps(r))
    if defect=='wrong_pilot':
        cp=next(pilot.rglob('config.json'));c=json.loads(cp.read_text());c['seed']=11;cp.write_text(json.dumps(c))
    if defect=='missing_pilot_evaluation':next(pilot.rglob('evaluation/metrics.json')).unlink()
    with pytest.raises(ValueError):freeze(pilot,tuning,tmp_path/'freeze.json',gpu_hours=10)


def test_freeze_can_exclude_documented_numerical_failure_candidate(tmp_path):
    pilot,tuning=protocol_fixture(tmp_path);run=tuning/'adam-5e-05'
    (run/'evaluation/metrics.json').unlink()
    (run/'result.json').write_text(json.dumps({'status':'failed','error':"FloatingPointError('Nonfinite gradient')"}))
    frozen=freeze(pilot,tuning,tmp_path/'freeze.json',gpu_hours=10)
    assert frozen['lr_by_algorithm']['adam']==1e-4
    assert len(frozen['failed_tuning_candidates'])==1
