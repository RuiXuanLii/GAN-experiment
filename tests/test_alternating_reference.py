"""The schedule diagnostic must account for player clocks and resume exactly."""
import torch
from tests.test_training import cfg
from experiments.alternating_reference import train_alternating


def test_alternating_reference_counts_only_applied_player_moments(tmp_path):
    c=cfg();c.update(training_schedule='alternating',critic_steps=5,oracle_budget=12,checkpoint_every=6,log_every=6)
    path=train_alternating(c,tmp_path/'run')
    s=torch.load(path/'checkpoint.pt',weights_only=False)
    assert (s['oracle_calls'],s['g_updates'],s['d_updates'])==(12,2,10)
    assert {x['n'] for x in s['g_optimizer']['states']}=={2}
    assert {x['n'] for x in s['d_optimizer']['states']}=={10}
    assert s['data']['batches']==12


def test_alternating_reference_resumption_matches_full_run(tmp_path):
    c=cfg();c.update(training_schedule='alternating',critic_steps=1,oracle_budget=12,checkpoint_every=6,log_every=6)
    full=train_alternating(c,tmp_path/'full')
    split=train_alternating(c,tmp_path/'split',stop_after=6)
    train_alternating(c,split,resume=True)
    a=torch.load(full/'checkpoint.pt',weights_only=False)
    b=torch.load(split/'checkpoint.pt',weights_only=False)
    for key in ['generator','critic']:
        for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
    assert a['g_updates']==b['g_updates']==6
    assert a['d_updates']==b['d_updates']==6


def test_alternating_resume_trims_log_rows_ahead_of_checkpoint(tmp_path):
    import csv
    c=cfg();c.update(training_schedule='alternating',critic_steps=1,oracle_budget=12,checkpoint_every=6,log_every=6)
    path=train_alternating(c,tmp_path/'run',stop_after=6)
    csv_path=path/'metrics.csv'
    with csv_path.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;row=next(reader)
    row['oracle_calls']='12'
    with csv_path.open('a',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
    train_alternating(c,path,resume=True)
    with csv_path.open() as f:calls=[int(r['oracle_calls']) for r in csv.DictReader(f)]
    assert calls==[6,12]
