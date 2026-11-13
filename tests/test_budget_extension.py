"""A lineage-recorded budget extension must equal uninterrupted training."""
import json
import torch
from tests.test_training import cfg
from experiments.alternating_reference import train_alternating
from experiments.extend_alternating import extend


def test_extended_budget_matches_full_run_and_preserves_parent(tmp_path):
    c=cfg();c.update(training_schedule='alternating',critic_steps=5,oracle_budget=12,checkpoint_every=6,log_every=6)
    parent=train_alternating(c,tmp_path/'parent')
    before=(parent/'checkpoint.pt').read_bytes()
    extended=extend(parent,tmp_path/'extended',24)
    assert (parent/'checkpoint.pt').read_bytes()==before
    full=train_alternating(dict(c,oracle_budget=24),tmp_path/'full')
    a=torch.load(extended/'checkpoint.pt',weights_only=False)
    b=torch.load(full/'checkpoint.pt',weights_only=False)
    for key in ['generator','critic']:
        for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
    assert a['oracle_calls']==b['oracle_calls']==24
    assert a['g_updates']==b['g_updates']==4
    assert a['d_updates']==b['d_updates']==20
    lineage=json.loads((extended/'lineage.json').read_text())
    assert lineage['parent_oracle_calls']==12
    assert lineage['config_changes']=={'oracle_budget':{'from':12,'to':24}}
