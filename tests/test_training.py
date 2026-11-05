import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ganopt.models import make_models
from ganopt.game import field, gradient_penalty
from ganopt.data import BatchStream
from ganopt.train import train
from ganopt.metrics import synthetic_metrics
from ganopt.optim import METHODS


def cfg(method='adam'):
    return {'protocol':'v1.0','stage':'smoke','dataset':'synthetic','algorithm':method,
            'seed':11,'split_seed':2026,'batch_size':8,'z_dim':4,'hidden':16,
            'base_channels':8,'lr':1e-4,'betas':[.5,.9],'eps':1e-8,'gp_lambda':10.,
            'oracle_budget':20,'log_every':10,'checkpoint_every':10,
            'extra_batch':'independent','lookahead_k':5,'lookahead_alpha':.5,
            'device':'cpu','threads':1,'data_root':'data','download':False,'eval_samples':128}


def test_game_signs_and_generator_live_graph():
    G=torch.nn.Linear(1,1,bias=False).double(); D=torch.nn.Linear(1,1,bias=False).double()
    with torch.no_grad(): G.weight.fill_(2); D.weight.fill_(3)
    x=torch.tensor([[4.]],dtype=torch.float64); z=torch.ones_like(x); a=torch.full_like(x,.5)
    grads,metrics=field(G,D,x,z,a,gp_lambda=0)
    torch.testing.assert_close(grads[0],torch.tensor([[-3.]],dtype=x.dtype))
    torch.testing.assert_close(grads[1],torch.tensor([[-2.]],dtype=x.dtype))
    assert metrics['g_loss']==pytest.approx(-6.)


def test_gradient_penalty_matches_analytic_linear_critic_derivative():
    D=torch.nn.Linear(2,1,bias=False).double()
    with torch.no_grad(): D.weight.copy_(torch.tensor([[3.,4.]],dtype=torch.float64))
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
    gp,_=gradient_penalty(D,x,-x,torch.full((2,1),.4,dtype=x.dtype))
    assert gp.item()==pytest.approx(16.)
    grad=torch.autograd.grad(gp,D.weight)[0]
    torch.testing.assert_close(grad,torch.tensor([[4.8,6.4]],dtype=x.dtype))


def test_repeated_field_does_not_accumulate_parameter_gradients():
    c=cfg(); G,D=make_models(c); stream=BatchStream(c); batch=stream.next()
    first,_=field(G,D,*batch,gp_lambda=10)
    second,_=field(G,D,*batch,gp_lambda=10)
    for a,b in zip(first,second): torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(p.grad is None for p in list(G.parameters())+list(D.parameters()))
    assert any(g.abs().sum()>0 for g in first[:len(list(G.parameters()))])


def test_data_stream_roundtrip_and_shared_oracle_sequence():
    c=cfg(); a=BatchStream(c); b=BatchStream(c)
    for x,y in zip(a.next(),b.next()): assert torch.equal(x,y)
    state=copy.deepcopy(a.state_dict()); expected=a.next(); a.load_state_dict(state)
    for x,y in zip(a.next(),expected): assert torch.equal(x,y)


@pytest.mark.parametrize('method',list(METHODS))
def test_checkpoint_resume_exactly_matches_uninterrupted(tmp_path,method):
    c=cfg(method); full=train(c,tmp_path/'full')
    train(c,tmp_path/'split',stop_after=10)
    resumed=train(c,tmp_path/'split',resume=True)
    a=torch.load(full/'checkpoint.pt',weights_only=False)
    b=torch.load(resumed/'checkpoint.pt',weights_only=False)
    for net in ['generator','critic']:
        for key in a[net]: torch.testing.assert_close(a[net][key],b[net][key],rtol=0,atol=0)
    assert a['oracle_calls']==b['oracle_calls']==20
    assert a['updates']==b['updates']==(10 if method=='extraadam' else 20)
    lines=(resumed/'metrics.jsonl').read_text().splitlines()
    assert len({json.loads(x)['oracle_calls'] for x in lines})==len(lines)
    for model in ['states']:
        for x,y in zip(a['optimizer'][model],b['optimizer'][model]):
            assert x['n']==y['n']; torch.testing.assert_close(x['m'],y['m'],rtol=0,atol=0)


def test_resume_rejects_changed_config(tmp_path):
    c=cfg(); train(c,tmp_path/'run',stop_after=10); c['lr']=2e-4
    with pytest.raises(ValueError,match='configuration'): train(c,tmp_path/'run',resume=True)


def test_resume_rejects_changed_source_hash(tmp_path):
    c=cfg(); path=train(c,tmp_path/'run',stop_after=10)
    chk=torch.load(path/'checkpoint.pt',weights_only=False); chk['source_hash']='wrong'
    torch.save(chk,path/'checkpoint.pt')
    with pytest.raises(ValueError,match='source'): train(c,tmp_path/'run',resume=True)


def test_image_network_cpu_gp_backward_without_dataset_download():
    c=cfg(); c.update(dataset='cifar10',z_dim=8,base_channels=8,batch_size=2)
    G,D=make_models(c); z=torch.randn(2,8); x=torch.randn(2,3,32,32); a=torch.rand(2,1,1,1)
    grads,metrics=field(G,D,x,z,a,gp_lambda=10)
    assert G(z).shape==x.shape
    assert all(torch.isfinite(g).all() for g in grads)
    assert metrics['g_grad_norm']>0 and metrics['d_grad_norm']>0
    assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout))
                   for m in list(G.modules())+list(D.modules()))


def test_synthetic_metrics_known_centers_and_far_points():
    theta=np.arange(8)*2*np.pi/8; centers=2*np.stack([np.cos(theta),np.sin(theta)],axis=1)
    good=synthetic_metrics(np.repeat(centers,10,axis=0))
    assert good['covered_modes']==8; assert good['quality_fraction']==1
    assert good['mean_nearest_distance']<1e-12
    bad=synthetic_metrics(np.ones((80,2))*100)
    assert bad['quality_fraction']==0 and bad['covered_modes']==0


def test_main_experiment_requires_freeze_and_cuda(tmp_path):
    c=cfg(); c.update(dataset='cifar10',stage='main')
    with pytest.raises(ValueError,match='freeze'): train(c,tmp_path/'run')
