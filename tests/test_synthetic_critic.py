"""Controlled critic-activation ablation: retain parameter initialization and G."""
import pytest
import torch
from ganopt.models import make_models
from ganopt.game import field
from ganopt.data import BatchStream
from tests.test_training import cfg


def test_relu_critic_ablation_preserves_initial_parameters_and_changes_field():
    c=cfg();torch.manual_seed(42);g0,d0=make_models(c)
    c['critic_activation']='relu';torch.manual_seed(42);g1,d1=make_models(c)
    for a,b in zip(list(g0.parameters())+list(d0.parameters()),list(g1.parameters())+list(d1.parameters()),strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    data=BatchStream(c).next()
    grads0,_=field(g0,d0,*data)
    grads1,metrics=field(g1,d1,*data)
    assert all(torch.isfinite(g).all() for g in grads1)
    assert metrics['g_grad_norm']>0 and metrics['d_grad_norm']>0
    assert any(not torch.equal(a,b) for a,b in zip(grads0,grads1,strict=True))


def test_unknown_synthetic_critic_activation_is_rejected():
    c=cfg();c['critic_activation']='typo'
    with pytest.raises(ValueError,match='critic_activation'):
        make_models(c)
