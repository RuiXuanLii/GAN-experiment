"""Independent mathematical oracles; none copies the production kernel."""
import copy
import numpy as np
import pytest
import torch
from ganopt.optim import Adam, ExtraAdam, OptimisticAdam, LookaheadMinmax, METHODS


def constant(g):
    return lambda: ([g.clone()], {})


def numpy_direction(g, m, v, n, b1=.5, b2=.9, eps=1e-8):
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*np.square(g)
    n += 1
    return (m/(1-b1**n))/(np.sqrt(v/(1-b2**n))+eps), m, v, n


def test_adam_matches_pytorch_multistep():
    p = torch.nn.Parameter(torch.tensor([.3,-.7], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    custom = Adam([p], lr=.02, betas=(.5,.9))
    trusted = torch.optim.Adam([q], lr=.02, betas=(.5,.9), eps=1e-8)
    for g in [[1.,-2.],[-.4,.3],[.8,3.],[-2.,.7]]:
        grad = torch.tensor(g,dtype=p.dtype)
        custom.step(constant(grad))
        trusted.zero_grad(set_to_none=True); q.grad=grad.clone(); trusted.step()
        torch.testing.assert_close(p,q,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('method',[Adam,OptimisticAdam])
def test_adam_family_against_numpy_recurrence(method):
    x=np.array([.7,-.2]); m=np.zeros(2); v=m.copy(); n=0; prev=m.copy()
    p=torch.nn.Parameter(torch.tensor(x,dtype=torch.float64)); opt=method([p],lr=.03)
    for g in [np.array([1.,-3.]),np.array([-.6,.8]),np.array([2.,.1])]:
        u,m,v,n=numpy_direction(g,m,v,n)
        x=x-.03*(u if method is Adam else 2*u-prev); prev=u.copy()
        opt.step(constant(torch.tensor(g,dtype=p.dtype)))
        np.testing.assert_allclose(p.detach().numpy(),x,rtol=1e-13,atol=1e-13)


def test_optimistic_first_step_uses_zero_previous_direction():
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    OptimisticAdam([p],lr=.1,eps=0).step(constant(torch.ones_like(p)))
    torch.testing.assert_close(p,torch.tensor([.8],dtype=p.dtype))


def test_extra_joint_extrapolation_restores_base_and_retains_two_moments():
    p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64)); seen=[]
    opt=ExtraAdam([p],lr=.1,eps=0)
    def field():
        seen.append(p.detach().clone())
        return [torch.stack([p[1],-p[0]]).detach()],{}
    opt.step(field)
    np.testing.assert_allclose(seen[1].numpy(),[.9,2.1])
    u,m,v,n=numpy_direction(np.array([2.,-1.]),np.zeros(2),np.zeros(2),0,eps=0)
    uc,m,v,n=numpy_direction(np.array([2.1,-.9]),m,v,n,eps=0)
    np.testing.assert_allclose(p.detach().numpy(),np.array([1.,2.])-.1*uc,rtol=1e-13)
    assert opt.state_dict()['states'][0]['n']==2
    np.testing.assert_allclose(opt.state_dict()['states'][0]['m'].numpy(),m)


def test_extra_nonfinite_correction_rolls_back_parameters_and_state():
    p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
    opt=ExtraAdam([p],lr=.1); opt.step(constant(torch.ones_like(p)))
    before=p.detach().clone(); state=copy.deepcopy(opt.state_dict()); calls=0
    def bad():
        nonlocal calls
        calls+=1
        return [torch.ones_like(p) if calls==1 else torch.full_like(p,float('nan'))],{}
    with pytest.raises(FloatingPointError): opt.step(bad)
    torch.testing.assert_close(p,before,rtol=0,atol=0)
    assert opt.state_dict()['states'][0]['n']==state['states'][0]['n']
    torch.testing.assert_close(opt.state_dict()['states'][0]['m'],state['states'][0]['m'])


def test_extra_closure_exception_restores_parameter():
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64)); calls=0
    opt=ExtraAdam([p],lr=.1)
    def bad():
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('intentional closure failure')
        return [torch.ones_like(p)],{}
    with pytest.raises(RuntimeError): opt.step(bad)
    assert p.item()==1.
    assert opt.state_dict()['states'][0]['n']==0


def test_lookahead_joint_sync_and_preserves_adam_moments():
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    q=torch.nn.Parameter(torch.tensor([-1.],dtype=torch.float64))
    opt=LookaheadMinmax([p,q],lr=.1,k=2,alpha=.5,eps=0)
    closure=lambda: ([torch.ones_like(p),-torch.ones_like(q)],{})
    opt.step(closure)
    assert p.item()==pytest.approx(.9); assert q.item()==pytest.approx(-.9)
    opt.step(closure)
    assert p.item()==pytest.approx(.9); assert q.item()==pytest.approx(-.9)
    assert [s['n'] for s in opt.state_dict()['states']]==[2,2]
    assert opt.state_dict()['inner_steps']==2


@pytest.mark.parametrize('cls',list(METHODS.values()))
def test_state_roundtrip_next_step_matches(cls):
    p=torch.nn.Parameter(torch.tensor([.3,-.4],dtype=torch.float64)); opt=cls([p],lr=.02)
    grad=torch.tensor([.4,-.9],dtype=p.dtype)
    opt.step(constant(grad)); opt.step(constant(-grad))
    q=torch.nn.Parameter(p.detach().clone()); other=cls([q],lr=.02)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    opt.step(constant(grad)); other.step(constant(grad))
    torch.testing.assert_close(p,q,rtol=0,atol=0)


@pytest.mark.parametrize('cls',list(METHODS.values()))
def test_no_state_tensor_requires_grad(cls):
    p=torch.nn.Parameter(torch.tensor([.3],dtype=torch.float64)); opt=cls([p])
    opt.step(lambda: ([torch.autograd.grad(p.square().sum(),p)[0]],{}))
    def walk(x):
        if isinstance(x,torch.Tensor): assert not x.requires_grad and x.grad_fn is None
        elif isinstance(x,dict):
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(opt.state_dict())


@pytest.mark.parametrize('kw',[{'lr':-1.},{'betas':(1.,.9)},{'eps':-1.}])
def test_invalid_hyperparameters_rejected(kw):
    with pytest.raises(ValueError): Adam([torch.nn.Parameter(torch.ones(1))],**kw)


def test_gradient_shape_mismatch_rejected_without_mutation():
    p=torch.nn.Parameter(torch.ones(2)); opt=Adam([p])
    with pytest.raises(ValueError): opt.step(constant(torch.ones(3)))
    assert torch.equal(p,torch.ones(2))


def test_bilinear_exact_eg_contraction_identity():
    # This tests the analytical EG model, not an unproved Adam contraction.
    w=np.array([1.,2.]); eta=.3; A=np.array([[0.,1.],[-1.,0.]])
    pred=w-eta*A@w; corrected=w-eta*A@pred
    np.testing.assert_allclose(corrected@corrected,(1-eta**2+eta**4)*(w@w))


def test_extra_matches_author_reference_with_zero_epsilon():
    # The historical reference places epsilon before variance bias correction.
    # epsilon=0 isolates identical mathematical rules; ordinary epsilon is
    # independently checked against torch.optim.Adam above.
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('gidel_reference',Path(__file__).parent/'reference/gidel_extragradient.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
    q=torch.nn.Parameter(p.detach().clone())
    custom=ExtraAdam([p],lr=.03,eps=0)
    reference=module.ExtraAdam([q],lr=.03,betas=(.5,.9),eps=0)
    def gradient(x): return torch.stack([x[1],-x[0]]).detach()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning) # reference PyTorch 0.4 overload only
        for _ in range(10):
            custom.step(lambda: ([gradient(p)],{}))
            reference.zero_grad(set_to_none=True); q.grad=gradient(q); reference.extrapolation()
            reference.zero_grad(set_to_none=True); q.grad=gradient(q); reference.step()
            torch.testing.assert_close(p,q,rtol=1e-13,atol=1e-13)
            assert custom.state_dict()['states'][0]['n']==reference.state[q]['step']


def test_state_history_does_not_alias_live_serialization():
    p=torch.nn.Parameter(torch.tensor([1.])); opt=OptimisticAdam([p])
    opt.step(constant(torch.ones_like(p))); saved=opt.state_dict()
    saved['previous'][0].fill_(999)
    assert opt.state_dict()['previous'][0].item()<2
