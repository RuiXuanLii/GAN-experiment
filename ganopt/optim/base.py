"""Pure Adam moment transitions and transactional joint-parameter updates.

No weight decay, AMSGrad, sparse gradients or per-player schedules are supported.
The closure returns gradients in the constructor's fixed parameter order.
"""
from __future__ import annotations
import copy
import math
from collections.abc import Callable, Iterable
import torch

Closure = Callable[[], tuple[list[torch.Tensor], dict]]


class AdamKernel:
    """Shared adaptive direction; subclasses decide when/how directions apply."""
    oracle_cost = 1

    def __init__(self, params: Iterable[torch.Tensor], lr: float=1e-4,
                 betas: tuple[float,float]=(.5,.9), eps: float=1e-8):
        self.params=list(params)
        if not self.params or len({id(p) for p in self.params})!=len(self.params):
            raise ValueError('Parameters must be nonempty and unique')
        if not math.isfinite(lr) or lr<=0 or not math.isfinite(eps) or eps<0:
            raise ValueError('lr must be positive; eps nonnegative, both finite')
        if len(betas)!=2 or any(not 0<=b<1 for b in betas):
            raise ValueError('betas must be in [0,1)')
        self.lr=float(lr); self.betas=tuple(betas); self.eps=float(eps)
        self.states=[{'m':torch.zeros_like(p),'v':torch.zeros_like(p),'n':0} for p in self.params]

    def _directions(self, grads: list[torch.Tensor]):
        """Return NEW states; never mutate states of a previous transaction."""
        if len(grads)!=len(self.params): raise ValueError('Gradient count mismatch')
        directions=[]; states=[]; b1,b2=self.betas
        with torch.no_grad():
            for p,g,s in zip(self.params,grads,self.states,strict=True):
                if g is None or g.shape!=p.shape or g.device!=p.device or g.dtype!=p.dtype:
                    raise ValueError('Gradient shape/device/dtype mismatch')
                if g.is_sparse: raise ValueError('Sparse gradients unsupported')
                if not torch.isfinite(g).all(): raise FloatingPointError('Nonfinite gradient')
                g=g.detach(); n=s['n']+1
                m=b1*s['m']+(1-b1)*g
                v=b2*s['v']+(1-b2)*g.square()
                if not torch.isfinite(m).all() or not torch.isfinite(v).all():
                    raise FloatingPointError('Nonfinite Adam moment (overflow)')
                u=(m/(1-b1**n))/(torch.sqrt(v/(1-b2**n))+self.eps)
                if not torch.isfinite(u).all(): raise FloatingPointError('Nonfinite direction')
                directions.append(u); states.append({'m':m,'v':v,'n':n})
        return directions,states

    def _commit(self, values):
        if any(not torch.isfinite(v).all() for v in values):
            raise FloatingPointError('Nonfinite updated parameter')
        with torch.no_grad():
            for p,v in zip(self.params,values,strict=True): p.copy_(v)

    def state_dict(self) -> dict:
        """Detached independent serialization, including subclass slow/history state."""
        out={'class':type(self).__name__,'lr':self.lr,'betas':self.betas,
             'eps':self.eps,'states':self.states}
        for key in ['previous','slow','inner_steps','k','alpha']:
            if hasattr(self,key): out[key]=getattr(self,key)
        return copy.deepcopy(out)

    def load_state_dict(self, state: dict):
        for key,value in [('class',type(self).__name__),('lr',self.lr),('eps',self.eps)]:
            if state[key]!=value: raise ValueError(f'Optimizer {key} mismatch')
        if tuple(state['betas'])!=self.betas or len(state['states'])!=len(self.params):
            raise ValueError('Optimizer configuration/parameter mismatch')
        state=copy.deepcopy(state)
        for s,p in zip(state['states'],self.params,strict=True):
            if not isinstance(s['n'],int) or s['n']<0: raise ValueError('Invalid state clock')
            for key in ['m','v']:
                if s[key].shape!=p.shape: raise ValueError('State shape mismatch')
                s[key]=s[key].to(device=p.device,dtype=p.dtype).detach()
                if not torch.isfinite(s[key]).all(): raise ValueError('Nonfinite optimizer state')
            if (s['v']<0).any(): raise ValueError('Negative variance state')
        for key in ['k','alpha']:
            if hasattr(self,key) and state[key]!=getattr(self,key):
                raise ValueError(f'Optimizer {key} mismatch')
        prepared={}
        for key in ['previous','slow']:
            if hasattr(self,key):
                if key not in state: raise ValueError(f'Missing {key} state')
                if len(state[key])!=len(self.params): raise ValueError('History count mismatch')
                prepared[key]=[]
                for x,p in zip(state[key],self.params,strict=True):
                    if x.shape!=p.shape or not torch.isfinite(x).all():
                        raise ValueError('History shape/finiteness mismatch')
                    prepared[key].append(x.to(device=p.device,dtype=p.dtype).detach())
        if 'inner_steps' in state and (not isinstance(state['inner_steps'],int) or state['inner_steps']<0):
            raise ValueError('Invalid inner step clock')
        self.states=state['states']
        for key,value in prepared.items(): setattr(self,key,value)
        if 'inner_steps' in state: self.inner_steps=state['inner_steps']
