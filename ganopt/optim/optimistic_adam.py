"""Daskalakis et al. (2018), Algorithm 1: optimism on ADAM DIRECTIONS."""
import torch
from .base import AdamKernel, Closure


class OptimisticAdam(AdamKernel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.previous=[torch.zeros_like(p) for p in self.params]

    def step(self, closure: Closure) -> dict:
        grads,metrics=closure()
        directions,new_states=self._directions(grads)
        self._commit([p.detach()-self.lr*(2*u-old) for p,u,old in
                      zip(self.params,directions,self.previous,strict=True)])
        self.states=new_states
        self.previous=[u.detach().clone() for u in directions]
        return metrics
