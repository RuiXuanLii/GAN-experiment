"""Joint Lookahead-Minmax with synchronous Adam inner updates.

Chavdarova et al. (2021) joint interpolation; inner moments are retained.
This controlled synchronous variant is not the paper's large-model benchmark.
"""
from .base import AdamKernel, Closure


class LookaheadMinmax(AdamKernel):
    def __init__(self,*args,k: int=5,alpha: float=.5,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(k,int) or k<1 or not 0<alpha<=1:
            raise ValueError('k must be positive integer; alpha in (0,1]')
        self.k=k; self.alpha=float(alpha); self.inner_steps=0
        self.slow=[p.detach().clone() for p in self.params]

    def step(self, closure: Closure) -> dict:
        grads,metrics=closure()
        directions,new_states=self._directions(grads)
        fast=[p.detach()-self.lr*u for p,u in zip(self.params,directions,strict=True)]
        sync=(self.inner_steps+1)%self.k==0
        values=[s+self.alpha*(f-s) for s,f in zip(self.slow,fast,strict=True)] if sync else fast
        self._commit(values)
        self.states=new_states; self.inner_steps+=1
        if sync: self.slow=[v.detach().clone() for v in values]
        return metrics
