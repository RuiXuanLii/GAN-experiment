"""Gidel et al. (2019), Algorithm 4, standard extrapolation branch.

Moments advance at BOTH evaluations. Correction starts from original parameters.
This differs from variants that discard prediction moments. A closure exception
rolls back the full transaction. Data RNG consumption is owned by the trainer.
"""
from .base import AdamKernel, Closure


class ExtraAdam(AdamKernel):
    oracle_cost=2

    def step(self, closure: Closure) -> dict:
        base=[p.detach().clone() for p in self.params]
        original_states=self.states
        try:
            grads,_=closure()
            directions,predicted_states=self._directions(grads)
            self._commit([p-self.lr*u for p,u in zip(base,directions,strict=True)])
            self.states=predicted_states
            grads,metrics=closure()  # recompute BOTH blocks at extrapolated joint point
            directions,corrected_states=self._directions(grads)
            self._commit([p-self.lr*u for p,u in zip(base,directions,strict=True)])
            self.states=corrected_states
            return metrics
        except Exception:
            self._commit(base)
            self.states=original_states
            raise
