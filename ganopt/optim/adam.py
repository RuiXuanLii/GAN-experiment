"""Adam, Kingma & Ba (2015), corrected-moment epsilon convention."""
from .base import AdamKernel, Closure


class Adam(AdamKernel):
    def step(self, closure: Closure) -> dict:
        grads,metrics=closure()
        directions,new_states=self._directions(grads)
        self._commit([p.detach()-self.lr*u for p,u in zip(self.params,directions,strict=True)])
        self.states=new_states
        return metrics
