"""Separate player losses for WGAN-GP, evaluated at one JOINT parameter point."""
import torch


def gradient_penalty(critic, real, fake, alpha):
    """GP with fake held fixed for the critic partial derivative.

create_graph=True is essential: the outer critic derivative differentiates
the input-gradient norm. No GP term is included in the generator loss.
"""
    interpolation=(alpha*real+(1-alpha)*fake.detach()).detach().requires_grad_(True)
    scores=critic(interpolation)
    input_grad=torch.autograd.grad(scores.sum(),interpolation,create_graph=True)[0]
    norms=input_grad.flatten(1).norm(2,dim=1)
    return (norms-1).square().mean(),norms


def field(generator, critic, real, noise, alpha, gp_lambda=10.):
    """Return detached gradients [G params, D params], and scalar diagnostics.

autograd.grad does not write parameter .grad; repeated closures cannot
silently accumulate gradients. Two distinct critic graphs implement the
player partial derivatives, and BOTH gradients precede ANY parameter update.
"""
    fake=generator(noise)
    g_loss=-critic(fake).mean()
    fake_score=critic(fake.detach()).mean(); real_score=critic(real).mean()
    gp,norms=gradient_penalty(critic,real,fake,alpha)
    d_loss=fake_score-real_score+gp_lambda*gp
    if not torch.isfinite(g_loss) or not torch.isfinite(d_loss):
        raise FloatingPointError('Nonfinite player loss')
    g_params=list(generator.parameters()); d_params=list(critic.parameters())
    gg=torch.autograd.grad(g_loss,g_params)
    dg=torch.autograd.grad(d_loss,d_params)
    gradients=[g.detach() for g in (*gg,*dg)]
    if any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Nonfinite game gradient')
    def norm(gs): return torch.sqrt(sum(g.square().sum() for g in gs)).item()
    return gradients,{'g_loss':g_loss.item(),'d_loss':d_loss.item(),'gp':gp.item(),
                      'critic_gap':(real_score-fake_score).item(),
                      'input_grad_norm':norms.detach().mean().item(),
                      'g_grad_norm':norm(gg),'d_grad_norm':norm(dg)}
