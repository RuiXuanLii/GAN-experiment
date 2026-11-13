# Round 3 algorithm and source review

Date: 2026-11-13. This review concerns definitions and experimental interpretation; it reports no new training results.

## Decision and scope

The primary comparison retains the existing **joint synchronous 1:1** implementations of Adam, ExtraAdam, Optimistic Adam and Lookahead-Minmax. Use the common hidden128/ReLU/GP0.1/batch64 setting and charge every full joint field evaluation. The successful archived Adam five-D-then-one-G runs are a separately labelled schedule/capability reference, not an additional contestant in an optimizer-only ranking. A successful alternating Adam reference does not imply that synchronous Adam must succeed.

The shared setting was developed through exploratory Adam diagnostics. Reporting seeds 11, 22 and 33 are new to this hidden128 setting; they are not described as never used anywhere in the project. This is a controlled eight-Gaussian comparison, not reproduction of the papers' image benchmarks.

## Exact implemented updates

Let w=(theta,phi) concatenate G and D parameters. The field is F(w)=(partial_theta L_G, partial_phi L_D); both blocks are evaluated before either is changed, and both losses use descent signs. GP appears only in L_D with generated samples held fixed for its critic partial derivative. Therefore the implemented WGAN-GP field is not generally the gradient field of one zero-sum objective.

Define A(g,s)=(u,s') as one Adam transition. For each parameter, n'=n+1, m'=beta1*m+(1-beta1)*g, v'=beta2*v+(1-beta2)*g^2, and u=[m'/(1-beta1^n')]/[sqrt(v'/(1-beta2^n'))+epsilon]. Here n counts actual moment transitions, not accepted parameter steps. Fixed betas=(0.5,0.9), epsilon=1e-8 and a constant learning rate are used.

| Method | Existing semantics |
| --- | --- |
| Adam | Compute (u,s')=A(F(w),s); accept w-eta*u jointly. |
| ExtraAdam | Save w. Evaluate F(w), advance both players' moments, and predict both blocks jointly. Evaluate the full field at that joint predicted point, advance both moment states again, and accept **original w minus eta times the second direction**. Prediction moments are retained; the correction is not taken from the predicted parameters. Main setting uses independent batches for the two evaluations. |
| Optimistic Adam | Compute the current Adam direction u and accept w-eta*(2*u-u_previous). Store u itself, not the raw gradient or the doubled update. Initial previous direction is zero. Both player histories advance once per joint step. |
| Lookahead-Minmax | Run five synchronous joint Adam inner updates from joint slow weights. Interpolate both players together: slow <- slow+0.5*(fast-slow), then fast <- slow. Retain Adam moments across interpolation. Before a synchronization the model contains fast parameters; every 6k evaluation and the 120k endpoint are synchronization boundaries. |

The corrected-moment epsilon convention matches the displayed equations. Historical author ExtraAdam code places epsilon before variance bias correction; the existing author-reference test uses epsilon=0 to isolate algorithm parity. See `docs/mathematical-notes-zh.md` for that implementation distinction.

## Why five-to-one is not an automatic substitution

Applying separate ExtraAdam instances to the active player freezes the opponent during prediction. Its correction consequently does not evaluate the original full joint prediction, even if discarded opponent gradients were computed and counted. Applying separate Lookahead instances also changes when the two players interpolate and which backtracked opponent each subsequent update observes.

Lookahead-Minmax does support unequal update ratios when its **joint** interpolation follows complete five-D-then-G cycles (Chavdarova Appendix G, Algorithms 3/5). This does not make an independent-player wrapper equivalent. A common multirate ExtraAdam comparison needs a separately specified prediction/correction construction. Round 3 avoids introducing that additional algorithmic variable.

## Compute, tuning and reporting

Each primary run receives 120,000 full field calls = 240,000 player partial-gradient calls. Adam, Optimistic Adam and LA each perform 120,000 joint inner updates; ExtraAdam performs 60,000 accepted joint updates. Every player's Adam moment clock ends at 120,000 in all four methods. LA additionally performs 24,000 joint interpolations without extra field calls. Equal oracle work does not mean equal parameter-update counts or equal elapsed time.

The final tuning plan gives every method the same three learning rates {1e-4, 3e-4, 1e-3}, seed9001, and 120k budget: 12 candidates. Freeze selections only after every candidate has completed or its numerical failure has been recorded. Hold LA k/alpha and all other shared settings fixed. Select using the prescribed last-five-checkpoint criterion at 6k cadence; do not select a best single checkpoint. This is an equal, limited learning-rate search, not exhaustive optimizer tuning. Run the selected configuration for every method on all three reporting seeds at 120k. A shared-rate comparison, if shown, answers a different controlled-hyperparameter question and must be labelled separately.

Report individual-seed endpoints, aggregate variability, and the last five checkpoints (96k, 102k, 108k, 114k, 120k). Keep coverage and quality together, alongside nearest-center distance and per-mode counts. Preserve the established 0.15 quality threshold, 1% coverage threshold and fixed 2,048 evaluation-noise samples. Retain failed candidates and seeds. Synthetic quality is not FID/KID, a convergence proof or evidence of CIFAR-10 performance. Concurrent CPU timings are not a reliable speed ranking.

## Primary sources checked

Page numbers below are printed paper pages, starting at 1.

- Kingma and Ba, *Adam: A Method for Stochastic Optimization*: [PDF](https://arxiv.org/pdf/1412.6980), Algorithm 1, p.2: moment updates and bias correction.
- Gidel et al., *A Variational Inequality Perspective on Generative Adversarial Networks*: [PDF](https://arxiv.org/pdf/1802.10551), Section 5 / Algorithm 4, p.8: standard ExtraAdam prediction, two moment transitions, and correction from the original iterate. The text explicitly does not provide a convergence proof for ExtraAdam. Author code: [Variational-Inequality-GAN](https://github.com/GauthierGidel/Variational-Inequality-GAN).
- Daskalakis et al., *Training GANs with Optimism*: [PDF](https://arxiv.org/pdf/1711.00141), Section 6 / Algorithm 1, pp.8-9: optimism on adaptive directions. The empirical discussion compares 1:1 OAdam with 1:1 and 1:5 Adam; it does not require the baseline's preferred schedule to be imposed on OAdam.
- Chavdarova et al., *Taming GANs with Lookahead-Minmax*: [PDF](https://fleuret.org/papers/chavdarova-et-al-iclr2021.pdf), Algorithm 1, p.4; Appendix G Algorithm 3, p.22; Algorithm 5, p.25; Section G.2, p.26: joint interpolation, unequal player-update ratios, retained Adam state, and the distinction from separately applying Lookahead to players. The current synchronous Adam inner method is a controlled variant of that framework. Author code: [LAGAN-Lookahead_Minimax](https://github.com/Chavdarova/LAGAN-Lookahead_Minimax).

## Verification required before interpreting results

Confirm existing joint-point/gradient-sign/GP tests; Adam parity; two-call ExtraAdam prediction, base-point correction and moment clocks; adaptive-direction optimism history; joint LA synchronization without moment resets; same-seed initialization and call-indexed data-stream pairing; exact budget counts; and checkpoint-resume equivalence including moments, history, fast/slow weights and data state. Numerical failures must remain visible rather than silently receiving a new seed or extra budget.
