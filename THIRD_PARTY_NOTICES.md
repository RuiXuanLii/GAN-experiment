# Academic attribution and external code

Custom algorithm implementations were written for this comparison based on:

- Kingma and Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015, https://arxiv.org/abs/1412.6980.
- Gidel et al., *A Variational Inequality Perspective on Generative Adversarial Networks*, ICLR 2019, https://arxiv.org/abs/1802.10551, Algorithm 4.
- Daskalakis et al., *Training GANs with Optimism*, ICLR 2018, https://arxiv.org/abs/1711.00141, Algorithm 1.
- Chavdarova et al., *Taming GANs with Lookahead-Minmax*, ICLR 2021, https://arxiv.org/abs/2006.14567.
- Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017, https://arxiv.org/abs/1704.00028.

`tests/reference/gidel_extragradient.py` is the author's unmodified MIT-licensed reference, downloaded from https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py . The full copyright/permission text remains in that file. Git blob SHA: d1aabae4e22b9b5ad80051a81a4cf0e260f42e14. Only used for a reference parity test; production code does not import it. Historical epsilon convention differs from modern corrected-moment convention, as documented in the mathematical notes.

The original LA/optimistic repositories were inspected for state conventions but are not redistributed wholesale. This is a controlled synchronous variant and does not claim benchmark reproduction of their original architecture and training schedule.

FID/KID evaluation uses torch-fidelity with its default pretrained Inception-compatible features: https://torch-fidelity.readthedocs.io/ . Cite that implementation and the original metric papers in the final report. No pretrained weights or CIFAR-10 images are redistributed in this starter package.
