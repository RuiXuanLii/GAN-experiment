"""Checkpointable CPU RNG streams; deterministic batches indexed by oracle call.

Image data sampling uses replacement, so there are no implicit epoch/worker
states to lose on resume. Downloads occur ONLY with config download=true.
"""
import hashlib
import numpy as np
import torch


def split_indices(size: int, seed: int, validation_size: int):
    indices=np.random.default_rng(seed).permutation(size)
    return indices[validation_size:],indices[:validation_size]


class BatchStream:
    def __init__(self,config):
        self.config=config
        self.batches=0
        self.rng=torch.Generator(device='cpu').manual_seed(config['seed']+1000)
        self.device=torch.device(config['device']); self.dataset=None; self.indices=None
        self.split_hash='synthetic-radius2-std0.05-eightmodes'
        if config['dataset']!='synthetic':
            from torchvision import datasets,transforms
            root=config['data_root']; download=config.get('download',False)
            if config['dataset']=='cifar10':
                self.dataset=datasets.CIFAR10(root=root,train=True,download=download)
                val_size=5000
            elif config['dataset']=='mnist':
                self.dataset=datasets.MNIST(root=root,train=True,download=download)
                val_size=6000
            else: raise ValueError('Unknown dataset')
            self.indices,self.validation=split_indices(len(self.dataset),config['split_seed'],val_size)
            self.split_hash=hashlib.sha256(self.indices.astype('<i8').tobytes()).hexdigest()
            self.transform=transforms.Compose([
                transforms.Resize((32,32)),
                transforms.Grayscale(3) if config['dataset']=='mnist' else transforms.Lambda(lambda x:x),
                transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

    def next(self):
        self.batches+=1
        b=self.config['batch_size']; g=self.rng
        if self.dataset is None:
            modes=torch.randint(8,(b,),generator=g)
            angle=modes.to(torch.float32)*(2*torch.pi/8)
            real=2*torch.stack([angle.cos(),angle.sin()],dim=1)+.05*torch.randn(b,2,generator=g)
        else:
            chosen=torch.randint(len(self.indices),(b,),generator=g).tolist()
            real=torch.stack([self.transform(self.dataset[int(self.indices[i])][0]) for i in chosen])
        noise=torch.randn(b,self.config['z_dim'],generator=g)
        alpha=torch.rand((b,)+(1,)*(real.ndim-1),generator=g)
        return tuple(x.to(self.device) for x in (real,noise,alpha))

    def state_dict(self):
        return {'rng':self.rng.get_state().clone(),'split_hash':self.split_hash,'batches':self.batches}

    def load_state_dict(self,state):
        if state['split_hash']!=self.split_hash: raise ValueError('Dataset split mismatch')
        self.rng.set_state(state['rng'].cpu())
        self.batches=state['batches']
