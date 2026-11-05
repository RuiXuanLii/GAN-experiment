"""Export real references and generated uint8 PNGs from a trusted image checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ganopt.data import split_indices
from ganopt.models import make_models
from ganopt.runtime import source_hash,validate_image_export,METRIC_PROTOCOL


def to_uint8(samples):
    """FP32 NCHW [-1,1] -> exact N H W C uint8 image convention."""
    if samples.ndim!=4 or samples.shape[1]!=3 or not torch.isfinite(samples).all():
        raise ValueError('Expected finite NCHW RGB samples')
    return ((samples.detach().cpu().clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(0,2,3,1).numpy()


def export(checkpoint,output,data_root,split='validation',count=2000,seed=123456,device='cpu',download=False):
    checkpoint=Path(checkpoint); output=Path(output)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False); c=state['config']
    if c['dataset'] not in ['cifar10','mnist']: raise ValueError('Requires an image-model checkpoint')
    if state['source_hash']!=source_hash(): raise ValueError('Checkpoint source mismatch')
    if split not in ['validation','test']: raise ValueError('Unknown reference split')
    validate_image_export(c,split,count,seed,state['oracle_calls'])
    if count<1000: raise ValueError('Image metric sample count must be >=1000')
    if output.exists() and any(output.iterdir()): raise FileExistsError('Export directory already exists')
    from torchvision import datasets,transforms
    Dataset=datasets.CIFAR10 if c['dataset']=='cifar10' else datasets.MNIST
    real_data=Dataset(root=data_root,train=split!='test',download=download)
    if split=='validation':
        _,indices=split_indices(len(real_data),c['split_seed'],5000 if c['dataset']=='cifar10' else 6000)
    else: indices=np.arange(len(real_data))
    if count>len(indices): raise ValueError('Requested more real images than reference split contains')
    indices=indices[:count]
    G,_=make_models(c); G.load_state_dict(state['generator']); G.to(device).eval()
    if c['dataset']=='mnist': transform=transforms.Compose([transforms.Resize((32,32)),transforms.Grayscale(3)])
    else: transform=lambda x:x
    for folder in ['real','fake']: (output/folder).mkdir(parents=True,exist_ok=True)
    for i,index in enumerate(indices): transform(real_data[int(index)][0]).save(output/'real'/f'{i:06d}.png')
    rng=torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        for start in range(0,count,64):
            z=torch.randn(min(64,count-start),c['z_dim'],generator=rng).to(device)
            for j,array in enumerate(to_uint8(G(z))): Image.fromarray(array).save(output/'fake'/f'{start+j:06d}.png')
    manifest={'dataset':c['dataset'],'reference_split':split,'count':count,'evaluation_seed':seed,
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'source_hash':state['source_hash'],'config':c,'oracle_calls':state['oracle_calls'],
              'metric_protocol':METRIC_PROTOCOL,
              'real_indices_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)); return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--data-root',default='data')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--count',type=int,default=2000);p.add_argument('--seed',type=int,default=123456)
    p.add_argument('--device',default='cpu');p.add_argument('--download',action='store_true')
    a=p.parse_args();print(export(a.checkpoint,a.output,a.data_root,a.split,a.count,a.seed,a.device,a.download))


if __name__=='__main__': main()
