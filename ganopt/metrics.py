"""Explicit synthetic diagnostics; never substitutes for pretrained image FID."""
import numpy as np


def synthetic_metrics(points):
    points=np.asarray(points)
    if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected finite nonempty N x 2 generated points')
    angles=np.arange(8)*2*np.pi/8
    centers=2*np.stack([np.cos(angles),np.sin(angles)],axis=1)
    dist=np.linalg.norm(points[:,None,:]-centers[None,:,:],axis=2)
    assign=dist.argmin(axis=1); nearest=dist.min(axis=1); good=nearest<.15
    counts=np.bincount(assign[good],minlength=8)
    covered=int((counts>=.01*len(points)).sum())
    proportions=np.bincount(assign,minlength=8)/len(points)
    nz=proportions[proportions>0]
    return {'covered_modes':covered,'quality_fraction':float(good.mean()),
            'mean_nearest_distance':float(nearest.mean()),
            'assignment_entropy':float(-(nz*np.log(nz)).sum())}
