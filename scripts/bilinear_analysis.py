"""Analytical GDA/EG/OGDA/LA-GDA models, not adaptive-Adam convergence claims."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',default='evidence/bilinear')
    p.add_argument('--eta',type=float,default=.3); p.add_argument('--steps',type=int,default=200)
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    eta=args.eta; A=np.array([[0.,1.],[-1.,0.]]); M=np.eye(2)-eta*A
    methods={'GDA':M,'EG':np.eye(2)-eta*A+eta**2*A@A,
             'LA-GDA k=2 alpha=.5':.5*np.eye(2)+.5*np.linalg.matrix_power(M,2),
             'LA-GDA k=5 alpha=.5':.5*np.eye(2)+.5*np.linalg.matrix_power(M,5)}
    trajectories={}; table=[]
    for method,B in methods.items():
        values=[np.array([1.,2.])]
        for _ in range(args.steps): values.append(B@values[-1])
        trajectories[method]=np.array(values)
        table.append({'method':method,'spectral_radius':float(max(abs(np.linalg.eigvals(B))))})
    values=[np.array([1.,2.])]; previous=np.zeros(2)
    for _ in range(args.steps):
        new=values[-1]-eta*(2*A@values[-1]-A@previous); previous=values[-1].copy(); values.append(new)
    trajectories['OGDA']=np.array(values)
    B=np.block([[np.eye(2)-2*eta*A,eta*A],[np.eye(2),np.zeros((2,2))]])
    table.append({'method':'OGDA','spectral_radius':float(max(abs(np.linalg.eigvals(B))))})
    error=abs(np.linalg.norm(trajectories['EG'][1])**2-(1-eta**2+eta**4)*5)
    result={'eta':eta,'steps':args.steps,'eg_identity_absolute_error':float(error),
            'scope':'fixed-step nonadaptive bilinear models','spectral_radii':table}
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    with (out/'spectral_radii.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method','spectral_radius']);w.writeheader();w.writerows(table)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for method,values in trajectories.items():
        inner=5 if 'k=5' in method else 2 if 'k=2' in method else 1
        calls=np.arange(len(values))* (2 if method=='EG' else inner)
        axes[0].semilogy(calls,np.linalg.norm(values,axis=1),label=method)
        if method!='GDA': axes[1].plot(values[:,0],values[:,1],label=method)
    axes[0].set(xlabel='Gradient-field evaluations',ylabel='Distance to equilibrium');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Min-player x',ylabel='Max-player y',aspect='equal');axes[1].legend(fontsize=7)
    fig.suptitle('Bilinear game: proven linear models (not Adam-family guarantees)')
    fig.savefig(out/'bilinear_dynamics.png',dpi=160);plt.close(fig); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
