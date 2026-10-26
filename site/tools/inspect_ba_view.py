"""Generate upright view candidates from the recorded BA geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    meta=json.loads((root/'data/ba.json').read_text())
    n=meta['points']+meta['cameras']
    xyz=np.fromfile(root/'data/ba-hgbp.bin',dtype='<f4').reshape(21,n,3)[-1,:meta['points']]
    # Estimate the vertical from the tower, not the sign-ambiguous PCA axes.
    tower=xyz[np.all((xyz>[-5,-135,-210])&(xyz<[65,-60,-45]),axis=1)]
    _,basis=np.linalg.eigh(np.cov(tower.T));up=basis[:,-1]
    if up[2]>0: up=-up
    right=np.array([1.,0.,0.]);right-=up*right.dot(up);right/=np.linalg.norm(right)
    forward=np.cross(right,up)
    view_basis=np.array([right,up,forward])
    print('Tower points:',len(tower),'basis:',view_basis.tolist(),'det:',np.linalg.det(view_basis))
    roi=np.all((xyz>[-250,-260,-240])&(xyz<[260,145,45]),axis=1)
    xyz=xyz[roi]@view_basis.T
    fig,axes=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    for ax,angle in zip(axes.flat,[-160,-135,-110,-45,20,45]):
        yaw=np.deg2rad(angle);pitch=np.deg2rad(8)
        x=xyz[:,0]*np.cos(yaw)+xyz[:,2]*np.sin(yaw)
        z=-xyz[:,0]*np.sin(yaw)+xyz[:,2]*np.cos(yaw)
        y=xyz[:,1]*np.cos(pitch)-z*np.sin(pitch)
        ax.scatter(x,y,c=xyz[:,1],s=.35,cmap='cividis',rasterized=True)
        ax.set_aspect('equal');ax.set_title(f'Tower upright, yaw={angle}');ax.grid(alpha=.15)
    fig.savefig(a.output,dpi=150)
    print(a.output)


if __name__=='__main__':main()
