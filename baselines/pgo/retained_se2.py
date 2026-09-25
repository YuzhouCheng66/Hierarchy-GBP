"""Retained SE2 scoring functions; provenance in scoring_provenance.json."""
from pathlib import Path
import numpy as np


def residual(x,y,z):
    c,s=np.cos(x[:,2]),np.sin(x[:,2]); d=y[:,:2]-x[:,:2]
    px=c*d[:,0]+s*d[:,1]-z[:,0]; py=-s*d[:,0]+c*d[:,1]-z[:,1]
    c,s=np.cos(z[:,2]),np.sin(z[:,2]); tx=c*px+s*py; ty=-s*px+c*py
    theta=np.arctan2(np.sin(y[:,2]-x[:,2]-z[:,2]),np.cos(y[:,2]-x[:,2]-z[:,2]))
    a=np.ones_like(theta); b=np.zeros_like(theta); mask=np.abs(theta)>=1e-12
    a[mask]=np.sin(theta[mask])/theta[mask]; b[mask]=(1-np.cos(theta[mask]))/theta[mask]
    return np.column_stack(((a*tx+b*ty)/(a*a+b*b),(-b*tx+a*ty)/(a*a+b*b),theta))


def load_graph(path):
    vertices={}; edges=[]
    for line in Path(path).read_text().splitlines():
        p=line.split()
        if not p: continue
        if p[0]=='VERTEX_SE2': vertices[int(p[1])]=list(map(float,p[2:5]))
        elif p[0]=='EDGE_SE2': edges.append((int(p[1]),int(p[2]),list(map(float,p[3:6])),list(map(float,p[6:12]))))
    ids=sorted(vertices); order={v:i for i,v in enumerate(ids)}
    return (np.array([vertices[v] for v in ids]),np.array([order[e[0]] for e in edges]),
            np.array([order[e[1]] for e in edges]),np.array([e[2] for e in edges]),np.array([e[3] for e in edges]))


def evaluate(graph,poses):
    init,i,j,z,w=graph; poses=np.asarray(poses,dtype=float)
    assert poses.shape==init.shape and np.all(np.isfinite(poses))
    r=residual(poses[i],poses[j],z); x,y,t=r.T
    q=w[:,0]*x*x+2*w[:,1]*x*y+2*w[:,2]*x*t+w[:,3]*y*y+2*w[:,4]*y*t+w[:,5]*t*t
    if q.min() < -1e-9: raise ValueError('Negative information quadratic')
    norms=np.sqrt(np.maximum(q,0))
    anchor_r=residual(init[:1],poses[:1],np.zeros((1,3)))
    anchor=float(.5e8*np.sum(anchor_r*anchor_r))
    return dict(raw_objective=float(.5*q.sum()),robust_objective=float(np.where(norms<=5,.5*q,5*(norms-2.5)).sum()),
                anchor_cost=anchor,active_huber_edges=int((norms>5).sum()),max_mahalanobis_norm=float(norms.max()))
