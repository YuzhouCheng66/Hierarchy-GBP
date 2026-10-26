"""Independently evaluate an exported Snavely BAL file (NumPy required)."""
import argparse
import json
from pathlib import Path
import numpy as np


def evaluate(path):
    with path.open() as stream:
        nc,np_,no=map(int,stream.readline().split())
        obs=np.array([list(map(float,stream.readline().split())) for _ in range(no)])
        values=np.fromstring(stream.read(),sep=' ')
    assert len(values)==nc*9+np_*3
    cameras=values[:nc*9].reshape(nc,9);points=values[nc*9:].reshape(np_,3)
    ci=obs[:,0].astype(int); pi=obs[:,1].astype(int)
    R=[]
    for camera in cameras:
        aa=camera[:3];angle=np.linalg.norm(aa)
        axis=aa/angle if angle else np.array([1.,0.,0.])
        x,y,z=axis; K=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        R.append(np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K))
    pc=np.einsum('nij,nj->ni',np.asarray(R)[ci],points[pi])+cameras[ci,3:6]
    uv=-pc[:,:2]/pc[:,2,None];r2=np.sum(uv**2,axis=1)
    distortion=1+cameras[ci,7]*r2+cameras[ci,8]*r2**2
    projected=cameras[ci,6,None]*distortion[:,None]*uv
    errors=np.sum((projected-obs[:,2:4])**2,axis=1)
    return float(errors.sum())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bal',type=Path);p.add_argument('native_metrics',type=Path)
    a=p.parse_args();cost=evaluate(a.bal);reference=json.loads(a.native_metrics.read_text())['final_cost']
    error=abs(cost/reference-1)
    assert error<1e-8,(cost,reference,error)
    print(f'Exported BAL reprojection cost {cost:.12g}; native {reference:.12g}; relative error {error:.3g}')
