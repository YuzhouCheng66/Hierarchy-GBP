from pathlib import Path
import sys,json,time
import numpy as np
import mpmath as mp
path=Path(sys.argv[1]);lines=iter(path.read_text().splitlines());own,ref,cutoff=map(float,next(lines).split());matrices=[]
for _ in range(5):
    rows,cols=map(int,next(lines).split());matrices.append(np.array([list(map(float,next(lines).split())) for _ in range(rows)]))
h,p,r,z,s=matrices
mp.mp.dps=70
def matrix(a):return mp.matrix([[mp.mpf(float(x)) for x in row] for row in a])
rh=matrix(r);zh=matrix(z);hh=matrix(h);ph=matrix(p)
out=dict(own_fp64=own,reference_fp64=ref,cutoff=cutoff,rows=h.shape[0],h_max=float(np.max(np.abs(h))),z_max=float(np.max(np.abs(z))),innovation_condition=float(np.linalg.cond(s)),precision_decimal_digits=mp.mp.dps)
for name,innovation in [('exact_root_model',mp.eye(len(r))+zh*zh.T),('exact_dense_prior_model',mp.eye(len(r))+hh*ph*hh.T),('rounded_dense_innovation',matrix(s))]:
    t=time.time();value=(rh.T*mp.lu_solve(innovation,rh))[0];out[name]=dict(chi=str(value),reject=value>cutoff,elapsed_s=time.time()-t)
    print(name,out[name],flush=True)
path.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
