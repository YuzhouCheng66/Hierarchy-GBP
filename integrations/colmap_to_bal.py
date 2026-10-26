"""Export supported COLMAP text reconstructions to standard, per-image BAL."""
import argparse
import math
from pathlib import Path


def rotation_vector(q):
    # F*R(q), F=diag(1,-1,-1), is a pi rotation about X times q.
    w, x, y, z = q
    w, x, y, z = -x, w, -z, y
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if not norm: raise ValueError('Invalid quaternion')
    w,x,y,z = (v/norm for v in (w,x,y,z))
    if w < 0: w,x,y,z = -w,-x,-y,-z
    s = math.sqrt(x*x+y*y+z*z)
    angle = 2*math.atan2(s,w)
    return [v*angle/s for v in (x,y,z)] if s>1e-15 else [0.,0.,0.]


def convert(folder, output, allow_independent):
    cameras={}
    for line in (folder/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        cid,model,_,_,*params=line.split(); params=list(map(float,params))
        if model not in ('SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL'):
            raise ValueError(f'{model} is not exactly represented by the supported BAL camera model')
        f,cx,cy=params[:3]; k1=params[3] if len(params)>3 else 0.; k2=params[4] if len(params)>4 else 0.
        cameras[int(cid)]=(f,cx,cy,k1,k2)
    points={}
    for line in (folder/'points3D.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        values=line.split(); points[int(values[0])]=list(map(float,values[1:4]))
    # Image records occupy two lines; the observation line is allowed to be empty.
    lines=(folder/'images.txt').read_text().splitlines(); images=[]; i=0
    while i<len(lines):
        line=lines[i]; i+=1
        if not line.strip() or line.lstrip().startswith('#'): continue
        values=line.split(maxsplit=9)
        if len(values)<10 or i>=len(lines): raise ValueError('Malformed COLMAP image record')
        obs=lines[i].split(); i+=1
        if len(obs)%3: raise ValueError('Malformed COLMAP observation list')
        images.append((int(values[0]),list(map(float,values[1:5])),list(map(float,values[5:8])),int(values[8]),obs))
    if not allow_independent and len({v[3] for v in images})!=len(images):
        raise ValueError('Shared intrinsics detected. BAL optimizes independent per-image cameras; explicitly pass --independent-intrinsics to accept this change.')
    images.sort(key=lambda v:v[0]); point_ids=sorted(points); index={p:i for i,p in enumerate(point_ids)}
    observations=[]; parameters=[]
    for ci,(_,q,t,cid,obs) in enumerate(images):
        f,cx,cy,k1,k2=cameras[cid]
        parameters.extend(rotation_vector(q)+[t[0],-t[1],-t[2],f,k1,k2])
        for j in range(0,len(obs),3):
            x,y,p=obs[j:j+3]; p=int(p)
            if p==-1: continue
            if p not in index: raise ValueError('Observation references missing point')
            observations.append((ci,index[p],float(x)-cx,cy-float(y)))
    with output.open('x') as out:
        out.write(f'{len(images)} {len(points)} {len(observations)}\n')
        for c,p,x,y in observations: out.write(f'{c} {p} {x:.17g} {y:.17g}\n')
        for value in parameters: out.write(f'{value:.17g}\n')
        for p in point_ids:
            for value in points[p]: out.write(f'{value:.17g}\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text_model',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--independent-intrinsics',action='store_true',help='Accept per-image intrinsics when COLMAP shares camera IDs')
    args=parser.parse_args(); convert(args.text_model,args.output,args.independent_intrinsics)
