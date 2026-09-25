"""Retained vectorized SE3 scorer; only the module qualifier was localized."""
import numpy as np


def normalize(q):
    return q/np.linalg.norm(q,axis=-1,keepdims=True)


def conjugate(q):
    return q*np.array([-1.,-1.,-1.,1.])


def multiply(a,b):
    v=a[:,3,None]*b[:,:3]+b[:,3,None]*a[:,:3]+np.cross(a[:,:3],b[:,:3])
    w=a[:,3]*b[:,3]-np.einsum('ni,ni->n',a[:,:3],b[:,:3])
    return np.column_stack((v,w))


def rotate(q,v):
    cross=2*np.cross(q[:,:3],v)
    return v+q[:,3,None]*cross+np.cross(q[:,:3],cross)


def residual(i,j,z):
    qi,qj,qz=(normalize(p[:,3:]) for p in (i,j,z))
    q=normalize(multiply(conjugate(qz),multiply(conjugate(qi),qj)))
    q=np.where(q[:,3,None]<0,-q,q)
    length=np.linalg.norm(q[:,:3],axis=1)
    theta=2*np.arctan2(length,q[:,3])
    factor=np.full_like(theta,2.)
    np.divide(theta,length,out=factor,where=length>1e-15)
    phi=q[:,:3]*factor[:,None]
    t=rotate(conjugate(qz),rotate(conjugate(qi),j[:,:3]-i[:,:3])-z[:,:3])
    coefficient=1./12.+theta**2/720.+theta**4/30240.
    large=theta>1e-4
    angle=theta[large]
    coefficient[large]=(1.-.5*angle/np.tan(.5*angle))/angle**2
    first=np.cross(phi,t)
    translation=t-.5*first+coefficient[:,None]*np.cross(phi,first)
    return np.column_stack((translation,phi))


def load_g2o(path):
    vertices,edges={},[]
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            p=line.split()
            if not p or p[0].startswith('#'): continue
            if p[0]=='VERTEX_SE3:QUAT': vertices[int(p[1])]=list(map(float,p[2:9]))
            elif p[0]=='EDGE_SE3:QUAT': edges.append((int(p[1]),int(p[2]),list(map(float,p[3:]))))
            else: raise ValueError(f'Unexpected g2o tag {p[0]}')
    ids=sorted(vertices)
    lookup={key:index for index,key in enumerate(ids)}
    ends=np.array([[lookup[e[0]],lookup[e[1]]] for e in edges])
    values=np.array([e[2] for e in edges])
    info=np.zeros((len(edges),6,6))
    rows,cols=np.triu_indices(6)
    info[:,rows,cols]=values[:,7:]
    info[:,cols,rows]=values[:,7:]
    return np.array([vertices[i] for i in ids]),ends,values[:,:7],info


def score_se3(graph, pose):
    initial, ends, z, info = graph
    pose = np.asarray(pose, dtype=float)
    if pose.shape != initial.shape or not np.isfinite(pose).all():
        raise ValueError('Invalid pose dimensions or values')
    error = residual(pose[ends[:, 0]], pose[ends[:, 1]], z)
    q = np.einsum('ni,nij,nj->n', error, info, error)
    if not np.isfinite(q).all() or q.min() < -1e-8:
        raise ValueError('Invalid information quadratic')
    q = np.maximum(q, 0)
    norm = np.sqrt(q)
    anchor_error = residual(initial[:1], pose[:1], np.array([[0., 0., 0., 0., 0., 0., 1.]]))
    return dict(raw_objective=float(.5 * q.sum()),
                robust_objective=float(np.where(norm <= 5, .5 * q, 5 * (norm - 2.5)).sum()),
                anchor_cost=float(.5e8 * np.sum(anchor_error ** 2)))
