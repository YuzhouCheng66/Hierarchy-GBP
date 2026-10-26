"""Pack verified, real BA captures for a shared-clock browser replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load(folder):
    meta=read(folder/'scene.json')
    n=meta['points']+meta['cameras']
    xyz=np.fromfile(folder/'scene.bin',dtype='<f4').reshape(-1,n,3).astype(float)
    assert len(xyz)==len(meta['frames'])==21
    assert meta['frames']==list(range(21)) and np.isfinite(xyz).all()
    return meta,xyz


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['hgbp','powerba','hgbp-clock','powerba-clock','input','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    hm,h=load(a.hgbp); pm,power=load(a.powerba)
    assert hm==pm
    # The solvers use different global normalization scales. Determine one
    # constant from the common initial geometry, then apply it to every frame.
    norms=np.linalg.norm(power[0],axis=1)
    valid=(norms>1e-4)&(norms<1e6)
    display_scale=float(np.median(np.linalg.norm(h[0,valid],axis=1)/norms[valid]))
    power*=display_scale
    initial_error=float(np.max(np.linalg.norm(h[0]-power[0],axis=1)/np.maximum(1,np.linalg.norm(h[0],axis=1))))
    assert initial_error<2e-6, initial_error
    hn=read(a.hgbp/'native.json'); pn=read(a.powerba/'ba_log.json')
    hc=read(a.hgbp_clock); pc=read(a.powerba_clock)
    assert abs(hn['final_cost']/hc['final_cost']-1)<1e-6
    assert abs(pn['cost'][-1]/pc['cost'][-1]-1)<1e-8
    # A single common rigid display transform; never a per-frame alignment.
    center=np.median(h[-1],axis=0)
    centered=h[-1]-center
    inside=np.all(np.abs(centered)<np.quantile(np.abs(centered),.98,axis=0),axis=1)
    _,vectors=np.linalg.eigh(centered[inside].T@centered[inside])
    rotation=vectors[:,::-1]
    if np.linalg.det(rotation)<0: rotation[:,2]*=-1
    for name,xyz in [('ba-hgbp.bin',h),('ba-powerba.bin',power)]:
        ((xyz-center)@rotation).astype('<f4').tofile(a.output/name)
    def times(values,total):
        values=np.asarray(values,dtype=float)
        return [0.,*(np.cumsum(values)/values.sum()*total).tolist()]
    power_cost=[]
    for cost,success in zip(pc['cost'],pc['step_is_successful']):
        power_cost.append(2*cost if success else power_cost[-1])
    meta=dict(dataset='Venice1778',view='ba-view.json',points=hm['points'],cameras=hm['cameras'],
        full_points=hm['landmarks'],observations=5001946,
        input_sha256=hashlib.file_digest(a.input.open('rb'),'sha256').hexdigest(),
        hgbp=dict(binary='ba-hgbp.bin',frames=list(range(21)),times=times(hc['outer_sec'],5.89),costs=hc['costs'],total=5.89),
        baseline=dict(name='PowerBA',threads=16,binary='ba-powerba.bin',frames=list(range(21)),
            times=times(np.diff([0,*pc['cumulative_time'][1:]]),20.32),costs=power_cost,total=20.32),
        provenance=dict(timing='Archived native outer-time fractions calibrated to paper Table 2 totals (H-GBP 5.89 s, PowerBA 20.32 s). Geometry capture overhead is not included; this is not a new runtime measurement.',
            geometry='Actual accepted camera/landmark states captured after each of 20 outer iterations. Rejected PowerBA trials retain the preceding accepted geometry. Capture final costs reproduce the reported configurations.',
            interpolation='Linear display interpolation between recorded outer states; never used to compute costs or report convergence.',
            sampling=f'{hm["points"]:,} uniform landmark-index samples and all 1,778 camera centers. One fixed initial-frame scale aligns the solvers\' global gauges, followed by a shared fixed PCA display rotation. No per-frame alignment.',
            initial_geometry_max_difference=initial_error,
            baseline_display_scale=display_scale,display_rotation=rotation.tolist(),paper='https://arxiv.org/abs/2610.06978v1'))
    (a.output/'ba.json').write_text(json.dumps(meta,indent=2)+'\n', newline='\n')
    print(f'BA replay: initial max difference {initial_error}; final costs {hn["final_cost"]}, {pn["cost"][-1]*2}')


if __name__=='__main__': main()
