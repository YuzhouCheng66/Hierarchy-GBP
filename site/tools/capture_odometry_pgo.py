"""Replay Globe10k from measured odometry, without altering graph measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.common import process_affinity
from benchmarks.run import solver_command, environment_for
from replay_pgo import graph


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input','work','hgbp','g2o','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reuse-captures',action='store_true',help='Repack and revalidate existing captures without rerunning solvers')
    a=p.parse_args()
    if not a.reuse_captures: a.work.mkdir(parents=True,exist_ok=False)
    seed=a.work/'Globe10k-odometry.g2o'
    if not a.reuse_captures:
        subprocess.run([sys.executable,str(ROOT/'benchmarks/datasets/reseed_globe_odometry.py'),
            '--input',str(a.input),'--output',str(seed)],check=True)
    def edges(path): return [s for s in path.read_text().splitlines() if s.startswith('EDGE_')]
    assert sorted(edges(seed))==sorted(edges(a.input))
    config=read(ROOT/'benchmarks/configs/pgo.json')
    profile=config['profiles']['SE3']
    env,_=environment_for(profile['environment'],16)
    command=solver_command('pgo',config,'SE3',a.hgbp.resolve(),seed,a.work/'hgbp.json',16,'0xffff')
    if not a.reuse_captures:
        with process_affinity('0xffff'),(a.work/'hgbp.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=500)
    env,_=environment_for({'OMP_NUM_THREADS':'{threads}','OMP_DYNAMIC':'FALSE','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},1)
    command=[str(a.g2o.resolve()),'--input',str(seed),'--output-json',str(a.work/'g2o.json'),'--num-outer','20','--huber-delta','5']
    if not a.reuse_captures:
        print('Running continuous g2o reference (20 outer)',flush=True)
        with process_affinity('0xffff'),(a.work/'g2o.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    native=read(a.work/'g2o.json');h=read(a.work/'hgbp.json');poses=read(a.work/'hgbp_poses.json')
    vertices,loops=graph(seed);ids=[v[0] for v in vertices]
    selected=list(range(len(ids)))
    loop_sample=[loops[i] for i in np.linspace(0,len(loops)-1,450,dtype=int)]
    lookup={ids[i]:j for j,i in enumerate(selected)}
    states=[np.array([vertices[i][1:] for i in selected],dtype='<f4')];frames=[0];previous=seed
    for outer in range(1,9):
        output=a.work/f'frame-{outer}.json'
        command=[str(a.g2o.resolve()),'--input',str(previous),'--output-json',str(output),'--num-outer','1','--huber-delta','5']
        if not a.reuse_captures:
            with process_affinity('0xffff'),(a.work/f'frame-{outer}.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150)
        actual=read(output)['final_objective']['raw'];reference=native['outer_iterations'][outer-1]['raw_cost']
        assert abs(actual/reference-1)<1e-7,(outer,actual,reference)
        previous=output.with_suffix('.g2o');v,_=graph(previous)
        assert [v_[0] for v_ in v]==ids
        states.append(np.array([v[i][1:] for i in selected],dtype='<f4'));frames.append(outer)
        print(f'Checkpoint {outer}: cost {actual:.8g}',flush=True)
    final,_=graph((a.work/'g2o.json').with_suffix('.g2o'))
    states.append(np.array([final[i][1:] for i in selected],dtype='<f4'));frames.append(20)
    np.stack(states).tofile(a.output/'pgo-g2o.bin')
    np.asarray([[frame[i][:3] for i in selected] for frame in poses['mg_pose_history']],dtype='<f4').tofile(a.output/'pgo-hgbp.bin')
    ht=[r['outer_total_sec'] for r in h['mg_history'][1:]];dt=[r['timing']['total_sec'] for r in native['outer_iterations']]
    meta=dict(dataset='Globe10k',nodes=len(ids),factors=len(edges(seed)),points=len(selected),cameras=0,
        loop_edges=[[lookup[i],lookup[j]] for i,j in loop_sample],input_sha256=hashlib.file_digest(seed.open('rb'),'sha256').hexdigest(),
        source_input_sha256=hashlib.file_digest(a.input.open('rb'),'sha256').hexdigest(),
        hgbp=dict(binary='pgo-hgbp.bin',frames=list(range(21)),times=[0.,*np.cumsum(ht).tolist()],
            costs=[r['nonlinear_objective'] for r in h['mg_history']],total=sum(ht)),
        baseline=dict(binary='pgo-g2o.bin',name='g2o-Cholesky',threads=1,frames=frames,times=[0.,*np.cumsum(dt).tolist()],
            costs=[native['initial_objective']['raw'],*[r['raw_cost'] for r in native['outer_iterations']]],total=sum(dt)),
        initialization='Integrated consecutive-edge odometry; all measurements and information matrices unchanged.',
        provenance=dict(timing='Native per-outer solver wall times from fresh continuous 20-outer runs on the common odometry initialization. These demo timings are separate from the paper table.',
            geometry='Recorded H-GBP states for every outer; g2o keyframes 0 through 8 and 20, verified against the continuous run. The two methods start from identical integrated odometry.',
            interpolation='Display-only interpolation between recorded keyframes; cost is evaluated only at completed iterations.',
            sampling='All 10,000 vertices and 450 uniformly sampled loop closures. No artificial drift or noise added.',
            paper='https://arxiv.org/abs/2610.06978v1'))
    (a.output/'pgo.json').write_text(json.dumps(meta,indent=2)+'\n',newline='\n')
    print(f'PGO odometry demo: H-GBP {sum(ht):.4f}s cost {meta["hgbp"]["costs"][-1]:.8g}; g2o {sum(dt):.4f}s cost {meta["baseline"]["costs"][-1]:.8g}',flush=True)


if __name__=='__main__':main()
