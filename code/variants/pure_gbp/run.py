"""Native all-GBP PGO runs and paired comparisons to the unchanged main policy."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('hgbp_runner',ROOT/'benchmarks/run.py')
official=importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def run(name,variant,repeat,args,policy,catalog,coarse):
    record=catalog['datasets'][name]
    if record['suite']!='pgo':raise ValueError('pure_gbp is PGO-only')
    data=args.data_root/record['path']
    if official.sha256(data)!=record['sha256']:raise ValueError('Input SHA-256 mismatch: '+name)
    build=args.main_build if variant=='main' else args.build_dir
    exe=build/'pgo'/(record['space'].lower()+'_benchmark.exe')
    out=args.output_root/f'{name}_{variant}_t{args.threads}_r{repeat}'
    out.mkdir(parents=True,exist_ok=False)
    command=official.solver_command('pgo',policy,record['space'],exe,data,out/'result.json',args.threads,args.affinity_mask)
    command.remove('--write-poses')
    env,changes=official.environment_for(policy['profiles'][record['space']]['environment'],args.threads)
    env={k:v for k,v in env.items() if not k.startswith('COARSE_LAB_')}
    if variant=='pure':
        extra={'HGBP_PURE_'+key.upper():str(value) for key,value in coarse.items()}
        env.update(extra);changes.update(extra)
        if args.trace:env['HGBP_PURE_TRACE']=str(out/'coarse.jsonl')
    captured=dict(dataset=name,variant=variant,repeat=repeat,threads=args.threads,
                  input_sha256=record['sha256'],binary_sha256=official.sha256(exe),
                  command=command,environment=changes,coarse_policy=coarse if variant=='pure' else None)
    print(f'Start {name} {variant} repeat={repeat}',flush=True)
    start=time.perf_counter()
    try:
        with (out/'stdout.txt').open('w',encoding='utf-8') as log,official.process_affinity(args.affinity_mask):
            process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        captured['returncode']=process.returncode
    except subprocess.TimeoutExpired:
        captured.update(returncode=None,error='timeout')
    captured['process_sec']=time.perf_counter()-start
    if captured['returncode']==0:
        native=official.load_json(out/'result.json');history=native['mg_history']
        if len(history)!=21 or history[-1]['outer']!=20:raise ValueError('Incomplete outer20 run')
        backend=native['config']['coarse_linear_backend'] if record['space']=='SE3' else native['coarse_linear_backend']
        if (backend=='pure_gbp')!=(variant=='pure'):raise ValueError('Wrong coarse solver binary: '+str(exe))
        if record['space']=='SE3' and (native['config']['smoother']!='gbp' or native['config']['final_direct_polish_steps']!=0):
            raise ValueError('Unexpected fine solver/polish')
        captured.update(native_sec=native.get('hgbp_solver_wall_sec',native['solver_wall_sec']),
                        cost=history[-1]['nonlinear_objective'],native_config=native['config'],backend=backend,
                        sweeps=sum(row.get('full_precision_sweeps',0)+row.get('eta_only_sweeps',0) for row in history),
                        coarse_sec=sum(row.get('coarse_solve_sec',0) for row in history))
    official.write_json(out/'record.json',captured)
    print({key:captured.get(key) for key in ('dataset','variant','returncode','native_sec','cost')},flush=True)
    if captured['returncode']!=0:raise RuntimeError('Native run failed: '+str(out/'stdout.txt'))
    return captured


def summarize(records):
    rows=[]
    for name in dict.fromkeys(row['dataset'] for row in records):
        row=dict(dataset=name)
        for variant in ('main','pure'):
            selected=[r for r in records if r['dataset']==name and r['variant']==variant]
            if not selected:continue
            if len(set(r['binary_sha256'] for r in selected))!=1:raise ValueError('Binary changed during comparisons')
            row[variant]={key:statistics.median(r[key] for r in selected) for key in ('native_sec','process_sec','cost','coarse_sec','sweeps')}
            row[variant].update(runs=len(selected),time_range=[min(r['native_sec'] for r in selected),max(r['native_sec'] for r in selected)])
        if 'main' in row and 'pure' in row:
            row['pure_over_main']=row['pure']['native_sec']/row['main']['native_sec']
            row['cost_ratio']=row['pure']['cost']/row['main']['cost']
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets',nargs='*')
    parser.add_argument('--threads',type=int,choices=(1,16),default=16)
    parser.add_argument('--build-dir',type=Path,default=HERE/'build')
    parser.add_argument('--main-build',type=Path,help='also run the default coarse-direct binary from this build root')
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/pgo')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--repeats',type=official.positive_int,default=1)
    parser.add_argument('--timeout',type=official.positive_seconds,default=2000.)
    parser.add_argument('--affinity-mask',type=official.affinity_mask,default='0xffff')
    parser.add_argument('--trace',action='store_true',help='diagnostics only; do not mix trace timings with speed runs')
    args=parser.parse_args()
    args.build_dir=args.build_dir.resolve();args.data_root=args.data_root.resolve();args.output_root=args.output_root.resolve()
    if args.main_build:args.main_build=args.main_build.resolve()
    if args.output_root.exists():raise FileExistsError(args.output_root)
    catalog=official.load_json(ROOT/'benchmarks/configs/datasets.json')
    names=args.datasets or catalog['subsets']['pgo']['all']
    if len(names)!=len(set(names)):raise ValueError('Duplicate dataset names')
    policy=official.load_json(ROOT/'benchmarks/configs/pgo.json')
    coarse=official.load_json(HERE/'config.json')['coarse']
    args.output_root.mkdir(parents=True)
    official.write_json(args.output_root/'protocol.json',dict(fine_policy=policy,coarse_policy=coarse,threads=args.threads,
        repeats=args.repeats,trace=args.trace,affinity_mask=args.affinity_mask,timeout=args.timeout))
    variants=['main','pure'] if args.main_build else ['pure']
    records=[]
    for repeat in range(args.repeats):
        for name in names:
            for variant in variants if repeat%2==0 else variants[::-1]:
                records.append(run(name,variant,repeat,args,policy,catalog,coarse))
                official.write_json(args.output_root/'summary.json',summarize(records))
    official.write_json(args.output_root/'summary.json',summarize(records))
    print(json.dumps(summarize(records),indent=2),flush=True)


if __name__=='__main__':main()
