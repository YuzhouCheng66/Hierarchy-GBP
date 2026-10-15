"""Archive paired native results, including failures, without build products."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=HERE/'evidence')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for directory in args.runs:
        records.extend(json.loads(path.read_text()) for path in sorted(directory.glob('*/record.json')))
    rows=[]
    for name in dict.fromkeys(r['dataset'] for r in records):
        row={'dataset':name}
        for variant in ('main','pure'):
            selected=[r for r in records if r['dataset']==name and r['variant']==variant]
            valid=[r for r in selected if r['returncode']==0]
            value={'runs':len(selected),'completed':len(valid),'binary_sha256':sorted(set(r['binary_sha256'] for r in selected))}
            if valid:
                value.update({key:statistics.median(r[key] for r in valid) for key in ('native_sec','process_sec','cost','coarse_sec','sweeps')})
                value['time_range']=[min(r['native_sec'] for r in valid),max(r['native_sec'] for r in valid)]
            value['failures']=[{'error':r.get('error','native_error'),'returncode':r['returncode'],'process_sec':r['process_sec']} for r in selected if r['returncode']!=0]
            row[variant]=value
        if row['main']['completed'] and row['pure']['completed']:
            row['pure_over_main']=row['pure']['native_sec']/row['main']['native_sec']
            row['cost_ratio']=row['pure']['cost']/row['main']['cost']
        rows.append(row)
    comparable=[r for r in rows if 'pure_over_main' in r]
    if not comparable:raise ValueError('No completed main/pure pairs to summarize')
    summary={'metric':'native optimization wall seconds; ratio=pure/main',
             'threads':sorted(set(r['threads'] for r in records)),
             'geometric_mean_ratio':math.exp(statistics.mean(math.log(r['pure_over_main']) for r in comparable)),
             'ratio_range':[min(r['pure_over_main'] for r in comparable),max(r['pure_over_main'] for r in comparable)],
             'maximum_cost_ratio':max(r['cost_ratio'] for r in comparable),'datasets':rows}
    write(args.output/'summary.json',summary);write(args.output/'runs.json',records)
    files={ROOT/'CMakeLists.txt',ROOT/'code/CMakeLists.txt',ROOT/'benchmarks/configs/pgo.json',ROOT/'benchmarks/configs/datasets.json',ROOT/'benchmarks/configs/data_release.json'}
    files.update((ROOT/'code/include').rglob('*.h'))
    for suffix in ('*.cpp','*.h'):files.update((ROOT/'code/src').glob(suffix))
    files.update((ROOT/'code/tests').glob('*.cpp'))
    files.update(path for path in HERE.rglob('*') if path.is_file() and path.suffix in ('.h','.cpp','.py','.ps1','.json','.cmake')
                 and not any(p in ('build','results','evidence','__pycache__') or p.startswith('build-') for p in path.relative_to(HERE).parts))
    files.add(HERE/'CMakeLists.txt')
    write(args.output/'source_sha256.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    with zipfile.ZipFile(args.output/'native_results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for directory in args.runs:
            for path in sorted(directory.rglob('*')):
                if path.is_file() and path.name in ('result.json','stdout.txt','protocol.json'):
                    archive.write(path,directory.name+'/'+path.relative_to(directory).as_posix())
    print(json.dumps({key:value for key,value in summary.items() if key!='datasets'},indent=2))
