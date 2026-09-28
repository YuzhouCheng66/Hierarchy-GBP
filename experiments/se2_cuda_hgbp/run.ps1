param(
    [Parameter(Mandatory=$true)][string]$DataDirectory,
    [string[]]$Datasets=@('FR079','FRH','M3500'),
    [ValidateSet('reference','transportsegment')][string[]]$Variants=@('reference','transportsegment'),
    [ValidateRange(0,100)][int]$Repeats=9,
    [string]$OutputDirectory='results/timing',
    [switch]$Shadow,
    [switch]$ProfileClock,
    [string]$NsightSystems='',
    [ValidateSet('graph','node')][string]$NsightGraphTrace='graph',
    [ValidateRange(0,3600)][int]$NsightDurationSeconds=0,
    [ValidateSet('','memcheck','synccheck','racecheck')][string]$Sanitizer='',
    [ValidateRange(0,2147483647)][int]$SanitizerLaunchCount=0,
    [ValidateRange(0,2147483647)][int]$SanitizerLaunchSkip=0
)
$ErrorActionPreference='Stop'
if($NsightSystems -and ($Sanitizer -or $Shadow -or $ProfileClock)) { throw 'Run profilers separately' }
if($NsightDurationSeconds -and !$NsightSystems) { throw 'Duration requires Nsight Systems' }
if(($SanitizerLaunchCount -or $SanitizerLaunchSkip) -and !$Sanitizer) { throw 'Launch limits require a sanitizer' }
$root=(Resolve-Path "$PSScriptRoot/../..").Path
$config=Get-Content "$root/configs/pgo.json" -Raw | ConvertFrom-Json
$manifest=Get-Content "$root/configs/datasets.json" -Raw | ConvertFrom-Json
if(![IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory=Join-Path $PSScriptRoot $OutputDirectory }
if(Test-Path $OutputDirectory) { throw "Refuse overwrite: $OutputDirectory" }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
Get-ChildItem Env: | Where-Object {$_.Name -match '^(GBP_|HGBP_|OMP_|MKL_|OPENBLAS_)'} | ForEach-Object {
    [Environment]::SetEnvironmentVariable($_.Name,$null,'Process')
}
foreach($p in $config.profiles.SE2.environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($p.Name,$p.Value.Replace('{threads}','16'),'Process')
}
if($ProfileClock) {$env:HGBP_RESIDENT_CLOCK='1'}
$policy=@($config.profiles.SE2.arguments | ForEach-Object {$_.Replace('{threads}','16').Replace('{affinity_mask}','0xFFFF')})
git -C $root diff --quiet 7b9be582c77e1c6e878af5db023ce56d7c39af07 -- src include configs CMakeLists.txt
if($LASTEXITCODE) { throw 'Production source changed' }
$records=@()
foreach($dataset in $Datasets) {
    $data=Join-Path $DataDirectory "$dataset.g2o"
    if((Get-FileHash $data).Hash.ToLowerInvariant() -ne $manifest.datasets.$dataset.sha256) { throw 'Data hash mismatch' }
    for($repeat=0;$repeat -le $Repeats;++$repeat) {
        $order=@($Variants)
        if($repeat%2) {[array]::Reverse($order)}
        foreach($variant in $order) {
            foreach($key in @('HGBP_OVERLAP_BASIS','HGBP_CACHED_COARSE_PATTERN',
                'HGBP_COARSE_SHADOW','HGBP_RESIDENT_THREADS','HGBP_TRANSPORT_EVENTS')) {
                [Environment]::SetEnvironmentVariable($key,$null,'Process')
            }
            $env:HGBP_CUDA_BACKEND='gpu'; $env:HGBP_CUDA_SHADOW=$(if($Shadow){'1'}else{'0'})
            if($variant -eq 'reference') { $name='se2_benchmark.exe' }
            else {
                $name='se2_cuda_benchmark.exe'
                $env:HGBP_TRANSPORT_EVENTS='0'
                $env:HGBP_OVERLAP_BASIS='1'
                $env:HGBP_CACHED_COARSE_PATTERN='1'
                $env:HGBP_RESIDENT_THREADS='32'
                if($Shadow) { $env:HGBP_COARSE_SHADOW='1' }
            }
            $exe="$PSScriptRoot/build/pgo/$name"
            $tag="$dataset.$variant.$repeat"; $json="$OutputDirectory/$tag.json"
            $watch=[Diagnostics.Stopwatch]::StartNew()
            if($NsightSystems) {
                $nsysExtra=@()
                if($NsightDurationSeconds) { $nsysExtra+="--duration=$NsightDurationSeconds" }
                $output=& $NsightSystems profile --trace=cuda,nvtx "--cuda-graph-trace=$NsightGraphTrace" --sample=none --cpuctxsw=none --force-overwrite=false @nsysExtra --output "$OutputDirectory/$tag" $exe --problem-file $data --out-json $json @policy 2>&1
            } elseif($Sanitizer) {
                $output=& compute-sanitizer --tool $Sanitizer --launch-count $SanitizerLaunchCount --launch-skip $SanitizerLaunchSkip --error-exitcode 9 --log-file "$OutputDirectory/$tag.$Sanitizer.log" $exe --problem-file $data --out-json $json @policy 2>&1
            } else { $output=& $exe --problem-file $data --out-json $json @policy 2>&1 }
            $exitCode=$LASTEXITCODE; $watch.Stop()
            $output | Set-Content "$OutputDirectory/$tag.log" -Encoding UTF8
            $lower=$null; $upper=$null
            $context=0.
            foreach($line in $output) {if("$line".StartsWith('CUDA_LOWER_SUMMARY ')) {$lower="$line".Substring(19)|ConvertFrom-Json}}
            foreach($line in $output) {if("$line".StartsWith('CUDA_UPPER_SUMMARY ')) {$upper="$line".Substring(19)|ConvertFrom-Json}}
            foreach($line in $output) {if("$line".StartsWith('CUDA_CONTEXT_MS ')) {$context=[double]"$line".Substring(16)}}
            $phase=$null; $topology=$null
            foreach($line in $output) {
                if("$line".StartsWith('CUDA_PHASE_SUMMARY ')) {
                    $phase=[ordered]@{}
                    foreach($match in [regex]::Matches("$line",'([a-z_]+)=(\d+)')) { $phase[$match.Groups[1].Value]=[long]$match.Groups[2].Value }
                }
                if("$line".StartsWith('CUDA_TOPOLOGY ')) {
                    $topology=[ordered]@{mode=$env:HGBP_TOPOLOGY_MODE}
                    foreach($match in [regex]::Matches("$line",'([a-z_]+)=(\d+)')) { $topology[$match.Groups[1].Value]=[long]$match.Groups[2].Value }
                }
            }
            $record=[ordered]@{dataset=$dataset;variant=$variant;repeat=$repeat;shadow=[bool]$Shadow;
                exit_code=$exitCode;json=$json;process_ms=$watch.Elapsed.TotalMilliseconds;sanitizer=$Sanitizer;
                sanitizer_launch_count=$SanitizerLaunchCount;sanitizer_launch_skip=$SanitizerLaunchSkip;
                profile_clock=[bool]$ProfileClock;nsight_systems=$NsightSystems;
                overlap_basis=($env:HGBP_OVERLAP_BASIS -eq '1');
                cached_coarse_pattern=($env:HGBP_CACHED_COARSE_PATTERN -eq '1');
                gpu_coarse_assembly=($env:HGBP_GPU_COARSE_ASSEMBLY -eq '1');
                compact_coarse=($env:HGBP_COARSE_COMPACT -eq '1');
                sense_acquire=($env:HGBP_SENSE_ACQUIRE -eq '1');
                resident_threads=$env:HGBP_RESIDENT_THREADS;
                topology_mode=$env:HGBP_TOPOLOGY_MODE;topology=$topology;
                lower_stage_timing_available=($env:HGBP_TRANSPORT_EVENTS -ne '0');
                belief_rebase_period=$env:HGBP_BELIEF_REBASE_PERIOD;
                nsight_cuda_graph_trace=$(if($NsightSystems){$NsightGraphTrace}else{''});
                nsight_duration_seconds=$NsightDurationSeconds;phase=$phase;phase_eta_threads=$env:HGBP_PHASE_ETA_THREADS;phase_static=($env:HGBP_PHASE_STATIC -eq '1');
                lower=$lower;upper=$upper;device_upper=$env:HGBP_DEVICE_UPPER;cycle_fused=$env:HGBP_CYCLE_FUSED;cycle_parallel_rhs=$env:HGBP_CYCLE_PARALLEL_RHS;cuda_context_ms=$context;sha256=(Get-FileHash $exe).Hash.ToLowerInvariant();
                dataset_sha256=$manifest.datasets.$dataset.sha256;reference_commit=(git -C $root rev-parse HEAD).Trim()}
            if(!$exitCode) {
                $r=Get-Content $json -Raw|ConvertFrom-Json
                if($r.mg_history.Count -ne 21) {throw 'Incomplete outer20'}
                if(!$NsightSystems -and $variant -notin @('reference','cpu_pruned') -and ($lower.calls -ne 60 -or $lower.fallback_batches -ne 0)) {throw 'Missing GPU sweeps or CPU fallback'}
                if($NsightSystems -and !$lower) { Write-Warning 'Nsight retained target stdout inside report; use kernel trace and JSON for this diagnostic only' }
                $record.solver_ms=1000*$r.solver_wall_sec; $record.cost=$r.mg_history[-1].nonlinear_objective
            }
            $records+=$record
            $records|ConvertTo-Json -Depth 6|Set-Content "$OutputDirectory/runs.json" -Encoding UTF8
            Write-Host "$tag exit=$exitCode solver_ms=$($record.solver_ms) cost=$($record.cost)"
            if($exitCode) {throw "Run failed: $tag"}
        }
    }
}
