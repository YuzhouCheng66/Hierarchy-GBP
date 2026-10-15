param(
    [Parameter(Mandatory=$true)][string]$DependencyRoot,
    [Parameter(Mandatory=$true)][string]$RuntimeDir,
    [string]$Ninja,
    [ValidateRange(1, 1024)][int]$Jobs=12,
    [string[]]$Targets=@('se2_benchmark','se2_cuda_benchmark','se2_cuda_failure_test','se2_cuda_layout_test','se2_cuda_overlap_test','se2_cuda_coarse_test'),
    [switch]$SkipConfigure
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../../cmake/msvc.ps1')
$compiler=Initialize-HgbpMsvc -ForceEnvironment
if(!$Ninja) { $Ninja=(Get-Command ninja -ErrorAction Stop).Source }
$build=Join-Path $PSScriptRoot 'build'
$argsCmake=@('--fresh','-S',$PSScriptRoot,'-B',$build,'-G','Ninja',"-DCMAKE_MAKE_PROGRAM=$Ninja",
    '-DCMAKE_BUILD_TYPE=Release',"-DCMAKE_CXX_COMPILER=$compiler","-DCMAKE_CUDA_HOST_COMPILER=$compiler",
    "-DHGBP_DEPENDENCY_ROOT=$DependencyRoot")
if(!$SkipConfigure) {
    & cmake @argsCmake
    if($LASTEXITCODE) { throw 'configure failed' }
}
& cmake --build $build --target @Targets -j $Jobs
if($LASTEXITCODE) { throw 'build failed' }
# Resolve the same runtime-directory override as before, then avoid replacing
# identical DLLs held by a profiler. Changed, locked dependencies still fail.
$dlls=@{}
foreach($directory in @("$DependencyRoot/vcpkg_installed/x64-windows/bin",$RuntimeDir)) {
    foreach($file in Get-ChildItem -LiteralPath $directory -Filter '*.dll' -File) {
        $dlls[$file.Name]=$file.FullName
    }
}
foreach($name in $dlls.Keys) {
    $source=$dlls[$name]
    $destination=Join-Path "$build/pgo" $name
    if((Test-Path -LiteralPath $destination) -and
        (Get-FileHash -LiteralPath $source).Hash -eq (Get-FileHash -LiteralPath $destination).Hash) {
        continue
    }
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
