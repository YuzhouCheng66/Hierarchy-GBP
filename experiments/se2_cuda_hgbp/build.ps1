param(
    [Parameter(Mandatory=$true)][string]$DependencyRoot,
    [Parameter(Mandatory=$true)][string]$RuntimeDir,
    [string]$Ninja,
    [int]$Jobs=12,
    [string[]]$Targets=@('se2_benchmark','se2_cuda_benchmark','se2_cuda_failure_test','se2_cuda_layout_test','se2_cuda_overlap_test','se2_cuda_coarse_test'),
    [switch]$SkipConfigure
)
$ErrorActionPreference='Stop'
$vswhere="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs=& $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars=Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$lines=& $env:ComSpec /d /s /c "`"$vcvars`" >nul && set"
if($LASTEXITCODE) { throw 'vcvars failed' }
foreach($line in $lines) {
    $pos=$line.IndexOf('=')
    if($pos -gt 0) { [Environment]::SetEnvironmentVariable($line.Substring(0,$pos),$line.Substring($pos+1),'Process') }
}
$tool=Get-ChildItem (Join-Path $vs 'VC\Tools\MSVC') -Directory | Sort-Object Name -Descending | Select-Object -First 1
$bin=Join-Path $tool.FullName 'bin\Hostx64\x64'
$compiler=(Join-Path $bin 'cl.exe') -replace '\\','/'
$env:Path="$bin;$env:Path"
$env:LIB="$(Join-Path $tool.FullName 'lib\x64');$env:LIB"
$env:INCLUDE="$(Join-Path $tool.FullName 'include');$env:INCLUDE"
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
