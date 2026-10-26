param(
    [Parameter(Mandatory=$true)][string]$DependencyRoot,
    [Parameter(Mandatory=$true)][string]$RootBADir,
    [Parameter(Mandatory=$true)][string]$RootBALibrary,
    [Parameter(Mandatory=$true)][string]$RootBAPrefix,
    [string]$BuildDirectory='build-integrations',
    [string]$AbseilPrefix='',
    [ValidateRange(1,64)][int]$Jobs=2,
    [switch]$CapturePowerBA
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
. "$root/code/cmake/msvc.ps1"
$compiler=Initialize-HgbpMsvc
$build=[IO.Path]::GetFullPath((Join-Path $root $BuildDirectory))
if([IO.Path]::IsPathRooted($BuildDirectory)) {$build=[IO.Path]::GetFullPath($BuildDirectory)}
& cmake -S $PSScriptRoot -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_CXX_COMPILER=$compiler" "-DHGBP_DEPENDENCY_ROOT=$DependencyRoot" `
    "-DHGBP_ROOTBA_DIR=$RootBADir" "-DHGBP_ROOTBA_LIBRARY=$RootBALibrary" `
    "-DHGBP_ROOTBA_PREFIX=$RootBAPrefix" "-DHGBP_CAPTURE_POWERBA=$($CapturePowerBA.IsPresent)" `
    "-DHGBP_ABSL_PREFIX=$AbseilPrefix" -DBUILD_TESTING=OFF
if($LASTEXITCODE) {throw 'Adapter configuration failed'}
$targets=@('ba_file'); if($CapturePowerBA) {$targets+='powerba_capture'}
& cmake --build $build --target @targets -j $Jobs
if($LASTEXITCODE) {throw 'Adapter build failed'}
Copy-Item -Path "$DependencyRoot/vcpkg_installed/x64-windows/bin/*.dll" -Destination $build -Force
foreach($name in @('tbb12','fmt','glog','gflags')) {
    Copy-Item -LiteralPath "$RootBAPrefix/bin/$name.dll" -Destination $build -Force
}
if($CapturePowerBA) {Copy-Item -Path "$AbseilPrefix/bin/absl*.dll" -Destination $build -Force}
