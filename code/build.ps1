param(
    [ValidateSet("Release", "RelWithDebInfo")]
    [string]$Config = "Release",
    [ValidateRange(1, 1024)][int]$Jobs = 12,
    [string]$BuildDirectory = "build",
    [string]$DependencyRoot = $env:HGBP_DEPENDENCY_ROOT,
    [string]$EigenDir = $env:HGBP_EIGEN_DIR,
    [string]$VcpkgInstalledDir = $env:HGBP_VCPKG_INSTALLED_DIR,
    [string]$RootBADir = $env:HGBP_ROOTBA_DIR,
    [string]$PGORuntimeDir = $env:HGBP_PGO_RUNTIME,
    [switch]$PGOOnly,
    [switch]$PureGBP,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ($PureGBP -and !$PGOOnly) { throw "PureGBP is a PGO-only experiment; also pass -PGOOnly." }

if ($DependencyRoot) {
    if (!$EigenDir) {
        $EigenDir = Join-Path $DependencyRoot "external\eigen"
    }
    if (!$VcpkgInstalledDir) {
        $VcpkgInstalledDir = Join-Path $DependencyRoot "vcpkg_installed\x64-windows"
    }
    if (!$RootBADir) {
        $RootBADir = Join-Path $DependencyRoot "external_baselines\rootba"
    }
}

function Resolve-DependencyDirectory {
    param(
        [string]$Path,
        [string]$Name
    )

    if (!$Path -or !(Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Name is missing. Set -DependencyRoot or pass its explicit path."
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

$EigenDir = Resolve-DependencyDirectory $EigenDir "Eigen"
$VcpkgInstalledDir = Resolve-DependencyDirectory $VcpkgInstalledDir "vcpkg installed tree"
if (!$PGOOnly) { $RootBADir = Resolve-DependencyDirectory $RootBADir "RootBA checkout" }
$PGORuntimeDir = Resolve-DependencyDirectory $PGORuntimeDir "PGO runtime (OpenBLAS with LAPACKE)"

# Check files as well as directories before launching a long build.
$requiredFiles = @((Join-Path $EigenDir "Eigen/Core"),
    (Join-Path $VcpkgInstalledDir "include/suitesparse/cholmod.h"),
    (Join-Path $PGORuntimeDir "openblas.dll"))
foreach ($name in @('cholmod', 'suitesparseconfig', 'amd', 'colamd', 'camd', 'ccolamd', 'lapack', 'openblas')) {
    $requiredFiles += Join-Path $VcpkgInstalledDir "lib/$name.lib"
}
if (!$PGOOnly) {
    $requiredFiles += Join-Path $RootBADir "src/rootba/bal/bal_problem.hpp"
    $requiredFiles += Join-Path $RootBADir "build-win/src/rootba/Release/rootba.lib"
    foreach ($name in @('tbb12', 'fmt', 'glog', 'gflags')) {
        $requiredFiles += Join-Path $RootBADir "conda_env/Library/lib/$name.lib"
        $requiredFiles += Join-Path $RootBADir "conda_env/Library/bin/$name.dll"
    }
}
$missing = @($requiredFiles | Where-Object { !(Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count) { throw "Missing build prerequisites:`n$($missing -join "`n")" }

if ([System.IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDir = [System.IO.Path]::GetFullPath($BuildDirectory)
} else {
    $BuildDir = [System.IO.Path]::GetFullPath((Join-Path $Root $BuildDirectory))
}

function Invoke-CMakeBuild {
    $configureArgs = @(
        "-S", $Root,
        "-B", $BuildDir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release",
        "-DCMAKE_CXX_COMPILER=$MsvcCompilerCMake",
        "-DHGBP_EIGEN_DIR=$EigenDir",
        "-DHGBP_VCPKG_INSTALLED_DIR=$VcpkgInstalledDir",
        "-DHGBP_ROOTBA_DIR=$RootBADir",
        "-DHGBP_BUILD_BA=$(!$PGOOnly)"
        "-DHGBP_PURE_GBP=$($PureGBP.IsPresent)"
    )

    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed with exit code $LASTEXITCODE."
    }

    & cmake --build $BuildDir --config $Config -j $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake build failed with exit code $LASTEXITCODE."
    }

    # Keep the validated PGO LAPACKE runtime separate from BA's dependency DLLs.
    Copy-Item -Force -Path (Join-Path $VcpkgInstalledDir "bin/*.dll") -Destination (Join-Path $BuildDir "pgo")
    Copy-Item -Force -Path (Join-Path $PGORuntimeDir "*.dll") -Destination (Join-Path $BuildDir "pgo")
    if (!$PGOOnly) {
        $baBin = Join-Path $BuildDir "ba"
        $runtimeBin = Join-Path $RootBADir "conda_env/Library/bin"
        # Do not shadow the system MSVC/OpenMP runtime with conda's older DLLs.
        foreach ($name in @('tbb12.dll', 'fmt.dll', 'glog.dll', 'gflags.dll')) {
            Copy-Item -Force -LiteralPath (Join-Path $runtimeBin $name) -Destination $baBin
        }
        Copy-Item -Force -Path (Join-Path $VcpkgInstalledDir "bin/*.dll") -Destination $baBin
    }
}

. (Join-Path $PSScriptRoot 'cmake/msvc.ps1')
$MsvcCompilerCMake = Initialize-HgbpMsvc
foreach ($command in @('cmake', 'ninja')) { $null = Get-Command $command -ErrorAction Stop }
if ($CheckOnly) {
    [ordered]@{
        Compiler = $MsvcCompilerCMake
        Eigen = $EigenDir
        SuiteSparse = $VcpkgInstalledDir
        PGORuntime = $PGORuntimeDir
        RootBA = $(if ($PGOOnly) { $null } else { $RootBADir })
        BuildDirectory = $BuildDir
        Scope = 'Required files and tools only; not a compile or runtime certification.'
    } | ConvertTo-Json
    return
}

Invoke-CMakeBuild
