param(
    [ValidateSet("Release", "RelWithDebInfo")]
    [string]$Config = "Release",
    [int]$Jobs = 12,
    [string]$BuildDirectory = "build",
    [string]$DependencyRoot = $env:HGBP_DEPENDENCY_ROOT,
    [string]$EigenDir = $env:HGBP_EIGEN_DIR,
    [string]$VcpkgInstalledDir = $env:HGBP_VCPKG_INSTALLED_DIR,
    [string]$RootBADir = $env:HGBP_ROOTBA_DIR
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

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
$RootBADir = Resolve-DependencyDirectory $RootBADir "RootBA checkout"

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
        "-DHGBP_ROOTBA_DIR=$RootBADir"
    )

    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed with exit code $LASTEXITCODE."
    }

    & cmake --build $BuildDir --config $Config -j $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake build failed with exit code $LASTEXITCODE."
    }

    $vcpkgBin = Join-Path $VcpkgInstalledDir "bin"
    if (Test-Path -LiteralPath $vcpkgBin -PathType Container) {
        Copy-Item -Force -Path (Join-Path $vcpkgBin "*.dll") -Destination $BuildDir
    }
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path -LiteralPath $vswhere)) {
    throw "Could not find vswhere.exe. Install Visual Studio 2022 C++ Build Tools."
}
$vsInstall = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsInstall) {
    throw "Could not find a Visual Studio installation with C++ tools."
}

if (!$env:VSCMD_ARG_TGT_ARCH) {
    $vcvars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
    if (!(Test-Path -LiteralPath $vcvars)) {
        throw "Could not find vcvars64.bat at $vcvars."
    }

    $vcEnvironment = & $env:ComSpec /d /s /c "`"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio environment setup failed with exit code $LASTEXITCODE."
    }
    foreach ($line in $vcEnvironment) {
        $separator = $line.IndexOf("=")
        if ($separator -gt 0) {
            $name = $line.Substring(0, $separator)
            $value = $line.Substring($separator + 1)
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Some VS installations keep vcvars on an older toolset and omit the MSVC
# library directory. Select the newest installed x64 toolset explicitly.
$msvcToolsRoot = Join-Path $vsInstall "VC\Tools\MSVC"
$msvcToolsDir = Get-ChildItem -LiteralPath $msvcToolsRoot -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (!$msvcToolsDir) {
    throw "Could not find MSVC tools under $msvcToolsRoot."
}
$msvcBinDir = Join-Path $msvcToolsDir.FullName "bin\Hostx64\x64"
$msvcLibDir = Join-Path $msvcToolsDir.FullName "lib\x64"
$msvcIncludeDir = Join-Path $msvcToolsDir.FullName "include"
$msvcCompiler = Join-Path $msvcBinDir "cl.exe"
foreach ($requiredPath in @($msvcCompiler, $msvcLibDir, $msvcIncludeDir)) {
    if (!(Test-Path -LiteralPath $requiredPath)) {
        throw "Required MSVC path is missing: $requiredPath"
    }
}
$env:Path = "$msvcBinDir;$env:Path"
$env:LIB = "$msvcLibDir;$env:LIB"
$env:INCLUDE = "$msvcIncludeDir;$env:INCLUDE"
$MsvcCompilerCMake = $msvcCompiler -replace "\\", "/"

Invoke-CMakeBuild
