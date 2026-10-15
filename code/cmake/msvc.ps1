# Shared Windows toolchain setup; no downloads or dependency installation.
function Initialize-HgbpMsvc {
    param([switch]$ForceEnvironment)

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (!(Test-Path -LiteralPath $vswhere)) {
        throw "Could not find vswhere.exe. Install Visual Studio 2022 C++ Build Tools."
    }
    $vsInstall = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or !$vsInstall) {
        throw "Could not find a Visual Studio installation with C++ tools."
    }
    if ($ForceEnvironment -or !$env:VSCMD_ARG_TGT_ARCH) {
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
                [Environment]::SetEnvironmentVariable($line.Substring(0, $separator),
                    $line.Substring($separator + 1), "Process")
            }
        }
    }
    if ($env:VSCMD_ARG_TGT_ARCH -ne "x64") {
        throw "Use an x64 Visual Studio environment, not $env:VSCMD_ARG_TGT_ARCH."
    }

    # Preserve the validated newest-installed toolset selection and search order.
    $msvcToolsRoot = Join-Path $vsInstall "VC\Tools\MSVC"
    $msvcToolsDir = Get-ChildItem -LiteralPath $msvcToolsRoot -Directory |
        Sort-Object Name -Descending | Select-Object -First 1
    if (!$msvcToolsDir) { throw "Could not find MSVC tools under $msvcToolsRoot." }
    $bin = Join-Path $msvcToolsDir.FullName "bin\Hostx64\x64"
    $lib = Join-Path $msvcToolsDir.FullName "lib\x64"
    $include = Join-Path $msvcToolsDir.FullName "include"
    $compiler = Join-Path $bin "cl.exe"
    foreach ($requiredPath in @($compiler, $lib, $include)) {
        if (!(Test-Path -LiteralPath $requiredPath)) {
            throw "Required MSVC path is missing: $requiredPath"
        }
    }
    $env:Path = "$bin;$env:Path"
    $env:LIB = "$lib;$env:LIB"
    $env:INCLUDE = "$include;$env:INCLUDE"
    return $compiler -replace "\\", "/"
}
