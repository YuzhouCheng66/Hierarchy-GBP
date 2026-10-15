param(
    [string]$DependencyRoot=$env:HGBP_DEPENDENCY_ROOT,
    [string]$PGORuntimeDir=$env:HGBP_PGO_RUNTIME,
    [string]$BuildDirectory=(Join-Path $PSScriptRoot 'build'),
    [int]$Jobs=3
)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot '../../build.ps1') -PGOOnly -PureGBP `
    -DependencyRoot $DependencyRoot -PGORuntimeDir $PGORuntimeDir `
    -BuildDirectory $BuildDirectory -Jobs $Jobs
