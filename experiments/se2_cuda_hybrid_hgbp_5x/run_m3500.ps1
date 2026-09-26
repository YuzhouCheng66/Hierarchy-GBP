param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [string]$Executable = ".\bin\se2_cuda_hybrid_hgbp.exe",

    [Parameter(Mandatory = $true)]
    [string]$CholmodBin,

    [ValidateSet(15, 20)]
    [int]$GroupSize = 20
)

$ErrorActionPreference = "Stop"
$env:HGBP_CHOLMOD_BIN = (Resolve-Path -LiteralPath $CholmodBin).Path

& $Executable `
    --problem-file $Dataset `
    --hybrid-hgbp `
    --hybrid-dense-coarse `
    --kernel gbp_persistent `
    --incoming-layout exact `
    --schur-kernel inverse `
    --coop-block-policy work_cap `
    --persistent-threads 32 `
    --sweeps 100 `
    --fixed-lambda-start 60 `
    --hybrid-cycles 20 `
    --group-size $GroupSize `
    --huber-delta 5

if ($LASTEXITCODE -ne 0) {
    throw "se2_cuda_hybrid_hgbp exited with code $LASTEXITCODE"
}
