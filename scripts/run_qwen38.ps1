# Generate traces with qwen3.8:27b and train the GNN guard on them.
# Usage (from anywhere):  powershell -File scripts\run_qwen38.ps1 [-Limit 2] [-SkipTrace]
param(
    [string]$Model   = "qwen3.8:27b",
    [string]$DataDir = "data/tracegen_qwen38",
    [string]$OutDir  = "results/tracegen_qwen38",
    [string]$Devices = "cuda:0,cuda:1",
    [int]$Limit      = 0,      # >0: only the first N user tasks per suite (smoke test)
    [switch]$SkipTrace         # skip trace generation, only build + train
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # repo root

function Run-Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit code $LASTEXITCODE)" }
}

# 0. Model check: look for "tools" under Capabilities
Run-Step "0. Check model" { ollama show $Model }
Write-Host "Confirm 'tools' is listed under Capabilities above. Continuing in 5 s (Ctrl+C to abort)." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 1. Seed the new data folder with the existing prompts (no re-generation)
Write-Host "`n=== 1. Seed $DataDir ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $DataDir | Out-Null
foreach ($f in 'originals', 'candidates', 'verified_variants', 'rejected_variants', 'generate_log') {
    $dst = Join-Path $DataDir "$f.jsonl"
    if (-not (Test-Path $dst)) { Copy-Item "data\tracegen\$f.jsonl" $dst }
}

# 2. Generate traces (benign + attacked), model split across both GPUs. Resumes if re-run.
if (-not $SkipTrace) {
    $traceArgs = @("trace", "--out", $DataDir, "--agents", $Model, "--gpu-mode", "split", "--attacks")
    if ($Limit -gt 0) { $traceArgs += @("--limit", $Limit) }
    Run-Step "2. Generate traces" { python -m tracegen @traceArgs }
}

# 3. Build the dataset (splits and report)
$buildArgs = @("build", "--out", $DataDir, "--attacks")
if ($Limit -gt 0) { $buildArgs += @("--limit", $Limit) }
Run-Step "3. Build dataset" { python -m tracegen @buildArgs }

# 4. Train and evaluate the GNN
Run-Step "4. Train GNN" {
    python scripts/run_tracegen_gnn.py --tracegen $DataDir --outdir $OutDir --devices $Devices
}

Write-Host "`nDone. Report: $OutDir\REPORT.md" -ForegroundColor Green
