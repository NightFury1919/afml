# cleanup_phase1.ps1 -- run from anywhere; operates on the folder this file sits in (repo root).
# Preview by default. Use -Apply to execute.
# Moves root-level clutter into archive\ (git mv if tracked, Move-Item if not),
# untracks .pkl / zip / checkpoint files (files stay on disk), adds .gitignore lines.
# Never runs 'git add -A' and never commits. Files referenced by code in subfolders are SKIPPED.
param([switch]$Apply)
Set-Location $PSScriptRoot
if (-not (Test-Path .git)) { Write-Host "No .git here; put this file in the repo root."; exit 1 }

$staged = git diff --cached --name-only
if ($staged) { Write-Host "Staged changes already exist; commit or reset them first:"; $staged; exit 1 }

$mode = if ($Apply) { "APPLY" } else { "PREVIEW (nothing changes; use -Apply to run)" }
Write-Host "=== cleanup_phase1: $mode ===`n"

$tracked = New-Object 'System.Collections.Generic.HashSet[string]'
git ls-files | ForEach-Object { [void]$tracked.Add($_) }
$code = git ls-files '*.py' '*.ipynb' '*.bat' '*.ps1' | Where-Object { $_ -match '[\\/]' -and $_ -notlike 'archive/*' }

$groups = @(
  @{ Dest='archive\kraken_calibration'; Items=@(
      @{P='cpcv_groups_calibration*.csv';          K='cpcv_groups_calibration'},
      @{P='kraken_detection_power*.csv';           K='kraken_detection_power'},
      @{P='kraken_target_bars_calibration*.csv';   K='kraken_target_bars_calibration'},
      @{P='trial_sharpe_skew*.csv';                K='trial_sharpe_skew'} ) },
  @{ Dest='archive\run_logs'; Items=@(
      @{P='pooled_log.txt';                        K='pooled_log'},
      @{P='window*_log.txt';                       K='_log.txt'},
      @{P='pool_kraken_pbo_20window_results.txt';  K='pool_kraken_pbo'},
      @{P='pytest_alignment_null.txt';             K='pytest_alignment_null'},
      @{P='pytest_calibrate.txt';                  K='pytest_calibrate'},
      @{P='pytest_power.txt';                      K='pytest_power'},
      @{P='ch09_*.txt';                            K='ch09_full_diag'},
      @{P='signed_flow_run.txt';                   K='signed_flow_run'},
      @{P='diff_ch12_readme.txt';                  K='diff_ch12_readme'},
      @{P='afml_py_files.txt';                     K='afml_py_files'} ) },
  @{ Dest='archive\migration_scripts'; Items=@(
      @{P='migrate_*.ps1';                         K='migrate_'},
      @{P='run_nb.py';                             K='run_nb'},
      @{P='package-lock.json';                     K='package-lock'} ) }
)

$moved = 0; $skipped = 0
foreach ($g in $groups) {
  foreach ($it in $g.Items) {
    $hits = @()
    if ($code) { $hits = @(Select-String -Path $code -SimpleMatch -Pattern $it.K -List -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }) }
    $files = @(Get-ChildItem -File -Filter $it.P)
    if ($files.Count -eq 0) { continue }
    if ($hits.Count -gt 0) {
      Write-Host ("KEEP  {0}  ({1} files) -- referenced in: {2}" -f $it.P, $files.Count, (($hits | Select-Object -First 3) -join ', '))
      $skipped += $files.Count; continue
    }
    foreach ($f in $files) {
      $dest = Join-Path $g.Dest $f.Name
      $how = if ($tracked.Contains($f.Name)) { 'git mv' } else { 'move ' }
      Write-Host ("{0} {1}  ->  {2}" -f $how, $f.Name, $g.Dest)
      if ($Apply) {
        New-Item -ItemType Directory -Force -Path $g.Dest | Out-Null
        if ($how -eq 'git mv') { git mv -- $f.Name $dest } else { Move-Item -LiteralPath $f.FullName -Destination $dest }
      }
      $moved++
    }
  }
}

Write-Host "`n--- untrack (files stay on disk) ---"
$untrack = @($tracked | Where-Object { $_ -match '\.pkl$' -or $_ -eq 'ch08/files.zip' -or $_ -match '\.ipynb_checkpoints/' })
$untrack | ForEach-Object { Write-Host "rm --cached  $_" }
Write-Host ("{0} tracked files to untrack" -f $untrack.Count)
if ($Apply -and $untrack.Count -gt 0) { git rm --cached -q -- $untrack }

Write-Host "`n--- .gitignore lines ---"
$lines = @('*.pkl', '.ipynb_checkpoints/', 'ch08/files.zip', 'portfolio/stock_data/',
           'portfolio/sp_500_historical_components.csv', 'portfolio/prices_daily.csv', 'portfolio/prices_summary.csv')
$existing = @(Get-Content .gitignore)
foreach ($l in $lines) {
  if ($existing -contains $l) { Write-Host "present  $l" }
  else { Write-Host "add      $l"; if ($Apply) { Add-Content -Path .gitignore -Value $l } }
}

Write-Host ("`nSummary: {0} to move, {1} kept (referenced), {2} to untrack." -f $moved, $skipped, $untrack.Count)
if ($Apply) { Write-Host "Done. Review with: git status --short ; then commit with explicit paths (git add .gitignore, then git commit)." }