# demo_video.ps1  --  silent screen-recording walkthrough for the QF-Portfolio project
# Run from C:\ws\AFML\portfolio in a PowerShell terminal with the mlfinlab conda env active.
#   1) Save this file in portfolio\
#   2) conda activate mlfinlab
#   3) Start your screen recorder, then run:  .\demo_video.ps1
# SAFE BY DESIGN: only reads files and runs the execute_plan.py DRY RUN. It never passes --execute,
# never runs git, never calls daily_run.py (so no plan/scoreboard file is rewritten), never prints keys.

# ---------------- EDIT THESE IF NEEDED ----------------
$PauseShort = 4      # seconds on a caption screen
$PauseLong  = 8      # seconds on a screen with lots to read
$DryRunCmd  = "python execute_plan.py --plans-dir accounts\etf_only\plans --orders-dir accounts\etf_only\orders"   # dry run for the ETF-only account (NO --execute)
$ShowDryRun = $true                      # set $false to skip the live dry run and just show files

$WaitForEnter = $true                     # $true: every screen waits for you to press Enter; $false: timed pauses
# -------------------------------------------------------

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

if ($env:CONDA_DEFAULT_ENV -ne "mlfinlab") {
    Write-Host "Activate the mlfinlab env first (conda activate mlfinlab). Stopping." -ForegroundColor Red
    exit 1
}
if ($DryRunCmd -match "--execute") {
    Write-Host "DryRunCmd contains --execute. Remove it. Stopping." -ForegroundColor Red
    exit 1
}

$LogFile = Join-Path $PSScriptRoot ("demo_output_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
try { Start-Transcript -Path $LogFile | Out-Null } catch { Write-Host "(could not start the output log)" -ForegroundColor DarkGray }

function Wait-Step([int]$Seconds) {
    if ($WaitForEnter) {
        Write-Host ""
        Write-Host "[Enter = next]" -ForegroundColor DarkGray
        [void](Read-Host)
    } else {
        Start-Sleep -Seconds $Seconds
    }
}

function Show-Caption([string]$Title, [string]$Line) {
    Clear-Host
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host "  $Line" -ForegroundColor White
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ""
}

function Newest([string]$Filter) {
    Get-ChildItem -Path . -Recurse -Depth 3 -Filter $Filter -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
}

function Show-FileHead($File, [int]$Lines = 25) {
    if ($File) {
        Write-Host ("--- " + (Resolve-Path -Relative $File.FullName) + " ---") -ForegroundColor DarkGray
        Get-Content $File.FullName -TotalCount $Lines
    } else {
        Write-Host "(file not found - skipping)" -ForegroundColor DarkGray
    }
}

function Show-FileTail([string]$Path, [int]$Lines = 15) {
    if (Test-Path $Path) {
        Write-Host ("--- $Path (last $Lines lines) ---") -ForegroundColor DarkGray
        Get-Content $Path -Tail $Lines
    } else {
        Write-Host "($Path not found - skipping)" -ForegroundColor DarkGray
    }
}

Clear-Host
Write-Host "Recording should be running. Press Enter to begin the walkthrough..." -ForegroundColor Green
[void](Read-Host)

# ---- 1. Title ----
Show-Caption "QF-Portfolio: AFML-based paper trading project" `
             "Paper accounts side by side: an ETF-only account and a SPY-only baseline, built on the AFML codebase."
Wait-Step $PauseShort

# ---- 2. Strategy config ----
Show-Caption "1. The strategy settings: ETF-only account" `
             "48 ETFs weighted by inverse volatility, 10% cap per ETF, no separate SPY sleeve (spy_share 0)."
if (Test-Path "accounts\etf_only\run_config.json") { Get-Content "accounts\etf_only\run_config.json" } else { Write-Host "(accounts\etf_only\run_config.json not found)" }
Wait-Step $PauseLong
Show-Caption "1b. The baseline: SPY-only account" "The comparison account holds only SPY (spy_share 1). Same dates, same rules otherwise."
if (Test-Path "accounts\spy_only\run_config.json") { Get-Content "accounts\spy_only\run_config.json" } else { Write-Host "(accounts\spy_only\run_config.json not found)" }
Wait-Step $PauseLong

# ---- 3. Nightly job ----
Show-Caption "2. The nightly job (9:30 PM, Windows Task Scheduler)" `
             "Updates prices, scores the shadow model, writes the daily plan. It never places orders."
if (Test-Path "nightly_all.bat") { Get-Content "nightly_all.bat" } else { Write-Host "(nightly_all.bat not found)" }
Wait-Step $PauseLong

Show-Caption "2b. Nightly log" "Each night the shadow model is scored on the new date."
Show-FileTail "shadow\nightly_log.txt" 12
Wait-Step $PauseLong

# ---- 4. Daily plan ----
Show-Caption "3. The daily run: the ETF-only account" "Health check, then a plan, then a scoreboard. Nothing here sends an order."
$report = Get-ChildItem "accounts\etf_only\runs" -Filter "run_*.md" -File -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
Show-FileHead $report 40
Wait-Step $PauseLong

# ---- 5. Execution safety ----
Show-Caption "4. Order step is human-gated" `
             "Dry run by default. Real orders need --execute --confirm <plan date>. Below is a DRY RUN for the ETF-only account."
Wait-Step $PauseShort
if ($ShowDryRun -and (Test-Path ".\use_account.ps1")) {
    # points this window at the ETF-only account; nothing secret is printed
    $env:APCA_API_KEY_ID     = [Environment]::GetEnvironmentVariable("APCA_ETF_KEY_ID", "User")
    $env:APCA_API_SECRET_KEY = [Environment]::GetEnvironmentVariable("APCA_ETF_SECRET", "User")
    Write-Host "Account: AFML_Ethan (ETF-only).  status / equity / cash / buying power:" -ForegroundColor Cyan
    $acct = @(python check_account.py 2>&1 | ForEach-Object { "$_" })
    if ($acct.Count -gt 0) { Write-Host $acct[0] }
    Write-Host ""
}
if ($ShowDryRun) {
    Write-Host "> $DryRunCmd" -ForegroundColor Green
    $dry = @(Invoke-Expression "$DryRunCmd 2>&1" | ForEach-Object { "$_" })
    $full = Join-Path $PSScriptRoot ("demo_dryrun_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
    $dry | Set-Content -Path $full
    $at = [Array]::IndexOf($dry, "Orders it would place:")
    if ($at -ge 0) {
        $end = $at + 1
        while ($end -lt $dry.Count -and $dry[$end].Trim() -ne "") { $end++ }
        $dry[0..$at] | ForEach-Object { Write-Host $_ }                          # header and every check result
        $dry[($at + 1)..([Math]::Min($at + 6, $end - 1))] | ForEach-Object { Write-Host $_ }   # the first orders
        if ($end - $at - 1 -gt 6) { Write-Host "  ... and $($end - $at - 7) more orders" -ForegroundColor DarkGray }
        $dry[$end..($dry.Count - 1)] | ForEach-Object { Write-Host $_ }          # the verdict
    } else {
        $dry | ForEach-Object { Write-Host $_ }
    }
} else {
    Write-Host "(live dry run skipped)" -ForegroundColor DarkGray
}
Wait-Step $PauseLong

Show-Caption "4b. First paper fills (Tue 2026-10-20)" "Ledger of the first 48 orders (status is as submitted). The reconcile step confirmed all 48 filled within 8 cents of plan."
Show-FileHead (Newest "orders_*.json") 20
Wait-Step $PauseLong

# ---- 6. Alpaca page (manual switch) ----
Show-Caption "5. The live paper account" "Switching to the Alpaca paper dashboard (positions + equity)."
Write-Host "Alt-Tab to the browser with the Alpaca PAPER positions page now (keys/account number off screen)." -ForegroundColor Yellow
Write-Host "Take as long as you like. When you are back in this window, press Enter to continue." -ForegroundColor Green
[void](Read-Host)

# ---- 7. Honesty about results ----
Show-Caption "6. Shadow model: honest status" `
             "Frozen 2026-10-16, not trading. No edge shown so far. Formal check ~Oct 2029 (36 periods)."
Wait-Step $PauseLong

# ---- 8. Next ----
Show-Caption "7. What's next" `
             "Monday: convert to ETF-only and fund the SPY baseline. Later: `$2,000 shorting experiment, Ch14 metrics (PSR, DSR, drawdown)."
Wait-Step $PauseLong

Show-Caption "Thanks for watching" "Stop the recording now."
Wait-Step $PauseShort
try { Stop-Transcript | Out-Null; Write-Host "Output also saved to $LogFile" -ForegroundColor DarkGray } catch { }
