# install_nightly_task.ps1
#
# WHY THIS EXISTS
# The nightly job (price update + shadow scoring) has to run every evening on its own, even
# when nobody is logged in. On Windows that is a Task Scheduler task. Clicking it together by
# hand is easy to get subtly wrong (a missed start that never catches up, a laptop that will
# not wake, a task that only runs while someone is logged on). This script creates the task
# with the right settings in one command, so a second machine ends up set up the same way.
#
# WHAT IT DOES
#   * Creates (or replaces) one Task Scheduler task that runs a launcher in this folder daily.
#   * Runs whether or not the user is logged on, so it asks for the Windows account password.
#   * Runs a missed start as soon as the machine is back, wakes the machine to run, and
#     retries 3 times, 5 minutes apart, if the launcher fails.
#   * Does NOT touch Alpaca keys, accounts, or git. It never runs anything itself unless you
#     pass -RunNow.
#
# HOW TO USE (PowerShell 5.1 is fine; do this from the portfolio folder)
#   .\install_nightly_task.ps1 -DryRun          show what would be created, create nothing
#   .\install_nightly_task.ps1                  create the task (asks for your Windows password)
#   .\install_nightly_task.ps1 -RunNow          create it, start it once, show the result
#   .\install_nightly_task.ps1 -Remove          delete the task again
#
# REHEARSAL: to try it without touching a real task, give it another name:
#   .\install_nightly_task.ps1 -TaskName "AFML rehearsal" -RunNow
#   .\install_nightly_task.ps1 -TaskName "AFML rehearsal" -Remove
#
# If Windows says "Access is denied", open PowerShell with "Run as administrator" and retry.
#
# WHICH LAUNCHER
# The default, nightly_run.bat, is keyless: prices and shadow scoring only. nightly_all.bat
# also runs the Alpaca accounts and the stock job, so only choose it on the machine that owns
# those accounts and keys:   -Launcher nightly_all.bat
# Never schedule nightly_all.bat on two machines that share one Alpaca account.

param(
    [string]$TaskName = "AFML nightly shadow",
    [ValidateSet("nightly_run.bat", "nightly_all.bat")]
    [string]$Launcher = "nightly_run.bat",
    [string]$At = "20:30",          # machine local time, 24-hour clock
    [switch]$DryRun,
    [switch]$RunNow,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

# The folder this script lives in is the portfolio folder. Everything is derived from it, so
# the repo can sit anywhere on any machine.
$portfolio = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcherPath = Join-Path $portfolio $Launcher

if ($Remove) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) {
        Write-Host "No task named '$TaskName'. Nothing to remove."
        exit 0
    }
    if ($DryRun) {
        Write-Host "DRY RUN: would remove task '$TaskName'."
        exit 0
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed task '$TaskName'."
    exit 0
}

if (-not (Test-Path $launcherPath)) {
    throw "Cannot find $launcherPath. Run this script from the portfolio folder of the repo."
}

# Task Scheduler uses the machine's clock. The job is meant to run at 8:30 PM Pacific, so say
# so loudly if this machine is set to another time zone.
$tz = (Get-TimeZone).Id
Write-Host "Machine time zone: $tz"
if (($tz -notlike "Pacific*") -and ($tz -notlike "*Los_Angeles*")) {
    Write-Warning "This machine is not on Pacific time. $At will be $At in '$tz', not Pacific."
}

if ($Launcher -eq "nightly_all.bat") {
    Write-Warning "nightly_all.bat uses Alpaca keys and the stock job. Only use it on the machine that owns those accounts."
}

$user = "$env:USERDOMAIN\$env:USERNAME"
try {
    $parsed = [datetime]::ParseExact($At, "HH:mm", [Globalization.CultureInfo]::InvariantCulture)
} catch {
    throw "-At must be a 24-hour time like 20:30. You gave '$At'."
}

$action = New-ScheduledTaskAction -Execute $launcherPath -WorkingDirectory $portfolio
$trigger = New-ScheduledTaskTrigger -Daily -At $parsed
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew

Write-Host ""
Write-Host "Task name:      $TaskName"
Write-Host "Runs:           $launcherPath"
Write-Host "Start in:       $portfolio"
Write-Host "Every day at:   $At (machine local time)"
Write-Host "Runs as:        $user, whether or not logged on"
Write-Host "Missed start:   runs as soon as the machine is back"
Write-Host "Wake to run:    yes"
Write-Host "On failure:     retry 3 times, 5 minutes apart"

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN: nothing was created."
    exit 0
}

# A task that runs while nobody is logged on needs the account password. It is passed straight
# to Task Scheduler and is not written to any file or log.
Write-Host ""
$secure = Read-Host "Windows password for $user" -AsSecureString
$plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
if ([string]::IsNullOrEmpty($plain)) {
    throw "A password is required for a task that runs while nobody is logged on. This Windows account may have no password set."
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -User $user -Password $plain -Force | Out-Null
$plain = $null

$task = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Host ""
Write-Host "Created task '$TaskName'. State: $($task.State). Next run: $($info.NextRunTime)"

if ($RunNow) {
    Write-Host "Starting it once now..."
    Start-ScheduledTask -TaskName $TaskName
    # Give the launcher time to finish, then report. A prices-only run takes well under a minute.
    $deadline = (Get-Date).AddMinutes(3)
    do {
        Start-Sleep -Seconds 5
        $state = (Get-ScheduledTask -TaskName $TaskName).State
    } while ($state -eq "Running" -and (Get-Date) -lt $deadline)
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "State: $state. Last Run Result: $($info.LastTaskResult)  (0 means success)"
    Write-Host "Now read the end of shadow\nightly_stdout.txt for a dated line with no FAILED."
}
