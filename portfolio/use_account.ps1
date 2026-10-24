function Use-Account {
    param([Parameter(Mandatory)][ValidateSet('etf','spy','short')]$Name)
    $p = "APCA_$($Name.ToUpper())"
    $env:APCA_API_KEY_ID     = [Environment]::GetEnvironmentVariable("${p}_KEY_ID", "User")
    $env:APCA_API_SECRET_KEY = [Environment]::GetEnvironmentVariable("${p}_SECRET", "User")
    if (-not $env:APCA_API_KEY_ID) { Write-Host "No keys stored for '$Name'." -ForegroundColor Red; return }
    Write-Host "Active account: $Name (key starts $($env:APCA_API_KEY_ID.Substring(0,4)))" -ForegroundColor Green
    python check_account.py
}