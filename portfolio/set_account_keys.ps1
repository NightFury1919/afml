# set_account_keys.ps1 -- store ONE Alpaca paper account's keys as persistent Windows user variables.
# Run it as a file (do NOT paste it into the terminal, pasted lines get eaten by the prompts):
#     .\set_account_keys.ps1 etf        # AFML_Ethan   -> APCA_ETF_*   (also the old APCA_API_* defaults)
#     .\set_account_keys.ps1 spy        # AFML SPY     -> APCA_SPY_*
#     .\set_account_keys.ps1 short      # AFML2K       -> APCA_SHORT_*
# It asks for the key id, then the secret (hidden). Nothing is printed or saved to command history.
param([Parameter(Mandatory)][ValidateSet('etf', 'spy', 'short')][string]$Name)

$prefix = "APCA_$($Name.ToUpper())"
$id = (Read-Host "Key id for '$Name' (starts with PK)").Trim()
$secure = Read-Host "Secret for '$Name' (typing is hidden)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$secret = ([Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)).Trim()
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

if ($id -notmatch '^P[KA][A-Z0-9]{18,}$') {
    Write-Host "That key id does not look right (length $($id.Length)). Nothing was saved." -ForegroundColor Red
    return
}
if ($secret.Length -lt 30 -or $secret -match '\s') {
    Write-Host "That secret does not look right (length $($secret.Length)). Nothing was saved." -ForegroundColor Red
    return
}

$targets = New-Object System.Collections.ArrayList
[void]$targets.Add(@("${prefix}_KEY_ID", "${prefix}_SECRET"))
if ($Name -eq 'etf') { [void]$targets.Add(@("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")) }   # the old defaults the nightly job reads
foreach ($t in $targets) {
    [Environment]::SetEnvironmentVariable($t[0], $id,     "User")
    [Environment]::SetEnvironmentVariable($t[1], $secret, "User")
    Write-Host "Saved $($t[0]) (starts $($id.Substring(0,4)), length $($id.Length)) and $($t[1])" -ForegroundColor Green
}
Write-Host "Now run:  . .\use_account.ps1 ; Use-Account $Name" -ForegroundColor Cyan
