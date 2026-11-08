@echo off
rem nightly_all.bat -- one nightly run for all accounts. Places NO orders and runs NO git.
rem   1. update prices and score the shadow model (account-independent)
rem   2. for each account: health check, plan, scoreboard, report, using THAT account's keys, config and folders
rem The old single-account line (daily_run.py with the default keys) was retired on 2026-10-23, see nightly_all_old.bat.
set PY=C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
cd /d C:\ws\AFML\portfolio
"%PY%" nightly_run.py >> shadow\nightly_stdout.txt 2>&1

rem ---- account A: AFML_Ethan, the ETF-only account ----
if "%APCA_ETF_KEY_ID%"=="" (
  echo [%date% %time%] etf: no keys in the environment, skipped >> accounts\etf_only\daily_stdout.txt
) else (
  set "APCA_API_KEY_ID=%APCA_ETF_KEY_ID%"
  set "APCA_API_SECRET_KEY=%APCA_ETF_SECRET%"
  "%PY%" daily_run.py --no-update --from-alpaca --config accounts\etf_only\run_config.json --plans-dir accounts\etf_only\plans --runs-dir accounts\etf_only\runs >> accounts\etf_only\daily_stdout.txt 2>&1
)

rem ---- account B: AFML SPY, the SPY-only baseline ----
if "%APCA_SPY_KEY_ID%"=="" (
  echo [%date% %time%] spy: no keys in the environment, skipped >> accounts\spy_only\daily_stdout.txt
) else (
  set "APCA_API_KEY_ID=%APCA_SPY_KEY_ID%"
  set "APCA_API_SECRET_KEY=%APCA_SPY_SECRET%"
  "%PY%" daily_run.py --no-update --from-alpaca --config accounts\spy_only\run_config.json --plans-dir accounts\spy_only\plans --runs-dir accounts\spy_only\runs >> accounts\spy_only\daily_stdout.txt 2>&1
)

rem Account C (AFML2K, shorting) is not here on purpose: it has no short path yet.
rem ---- account C: AFML_MOM, the momentum account (inert until sleeve_n, sleeve_share and start are set) ----
if "%APCA_MOM_KEY_ID%"=="" (
  echo [%date% %time%] mom: no keys in the environment, skipped >> accounts\momentum\daily_stdout.txt
) else (
  set "APCA_API_KEY_ID=%APCA_MOM_KEY_ID%"
  set "APCA_API_SECRET_KEY=%APCA_MOM_SECRET%"
  "%PY%" momentum_daily.py --no-update --from-alpaca --config accounts\momentum\run_config.json --plans-dir accounts\momentum\plans --runs-dir accounts\momentum\runs >> accounts\momentum\daily_stdout.txt 2>&1
)
