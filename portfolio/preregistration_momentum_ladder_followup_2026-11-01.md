# Follow-up note to `preregistration_momentum_ladder.md` (dated 2026-11-01)

This note does NOT change the frozen pre-registration. Nothing in that file was edited, and no rule, variant or price file
changed. It records three things a reader of the frozen file might otherwise trip over.

## 1. The run happened before the freeze commit
The single pre-registered run was made on 2026-10-31 with `run_momentum_ladder.py` on `prices_daily_asof_2026-10-30.csv`
(SHA-256 `dff1a011fee8a8f662d94d166f58c6ddd685a9d5ac819407e3970f1d2160b30c`). At that moment the working copy of the
pre-registration already said `Status: FROZEN` on a line by itself, and the hash matched. The commit that put the frozen text
into the repository is `9e9c2a6`. The earlier commit `d1b3581` ("Freeze momentum ladder pre-registration ...") actually held the
DRAFT text by mistake. So the repository history alone does not show the frozen text before the run; the working copy did, and
it did not change afterwards. The same disclosure is written in the results block at the bottom of `momentum_ladder.py`.
The sentence "Nothing below has been run on the real prices" (line 5 of the frozen file) was true when the text was written and
is no longer true; the run it describes is the one recorded in `momentum_results\`.

## 2. The "$900 floor" in the motivation paragraph
Line 19 asks at what size "the chance of hitting the $900 floor" becomes too high. That is leftover wording from the draft.
The account's loss limit is $500 (-50%), as the boss said on 2026-10-31, and the rules and the live floor use $500
(see the "boss answered" paragraph and the numbered decisions). The $900 figure that still matters is the -10% level used by the
touch-floor screen (the "End-below-floor" and "Touch-floor" definitions), which is a screening rule, not the account's floor.

## 3. Buying power for accounts under $2,000
The frozen file says to treat buying power as cash only. Alpaca support (2026-10-31) said accounts under $2,000 have cash
buying power only: no margin, no shorting, 1x. Alpaca's public help page separately says sale proceeds in such accounts can be
used the same day. Which of the two holds on the paper API for a sell followed by a buy on the same day has not been tested
live. This affects how the live account is traded (see `execute_plan.py`), not the pre-registered test, which assumed no leverage.

## 4. Checking the price file's hash on another machine (line endings)
The SHA-256 in the frozen file (`dff1a011...`) is the hash of the file as it sits on the Windows machine, which has CRLF line
endings. Git stores the same file with LF endings, so the committed copy hashes differently (`cb259897...` on a Linux clone).
Converting the committed copy to CRLF gives exactly `dff1a011fee8a8f662d94d166f58c6ddd685a9d5ac819407e3970f1d2160b30c`, so
the data are identical; only the line endings differ. Anyone re-checking the hash from a fresh clone on Linux or macOS must
convert to CRLF first (or hash the Windows working copy). The run itself used the Windows copy, so nothing about the results
changes.
