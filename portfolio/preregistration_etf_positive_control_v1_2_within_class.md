# Pre-registration: positive control v1.2, within-class ranking (DRAFT, not yet run)

Status: **draft for approval. Nothing below has been run.** Freeze it (commit it) before the run.

## Question
Does ranking, labeling and planting skill inside each of the 6 asset classes (breadth about 30.2) detect a
planted edge more often than the single pool of 48 (breadth about 16.5), in the same worlds?

## Design (everything else is identical to v1.1)
- Worlds: the v1.1 demeaned block bootstrap, the same 500 seeds (20261004 to 20261503), block 63.
- IC levels: 0, 0.02, 0.03, 0.05, 0.075, 0.10 (nominal).
- Command: `python positive_control_etf.py --workers 4 --demean --within-class`
- Within-class pieces: features ranked within class, labels beat the class median, class needs at least 3
  eligible ETFs per date, planted tilt = within-class standardized mom_12_1 rank. Classes: etf_classes.CLASS_OF.
- Statistic and detection rule unchanged: out-of-fold Spearman IC on dates 21 apart, t above the 95th percentile
  of this variant's own IC = 0 worlds.
- Results file: positive_control_etf_demeaned_within_class_results.csv (the script appends and resumes, and
  never touches the v1 or v1.1 files).

## What is compared (against the recorded v1.1 pooled result)
Pooled v1.1: null95 2.972; power 0.140 / 0.290 / 0.722 at nominal IC 0.02 / 0.03 / 0.05; false-positive rate
of t >= 1.96 at IC 0 of 26.0%; mean null model IC 0.028.

1. Primary: power at nominal IC 0.03 and 0.05, each against its own null95.
2. Secondary: false-positive rate of t >= 1.96 at IC 0, mean null model IC, mean oracle IC at IC 0.

## Pass rule (proposed, frozen at commit)
Within-class "helps" if BOTH hold:
- power at nominal IC 0.03 is at least 10 percentage points above 0.290 (that is, at least 0.39), and
- power at nominal IC 0.05 is not lower than 0.722 minus 5 points (at least 0.67).
Otherwise it is "no clear gain" and the pooled design stays the default. Either result is recorded.

## Caveats stated up front
- Nominal IC is not the same quantity in the two variants (a within-class skill versus a skill over one pool),
  so also read the realized oracle IC columns. Power against each variant's own null is the fair comparison;
  the raw t values are not.
- v1.1 itself did not have a clean no-edge world (26% false positives; leftover signal through vol_60). The
  per-variant null95 corrects for this inside each variant, but a contaminated null is still a weaker test.
  If the within-class null is also contaminated, say so in the result.
- Small classes (Currencies 3, Commodities 4) are noisy and carry fixed weight; this run does not test the
  portfolio-level cost of that.
- Not a test of live profitability.
