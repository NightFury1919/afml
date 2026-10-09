"""Dry-run paper-trading runner: prices -> model -> target weights -> the orders it WOULD place.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python paper_runner.py                                   # empty account, $1,000
    python paper_runner.py --equity 1000 --holdings-file holdings.json
    holdings.json: {"SPY": 300.0, "GLD": 150.0}              # current market value in dollars

What it does, in order (all settings live in PROTOCOL below, see paper_protocol_v1.md)
    1. Loads the newest prices_daily_asof_*.csv and removes the SH hedge from the ranked universe.
    2. Builds the five ranked features for every date and the relative 21-day labels. The TRAINING set
       is every row whose label was already resolved by the last price date. The LIVE set is the
       last price date, which has features but no label yet.
    3. Fits the baseline logistic regression (C = 1) on the training rows and scores the live rows.
    4. portfolio_construction.target_weights() keeps the top fifth by score, equal weight.
    5. rebalance.rebalance_orders() turns the target weights and your current holdings into orders.
    6. Writes plans/plan_<last price date>.json and prints a summary. It never overwrites a plan.

What it does NOT do: it never sends an order and imports no trading client (a test checks that).
Holdings come from a file you provide, not from Alpaca. Reading the paper account is a separate,
later step.

Known limits of this first version (stated so nothing is assumed):
    - Pooled ranking only. Within-class ranking waits for the v1.2 comparison and class budgets.
    - Refits the model from scratch every run on all resolved labels, with no purging between
      training rows and the live row (the live row has no label, so nothing leaks into it).
    - No time decay (decided in principle, not yet wired in), no costs, no shorting.
    - Training rows overlap heavily (21-day labels sampled daily), as in the positive control.
"""
import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from etf_features import DEFAULT_WINDOWS, FEATURE_NAMES, compute_features
from etf_labels import make_labels
from panel_data import drop_hedge, load_prices
from portfolio_construction import target_weights
from rebalance import rebalance_orders


@dataclass(frozen=True)
class Protocol:
    horizon: int = 21               # label horizon in trading days
    min_assets: int = 10            # ETFs with all five features needed on a date
    C: float = 1.0                  # logistic regression strength (baseline, untuned)
    top_share: float = 0.2          # keep the top fifth by score
    max_weight: float = None        # per-ETF cap as a share of equity; None = no cap
    exposure: float = 1.0           # confidence scalar; 1.0 until the ramp is decided
    mode: str = "fractional"        # "fractional" (dollars) or "whole" (shares)
    min_notional: float = 1.0       # smallest dollar order worth sending
    ranking: str = "pooled"         # only "pooled" is wired in
    max_stale_days: int = 5         # warn if the last price date is older than this


PROTOCOL = Protocol()


def training_and_live(prices, protocol=PROTOCOL):
    """(train, live). train: features + fwd_ret/excess_ret/label/t1, labels resolved by the last
    price date. live: the five features on the last price date only."""
    px = drop_hedge(prices)
    feats = compute_features(px, DEFAULT_WINDOWS, protocol.min_assets)
    last = px.index[-1]
    live = feats[feats.index.get_level_values("date") == last]
    if len(live) < protocol.min_assets:
        raise ValueError(f"need at least {protocol.min_assets} ETFs with all five features on "
                         f"{last.date()}, found {len(live)}")
    labels = make_labels(px, feats.index, protocol.horizon, protocol.min_assets)
    train = feats.reindex(labels.index).join(labels)
    return train, live[FEATURE_NAMES]


def score_live(train, live, protocol=PROTOCOL):
    """Probability that each live ETF beats the cross-sectional median, indexed by ETF."""
    y = train["label"]
    if y.nunique() < 2:
        raise ValueError("training labels contain only one class")
    model = LogisticRegression(C=protocol.C, max_iter=200)
    model.fit(train[FEATURE_NAMES].to_numpy(), y.to_numpy())
    proba = model.predict_proba(live[FEATURE_NAMES].to_numpy())[:, 1]
    return pd.Series(proba, index=live.index.get_level_values("asset"), name="score")


def make_plan(prices, holdings, equity, protocol=PROTOCOL, today=None):
    """Everything the runner decides, as a plain dict. Sends nothing."""
    if not equity > 0:
        raise ValueError("equity must be positive")
    if protocol.ranking != "pooled":
        raise ValueError("only pooled ranking is wired in")
    train, live = training_and_live(prices, protocol)
    scores = score_live(train, live, protocol)
    weights = target_weights(scores, top_share=protocol.top_share, max_weight=protocol.max_weight,
                             exposure=protocol.exposure)
    last_prices = prices.iloc[-1].dropna().to_dict()
    orders, cash_after = rebalance_orders(
        weights, holdings, equity, prices=last_prices if protocol.mode == "whole" else None,
        mode=protocol.mode, min_notional=protocol.min_notional)

    last = prices.index[-1]
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    warnings = []
    age = (today - last).days
    if age > protocol.max_stale_days:
        warnings.append(f"price data is {age} days old (last date {last.date()})")
    return dict(as_of=last.date().isoformat(), protocol=dataclasses.asdict(protocol),
                n_train=int(len(train)), n_live=int(len(live)),
                scores={k: float(v) for k, v in scores.items()},
                weights={k: float(v) for k, v in weights.items()},
                equity=float(equity), holdings={k: float(v) for k, v in holdings.items()},
                orders=orders, cash_after=float(cash_after), warnings=warnings)


def write_plan(plan, folder):
    """Save the plan as plan_<as_of>.json. Raises FileExistsError rather than overwrite one."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"plan_{plan['as_of']}.json"
    if path.exists():
        raise FileExistsError(f"{path.name} already exists; plans are never overwritten")
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def main():
    from update_prices import latest_snapshot

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(here), help="folder holding the price snapshots")
    ap.add_argument("--plans-dir", default=str(here / "plans"))
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--holdings-file", default=None, help="JSON {symbol: dollars held}")
    args = ap.parse_args()

    src = latest_snapshot(args.folder)
    prices = load_prices(src)
    holdings = json.loads(Path(args.holdings_file).read_text(encoding="utf-8")) if args.holdings_file else {}
    plan = make_plan(prices, holdings, args.equity)

    print(f"DRY RUN. No orders are sent. Prices: {src.name}  (as of {plan['as_of']})")
    print(f"Trained on {plan['n_train']:,} resolved rows; scored {plan['n_live']} ETFs.\n")
    top = sorted(plan["scores"].items(), key=lambda kv: -kv[1])
    print("Top scores: " + ", ".join(f"{k} {v:.3f}" for k, v in top[:12]))
    print("\nTarget weights:")
    for k, v in sorted(plan["weights"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:6s} {v:6.1%}   ${v * plan['equity']:,.2f}")
    print("\nOrders it would place:")
    for o in plan["orders"] or [{"note": "none"}]:
        print("  ", o)
    print(f"\nCash after: ${plan['cash_after']:,.2f}")
    for w in plan["warnings"]:
        print("WARNING:", w)
    try:
        print(f"Saved {write_plan(plan, args.plans_dir)}")
    except FileExistsError as e:
        print(f"Plan not saved: {e}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_paper_runner.py -v
#
# collected 17 items
#
# test_paper_runner.py::test_training_labels_are_all_resolved_by_the_last_date PASSED [  5%]
# test_paper_runner.py::test_training_uses_every_label_already_resolved_up_to_the_last_date PASSED [ 11%]
# test_paper_runner.py::test_live_rows_are_only_the_last_date_and_carry_no_labels PASSED [ 17%]
# test_paper_runner.py::test_live_features_equal_the_feature_pipeline_on_the_last_date PASSED [ 23%]
# test_paper_runner.py::test_no_lookahead_older_data_gives_a_prefix_of_the_same_training_rows PASSED [ 29%]
# test_paper_runner.py::test_scores_are_probabilities_for_exactly_the_live_assets_and_repeatable PASSED [ 35%]
# test_paper_runner.py::test_empty_account_plan_buys_the_top_fifth_equally_and_spends_all_the_cash PASSED [ 41%]
# test_paper_runner.py::test_running_again_with_the_resulting_holdings_orders_nothing PASSED [ 47%]
# test_paper_runner.py::test_exposure_below_one_leaves_the_rest_in_cash PASSED [ 52%]
# test_paper_runner.py::test_the_hedge_is_never_scored_or_bought_and_a_held_hedge_is_closed PASSED [ 58%]
# test_paper_runner.py::test_whole_share_mode_spends_no_more_than_equity PASSED [ 64%]
# test_paper_runner.py::test_too_few_scored_etfs_is_an_error PASSED        [ 70%]
# test_paper_runner.py::test_bad_inputs_are_rejected PASSED                [ 76%]
# test_paper_runner.py::test_stale_data_produces_a_warning_but_still_a_plan PASSED [ 82%]
# test_paper_runner.py::test_the_plan_records_the_protocol_it_was_made_under PASSED [ 88%]
# test_paper_runner.py::test_making_a_plan_never_loads_any_trading_client PASSED [ 94%]
# test_paper_runner.py::test_write_plan_round_trips_and_never_overwrites PASSED [100%]
#
# 17 passed in 2.80s
# Mutation checks (sandbox): top share ignored, exposure ignored, live row from the wrong date, hedge
#   not removed, holdings ignored, plan overwrite allowed, half the training rows, and labels built from
#   older data each made tests fail; original restored. The last one survived the first version of the
#   tests, so test_training_uses_every_label_already_resolved_up_to_the_last_date was added.
# CLI smoke test (sandbox, synthetic snapshot): printed the plan, saved it, and refused to overwrite.
# NOT yet run on the real snapshot. Do that on your machine: python paper_runner.py
# ---------------------------------------------------------------------------
