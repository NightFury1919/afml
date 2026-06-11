import numpy as np
import pandas as pd

# Meta-Labeling — AFML Chapter 3, Sections 3.6-3.9, pages 50-54
#
# Meta-labeling is a two-stage approach to building trading strategies:
#
# Stage 1 — Primary model:
#   A simple model (or rule) that decides the SIDE of each trade (+1 = long, -1 = short).
#   It doesn't need to be highly accurate — it just needs to identify the direction.
#
# Stage 2 — Meta-model (meta-labeling):
#   A second model that looks at each primary signal and predicts whether to ACT on it.
#   It outputs a position size between 0 and 1:
#     0 = don't trade (primary model is probably wrong here)
#     1 = trade at full size (primary model is probably right here)
#
# --- Why does this work better than one model? ---
# Separating the "which direction" and "whether to trade" decisions allows each model
# to specialize. The meta-model learns to recognize when the primary model's signals
# are reliable vs when they should be ignored.
#
# --- How labels change with meta-labeling ---
# Without meta-labeling: bin ∈ {-1, 0, 1} — direction of price move
# With meta-labeling:    bin ∈ {0, 1}     — did the primary model's bet pay off?
#
# This turns the problem into a binary classification — much easier for ML models.


def get_events_meta(close, t_events, pt_sl, trgt, min_ret, t1=False, side=None):
    # Expanded getEvents with Meta-Labeling — AFML Chapter 3, Snippet 3.6, page 51
    #
    # Extension of get_events() from triple_barrier.py that adds support for
    # a primary model's side predictions. When side is provided, the barriers
    # become asymmetric — profit target and stop loss are applied relative to
    # the predicted direction rather than always assuming a long position.
    #
    # --- What changes with meta-labeling ---
    # Without side: barriers are symmetric (same width up and down, assume long)
    # With side:    barriers follow the primary model's direction.
    #               If primary says long (+1): upper = profit, lower = stop loss
    #               If primary says short (-1): upper = stop loss, lower = profit
    #
    # --- Inputs ---
    # close    : pd.Series — closing prices indexed by datetime
    # t_events : pd.DatetimeIndex — candidate entry dates (from CUSUM filter)
    # pt_sl    : list [pt_multiplier, sl_multiplier]
    #            Multipliers for profit target and stop loss barriers.
    #            e.g. [1, 1] = symmetric, [2, 1] = wider profit target
    # trgt     : pd.Series — barrier width per event (typically daily volatility)
    # min_ret  : float — minimum target return to include an event
    # t1       : pd.Series or False — vertical barrier timestamps
    # side     : pd.Series or None — primary model's predicted side per event
    #            +1 = primary model predicts price will rise (long)
    #            -1 = primary model predicts price will fall (short)
    #            None = no primary model, use symmetric barriers (same as get_events)
    #
    # --- Output ---
    # pd.DataFrame with columns:
    #   t1   : timestamp of first barrier touch
    #   trgt : barrier width used
    #   side : primary model's side (only if side was provided)

    # Import apply_pt_sl_on_t1 from triple_barrier to avoid code duplication
    from .triple_barrier import apply_pt_sl_on_t1

    # Step 1: Filter trgt to event dates and apply minimum return filter
    trgt = trgt.reindex(t_events, method='bfill')
    trgt = trgt[trgt > min_ret]

    # Step 2: Set up vertical barriers
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Step 3: Determine side and barrier multipliers
    # If side is None: assume long (+1) for all events, symmetric barriers
    # If side is provided: use primary model's predictions, may be asymmetric
    if side is None:
        # No primary model — assume long, use same multiplier for both barriers
        side_  = pd.Series(1., index=trgt.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]  # symmetric: same multiplier both sides
    else:
        # Primary model provided — use its side predictions
        # pt_sl[:2] applies the two multipliers as given
        side_  = side.loc[trgt.index]  # align side to filtered event dates
        pt_sl_ = pt_sl[:2]             # [profit_target_mult, stop_loss_mult]

    # Step 4: Build events DataFrame
    events = pd.concat(
        {'t1': t1, 'trgt': trgt, 'side': side_},
        axis=1
    ).dropna(subset=['trgt'])

    # Step 5: Find first barrier touch (single-threaded)
    df0 = apply_pt_sl_on_t1(
        close=close,
        events=events,
        pt_sl=pt_sl_
    )

    # Step 6: Take earliest touch across all barriers
    events['t1'] = df0.dropna(how='all').min(axis=1)

    # Drop side column if no primary model was used
    if side is None:
        events = events.drop('side', axis=1)

    return events


def get_bins_meta(events, close):
    # Expanded getBins with Meta-Labeling — AFML Chapter 3, Snippet 3.7, pages 51-52
    #
    # Extension of get_bins() from triple_barrier.py that handles meta-labeling.
    # The key difference is how labels are assigned when a primary model's side
    # is available:
    #
    # Case 1 — No side (standard triple barrier):
    #   bin = sign(return)
    #   bin ∈ {-1, 0, 1} — did price go up, stay flat, or go down?
    #
    # Case 2 — With side (meta-labeling):
    #   bin = 1 if the primary model's bet was correct (made money)
    #   bin = 0 if the primary model's bet was wrong (lost money)
    #   bin ∈ {0, 1} — was the primary model RIGHT or WRONG?
    #
    # This converts the problem to binary classification — much easier for ML.
    #
    # --- Inputs ---
    # events : pd.DataFrame — output of get_events_meta(), with columns:
    #            t1   : first barrier touch timestamp
    #            trgt : barrier width
    #            side : primary model's side (optional)
    # close  : pd.Series — closing prices indexed by datetime
    #
    # --- Output ---
    # pd.DataFrame with columns:
    #   ret : actual return from entry to first barrier touch
    #   bin : label
    #         Without side: -1 (loss), 0 (neutral), +1 (win) based on direction
    #         With side: 0 (primary model wrong) or 1 (primary model right)

    # Step 1: Drop events with no barrier touch
    events_ = events.dropna(subset=['t1'])

    # Step 2: Build aligned price series covering entry and exit dates
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    # Step 3: Compute returns from entry to first barrier touch
    out = pd.DataFrame(index=events_.index)

    # Get exit prices using nearest available bar
    exit_prices = []
    for t1 in events_['t1'].values:
        idx = close.index.searchsorted(t1)
        idx = min(idx, len(close) - 1)
        exit_prices.append(close.iloc[idx])

    entry_prices = close.reindex(events_.index, method='bfill').values
    out['ret'] = (np.array(exit_prices) / entry_prices) - 1

    # Step 4: Assign labels
    if 'side' in events_:
        # Meta-labeling case: multiply return by side
        # If primary said long (+1) and price went up → positive return → correct
        # If primary said short (-1) and price went down → negative return × -1 → correct
        out['ret'] = out['ret'] * events_['side']   # sign-adjusted return
        out['bin'] = np.sign(out['ret'])             # +1 if correct, -1 if wrong

        # Convert to binary: wrong (≤0) → 0, correct (>0) → 1
        # A return of exactly 0 (vertical barrier, neutral) counts as wrong
        out.loc[out['ret'] <= 0, 'bin'] = 0
    else:
        # Standard case: label by direction of price move
        out['bin'] = np.sign(out['ret'])

    return out


def drop_labels(events, min_pct=0.05):
    # Dropping Under-Populated Labels — AFML Chapter 3, Snippet 3.8, page 53
    #
    # Some ML classifiers perform poorly when class labels are highly imbalanced.
    # For example, if 95% of labels are +1 and only 5% are -1, a model can achieve
    # 95% accuracy by always predicting +1 — without learning anything useful.
    #
    # This function recursively removes observations associated with extremely rare
    # labels until either:
    #   - All remaining labels appear in at least min_pct of cases, OR
    #   - Only 2 labels remain (can't drop further without losing all signal)
    #
    # --- When to use this ---
    # Apply after get_bins_meta() if your label distribution is heavily skewed.
    # Typical use: drop labels that appear in less than 5% of observations.
    #
    # --- Inputs ---
    # events  : pd.DataFrame — must have a 'bin' column with labels
    # min_pct : float — minimum fraction of cases a label must appear in
    #           Default 0.05 = 5%. Labels appearing less often get dropped.
    #
    # --- Output ---
    # pd.DataFrame — same as events but with rare-label rows removed

    while True:
        # Compute fraction of cases each label accounts for
        df0 = events['bin'].value_counts(normalize=True)

        # Stop if all labels meet the minimum threshold
        # or if only 2 labels remain (can't drop any more)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break

        # Drop all observations with the rarest label
        rarest_label = df0.idxmin()
        print(f"Dropping label {rarest_label} ({df0.min()*100:.1f}% of cases)")
        events = events[events['bin'] != rarest_label]

    return events
