import sys
import os
import time

# Add ch02, ch03, ch04, and project root to path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ch02 = os.path.join(root, 'ch02')
ch03 = os.path.join(root, 'ch03')
ch04 = os.path.dirname(__file__)
sys.path.insert(0, root)
sys.path.insert(0, ch02)
sys.path.insert(0, ch03)
sys.path.insert(0, ch04)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import bars      # from ch02
import labeling  # from ch03

from sample_weights.co_events             import mp_num_co_events
from sample_weights.uniqueness            import get_average_uniqueness
from sample_weights.indicator_matrix      import get_ind_matrix
from sample_weights.avg_uniqueness_matrix import get_avg_uniqueness
from sample_weights.sequential_bootstrap  import seq_bootstrap
from sample_weights.monte_carlo                  import get_rnd_t1, aux_mc, main_mc
from sample_weights.return_attribution           import get_sample_weights
from sample_weights.time_decay                   import get_time_decay
from sample_weights.real_data_bootstrap_comparison import compare_bootstrap_on_real_events

sns.set_style("whitegrid")


if __name__ == '__main__':

    # =============================================================================
    # Load Raw Tick Data and Generate Dollar Bars + Events
    # =============================================================================
    data_path = os.path.join(ch02, 'input_data', 'BTCTUSD-trades-2026-03.csv')

    print("Loading tick data...")
    raw = pd.read_csv(data_path, header=None,
                      names=['TradeID', 'Price', 'Volume', 'QuoteVolume',
                             'Timestamp', 'IsBuyerMaker', 'IsBestMatch'])

    raw['Date']  = pd.to_datetime(raw['Timestamp'], unit='us')
    raw['Label'] = raw['IsBuyerMaker'].apply(lambda x: -1 if x else 1)

    df = raw[['Date', 'Price', 'Volume', 'Label']].copy()
    df['Dollar'] = df['Price'] * df['Volume']
    df = bars.delta(df)

    print(f"Loaded {len(df):,} ticks")

    print("\nGenerating dollar bars...")
    dollar_bars = bars.dollar_bars(df, thresh=10000)
    dollar_bars = dollar_bars.set_index('Date')
    close = dollar_bars['Close']
    print(f"Dollar bars: {len(close)} bars")

    print("\nApplying CUSUM filter...")
    cusum_df = pd.DataFrame({'Date': close.index, 'Price': close.values})
    events_dates = bars.cusum_filter(cusum_df, h=500)
    print(f"CUSUM events: {len(events_dates)}")

    print("\nComputing daily volatility and triple barrier events...")
    daily_vol = labeling.get_daily_vol(close, span0=100)
    t1_series = labeling.add_vertical_barrier(close, events_dates, num_days=3)
    tb_events = labeling.get_events(
        close=close, t_events=events_dates, pt_sl=[1, 1],
        trgt=daily_vol, min_ret=0.005, t1=t1_series
    )
    tb_events = tb_events.dropna(subset=['t1'])
    print(f"Triple barrier events with valid t1: {len(tb_events)}")

    # =============================================================================
    # Section 4.2/4.5 — Concurrency and Average Uniqueness
    # =============================================================================
    print("\nComputing concurrency and average uniqueness...")
    tw = get_average_uniqueness(close, tb_events, num_threads=1)
    print(f"Average uniqueness across all events: {tw.mean():.4f}")
    print(f"Min: {tw.min():.4f}  Max: {tw.max():.4f}")

    # --- Plot 1: Average Uniqueness Distribution ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(tw.values, bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax.axvline(tw.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {tw.mean():.3f}')
    ax.set_title("Distribution of Average Uniqueness — Real Triple Barrier Events", fontsize=12)
    ax.set_xlabel("Average Uniqueness (tW)")
    ax.set_ylabel("Number of Events")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Section 4.5.3-4.5.4 — Standard vs Sequential Bootstrap (Figure 4.2 replica)
    # =============================================================================
    # Uses main_mc() from monte_carlo.py, which supports multiprocessing via
    # num_threads. Each trial is fully independent (fresh random data, fresh
    # bootstrap draws), making this an ideal candidate for parallelization.
    #
    # IMPORTANT — Windows multiprocessing requirement:
    # This whole section must run inside `if __name__ == '__main__':` (see the
    # bottom of this file) or multiprocessing will fail/hang on Windows. If you
    # don't need the speedup, set NUM_THREADS = 1 below and it runs identically
    # to before, just single-threaded.
    NUM_THREADS = 6   # set to 1 to disable multiprocessing; my machine has 6 cores

    print(f"\nRunning Monte Carlo comparison: standard vs sequential bootstrap "
          f"(num_threads={NUM_THREADS})...")
    mc_start = time.time()
    mc_result = main_mc(
        num_obs=10, num_bars=100, max_h=5,
        num_iters=300, num_threads=NUM_THREADS
    )
    mc_elapsed = time.time() - mc_start
    print(f"Monte Carlo runtime: {mc_elapsed:.2f}s  (num_threads={NUM_THREADS})")
    std_u_vals = mc_result['stdU'].tolist()
    seq_u_vals = mc_result['seqU'].tolist()

    print(f"Standard bootstrap — mean: {np.mean(std_u_vals):.4f}, median: {np.median(std_u_vals):.4f}")
    print(f"Sequential bootstrap — mean: {np.mean(seq_u_vals):.4f}, median: {np.median(seq_u_vals):.4f}")

    # --- Plot 2: Figure 4.2 replica — histogram comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(std_u_vals, bins=20, alpha=0.5, color='grey', label='Standard bootstrap', density=True)
    ax.hist(seq_u_vals, bins=20, alpha=0.5, color='steelblue', label='Sequential bootstrap', density=True)
    ax.axvline(np.median(std_u_vals), color='grey', linestyle='--', linewidth=2)
    ax.axvline(np.median(seq_u_vals), color='steelblue', linestyle='--', linewidth=2)
    ax.set_title(f"Monte Carlo: Standard vs Sequential Bootstrap Uniqueness (300 trials)", fontsize=12)
    ax.set_xlabel("Average Uniqueness")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Section 4.5.3-4.5.4 — Standard vs Sequential Bootstrap on REAL data
    # =============================================================================
    # The plot above proves the GENERAL claim using synthetic, randomly generated
    # overlap scenarios. This plot answers the more practical question: how much
    # does sequential bootstrap actually help on MY OWN real, labeled events?
    #
    # We subsample down to a small, contiguous block of real events (kept small
    # so this runs in a few seconds, not minutes) and repeatedly bootstrap from
    # that real overlap structure.
    print("\nRunning bootstrap comparison on REAL triple barrier events...")
    real_result = compare_bootstrap_on_real_events(
        close, tb_events, max_events=12, n_trials=15, seed=42
    )
    print(f"Real events used: {real_result['n_events']}  (bars spanned: {real_result['n_bars']})")
    print(f"Standard bootstrap (real data) — mean: {np.mean(real_result['std_vals']):.4f}")
    print(f"Sequential bootstrap (real data) — mean: {np.mean(real_result['seq_vals']):.4f}")

    # --- Plot: Standard vs Sequential Bootstrap on real events ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(real_result['std_vals'], bins=10, alpha=0.5, color='grey',
            label='Standard bootstrap (real data)', density=True)
    ax.hist(real_result['seq_vals'], bins=10, alpha=0.5, color='seagreen',
            label='Sequential bootstrap (real data)', density=True)
    ax.axvline(np.mean(real_result['std_vals']), color='grey', linestyle='--', linewidth=2)
    ax.axvline(np.mean(real_result['seq_vals']), color='seagreen', linestyle='--', linewidth=2)
    ax.set_title(
        f"Real Data: Standard vs Sequential Bootstrap "
        f"({real_result['n_events']} of {len(tb_events)} real events, 15 trials)",
        fontsize=12
    )
    ax.set_xlabel("Average Uniqueness")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Section 4.6 — Sample Weights by Return Attribution
    # =============================================================================
    print("\nComputing sample weights by return attribution...")
    sample_w = get_sample_weights(close, tb_events, num_threads=1)
    print(f"Sample weights sum: {sample_w.sum():.4f} (should equal {len(sample_w)})")
    print(f"Mean weight: {sample_w.mean():.4f}")

    # --- Plot 3: Sample Weight Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sample Weights by Absolute Return Attribution (Section 4.6)", fontsize=12)

    axes[0].hist(sample_w.values, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[0].axvline(1.0, color='black', linestyle='--', label='Mean weight = 1.0')
    axes[0].set_title("Weight Distribution")
    axes[0].set_xlabel("Sample Weight")
    axes[0].set_ylabel("Number of Events")
    axes[0].legend()

    axes[1].scatter(sample_w.index, sample_w.values, alpha=0.5, s=15, color='darkorange')
    axes[1].set_title("Sample Weight Over Time")
    axes[1].set_xlabel("Event Date")
    axes[1].set_ylabel("Sample Weight")
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Section 4.7 — Time Decay Curves
    # =============================================================================
    print("\nComputing time decay curves for several clf_last_w values...")

    fig, ax = plt.subplots(figsize=(12, 6))
    for clw, color in zip([1.0, 0.5, 0.0, -0.5], ['green', 'blue', 'orange', 'red']):
        decay = get_time_decay(tw, clf_last_w=clw)
        ax.plot(decay.index, decay.values, label=f'clf_last_w = {clw}', color=color, linewidth=1.5)

    ax.set_title("Time Decay Curves for Different clf_last_w Values (Section 4.7)", fontsize=12)
    ax.set_xlabel("Event Date")
    ax.set_ylabel("Decayed Weight")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Summary
    # =============================================================================
    print("\n" + "="*60)
    print("CHAPTER 4 SUMMARY")
    print("="*60)
    print(f"Total triple barrier events analyzed: {len(tb_events)}")
    print(f"Average uniqueness (tW):              {tw.mean():.4f}")
    print(f"Standard bootstrap uniqueness (synthetic MC): {np.mean(std_u_vals):.4f}")
    print(f"Sequential bootstrap uniqueness (synthetic MC): {np.mean(seq_u_vals):.4f}")
    print(f"Standard bootstrap uniqueness (REAL data):    {np.mean(real_result['std_vals']):.4f}")
    print(f"Sequential bootstrap uniqueness (REAL data):  {np.mean(real_result['seq_vals']):.4f}")
    print(f"Improvement from sequential bootstrap (real data): "
          f"{(np.mean(real_result['seq_vals'])/np.mean(real_result['std_vals']) - 1)*100:.1f}%")
    print(f"Sample weight (return attribution) mean: {sample_w.mean():.4f}")

    # =============================================================================
    # TDD TEST RESULTS — Chapter 4
    # pytest ch04/tests/test_ch04.py -v
    # Run date: 2026-06-23  |  Python 3.10.20  |  pytest 9.0.3
    # =============================================================================
    # TestMpNumCoEvents::test_known_concurrency_values                       PASSED
    # TestMpNumCoEvents::test_overlap_bar_has_highest_concurrency            PASSED
    # TestMpNumCoEvents::test_returns_series                                 PASSED
    # TestMpNumCoEvents::test_open_event_uses_last_bar_as_end                PASSED
    # TestMpSampleTw::test_known_uniqueness_values                           PASSED
    # TestMpSampleTw::test_non_overlapping_event_has_uniqueness_one          PASSED
    # TestMpSampleTw::test_uniqueness_bounded_between_zero_and_one           PASSED
    # TestGetAverageUniqueness::test_returns_series                          PASSED
    # TestGetAverageUniqueness::test_matches_known_values                    PASSED
    # TestGetAverageUniqueness::test_output_length_matches_events            PASSED
    # TestGetIndMatrix::test_matches_book_example                            PASSED
    # TestGetIndMatrix::test_shape_is_bars_by_events                         PASSED
    # TestGetIndMatrix::test_only_zeros_and_ones                             PASSED
    # TestGetIndMatrix::test_single_observation_no_overlap                   PASSED
    # TestGetAvgUniqueness::test_matches_book_example                        PASSED
    # TestGetAvgUniqueness::test_matches_bar_by_bar_method                   PASSED
    # TestGetAvgUniqueness::test_returns_series                              PASSED
    # TestSeqBootstrap::test_default_sample_length                           PASSED
    # TestSeqBootstrap::test_custom_sample_length                            PASSED
    # TestSeqBootstrap::test_all_drawn_values_are_valid_columns              PASSED
    # TestSeqBootstrap::test_deterministic_with_seed                        PASSED
    # TestSeqBootstrap::test_probability_after_first_draw_matches_book       PASSED
    # TestSeqBootstrap::test_already_drawn_observation_gets_lowest_probability PASSED
    # TestSeqBootstrap::test_non_overlapping_observation_gets_highest_probability PASSED
    # TestGetRndT1::test_output_is_sorted                                    PASSED
    # TestGetRndT1::test_correct_number_of_observations                      PASSED
    # TestGetRndT1::test_durations_within_bounds                             PASSED
    # TestGetRndT1::test_start_bars_within_bounds                            PASSED
    # TestAuxMc::test_returns_dict_with_expected_keys                        PASSED
    # TestAuxMc::test_uniqueness_values_in_valid_range                       PASSED
    # TestAuxMc::test_sequential_tends_to_beat_standard_on_average           PASSED
    # TestMpSampleW::test_known_weight_values                                PASSED
    # TestMpSampleW::test_weights_are_non_negative                           PASSED
    # TestMpSampleW::test_largest_return_gets_largest_weight                 PASSED
    # TestGetSampleWeights::test_weights_sum_to_number_of_observations       PASSED
    # TestGetSampleWeights::test_returns_series                              PASSED
    # TestGetSampleWeights::test_weights_non_negative                       PASSED
    # TestGetTimeDecay::test_no_decay_when_clf_last_w_is_one                 PASSED
    # TestGetTimeDecay::test_known_values_clf_last_w_half                    PASSED
    # TestGetTimeDecay::test_known_values_clf_last_w_zero                   PASSED
    # TestGetTimeDecay::test_known_values_negative_clf_last_w_hard_excludes  PASSED
    # TestGetTimeDecay::test_newest_observation_always_gets_weight_one       PASSED
    # TestGetTimeDecay::test_weights_never_negative                          PASSED
    # TestGetTimeDecay::test_monotonically_increasing_with_time              PASSED
    # TestCompareBootstrapOnRealEvents::test_returns_expected_keys           PASSED
    # TestCompareBootstrapOnRealEvents::test_n_events_never_exceeds_max_events PASSED
    # TestCompareBootstrapOnRealEvents::test_no_subsampling_when_fewer_events_than_cap PASSED
    # TestCompareBootstrapOnRealEvents::test_correct_number_of_trials        PASSED
    # TestCompareBootstrapOnRealEvents::test_uniqueness_values_in_valid_range PASSED
    # TestCompareBootstrapOnRealEvents::test_drops_events_with_unresolved_t1 PASSED
    # TestCompareBootstrapOnRealEvents::test_reproducible_with_seed          PASSED
    # TestCompareBootstrapOnRealEvents::test_runs_within_reasonable_time     PASSED
    # TestCompareBootstrapOnRealEvents::test_indicator_matrix_shape_matches_n_events PASSED
    # -----------------------------------------------------------------------------
    # 53 passed in 21.97s
    # =============================================================================
