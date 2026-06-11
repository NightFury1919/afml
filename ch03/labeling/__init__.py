# labeling/__init__.py
# Chapter 3 — Labeling
# Exposes all labeling functions at the package level so you can write:
#   import labeling
#   labeling.fixed_time_horizon(...)
#   labeling.get_daily_vol(...)
#   labeling.get_events(...)
#   labeling.get_events_meta(...)
#   etc.

from .returns import fixed_time_horizon

from .triple_barrier import (
    get_daily_vol,
    add_vertical_barrier,
    get_events,
    apply_pt_sl_on_t1,
    get_bins,
)

from .meta_labeling import (
    get_events_meta,
    get_bins_meta,
    drop_labels,
)