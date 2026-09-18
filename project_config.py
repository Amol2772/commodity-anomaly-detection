"""
Shared configuration for the commodity anomaly-detection project.

Both notebooks import from this file so parameters (ticker list, rolling window,
thresholds) can't silently drift apart between exploration and detection.

If you re-tune Z_THRESHOLD using the sweep in the EDA notebook (section 7),
change it HERE ONLY — both notebooks pick it up automatically on their next run.

Place this file in the same folder as the notebooks and the data/ directory.
"""

import numpy as np

TICKERS = {
    "gold": "Gold (GC=F)",
    "silver": "Silver (SI=F)",
    "wti_crude": "WTI Crude (CL=F)",
}

ROLL_WINDOW = 60       # rolling window (trading days) for the baseline z-score and volatility
Z_THRESHOLD = 4.0      # robust (MAD-based) z-score threshold — tune via EDA notebook, section 7
MAD_SCALE = 1.4826     # scales MAD to be comparable to a normal-distribution std
ANNUALISATION = np.sqrt(252)

IF_CONTAMINATION = 0.02  # Isolation Forest expected anomaly fraction — see anomaly notebook, section 2


def rolling_mad(series, window):
    """Rolling Median Absolute Deviation.

    Far more resistant to a single extreme value distorting the estimate than a
    rolling std is — see the EDA notebook, section 7, for why that matters here
    (WTI's 2020-04-20 crash was inflating the rolling std for ~2 months afterward).
    """
    return series.rolling(window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
