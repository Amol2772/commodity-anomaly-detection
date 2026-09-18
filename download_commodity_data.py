"""
Download daily OHLCV data for commodity futures relevant to the
Commerzbank Commodity Trading Desk project (precious metals + oil & gas),
and run basic sanity checks before any modelling starts.

Requires: pip install yfinance pandas
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

# --- Config ---------------------------------------------------------------

TICKERS = {
    "GC=F": "gold",
    "SI=F": "silver",
    "CL=F": "wti_crude",
}

START_DATE = "2015-01-01"
END_DATE = None  # None = up to today
OUTPUT_DIR = Path("data")
Z_SCORE_FLAG_THRESHOLD = 4.0  # daily returns beyond this many std devs get flagged for manual review


def download_ticker(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} — check the ticker symbol.")

    # Newer yfinance versions return MultiIndex columns like ('Close', 'GC=F')
    # even for a single ticker. Flatten to plain column names so df["Close"]
    # etc. come back as a Series rather than a one-column DataFrame.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "date"
    return df


def sanity_check(df: pd.DataFrame, name: str) -> dict:
    """Run basic data-quality checks and return a summary dict."""
    close = df["Close"]
    returns = close.pct_change().dropna()

    # 1. Missing business days (rough check — futures markets have their own
    #    holiday calendars, so treat this as a flag to inspect, not a hard error)
    full_range = pd.bdate_range(df.index.min(), df.index.max())
    missing_days = full_range.difference(df.index)

    # 2. Extreme single-day moves — often a genuine event, but sometimes a
    #    contract-roll artifact in continuous futures series. Worth eyeballing.
    z_scores = (returns - returns.mean()) / returns.std()
    flagged = returns[z_scores.abs() > Z_SCORE_FLAG_THRESHOLD]

    # 3. Any NaN/zero-volume rows that shouldn't be there
    nan_rows = df[df.isna().any(axis=1)]
    zero_volume_rows = df[df["Volume"] == 0] if "Volume" in df.columns else pd.DataFrame()

    summary = {
        "name": name,
        "rows": len(df),
        "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
        "missing_business_days": len(missing_days),
        "extreme_moves_flagged": len(flagged),
        "nan_rows": len(nan_rows),
        "zero_volume_rows": len(zero_volume_rows),
    }

    if len(flagged) > 0:
        print(f"\n[{name}] Largest flagged single-day moves (check for roll artifacts vs real news):")
        print(flagged.reindex(flagged.abs().sort_values(ascending=False).index).head(5))

    return summary


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summaries = []

    for ticker, name in TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        df = download_ticker(ticker, START_DATE, END_DATE)

        out_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")

        summary = sanity_check(df, name)
        summaries.append(summary)

    print("\n=== Sanity check summary ===")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
