from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-31", periods=240, freq="ME")

    regimes = np.repeat([0, 1, 2, 3], repeats=[70, 50, 60, 60])
    market_return = np.select(
        [regimes == 0, regimes == 1, regimes == 2, regimes == 3],
        [
            rng.normal(0.008, 0.025, len(dates)),
            rng.normal(-0.018, 0.060, len(dates)),
            rng.normal(0.014, 0.035, len(dates)),
            rng.normal(-0.004, 0.030, len(dates)),
        ],
    )
    volatility = np.select(
        [regimes == 0, regimes == 1, regimes == 2, regimes == 3],
        [
            rng.normal(0.030, 0.006, len(dates)),
            rng.normal(0.090, 0.020, len(dates)),
            rng.normal(0.045, 0.010, len(dates)),
            rng.normal(0.055, 0.012, len(dates)),
        ],
    )
    term_spread_change = rng.normal(0.0, 0.020, len(dates)) - 0.2 * market_return
    credit_spread_change = rng.normal(0.0, 0.015, len(dates)) - 0.4 * market_return + 0.3 * volatility

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "market_return": market_return,
            "volatility": volatility,
            "term_spread_change": term_spread_change,
            "credit_spread_change": credit_spread_change,
        }
    )

    output = Path("data/processed/sample_monthly_features.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()

