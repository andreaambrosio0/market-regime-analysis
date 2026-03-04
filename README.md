# Market Regime Analysis

Binary regime classification (**BULLISH** / **BEARISH**) for crypto markets using 8 focused signals. BTC drives 70% of the signal; altcoins provide 30% confirmation. When bullish, hold BTC. When bearish, hold cash.

## The 8 Signals

### BTC Signals (70% weight)

| # | Signal | Logic | What it captures |
|---|--------|-------|-----------------|
| 1 | **EMA Cross (10/21)** | 10-EMA > 21-EMA → bullish | Short-term momentum shift — turns first |
| 2 | **Price > 21 EMA** | Price above 21-EMA → bullish | Trend support — price holding above fast trend |
| 3 | **7d Momentum** | 7d return > 0 → bullish | Immediate direction — captures breakouts/breakdowns |
| 4 | **21d Trend** | 21d return > 0 → bullish | Short-term trend confirmation |
| 5 | **Volatility Regime** | RV 7d < RV 30d → bullish | Risk declining = calm, trending market |

### Alt Signals (30% weight)

| # | Signal | Logic | What it captures |
|---|--------|-------|-----------------|
| 6 | **Breadth 7d** | >50% coins positive 7d → bullish | Broad participation vs narrow rally |
| 7 | **Alt 7d Momentum** | Median alt 7d return > 0 → bullish | Risk appetite across market |
| 8 | **Drawdown Speed** | 7d drawdown intensity < -5% → bearish | Crash detection — liquidation cascades |

## How Regime Transitions Work

1. Each signal outputs +1 (bullish) or -1 (bearish)
2. BTC score = mean of 5 BTC signals; Alt score = mean of 3 alt signals
3. Composite = 0.70 × BTC score + 0.30 × Alt score
4. If composite > 0 → raw signal = BULLISH, else BEARISH
5. **2-day confirmation filter**: regime only flips after 2 consecutive days agree (prevents whipsaws)

At each transition, the system identifies which specific signals flipped and their numeric values (saved to `output/regime_transitions.csv`).

## Backtest

- **BULLISH** → 100% long BTC
- **BEARISH** → 100% cash (flat)
- **Benchmark**: Buy-and-hold BTC
- Next-day execution (signal on day t → trade on day t+1, no look-ahead bias)

## Setup

```bash
pip install -r requirements.txt
```

Place your data CSV at `data/market_data_daily_fixed.csv`.

## Run

```bash
python3 market_regime_analysis.py
```

## Output (~15 slides)

| Slide | Content |
|-------|---------|
| 1 | Current regime dashboard with all 8 signals |
| 2 | Full-history BTC price with regime overlay + EMA 10/21 |
| 3 | Composite score time series |
| 4 | Signal heatmap (8 signals over time) |
| 5 | Backtest equity curve: strategy vs buy-and-hold |
| 6 | Drawdown comparison |
| 7 | Year-by-year return comparison table |
| 8 | Rolling 90d Sharpe comparison |
| 9 | Monthly returns heatmap |
| 10 | Regime transition catalyst details |
| 11 | Breadth over time with regime |
| 12 | Volatility regime (RV 7d vs RV 30d) |
| 13-15 | Deep dives: BTC, ETH, SOL |

Additional outputs:
- `output/regime_data.csv` — full dataset with signals and regime labels
- `output/regime_transitions.csv` — every regime flip with catalyst details
- `output/all_slides_concatenated.png` — all slides stitched vertically

## Data

Daily crypto market data (54 assets, Dec 2020 – Mar 2026) from Coinbase.
