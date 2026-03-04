# Market Regime Analysis — Institutional Grade

Binary regime classification (**BULLISH** / **BEARISH**) for crypto markets using 18 sub-signals with BTC as the dominant signal (70% weight) and altcoins providing confirmation (30% weight).

## Methodology

### Signal Architecture (18 Sub-Signals)

**BTC Component (70% weight) — 10 signals:**
| Signal | Description |
|--------|-------------|
| 30d Trend | BTC 30-day return direction |
| 7d Momentum | Short-term momentum |
| Golden Cross | 50MA > 200MA |
| Price > 200MA | Long-term trend support |
| Vol Level | RV30d below expanding median |
| Vol Declining | RV30d < RV90d |
| Mid-term Trend | 60-day return direction |
| RV Term Structure | RV7d/RV90d ratio (contango vs inversion) |
| Hurst + Trend | Trend persistence confirmation via R/S exponent |
| Vol-Price Confirm | Volume confirms price direction |

**Altcoin Component (30% weight) — 8 signals:**
| Signal | Description |
|--------|-------------|
| Alt 30d Trend | Median altcoin 30-day return |
| Breadth 30d | % of assets with positive 30d return |
| Breadth 7d | % of assets with positive 7d return |
| Alt Momentum | Median altcoin 7d return |
| Alt Vol | Altcoin realized vol vs median |
| Correlation Convergence | Average pairwise correlation vs median |
| Return Dispersion | Cross-sectional return spread |
| Drawdown Intensity | Speed of price decline from highs |

### Institutional Analytics
- **Volatility Architecture**: RV term structure, variance risk premium (IV-RV), vol clustering/persistence
- **Momentum Persistence**: Hurst exponent (R/S method), cross-sectional return dispersion
- **Volume Dynamics**: Volume-price divergence, liquidity concentration
- **Cross-Asset Reflexivity**: Rolling beta of alts to BTC, pairwise correlation convergence
- **Structural Shifts**: Drawdown intensity, ADF stationarity testing
- **Per-Asset Scorecard**: 4-regime classification (Healthy Expansion, Euphoria, Capitulation, Accumulation)

### Backtest
- **BULLISH** → 100% long BTC
- **BEARISH** → 100% cash (flat)
- **Benchmark**: Buy-and-hold BTC from day 1
- Next-day execution (no look-ahead bias), 3-day confirmation filter

## Setup

```bash
pip install -r requirements.txt
```

Place your data CSV in `data/market_data_daily_fixed.csv`.

## Run

```bash
python3 market_regime_analysis.py
```

## Output (66 Slides + Concatenated PNG)

| Section | Slides | Content |
|---------|--------|---------|
| A: Executive Summary | 1–4 | Current regime dashboard, full history, composite score, BTC vs alt decomposition |
| B: Backtest Results | 5–15 | Equity curves, drawdowns, year-by-year returns/Sharpe/DD, rolling returns |
| C: Indicator Deep Dive | 16–22 | Moving averages, breadth, volatility, volume, signal heatmaps, agreement |
| D: Risk Analysis | 23–27 | Rolling Sharpe, monthly heatmaps, return distribution, rolling vol, regime duration |
| E: Coin Analysis | 28–43 | All-coin cumulative, regime returns, volatility, 8 deep dives, correlation matrices |
| F: Volatility Architecture | 44–47 | RV term structure, VRP (BTC/ETH), vol clustering, vol heatmap |
| G: Momentum & Persistence | 48–51 | Hurst exponent, return dispersion, signal IC, forward return validation |
| H: Volume & Liquidity | 52–54 | Volume-price divergence, liquidity concentration, volume dashboard |
| I: Cross-Asset Reflexivity | 55–58 | Alt beta to BTC, correlation convergence, beta heatmap, decoupling scatter |
| J: Structural Shifts | 59–61 | Drawdown intensity, ADF stationarity, enhanced regime overview |
| K: Per-Asset Scorecard | 62–65 | 4-regime stacked area, current table, transition flows, forward return validation |
| L: Enhanced Backtest | 66 | Full dashboard with equity, drawdowns, Sharpe, stats table |

- `output/all_slides_concatenated.png` — All 66 slides stitched vertically
- `output/regime_data.csv` — Full processed dataset with all regime labels and signals

## Data

Daily crypto market data (54 assets, Dec 2020 – Mar 2026) from Coinbase including:
- Price, volume, multi-horizon returns (1d to 365d)
- Forward returns (1d to 30d)
- Realized volatility (7d to 90d)
- Implied volatility (7d to 60d) — BTC and ETH only, from Sep 2021
- Volume moving averages and ratios
