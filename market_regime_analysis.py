"""
================================================================================
MARKET REGIME ANALYSIS & BACKTEST — Crypto Markets (v4 Simplified)
================================================================================

PURPOSE:
    Determine whether the crypto market is currently BULLISH or BEARISH using
    8 focused, rule-based signals. Hold BTC when bullish, hold cash when bearish.

HOW IT WORKS:
    8 independent signals each vote bullish (+1) or bearish (-1).
    - 5 signals look at BTC (70% weight): EMA cross, trend support, momentum, vol
    - 3 signals look at altcoins (30% weight): breadth, momentum, crash detection
    Composite > 0 → BULLISH. Otherwise → BEARISH.
    A 2-day confirmation filter prevents reacting to single-day noise.

SIGNALS:
    BTC (70%):
      1. EMA Cross (10/21)  — short-term momentum shift
      2. Price > 21 EMA     — trend support
      3. 7d Momentum        — immediate direction
      4. 21d Trend          — short-term trend confirmation
      5. Volatility Regime  — RV 7d < RV 30d = calm trending market

    Alts (30%):
      6. Breadth 7d         — broad participation vs narrow rally
      7. Alt 7d Momentum    — risk appetite across market
      8. Drawdown Speed     — crash/liquidation cascade detection

BACKTEST:
    BULLISH → 100% long BTC | BEARISH → 100% cash
    Benchmark: Buy-and-hold BTC from day 1
    Execution: signal on day t → trade on day t+1 (no look-ahead bias)

Data: Coinbase daily crypto data (54 assets, 2018 – Mar 2026)
Output: ~15 visualization slides + regime_data.csv + regime_transitions.csv
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from datetime import timedelta
import os
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "market_data_daily_fixed.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BTC_WEIGHT = 0.70
ALT_WEIGHT = 0.30

CORE_ALTS = ["eth", "sol", "xrp", "ada", "doge", "avax", "link", "dot", "bnb", "ltc",
             "uni", "near", "sui", "apt", "arb"]

# Colors
C_BULL = "#00d48a"
C_BEAR = "#ff4757"
C_BTC = "#f7931a"
C_STRATEGY = "#58a6ff"
C_CASH = "#666666"
C_BG = "#0a0e17"
C_PANEL = "#0e1420"
C_GRID = "#1a1f2e"
C_TEXT = "#c8cdd5"
C_TEXT_DIM = "#6b7280"

plt.rcParams.update({
    "figure.facecolor": C_BG,
    "axes.facecolor": C_PANEL,
    "axes.edgecolor": "#252a36",
    "axes.labelcolor": C_TEXT,
    "text.color": C_TEXT,
    "xtick.color": C_TEXT_DIM,
    "ytick.color": C_TEXT_DIM,
    "grid.color": C_GRID,
    "grid.alpha": 0.4,
    "font.family": "monospace",
    "font.size": 10,
})

SLIDE_NUM = [0]

SIGNAL_NAMES = {
    "btc_s_ema_cross": "EMA Cross 10/21",
    "btc_s_above_21ema": "Price > 21 EMA",
    "btc_s_momentum_7d": "7d Momentum",
    "btc_s_trend_21d": "21d Trend",
    "btc_s_vol_regime": "Volatility Regime",
    "alt_s_breadth_7d": "Breadth 7d",
    "alt_s_momentum_7d": "Alt 7d Momentum",
    "alt_s_drawdown": "Drawdown Speed",
}


def save_slide(fig, name, desc):
    """Save figure and print description."""
    SLIDE_NUM[0] += 1
    fname = f"{SLIDE_NUM[0]:02d}_{name}.png"
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  [{SLIDE_NUM[0]:02d}] {fname}")
    print(f"       -> {desc}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values(["time", "asset"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING (simplified — no institutional analytics)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df):
    """
    Build daily features for 8 signals.
    BTC (70%): EMA 10/21, price vs 21 EMA, 7d return, 21d return, RV 7d vs 30d.
    Alts (30%): breadth 7d, alt 7d momentum, drawdown intensity.
    """
    btc = df[df["asset"] == "btc"].set_index("time").sort_index()
    alts = df[df["asset"].isin(CORE_ALTS)].copy()
    all_assets = df.copy()

    # --- BTC features ---
    feat = pd.DataFrame(index=btc.index)
    feat["btc_price"] = btc["price"]
    feat["btc_return_1d"] = btc["return_1d"]
    feat["btc_return_7d"] = btc["return_7d"]
    feat["btc_rv_7d"] = btc["rv_7d"]
    feat["btc_rv_30d"] = btc["rv_30d"]

    # 21d return: compute from price if not in data
    if "return_21d" in btc.columns:
        feat["btc_return_21d"] = btc["return_21d"]
    else:
        feat["btc_return_21d"] = btc["price"].pct_change(21) * 100

    # EMAs: 10 and 21 for faster signals
    feat["btc_ema_10"] = btc["price"].ewm(span=10).mean()
    feat["btc_ema_21"] = btc["price"].ewm(span=21).mean()

    # --- Alt features ---
    alt_ret_7 = alts.groupby("time")["return_7d"].median().rename("alt_return_7d")

    # Breadth 7d (all assets)
    breadth_7d = all_assets.groupby("time").apply(
        lambda g: (g["return_7d"] > 0).mean(), include_groups=False
    ).rename("breadth_7d")

    # Drawdown intensity: 7d change in drawdown from 90d high
    rolling_high = feat["btc_price"].rolling(90).max()
    drawdown_pct = (feat["btc_price"] / rolling_high - 1) * 100
    feat["drawdown_intensity"] = drawdown_pct.diff(7)

    for s in [alt_ret_7, breadth_7d]:
        feat = feat.join(s, how="left")

    feat = feat.dropna(subset=["btc_return_7d", "btc_rv_7d", "btc_rv_30d",
                                "btc_ema_10", "btc_ema_21"])
    feat.index.name = "time"
    feat = feat.reset_index()

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIME INDICATOR (8 signals, 2-day confirmation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime(feat):
    """
    Binary regime: BULLISH or BEARISH from 8 signals.

    BTC (70% weight) — 5 signals:
      1. EMA Cross: 10-EMA > 21-EMA
      2. Price > 21 EMA
      3. 7d Momentum: 7d return > 0
      4. 21d Trend: 21d return > 0
      5. Volatility Regime: RV 7d < RV 30d (calm = bullish)

    Alt (30% weight) — 3 signals:
      6. Breadth 7d: >50% coins positive 7d
      7. Alt 7d Momentum: median alt 7d return > 0
      8. Drawdown Speed: 7d drawdown intensity < -5% = bearish

    Composite = 0.70 * mean(BTC signals) + 0.30 * mean(alt signals)
    2-day confirmation filter to reduce whipsaws.
    """
    f = feat.copy()

    # ── BTC SIGNALS (5) ──

    # 1. EMA Cross: 10-EMA > 21-EMA → bullish (short-term momentum shift)
    f["btc_s_ema_cross"] = np.where(f["btc_ema_10"] > f["btc_ema_21"], 1, -1)

    # 2. Price > 21 EMA → bullish (trend support)
    f["btc_s_above_21ema"] = np.where(f["btc_price"] > f["btc_ema_21"], 1, -1)

    # 3. 7d Momentum: 7d return > 0 → bullish (immediate direction)
    f["btc_s_momentum_7d"] = np.where(f["btc_return_7d"] > 0, 1, -1)

    # 4. 21d Trend: 21d return > 0 → bullish (short-term trend confirmation)
    f["btc_s_trend_21d"] = np.where(f["btc_return_21d"] > 0, 1, -1)

    # 5. Volatility Regime: RV 7d < RV 30d → bullish (risk declining)
    f["btc_s_vol_regime"] = np.where(f["btc_rv_7d"] < f["btc_rv_30d"], 1, -1)

    btc_signals = ["btc_s_ema_cross", "btc_s_above_21ema", "btc_s_momentum_7d",
                   "btc_s_trend_21d", "btc_s_vol_regime"]
    f["btc_score"] = f[btc_signals].mean(axis=1)

    # ── ALT SIGNALS (3) ──

    # 6. Breadth 7d: >50% coins positive 7d → bullish
    f["alt_s_breadth_7d"] = np.where(f["breadth_7d"] > 0.5, 1, -1)

    # 7. Alt 7d Momentum: median alt 7d return > 0 → bullish
    f["alt_s_momentum_7d"] = np.where(f["alt_return_7d"] > 0, 1, -1)

    # 8. Drawdown Speed: 7d drawdown intensity < -5% → bearish
    dd = f["drawdown_intensity"].fillna(0)
    f["alt_s_drawdown"] = np.where(dd < -5, -1,
                                    np.where(dd > 2, 1, 0)).astype(float)
    f["alt_s_drawdown"] = f["alt_s_drawdown"].replace(0, np.nan).ffill().fillna(0)

    alt_signals = ["alt_s_breadth_7d", "alt_s_momentum_7d", "alt_s_drawdown"]
    f["alt_score"] = f[alt_signals].mean(axis=1)

    # Composite
    f["raw_score"] = BTC_WEIGHT * f["btc_score"] + ALT_WEIGHT * f["alt_score"]
    f["raw_signal"] = np.where(f["raw_score"] > 0, 1, 0)

    # 2-day confirmation filter (faster signals deserve faster confirmation)
    f["signal_ma"] = f["raw_signal"].rolling(2, min_periods=1).mean()
    f["regime"] = np.where(f["signal_ma"] > 0.5, "BULLISH", "BEARISH")
    f["regime_num"] = np.where(f["regime"] == "BULLISH", 1, 0)

    return f


# ─────────────────────────────────────────────────────────────────────────────
# 4. CATALYST IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def identify_catalysts(feat):
    """
    For each regime transition, identify which signals flipped and their values.
    Returns a DataFrame of transitions and prints details to terminal.
    """
    f = feat.copy()
    f["prev_regime"] = f["regime"].shift(1)
    transitions = f[f["regime"] != f["prev_regime"]].copy()
    transitions = transitions.iloc[1:]  # skip first row (no previous)

    all_signals = list(SIGNAL_NAMES.keys())
    rows = []

    for idx, row in transitions.iterrows():
        date = row["time"]
        old_regime = row["prev_regime"]
        new_regime = row["regime"]

        # Find previous day to compare signals
        prev_idx = f.index[f.index.get_loc(idx) - 1]
        prev_row = f.loc[prev_idx]

        flipped = []
        details = []
        for sig in all_signals:
            old_val = prev_row[sig]
            new_val = row[sig]
            if old_val != new_val:
                flipped.append(SIGNAL_NAMES[sig])
                # Build detail string with numeric context
                detail = _signal_detail(sig, row, SIGNAL_NAMES[sig])
                details.append(detail)

        rows.append({
            "date": date,
            "from_regime": old_regime,
            "to_regime": new_regime,
            "raw_score": row["raw_score"],
            "btc_score": row["btc_score"],
            "alt_score": row["alt_score"],
            "signals_flipped": "; ".join(flipped) if flipped else "Confirmation filter",
            "details": " | ".join(details) if details else "2-day filter triggered transition",
        })

    transitions_df = pd.DataFrame(rows)
    return transitions_df


def _signal_detail(sig, row, name):
    """Build a human-readable detail string for a signal flip."""
    if sig == "btc_s_ema_cross":
        return f"{name}: 10-EMA ${row['btc_ema_10']:,.0f} vs 21-EMA ${row['btc_ema_21']:,.0f}"
    elif sig == "btc_s_above_21ema":
        return f"{name}: price ${row['btc_price']:,.0f} vs 21-EMA ${row['btc_ema_21']:,.0f}"
    elif sig == "btc_s_momentum_7d":
        return f"{name}: 7d return {row['btc_return_7d']:.1f}%"
    elif sig == "btc_s_trend_21d":
        return f"{name}: 21d return {row['btc_return_21d']:.1f}%"
    elif sig == "btc_s_vol_regime":
        return f"{name}: RV 7d {row['btc_rv_7d']:.1f}% vs RV 30d {row['btc_rv_30d']:.1f}%"
    elif sig == "alt_s_breadth_7d":
        return f"{name}: {row['breadth_7d']*100:.0f}% (threshold: 50%)"
    elif sig == "alt_s_momentum_7d":
        return f"{name}: median alt 7d return {row['alt_return_7d']:.2f}%"
    elif sig == "alt_s_drawdown":
        return f"{name}: 7d drawdown intensity {row['drawdown_intensity']:.1f}% (threshold: -5%)"
    return f"{name}: flipped"


def print_catalysts(transitions_df):
    """Print regime transitions with catalyst details to terminal."""
    print(f"\n  REGIME TRANSITIONS ({len(transitions_df)} total)")
    print(f"  {'='*80}")
    for _, row in transitions_df.iterrows():
        arrow = "BULL->BEAR" if row["to_regime"] == "BEARISH" else "BEAR->BULL"
        color = "red" if row["to_regime"] == "BEARISH" else "green"
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        print(f"\n  {arrow} on {date_str} (score: {row['raw_score']:+.3f})")
        print(f"    Signals flipped: {row['signals_flipped']}")
        if row["details"] and row["details"] != "2-day filter triggered transition":
            for detail in row["details"].split(" | "):
                print(f"      - {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(feat):
    """
    Strategy: long BTC when BULLISH, cash when BEARISH.
    Next-day execution to avoid look-ahead bias.
    """
    bt = feat[["time", "btc_price", "btc_return_1d", "regime", "regime_num"]].copy()
    bt = bt.dropna(subset=["btc_return_1d"]).reset_index(drop=True)

    bt["position"] = bt["regime_num"].shift(1).fillna(0)
    bt["strat_return"] = bt["position"] * bt["btc_return_1d"] / 100
    bt["bh_return"] = bt["btc_return_1d"] / 100

    bt["strat_equity"] = (1 + bt["strat_return"]).cumprod()
    bt["bh_equity"] = (1 + bt["bh_return"]).cumprod()

    bt["strat_peak"] = bt["strat_equity"].cummax()
    bt["strat_dd"] = (bt["strat_equity"] / bt["strat_peak"] - 1) * 100
    bt["bh_peak"] = bt["bh_equity"].cummax()
    bt["bh_dd"] = (bt["bh_equity"] / bt["bh_peak"] - 1) * 100

    bt["year"] = pd.to_datetime(bt["time"]).dt.year

    return bt


def compute_stats(returns, name="Strategy"):
    """Compute standard performance metrics."""
    total = (1 + returns).prod() - 1
    n_years = len(returns) / 365.25
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(365.25)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365.25) if returns.std() > 0 else 0
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate = (returns[returns != 0] > 0).mean() if (returns != 0).any() else 0
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf
    time_in = (returns != 0).mean()

    return {
        "name": name,
        "total_return": total * 100,
        "cagr": cagr * 100,
        "volatility": vol * 100,
        "sharpe": sharpe,
        "max_drawdown": max_dd * 100,
        "calmar": calmar,
        "win_rate": win_rate * 100,
        "profit_factor": profit_factor,
        "time_in_market": time_in * 100,
    }


def yearly_stats(bt):
    """Year-by-year performance comparison."""
    years = sorted(bt["year"].unique())
    rows = []
    for yr in years:
        sub = bt[bt["year"] == yr]
        if len(sub) < 10:
            continue
        s = compute_stats(sub["strat_return"], f"Strategy {yr}")
        b = compute_stats(sub["bh_return"], f"B&H {yr}")
        rows.append({
            "year": yr,
            "strat_return": s["total_return"],
            "bh_return": b["total_return"],
            "strat_sharpe": s["sharpe"],
            "bh_sharpe": b["sharpe"],
            "strat_max_dd": s["max_drawdown"],
            "bh_max_dd": b["max_drawdown"],
            "strat_vol": s["volatility"],
            "bh_vol": b["volatility"],
            "time_in_market": s["time_in_market"],
            "outperformance": s["total_return"] - b["total_return"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. COIN ANALYSIS (BTC, ETH, SOL only)
# ─────────────────────────────────────────────────────────────────────────────

def coin_analysis(df, coin):
    """Return a clean DataFrame for a single coin."""
    c = df[df["asset"] == coin].set_index("time").sort_index()
    cols = [col for col in ["price", "volume", "return_1d", "return_7d", "return_30d",
                             "rv_30d", "volume_ratio"] if col in c.columns]
    c = c[cols].copy()
    c["ema_10"] = c["price"].ewm(span=10).mean()
    c["ema_21"] = c["price"].ewm(span=21).mean()
    c["cum_return"] = (1 + c["return_1d"].fillna(0) / 100).cumprod()
    return c


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATION (~15 slides)
# ─────────────────────────────────────────────────────────────────────────────

INSIGHT_BOX = dict(boxstyle="round,pad=0.5", facecolor="#1a1f2e", edgecolor="#3a3f4e",
                   alpha=0.92, linewidth=0.8)


def add_insight(ax, x, y, text, fontsize=9, color="#e0e0e0", ha="left", va="top",
                transform=None):
    """Add an insight text box to an axis."""
    if transform is None:
        transform = ax.transAxes
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
            transform=transform, bbox=INSIGHT_BOX, linespacing=1.6)


def make_all_charts(feat, bt, df_raw, ystats, transitions_df):
    """Generate ~15 focused visualization slides."""

    times = pd.to_datetime(feat["time"])
    bt_times = pd.to_datetime(bt["time"])

    btc_ath = feat["btc_price"].max()
    btc_ath_date = feat.loc[feat["btc_price"].idxmax(), "time"]
    btc_current = feat.iloc[-1]["btc_price"]
    total_bull_days = (feat["regime"] == "BULLISH").sum()
    total_bear_days = (feat["regime"] == "BEARISH").sum()
    regime_switches = (feat["regime"] != feat["regime"].shift(1)).sum()

    all_signals = list(SIGNAL_NAMES.keys())

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 1: Current Regime Dashboard
    # ══════════════════════════════════════════════════════════════════════
    latest = feat.iloc[-1]
    regime = latest["regime"]
    color = C_BULL if regime == "BULLISH" else C_BEAR

    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35)

    # Main regime display
    ax_main = fig.add_subplot(gs[0, 1:3])
    ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 10); ax_main.axis("off")
    rect = FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.3",
                          facecolor=color, alpha=0.2, edgecolor=color, linewidth=3)
    ax_main.add_patch(rect)
    ax_main.text(5, 7.5, "CURRENT MARKET REGIME", ha="center", fontsize=13,
                 color=C_TEXT_DIM, fontweight="bold")
    ax_main.text(5, 4.8, regime, ha="center", fontsize=40, color=color, fontweight="bold")
    score_pct = (latest["raw_score"] + 1) / 2 * 100
    ax_main.text(5, 2.5, f"Signal Strength: {score_pct:.0f}%", ha="center",
                 fontsize=15, color=C_TEXT)
    ax_main.text(5, 1.0, f"Date: {latest['time'].strftime('%Y-%m-%d')}", ha="center",
                 fontsize=11, color=C_TEXT_DIM)

    # BTC signals panel
    ax_btc = fig.add_subplot(gs[0, 0]); ax_btc.axis("off")
    ax_btc.set_title("BTC SIGNALS (70%)", fontsize=11, fontweight="bold")
    btc_sigs_display = [
        ("EMA Cross 10/21", latest["btc_s_ema_cross"]),
        ("Price > 21 EMA", latest["btc_s_above_21ema"]),
        ("7d Momentum", latest["btc_s_momentum_7d"]),
        ("21d Trend", latest["btc_s_trend_21d"]),
        ("Vol Regime", latest["btc_s_vol_regime"]),
    ]
    for i, (name, val) in enumerate(btc_sigs_display):
        y = 0.85 - i * 0.16
        ax_btc.text(0.05, y, name, fontsize=10, color=C_TEXT_DIM, transform=ax_btc.transAxes)
        sym = "+" if val > 0 else "-"
        ax_btc.text(0.95, y, sym, fontsize=14, fontweight="bold",
                    color=C_BULL if val > 0 else C_BEAR, ha="right", transform=ax_btc.transAxes)

    # Alt signals panel
    ax_alt = fig.add_subplot(gs[0, 3]); ax_alt.axis("off")
    ax_alt.set_title("ALT SIGNALS (30%)", fontsize=11, fontweight="bold")
    alt_sigs_display = [
        ("Breadth 7d", latest["alt_s_breadth_7d"]),
        ("Alt 7d Mom", latest["alt_s_momentum_7d"]),
        ("DD Speed", latest["alt_s_drawdown"]),
    ]
    for i, (name, val) in enumerate(alt_sigs_display):
        y = 0.85 - i * 0.25
        ax_alt.text(0.05, y, name, fontsize=10, color=C_TEXT_DIM, transform=ax_alt.transAxes)
        sym = "+" if val > 0 else "-"
        ax_alt.text(0.95, y, sym, fontsize=14, fontweight="bold",
                    color=C_BULL if val > 0 else C_BEAR, ha="right", transform=ax_alt.transAxes)

    # 120-day price + regime
    ax_90 = fig.add_subplot(gs[1:, :])
    recent = feat.tail(120).copy()
    rt = pd.to_datetime(recent["time"])
    ax_90.plot(rt, recent["btc_price"], color=C_BTC, linewidth=1.8)
    for i in range(len(recent) - 1):
        c = C_BULL if recent.iloc[i]["regime"] == "BULLISH" else C_BEAR
        ax_90.axvspan(rt.iloc[i], rt.iloc[i+1], alpha=0.12, color=c, linewidth=0)
    ax_90.set_title("LAST 120 DAYS - BTC Price with Regime Overlay", fontsize=12,
                     fontweight="bold", pad=10)
    ax_90.set_ylabel("BTC Price ($)", fontsize=10)
    ax_90.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_90.grid(True, alpha=0.3)

    n_bear_sigs = sum(1 for _, v in btc_sigs_display + alt_sigs_display if v < 0)
    add_insight(ax_90, 0.01, 0.98,
                f"ANALYSIS: {n_bear_sigs}/8 signals bearish | "
                f"BTC at ${btc_current:,.0f} | "
                f"Breadth: {latest['breadth_7d']*100:.0f}%",
                fontsize=9)

    fig.suptitle("MARKET REGIME ANALYSIS - DASHBOARD", fontsize=16,
                 fontweight="bold", y=0.98, color="white")
    save_slide(fig, "dashboard",
               "Current regime with all 8 signals and 120-day price overlay.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 2: Full History BTC + Regime + EMA Lines (10/21)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(22, 7))
    ax.plot(times, feat["btc_price"], color=C_BTC, linewidth=1, alpha=0.9, label="BTC Price")
    ax.plot(times, feat["btc_ema_10"], color="#58a6ff", linewidth=0.7, alpha=0.6, label="10 EMA")
    ax.plot(times, feat["btc_ema_21"], color="#ff6b9d", linewidth=0.7, alpha=0.6, label="21 EMA")
    for i in range(len(feat) - 1):
        c = C_BULL if feat.iloc[i]["regime"] == "BULLISH" else C_BEAR
        ax.axvspan(times.iloc[i], times.iloc[i+1], alpha=0.08, color=c, linewidth=0)
    ax.set_title("BTC PRICE WITH REGIME OVERLAY + EMA 10/21", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("BTC Price ($)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    save_slide(fig, "btc_price_regime",
               "Full history BTC price (log) with regime overlay and 10/21 EMA lines.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 3: Composite Score Time Series
    # ══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 8), height_ratios=[2, 1],
                                     sharex=True)
    ax1.plot(times, feat["btc_price"], color=C_BTC, linewidth=1)
    ax1.set_ylabel("BTC Price ($)")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title("COMPOSITE SCORE vs BTC PRICE", fontsize=14, fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(times, feat["raw_score"], 0,
                     where=feat["raw_score"] > 0, color=C_BULL, alpha=0.4)
    ax2.fill_between(times, feat["raw_score"], 0,
                     where=feat["raw_score"] <= 0, color=C_BEAR, alpha=0.4)
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Composite Score")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    save_slide(fig, "composite_score",
               "Composite score (green=bullish, red=bearish) aligned with BTC price.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 4: Signal Heatmap (8 signals over time)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(22, 5))
    signal_data = feat[all_signals].values.T
    signal_labels = [SIGNAL_NAMES[s] for s in all_signals]

    cmap = LinearSegmentedColormap.from_list("regime", [C_BEAR, "#1a1f2e", C_BULL])
    im = ax.imshow(signal_data, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                   interpolation="nearest")

    # Thin down x-axis labels
    n_ticks = 12
    tick_positions = np.linspace(0, len(times) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([times.iloc[i].strftime("%Y-%m") for i in tick_positions],
                        rotation=45, ha="right")
    ax.set_yticks(range(len(signal_labels)))
    ax.set_yticklabels(signal_labels, fontsize=10)
    ax.set_title("SIGNAL HEATMAP (8 Signals Over Time)", fontsize=14,
                 fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, label="Bullish (+1) / Bearish (-1)", shrink=0.6)
    save_slide(fig, "signal_heatmap",
               "All 8 signals over time: green=bullish, red=bearish.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 5: Backtest Equity Curve
    # ══════════════════════════════════════════════════════════════════════
    strat_stats = compute_stats(bt["strat_return"], "Strategy")
    bh_stats = compute_stats(bt["bh_return"], "Buy & Hold")

    fig, ax = plt.subplots(figsize=(22, 7))
    ax.plot(bt_times, bt["strat_equity"], color=C_STRATEGY, linewidth=1.5,
            label=f"Strategy ({strat_stats['cagr']:.1f}% CAGR)")
    ax.plot(bt_times, bt["bh_equity"], color=C_BTC, linewidth=1.5, alpha=0.7,
            label=f"Buy & Hold ({bh_stats['cagr']:.1f}% CAGR)")

    # Shade cash periods
    for i in range(len(bt) - 1):
        if bt.iloc[i]["position"] == 0:
            ax.axvspan(bt_times.iloc[i], bt_times.iloc[i+1], alpha=0.05,
                       color=C_CASH, linewidth=0)

    ax.set_title("BACKTEST: Strategy vs Buy & Hold", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Equity (starting $1)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    add_insight(ax, 0.01, 0.15,
                f"Strategy: {strat_stats['total_return']:.0f}% total | "
                f"Sharpe {strat_stats['sharpe']:.2f} | "
                f"MaxDD {strat_stats['max_drawdown']:.1f}%\n"
                f"Buy&Hold: {bh_stats['total_return']:.0f}% total | "
                f"Sharpe {bh_stats['sharpe']:.2f} | "
                f"MaxDD {bh_stats['max_drawdown']:.1f}%",
                fontsize=10)

    save_slide(fig, "backtest_equity",
               "Equity curves: regime strategy vs buy-and-hold BTC.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 6: Backtest Drawdown Comparison
    # ══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 7), sharex=True)
    ax1.fill_between(bt_times, bt["strat_dd"], 0, color=C_STRATEGY, alpha=0.4)
    ax1.set_title("DRAWDOWN COMPARISON", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Strategy DD (%)")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(bt_times, bt["bh_dd"], 0, color=C_BEAR, alpha=0.4)
    ax2.set_ylabel("Buy & Hold DD (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    add_insight(ax1, 0.01, 0.15,
                f"Strategy max DD: {strat_stats['max_drawdown']:.1f}% vs "
                f"Buy&Hold max DD: {bh_stats['max_drawdown']:.1f}%",
                fontsize=9)

    save_slide(fig, "backtest_drawdown",
               "Drawdown comparison: strategy avoids major crashes by going to cash.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 7: Year-by-Year Return Comparison Table
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(16, max(4, len(ystats) * 0.6 + 2)))
    ax.axis("off")
    ax.set_title("YEAR-BY-YEAR PERFORMANCE COMPARISON", fontsize=14,
                 fontweight="bold", pad=20)

    headers = ["Year", "Strategy", "Buy&Hold", "Outperf", "S.Sharpe", "B.Sharpe",
               "S.MaxDD", "B.MaxDD", "Time In"]
    col_x = [0.02, 0.12, 0.24, 0.36, 0.48, 0.58, 0.68, 0.78, 0.90]

    y = 0.92
    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h, fontsize=10, fontweight="bold", color="white",
                transform=ax.transAxes)
    y -= 0.06
    ax.axhline(y=y, xmin=0.01, xmax=0.99, color=C_TEXT_DIM, linewidth=0.5,
               transform=ax.transAxes)

    for _, row in ystats.iterrows():
        y -= 0.06
        outperf_color = C_BULL if row["outperformance"] > 0 else C_BEAR
        vals = [
            f"{int(row['year'])}",
            f"{row['strat_return']:.1f}%",
            f"{row['bh_return']:.1f}%",
            f"{row['outperformance']:+.1f}pp",
            f"{row['strat_sharpe']:.2f}",
            f"{row['bh_sharpe']:.2f}",
            f"{row['strat_max_dd']:.1f}%",
            f"{row['bh_max_dd']:.1f}%",
            f"{row['time_in_market']:.0f}%",
        ]
        colors = [C_TEXT] * 3 + [outperf_color] + [C_TEXT] * 5
        for i, (v, clr) in enumerate(zip(vals, colors)):
            ax.text(col_x[i], y, v, fontsize=9, color=clr, transform=ax.transAxes)

    save_slide(fig, "yearly_returns",
               "Year-by-year strategy vs buy-and-hold: returns, Sharpe, drawdowns.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 8: Rolling 90d Sharpe Comparison
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(22, 6))
    roll_s = bt["strat_return"].rolling(90).apply(
        lambda x: x.mean() / x.std() * np.sqrt(365.25) if x.std() > 0 else 0)
    roll_b = bt["bh_return"].rolling(90).apply(
        lambda x: x.mean() / x.std() * np.sqrt(365.25) if x.std() > 0 else 0)
    ax.plot(bt_times, roll_s, color=C_STRATEGY, linewidth=1, label="Strategy")
    ax.plot(bt_times, roll_b, color=C_BTC, linewidth=1, alpha=0.7, label="Buy & Hold")
    ax.axhline(0, color="white", linewidth=0.3, alpha=0.5)
    ax.set_title("ROLLING 90-DAY SHARPE RATIO", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_slide(fig, "rolling_sharpe",
               "Rolling 90-day Sharpe: strategy maintains higher risk-adjusted returns.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 9: Monthly Returns Heatmap
    # ══════════════════════════════════════════════════════════════════════
    bt_monthly = bt.copy()
    bt_monthly["month"] = pd.to_datetime(bt_monthly["time"]).dt.to_period("M")
    monthly_rets = bt_monthly.groupby("month")["strat_return"].apply(
        lambda x: (1 + x).prod() - 1).reset_index()
    monthly_rets["year"] = monthly_rets["month"].dt.year
    monthly_rets["mon"] = monthly_rets["month"].dt.month

    pivot = monthly_rets.pivot_table(index="year", columns="mon",
                                      values="strat_return", aggfunc="first")
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(16, max(4, len(pivot) * 0.5 + 1)))
    cmap_mr = LinearSegmentedColormap.from_list("mr", [C_BEAR, "#1a1f2e", C_BULL])
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.1)
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap=cmap_mr,
                   vmin=-vmax * 100, vmax=vmax * 100)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > vmax * 0.3 else C_TEXT_DIM)

    ax.set_title("STRATEGY MONTHLY RETURNS HEATMAP", fontsize=14,
                 fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, label="Monthly Return (%)", shrink=0.6)
    save_slide(fig, "monthly_returns",
               "Strategy monthly returns: green=positive, red=negative.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 10: Signal Catalyst Detail (transitions)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(22, max(6, len(transitions_df) * 0.3 + 2)))
    ax.axis("off")
    ax.set_title("REGIME TRANSITIONS - CATALYST DETAILS", fontsize=14,
                 fontweight="bold", pad=20)

    y = 0.95
    # Show last 30 transitions (or all if fewer)
    show_df = transitions_df.tail(30)
    for _, row in show_df.iterrows():
        if y < 0.02:
            break
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        direction = "BEAR->BULL" if row["to_regime"] == "BULLISH" else "BULL->BEAR"
        clr = C_BULL if row["to_regime"] == "BULLISH" else C_BEAR
        ax.text(0.01, y, f"{date_str}  {direction}", fontsize=9, fontweight="bold",
                color=clr, transform=ax.transAxes)
        ax.text(0.22, y, f"Score: {row['raw_score']:+.3f}", fontsize=8,
                color=C_TEXT_DIM, transform=ax.transAxes)
        ax.text(0.38, y, row["signals_flipped"][:80], fontsize=8,
                color=C_TEXT, transform=ax.transAxes)
        y -= 0.03

    save_slide(fig, "regime_transitions",
               "Each regime transition with the specific signals that flipped.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 11: Breadth Over Time with Regime
    # ══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 7), height_ratios=[2, 1],
                                     sharex=True)
    ax1.plot(times, feat["btc_price"], color=C_BTC, linewidth=1)
    for i in range(len(feat) - 1):
        c = C_BULL if feat.iloc[i]["regime"] == "BULLISH" else C_BEAR
        ax1.axvspan(times.iloc[i], times.iloc[i+1], alpha=0.08, color=c, linewidth=0)
    ax1.set_ylabel("BTC Price ($)")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title("BREADTH (7d) vs BTC PRICE", fontsize=14, fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(times, feat["breadth_7d"] * 100, 50,
                     where=feat["breadth_7d"] > 0.5, color=C_BULL, alpha=0.3)
    ax2.fill_between(times, feat["breadth_7d"] * 100, 50,
                     where=feat["breadth_7d"] <= 0.5, color=C_BEAR, alpha=0.3)
    ax2.plot(times, feat["breadth_7d"] * 100, color="white", linewidth=0.8)
    ax2.axhline(50, color=C_TEXT_DIM, linewidth=0.5, linestyle="--")
    ax2.set_ylabel("% Coins Positive 7d")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    save_slide(fig, "breadth",
               "Market breadth: % of coins with positive 7d return. >50% = broad rally.")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDE 12: Volatility Regime (RV 7d vs RV 30d)
    # ══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 7), height_ratios=[2, 1],
                                     sharex=True)
    ax1.plot(times, feat["btc_price"], color=C_BTC, linewidth=1)
    for i in range(len(feat) - 1):
        c = C_BULL if feat.iloc[i]["regime"] == "BULLISH" else C_BEAR
        ax1.axvspan(times.iloc[i], times.iloc[i+1], alpha=0.08, color=c, linewidth=0)
    ax1.set_ylabel("BTC Price ($)")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title("VOLATILITY REGIME: RV 7d vs RV 30d", fontsize=14,
                  fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, feat["btc_rv_7d"], color="#ff6b9d", linewidth=0.8, label="RV 7d")
    ax2.plot(times, feat["btc_rv_30d"], color="#58a6ff", linewidth=0.8, label="RV 30d")
    rv_ratio = feat["btc_rv_7d"] / feat["btc_rv_30d"].replace(0, np.nan)
    for i in range(len(feat) - 1):
        if rv_ratio.iloc[i] > 1:  # inverted = bearish
            ax2.axvspan(times.iloc[i], times.iloc[i+1], alpha=0.15,
                        color=C_BEAR, linewidth=0)
    ax2.set_ylabel("Realized Volatility (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    save_slide(fig, "volatility_regime",
               "RV 7d vs RV 30d: when short-term vol < long-term vol = bullish (calm market).")

    # ══════════════════════════════════════════════════════════════════════
    # SLIDES 13-15: Top Coin Deep Dives (BTC, ETH, SOL)
    # ══════════════════════════════════════════════════════════════════════
    for coin in ["btc", "eth", "sol"]:
        if coin not in df_raw["asset"].unique():
            continue
        cd = coin_analysis(df_raw, coin)
        if len(cd) < 30:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle(f"{coin.upper()} DEEP DIVE", fontsize=16, fontweight="bold",
                     y=0.98, color="white")

        # Price + EMAs
        ax = axes[0, 0]
        ax.plot(cd.index, cd["price"], color=C_BTC if coin == "btc" else "#58a6ff",
                linewidth=1)
        ax.plot(cd.index, cd["ema_10"], color="#ff6b9d", linewidth=0.6, alpha=0.6,
                label="10 EMA")
        ax.plot(cd.index, cd["ema_21"], color="#ffd93d", linewidth=0.6, alpha=0.6,
                label="21 EMA")
        ax.set_title(f"{coin.upper()} Price + EMAs", fontsize=11, fontweight="bold")
        ax.set_ylabel("Price ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Cumulative return
        ax = axes[0, 1]
        ax.plot(cd.index, cd["cum_return"], color=C_STRATEGY, linewidth=1)
        ax.set_title(f"{coin.upper()} Cumulative Return", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Return (x)")
        ax.grid(True, alpha=0.3)

        # 7d & 30d returns
        ax = axes[1, 0]
        if "return_7d" in cd.columns:
            ax.plot(cd.index, cd["return_7d"], color="#58a6ff", linewidth=0.6,
                    alpha=0.7, label="7d Return")
        if "return_30d" in cd.columns:
            ax.plot(cd.index, cd["return_30d"], color="#ff6b9d", linewidth=0.6,
                    alpha=0.7, label="30d Return")
        ax.axhline(0, color="white", linewidth=0.3, alpha=0.5)
        ax.set_title(f"{coin.upper()} Returns", fontsize=11, fontweight="bold")
        ax.set_ylabel("Return (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Realized volatility
        ax = axes[1, 1]
        if "rv_30d" in cd.columns:
            ax.plot(cd.index, cd["rv_30d"], color="#ffd93d", linewidth=0.8, label="RV 30d")
        ax.set_title(f"{coin.upper()} Realized Volatility", fontsize=11, fontweight="bold")
        ax.set_ylabel("RV (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_slide(fig, f"deep_dive_{coin}",
                   f"{coin.upper()} deep dive: price + EMAs, cumulative return, returns, volatility.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CONCATENATED PNG
# ─────────────────────────────────────────────────────────────────────────────

def create_concatenated_png():
    """Stitch all slide PNGs into one tall vertical image."""
    import glob
    slide_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "[0-9][0-9]_*.png")))
    if not slide_files:
        print("  No slides found to concatenate.")
        return None

    images = [Image.open(f) for f in slide_files]
    max_width = max(img.width for img in images)

    resized = []
    for img in images:
        if img.width != max_width:
            ratio = max_width / img.width
            new_h = int(img.height * ratio)
            img = img.resize((max_width, new_h), Image.LANCZOS)
        resized.append(img)

    total_height = sum(img.height for img in resized)
    concat = Image.new("RGB", (max_width, total_height), color=(10, 14, 23))

    y_offset = 0
    for img in resized:
        concat.paste(img, (0, y_offset))
        y_offset += img.height

    out_path = os.path.join(OUTPUT_DIR, "all_slides_concatenated.png")
    concat.save(out_path, quality=95)
    print(f"  [saved] {out_path} ({max_width}x{total_height}px, {len(slide_files)} slides)")

    for img in images:
        img.close()
    for img in resized:
        try:
            img.close()
        except Exception:
            pass
    concat.close()

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 9. CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(feat, bt, ystats):
    latest = feat.iloc[-1]
    regime = latest["regime"]
    strat_stats = compute_stats(bt["strat_return"], "Strategy")
    bh_stats = compute_stats(bt["bh_return"], "Buy & Hold")

    sep = "=" * 72
    print(f"\n{sep}")
    print("  MARKET REGIME ANALYSIS - RESULTS")
    print(sep)

    print(f"\n  Date            : {latest['time'].strftime('%Y-%m-%d')}")
    color_tag = "G" if regime == "BULLISH" else "R"
    print(f"  CURRENT REGIME  : [{color_tag}] {regime}")
    print(f"  Raw Score       : {latest['raw_score']:+.3f}")
    print(f"  BTC Score       : {latest['btc_score']:+.3f} (70% weight)")
    print(f"  Alt Score       : {latest['alt_score']:+.3f} (30% weight)")

    print(f"\n  BTC SIGNALS (70% weight, 5 signals)")
    print(f"  {'_'*50}")
    for sig, name in [("btc_s_ema_cross", "EMA Cross 10/21"),
                      ("btc_s_above_21ema", "Price > 21 EMA"),
                      ("btc_s_momentum_7d", "7d Momentum"),
                      ("btc_s_trend_21d", "21d Trend"),
                      ("btc_s_vol_regime", "Vol Regime")]:
        val = latest[sig]
        sym = "  + BULLISH" if val > 0 else "  - BEARISH"
        detail = _signal_detail(sig, latest, name)
        print(f"    {name:<18} {sym}  ({detail})")

    print(f"\n  ALT SIGNALS (30% weight, 3 signals)")
    print(f"  {'_'*50}")
    for sig, name in [("alt_s_breadth_7d", "Breadth 7d"),
                      ("alt_s_momentum_7d", "Alt 7d Momentum"),
                      ("alt_s_drawdown", "Drawdown Speed")]:
        val = latest[sig]
        sym = "  + BULLISH" if val > 0 else "  - BEARISH"
        detail = _signal_detail(sig, latest, name)
        print(f"    {name:<18} {sym}  ({detail})")

    print(f"\n  BACKTEST RESULTS (Full Period)")
    print(f"  {'_'*50}")
    print(f"  {'Metric':<22} {'Strategy':>12} {'Buy & Hold':>12}")
    print(f"  {'_'*50}")
    for label, key, fmt in [
        ("Total Return", "total_return", ".1f"),
        ("CAGR", "cagr", ".1f"),
        ("Sharpe Ratio", "sharpe", ".2f"),
        ("Max Drawdown", "max_drawdown", ".1f"),
        ("Calmar Ratio", "calmar", ".2f"),
        ("Volatility", "volatility", ".1f"),
        ("Win Rate", "win_rate", ".1f"),
        ("Time in Market", "time_in_market", ".1f"),
    ]:
        sv = strat_stats[key]
        bv = bh_stats[key]
        pct = "%" if key not in ["sharpe", "calmar", "profit_factor"] else ""
        print(f"  {label:<22} {sv:>11{fmt}}{pct} {bv:>11{fmt}}{pct}")

    print(f"\n  YEAR-BY-YEAR")
    print(f"  {'_'*72}")
    print(f"  {'Year':<6} {'Strat':>8} {'B&H':>8} {'Outperf':>9} {'S.Sharpe':>9} "
          f"{'B.Sharpe':>9} {'S.MaxDD':>8} {'B.MaxDD':>8}")
    print(f"  {'_'*72}")
    for _, row in ystats.iterrows():
        print(f"  {int(row['year']):<6} {row['strat_return']:>7.1f}% {row['bh_return']:>7.1f}% "
              f"{row['outperformance']:>+8.1f}pp {row['strat_sharpe']:>8.2f} {row['bh_sharpe']:>8.2f} "
              f"{row['strat_max_dd']:>7.1f}% {row['bh_max_dd']:>7.1f}%")

    # Recent regime history
    print(f"\n  RECENT REGIME (last 14 days)")
    print(f"  {'_'*50}")
    for idx, row in feat.tail(14).iterrows():
        d = row["time"].strftime("%Y-%m-%d")
        r = row["regime"]
        s = row["raw_score"]
        marker = " >>>" if idx == feat.index[-1] else "    "
        dot = "[G]" if r == "BULLISH" else "[R]"
        print(f"  {marker} {d}  {dot} {r:<10} (score: {s:+.3f})")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("  MARKET REGIME ANALYSIS v4 - SIMPLIFIED")
    print("  Binary regime (BULLISH/BEARISH) | 8 Signals | BTC 70% + Alts 30%")
    print("=" * 72)

    print("\n[1/7] Loading data...")
    df = load_data(DATA_PATH)
    print(f"       {len(df):,} rows | {df['asset'].nunique()} assets | "
          f"{df['time'].min().date()} -> {df['time'].max().date()}")

    print("[2/7] Engineering features...")
    feat = build_features(df)
    print(f"       {len(feat)} daily observations | {feat.shape[1]} features")

    print("[3/7] Computing regime (8 signals, 2-day confirmation)...")
    feat = compute_regime(feat)
    bull_pct = (feat["regime"] == "BULLISH").mean() * 100
    print(f"       BULLISH: {bull_pct:.1f}% of days | BEARISH: {100-bull_pct:.1f}% of days")

    print("[4/7] Identifying regime transition catalysts...")
    transitions_df = identify_catalysts(feat)
    print(f"       {len(transitions_df)} regime transitions identified")

    print("[5/7] Running backtest (long BTC when bullish, cash when bearish)...")
    bt = run_backtest(feat)
    ystats = yearly_stats(bt)
    strat_final = bt["strat_equity"].iloc[-1]
    bh_final = bt["bh_equity"].iloc[-1]
    print(f"       Strategy: $1.00 -> ${strat_final:.2f} ({strat_final:.1f}x)")
    print(f"       Buy&Hold: $1.00 -> ${bh_final:.2f} ({bh_final:.1f}x)")

    print_report(feat, bt, ystats)
    print_catalysts(transitions_df)

    print(f"\n[6/7] Generating ~15 visualization slides...")
    make_all_charts(feat, bt, df, ystats, transitions_df)

    # Save data
    out_csv = os.path.join(OUTPUT_DIR, "regime_data.csv")
    feat.to_csv(out_csv, index=False)
    print(f"\n  [saved] {out_csv}")

    trans_csv = os.path.join(OUTPUT_DIR, "regime_transitions.csv")
    transitions_df.to_csv(trans_csv, index=False)
    print(f"  [saved] {trans_csv}")

    print("[7/7] Creating concatenated PNG...")
    create_concatenated_png()

    print(f"\nDone. All outputs -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
