from __future__ import annotations

import datetime as dt
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --- SAFETY GUARD: prevent huge Streamlit object dumps in the app body ---
import types

_ST_WRITE = st.write

def _safe_write(*args, **kwargs):
    # If someone accidentally does st.write(st) or st.write(st.sidebar) (DeltaGenerator),
    # Streamlit renders a massive API/object dump. Block that.
    if len(args) == 1:
        a = args[0]
        if isinstance(a, types.ModuleType) and getattr(a, "__name__", "") == "streamlit":
            return
        if a.__class__.__name__ == "DeltaGenerator":
            return
    return _ST_WRITE(*args, **kwargs)

st.write = _safe_write


# =============================
# SETTINGS
# =============================
USD = "USD"
CAD = "CAD"
TRADING_DAYS = 252

REQUIRED_COLS = [
    "Group ID",
    "Ticker",
    "Fund Type",
    "Currency",
    "TT",            # BUY / SELL
    "Total Amount",  # for BUY: total paid; for SELL: proceeds received (positive)
    "Shares",
    "Origin P/P",
    "Date",
]

BENCH_PRESETS = {
    "S&P 500 (SPY)": "SPY",
    "Total US Market (VTI)": "VTI",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Berkshire (BRK-B)": "BRK-B",
    "TSX 60 (XIU.TO)": "XIU.TO",
    "S&P 500 CAD (VFV.TO)": "VFV.TO",
}

CACHE_DIR = "data_cache"
DIV_DIR = os.path.join(CACHE_DIR, "dividends")
os.makedirs(DIV_DIR, exist_ok=True)


# =============================
# BASIC HELPERS
# =============================
def normalize_ticker(t: str) -> str:
    if pd.isna(t):
        return ""
    return str(t).strip().upper()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).map(normalize_ticker)
    df["TT"] = df["TT"].astype(str).str.strip().str.upper()
    df["Currency"] = df["Currency"].astype(str).str.strip().str.upper()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["Date"].isna().any():
        bad = int(df["Date"].isna().sum())
        raise ValueError(f"{bad} rows have invalid dates (could not parse).")

    if (~df["TT"].isin(["BUY", "SELL"])).any():
        bad_vals = df.loc[~df["TT"].isin(["BUY", "SELL"]), "TT"].unique().tolist()
        raise ValueError(f"Invalid TT values found: {bad_vals} (expected BUY/SELL).")

    if (~df["Currency"].isin([CAD, USD])).any():
        bad_vals = df.loc[~df["Currency"].isin([CAD, USD]), "Currency"].unique().tolist()
        raise ValueError(f"Invalid Currency values found: {bad_vals} (expected CAD/USD).")

    for c in ["Total Amount", "Shares", "Origin P/P"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["Total Amount", "Shares", "Origin P/P"]].isna().any().any():
        raise ValueError("Some rows have non-numeric Total Amount / Shares / Origin P/P.")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def date_to_py(d) -> dt.date:
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return pd.to_datetime(d).date()


def nearest_prev_trading_day(trading_days: List[dt.date], d: dt.date) -> Optional[dt.date]:
    if not trading_days:
        return None
    if d in trading_days:
        return d
    import bisect
    i = bisect.bisect_right(trading_days, d) - 1
    if i < 0:
        return None
    return trading_days[i]


def prev_available(series: pd.Series, d: dt.date) -> Optional[float]:
    if series is None or series.empty:
        return None
    if d in series.index:
        return float(series.loc[d])
    idx = [x for x in series.index if x <= d]
    if not idx:
        return None
    return float(series.loc[max(idx)])


def get_fx_for_date(fx: pd.Series, d: dt.date) -> float:
    v = prev_available(fx, d)
    return float(v) if v is not None else 1.0


def annualize_logret(logret: pd.Series, winsorize: bool) -> Tuple[float, float]:
    lr = logret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(lr) < 20:
        return 0.07, 0.15
    if winsorize and len(lr) >= 60:
        lo, hi = lr.quantile(0.01), lr.quantile(0.99)
        lr = lr.clip(lo, hi)
    mu = float(lr.mean()) * TRADING_DAYS
    sig = float(lr.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    # safety floors
    if not np.isfinite(mu):
        mu = 0.07
    if not np.isfinite(sig) or sig <= 0:
        sig = 0.15
    return mu, sig


# =============================
# STOOQ FETCHING (PRICES + FX)
# =============================
def _stooq_candidates(yahoo_ticker: str) -> List[str]:
    t = (yahoo_ticker or "").strip()
    if not t:
        return []
    u = t.upper()

    if u in {"CAD=X", "USDCAD=X", "USDCAD", "USD/CAD"}:
        return ["usdcad"]

    if u.endswith(".TO"):
        base = u[:-3]
        return [
            base.lower() + ".to",
            base.replace("-", ".").lower() + ".to",
            base.replace(".", "-").lower() + ".to",
        ]

    base = u.replace("-", ".")
    return [base.lower() + ".us", u.lower() + ".us"]


def _stooq_url(sym: str) -> str:
    return "https://stooq.com/q/d/l/?" + urllib.parse.urlencode({"s": sym, "i": "d"})


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def stooq_daily_df_for_ticker(yahoo_ticker: str) -> pd.DataFrame:
    candidates = _stooq_candidates(yahoo_ticker)
    for sym in candidates:
        url = _stooq_url(sym)
        for _ in range(3):
            try:
                df = pd.read_csv(url)
                if df is None or df.empty or "Date" not in df.columns:
                    return pd.DataFrame()
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date")
                df = df.set_index(df["Date"].dt.date)
                if "Close" not in df.columns:
                    return pd.DataFrame()
                return df
            except Exception:
                time.sleep(0.35)
    return pd.DataFrame()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_fx_usd_to_cad_stooq(start: dt.date, end: dt.date) -> pd.Series:
    df = stooq_daily_df_for_ticker("usdcad")
    if df.empty:
        raise RuntimeError("Could not download FX series from Stooq (usdcad).")

    s = df["Close"].astype(float).copy()
    s = s[(s.index >= start) & (s.index <= end)].sort_index()
    if s.empty:
        raise RuntimeError("FX series from Stooq is empty for requested date range.")
    return s


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_close_prices_stooq(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    out: Dict[str, pd.Series] = {}
    for t in sorted(set([x for x in tickers if x])):
        df = stooq_daily_df_for_ticker(t)
        if df.empty:
            continue
        s = df["Close"].astype(float).copy()
        s = s[(s.index >= start) & (s.index <= end)].sort_index()
        if not s.empty:
            out[t] = s
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


# =============================
# YFINANCE FALLBACK PRICES (RATE LIMIT RESILIENT)
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_close_prices_yf(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    tickers = sorted(set([t for t in tickers if t]))
    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers,
            start=str(start),
            end=str(end + dt.timedelta(days=1)),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            close = raw["Close"].copy()
        else:
            close = pd.DataFrame({tickers[0]: raw["Close"]})

        close.index = pd.to_datetime(close.index).date
        close = close.sort_index()
        close = close[(close.index >= start) & (close.index <= end)]
        return close
    except Exception:
        return pd.DataFrame()


def build_constant_price_series(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
    last_prices: Dict[str, float],
) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end).date
    out = {}
    for t in tickers:
        p = last_prices.get(t)
        if p is None or not np.isfinite(p) or p <= 0:
            continue
        out[t] = pd.Series(float(p), index=idx)
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


def fetch_close_prices_hybrid(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
    enable_yf_fallback: bool,
    const_fallback_prices: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    close_stooq = fetch_close_prices_stooq(tickers, start, end)
    have = set(close_stooq.columns) if not close_stooq.empty else set()
    missing = [t for t in tickers if t not in have]

    close_all = close_stooq

    if enable_yf_fallback and missing:
        time.sleep(0.25)
        close_yf = fetch_close_prices_yf(missing, start, end)
        if not close_yf.empty:
            close_all = close_all.join(close_yf, how="outer") if not close_all.empty else close_yf

    have2 = set(close_all.columns) if not close_all.empty else set()
    still_missing = [t for t in tickers if t not in have2]

    used_constant = []
    if still_missing:
        const_df = build_constant_price_series(still_missing, start, end, const_fallback_prices)
        if not const_df.empty:
            used_constant = list(const_df.columns)
            close_all = close_all.join(const_df, how="outer") if not close_all.empty else const_df

    close_all = close_all.sort_index()
    have3 = set(close_all.columns) if not close_all.empty else set()
    still_missing_final = [t for t in tickers if t not in have3]

    return close_all, still_missing_final, used_constant


# =============================
# DIVIDENDS (YFINANCE) WITH DISK CACHE
# =============================
def _div_cache_path(ticker: str) -> str:
    safe = normalize_ticker(ticker).replace("/", "_").replace("\\", "_").replace(":", "_")
    return os.path.join(DIV_DIR, f"{safe}.pkl")


def _load_div_disk_cache(ticker: str, max_age_days: int = 40) -> Optional[pd.Series]:
    path = _div_cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
        if (dt.datetime.now() - mtime).days > max_age_days:
            return None
        s = pd.read_pickle(path)
        return s if isinstance(s, pd.Series) else None
    except Exception:
        return None


def _save_div_disk_cache(ticker: str, s: pd.Series) -> None:
    try:
        pd.to_pickle(s, _div_cache_path(ticker))
    except Exception:
        pass


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=False)
def fetch_dividends_yf(tickers: List[str], start: dt.date, end: dt.date) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for t in sorted(set([x for x in tickers if x])):
        cached = _load_div_disk_cache(t)
        if cached is not None:
            s = cached.copy()
            s.index = pd.to_datetime(s.index).date
            s = s[(s.index >= start) & (s.index <= end)].sort_index()
            out[t] = s
            continue

        try:
            s_full = yf.Ticker(t).dividends
            if s_full is None or len(s_full) == 0:
                out[t] = pd.Series(dtype=float)
                _save_div_disk_cache(t, out[t])
                continue

            s_full = s_full.copy()
            s_full.index = pd.to_datetime(s_full.index).date
            s_full = s_full.sort_index()
            _save_div_disk_cache(t, s_full)

            s = s_full[(s_full.index >= start) & (s_full.index <= end)].sort_index()
            out[t] = s
            time.sleep(0.15)
        except Exception:
            out[t] = pd.Series(dtype=float)

    return out


# =============================
# ACB + REALIZED
# =============================
@dataclass
class PositionState:
    shares: float = 0.0
    acb_cad: float = 0.0
    realized_pl_cad: float = 0.0


def compute_acb_realized(df: pd.DataFrame, fx_usdcad: pd.Series) -> Tuple[pd.DataFrame, Dict[str, PositionState]]:
    df = df.copy()
    df["Date_d"] = df["Date"].dt.date
    df["FX_USD_to_CAD"] = df["Date_d"].apply(lambda d: get_fx_for_date(fx_usdcad, d))
    df["Amount_CAD"] = np.where(df["Currency"] == USD, df["Total Amount"] * df["FX_USD_to_CAD"], df["Total Amount"])

    states: Dict[str, PositionState] = {}
    realized_rows = []

    for i, row in df.iterrows():
        t = row["Ticker"]
        tt = row["TT"]
        sh = float(row["Shares"])
        amt_cad = float(row["Amount_CAD"])

        if t not in states:
            states[t] = PositionState()
        stt = states[t]

        if tt == "BUY":
            stt.shares += sh
            stt.acb_cad += amt_cad
            realized = 0.0
            avg_cost = stt.acb_cad / stt.shares if stt.shares > 0 else 0.0
        else:  # SELL
            if stt.shares + 1e-12 < sh:
                raise ValueError(f"Row {i}: SELL {sh} shares of {t}, but only {stt.shares} held.")
            avg_cost_before = stt.acb_cad / stt.shares if stt.shares > 0 else 0.0
            cost_sold = avg_cost_before * sh
            proceeds = amt_cad
            realized = proceeds - cost_sold

            stt.shares -= sh
            stt.acb_cad -= cost_sold
            stt.realized_pl_cad += realized
            avg_cost = stt.acb_cad / stt.shares if stt.shares > 0 else 0.0

        realized_rows.append((realized, avg_cost))

    df["Realized_PL_CAD"] = [x[0] for x in realized_rows]
    df["ACB_per_share_CAD_after"] = [x[1] for x in realized_rows]
    return df, states


# =============================
# DRIP
# =============================
def build_drip_events(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
    dividends: Dict[str, pd.Series],
    close_prices: pd.DataFrame,
    tx_df: pd.DataFrame,
) -> pd.DataFrame:
    if close_prices.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Currency", "TT", "Div_per_share", "Synthetic"])

    all_days = sorted(close_prices.index)
    rows = []

    for t in tickers:
        divs = dividends.get(t, pd.Series(dtype=float))
        if divs is None or divs.empty:
            continue

        t_ccy = CAD
        if (tx_df["Ticker"] == t).any():
            t_ccy = tx_df.loc[tx_df["Ticker"] == t, "Currency"].iloc[0]

        for pay_date, div_per_share in divs.items():
            d = date_to_py(pay_date)
            if d < start or d > end:
                continue
            td = nearest_prev_trading_day(all_days, d)
            if td is None:
                continue
            if t not in close_prices.columns or td not in close_prices.index:
                continue
            px = float(close_prices.at[td, t])
            if not np.isfinite(px) or px <= 0:
                continue
            rows.append({
                "Date": pd.Timestamp(d),
                "Ticker": t,
                "Currency": t_ccy,
                "TT": "DRIP",
                "Div_per_share": float(div_per_share),
                "Synthetic": True,
            })

    return pd.DataFrame(rows)


def apply_drip_to_events(
    events: pd.DataFrame,
    drip: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    if drip.empty or close_prices.empty:
        return events

    all_days = sorted(close_prices.index)

    base = events.copy()
    base["Date_d"] = pd.to_datetime(base["Date"]).dt.date
    base["Ticker"] = base["Ticker"].map(normalize_ticker)
    base["delta_shares"] = np.where(base["TT"] == "BUY", base["Shares"], -base["Shares"])
    base["TD"] = base["Date_d"].apply(lambda d: nearest_prev_trading_day(all_days, d))
    base = base.dropna(subset=["TD"]).copy()
    base["TD"] = base["TD"].astype(object)

    deltas = (
        base.groupby(["TD", "Ticker"])["delta_shares"]
        .sum()
        .unstack("Ticker")
        .reindex(all_days)
        .fillna(0.0)
        .cumsum()
        .fillna(0.0)
    )

    drip = drip.copy()
    drip["Date_d"] = pd.to_datetime(drip["Date"]).dt.date
    drip["TD"] = drip["Date_d"].apply(lambda d: nearest_prev_trading_day(all_days, d))
    drip = drip.dropna(subset=["TD"]).copy()
    drip["TD"] = drip["TD"].astype(object)

    rows = []
    for _, r in drip.iterrows():
        t = normalize_ticker(r["Ticker"])
        td = r["TD"]
        if t not in deltas.columns or t not in close_prices.columns or td not in close_prices.index:
            continue

        shares_held = float(deltas.at[td, t])
        if shares_held <= 0:
            continue

        px = float(close_prices.at[td, t])
        if not np.isfinite(px) or px <= 0:
            continue

        div_cash = shares_held * float(r["Div_per_share"])
        shares_added = div_cash / px
        if shares_added <= 0:
            continue

        rows.append({
            "Group ID": "DRIP",
            "Ticker": t,
            "Fund Type": "DRIP",
            "Currency": r["Currency"],
            "TT": "BUY",
            "Total Amount": div_cash,
            "Shares": shares_added,
            "Origin P/P": px,
            "Date": pd.Timestamp(r["Date_d"]),
            "Synthetic": True,
        })

    if not rows:
        return events

    drip_buys = pd.DataFrame(rows)
    for c in REQUIRED_COLS:
        if c not in drip_buys.columns:
            drip_buys[c] = np.nan

    combined = pd.concat([events, drip_buys], ignore_index=True)
    combined["Synthetic"] = combined.get("Synthetic", False).fillna(False)
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined


# =============================
# CASHFLOWS + BENCHMARK MATCHING
# =============================
def build_external_cashflow_series(tx_enriched: pd.DataFrame, index_days: List[dt.date]) -> pd.Series:
    cf = tx_enriched.copy()
    cf["Date_d"] = pd.to_datetime(cf["Date"]).dt.date
    cf = cf[cf["TT"].isin(["BUY", "SELL"])].copy()
    if "Synthetic" in cf.columns:
        cf = cf[~cf["Synthetic"].fillna(False)].copy()

    cf["CF_CAD"] = np.where(cf["TT"] == "BUY", cf["Amount_CAD"], -cf["Amount_CAD"])
    daily = cf.groupby("Date_d")["CF_CAD"].sum()

    out = pd.Series(0.0, index=index_days)
    for d, v in daily.items():
        if d in out.index:
            out.loc[d] += float(v)
    return out


def cashflow_matched_benchmark_value(bench_price_cad: pd.Series, cashflows_cad: pd.Series) -> pd.Series:
    bench_price_cad = bench_price_cad.sort_index().ffill().bfill()
    cashflows_cad = cashflows_cad.reindex(bench_price_cad.index).fillna(0.0)

    shares = 0.0
    values = []
    for d in bench_price_cad.index:
        px = float(bench_price_cad.loc[d])
        cf = float(cashflows_cad.loc[d])
        if px <= 0:
            values.append(np.nan)
            continue
        shares += cf / px
        values.append(shares * px)
    return pd.Series(values, index=bench_price_cad.index)


def time_weighted_returns(port_value: pd.Series, external_cashflows: pd.Series) -> pd.Series:
    V = port_value.replace(0.0, np.nan).ffill()
    CF = external_cashflows.reindex(V.index).fillna(0.0)

    prev = V.shift(1)
    r = (V - CF) / prev - 1.0
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    ok = (prev > 0) & (r > -0.9999) & np.isfinite(r)
    return r[ok]


# =============================
# PORTFOLIO VALUE SERIES (FIXED)
# =============================
def build_portfolio_value_series(
    tx_df: pd.DataFrame,
    close_prices: pd.DataFrame,
    fx_usdcad: pd.Series,
) -> pd.Series:
    if close_prices.empty:
        return pd.Series(dtype=float)

    close_prices = close_prices.sort_index().ffill().bfill()  # prevents 0-cliff days

    all_days = sorted(close_prices.index)
    tx = tx_df.copy()
    tx["Date_d"] = pd.to_datetime(tx["Date"]).dt.date
    tx["Ticker"] = tx["Ticker"].map(normalize_ticker)

    tx["delta_shares"] = np.where(tx["TT"] == "BUY", tx["Shares"], -tx["Shares"])
    tx["TD"] = tx["Date_d"].apply(lambda d: nearest_prev_trading_day(all_days, d))
    tx = tx.dropna(subset=["TD"]).copy()
    tx["TD"] = tx["TD"].astype(object)

    pos = (
        tx.groupby(["TD", "Ticker"])["delta_shares"]
        .sum()
        .unstack("Ticker")
        .reindex(all_days)
        .fillna(0.0)
        .cumsum()
        .fillna(0.0)
    )

    fx_aligned = pd.Series({d: get_fx_for_date(fx_usdcad, d) for d in all_days})

    value = pd.Series(0.0, index=all_days)
    for t in pos.columns:
        if t not in close_prices.columns:
            continue

        ccy = CAD
        if (tx["Ticker"] == t).any():
            ccy = tx.loc[tx["Ticker"] == t, "Currency"].iloc[0]

        mult = fx_aligned if ccy == USD else 1.0
        value += pos[t] * close_prices[t] * mult

    value = value.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return value


# =============================
# MONTE CARLO + CONTRIBUTIONS
# =============================
def build_monthly_lump_schedule(index_days: List[dt.date], monthly_amount: float, deposit_day: int) -> pd.Series:
    s = pd.Series(0.0, index=index_days)
    if monthly_amount <= 0 or not index_days:
        return s

    start = index_days[0]
    end = index_days[-1]
    cur = dt.date(start.year, start.month, 1)

    while cur <= end:
        d = dt.date(cur.year, cur.month, min(int(deposit_day), 28))
        dd = None
        for x in index_days:
            if x >= d:
                dd = x
                break
        if dd is not None and dd in s.index:
            s.loc[dd] += float(monthly_amount)

        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)

    return s


def build_smoothed_daily_schedule(index_days: List[dt.date], monthly_amount: float) -> pd.Series:
    """
    Smooth the monthly deposit across business days (constant daily contribution).
    monthly_amount is interpreted as a monthly contribution.
    We convert to an annual contribution and distribute across trading days:
        per_day = monthly * 12 / 252
    """
    s = pd.Series(0.0, index=index_days)
    if monthly_amount <= 0 or not index_days:
        return s
    per_day = float(monthly_amount) * 12.0 / TRADING_DAYS
    s.iloc[:] = per_day
    s.iloc[0] = 0.0  # don't add on day 0 (valuation date)
    return s


def mc_paths_with_deposits(
    start_value: float,
    mu_annual: float,
    sigma_annual: float,
    n_steps: int,
    n_sims: int,
    seed: int,
    deposits_by_step: np.ndarray,  # length n_steps+1
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt_ = 1.0 / TRADING_DAYS
    shocks = rng.normal(0.0, 1.0, size=(n_sims, n_steps))
    inc = (mu_annual - 0.5 * sigma_annual**2) * dt_ + sigma_annual * np.sqrt(dt_) * shocks

    V = np.full((n_sims, n_steps + 1), float(start_value), dtype=float)
    for k in range(1, n_steps + 1):
        V[:, k] = V[:, k - 1] * np.exp(inc[:, k - 1]) + float(deposits_by_step[k])
    return V


def fan_percentiles(paths: np.ndarray, ps: List[float]) -> Dict[float, np.ndarray]:
    return {p: np.percentile(paths, p, axis=0) for p in ps}


# =============================
# MAIN UI
# =============================
def main() -> None:
    st.set_page_config(page_title="TFSA Tracker + Monte Carlo", layout="wide")
    st.title("TFSA Investment Tracker (CAD) + Monte Carlo & Fan Chart")

    st.sidebar.header("1) Upload tracker file")
    up = st.sidebar.file_uploader("Upload TFSATracker.xlsx", type=["xlsx"])
    if not up:
        st.info("Upload your tracker spreadsheet (.xlsx) to begin.")
        return

    xls = pd.ExcelFile(up)
    sheet = st.sidebar.selectbox("Sheet", xls.sheet_names, index=0)
    raw = pd.read_excel(up, sheet_name=sheet)

    df = ensure_columns(raw)

    tickers_raw = raw["Ticker"].astype(str)
    bad_ws = tickers_raw[tickers_raw != tickers_raw.str.strip()]
    if len(bad_ws) > 0:
        st.warning("Tickers with leading/trailing spaces found:")
        st.write(pd.DataFrame({"Row": bad_ws.index, "Ticker": bad_ws.values}))
    else:
        st.sidebar.success("No ticker trailing/leading spaces detected ✅")

    start_date = date_to_py(df["Date"].min())
    today = dt.date.today()

    st.sidebar.header("2) Data settings")
    end_date = st.sidebar.date_input("Valuation end date", value=min(today, today), min_value=start_date, max_value=today)

    st.sidebar.header("3) Data sources")
    enable_yf_fallback = st.sidebar.checkbox(
        "Use Yahoo fallback for missing tickers",
        value=True,
        help="Only used for tickers Stooq can't fetch."
    )

    st.sidebar.header("4) Dividends / DRIP")
    enable_drip = st.sidebar.checkbox(
        "Enable automatic dividend reinvestment (DRIP) [approx]",
        value=True,
        help="Dividends from yfinance (cached). Reinvest on payment date at Close price."
    )

    st.sidebar.header("5) Benchmarks")
    picks = st.sidebar.multiselect(
        "Pick up to 3 benchmarks",
        options=list(BENCH_PRESETS.keys()),
        default=["Total US Market (VTI)"],
        max_selections=3,
    )
    custom_bench = st.sidebar.text_input("Optional custom benchmark ticker", value="").strip()
    bench_tickers = [BENCH_PRESETS[p] for p in picks]
    if custom_bench:
        bench_tickers.append(custom_bench)

    st.sidebar.header("6) Forecast contributions")
    monthly_deposit = st.sidebar.number_input("Monthly deposit (CAD)", value=0.0, step=50.0)
    deposit_style = st.sidebar.selectbox("Contribution style", ["Smoothed daily (recommended)", "Monthly lump sum"])
    deposit_day = st.sidebar.slider("Deposit day of month (lump sum)", 1, 28, 1)

    st.sidebar.header("7) Monte Carlo return model")
    prior_mode = st.sidebar.selectbox(
        "Prior for long-run expected return",
        ["Benchmark prior (recommended)", "Fixed prior", "No prior (use only my data)"]
    )
    prior_years = st.sidebar.slider("Prior lookback (years)", 3, 15, 10)
    winsor = st.sidebar.checkbox("Robustify (clip extreme daily returns)", value=True)
    shrink_user = st.sidebar.slider(
        "Shrink strength (0 = trust only my data, 1 = mostly long-run prior)",
        0.0, 1.0, 0.35, 0.05
    )
    prior_bench = st.sidebar.selectbox(
        "Benchmark used for prior (CAD-based)",
        ["VTI", "SPY", "QQQ", "VFV.TO", "XIU.TO"],
        index=0
    )

    show_mc_debug = st.sidebar.checkbox("Show Monte Carlo debug stats", value=False)

    portfolio_tickers = sorted(df["Ticker"].unique().tolist())
    tickers_all = sorted(list(set(portfolio_tickers + bench_tickers + [prior_bench])))

    const_prices = (
        df.sort_values("Date")
          .groupby("Ticker")["Origin P/P"]
          .last()
          .to_dict()
    )

    fx = fetch_fx_usd_to_cad_stooq(max(start_date - dt.timedelta(days=10), dt.date(1990, 1, 1)), end_date)

    lookback_start = max(end_date - dt.timedelta(days=365 * 5), start_date - dt.timedelta(days=30))

    close, still_missing, used_constant = fetch_close_prices_hybrid(
        tickers_all,
        lookback_start,
        end_date,
        enable_yf_fallback=enable_yf_fallback,
        const_fallback_prices=const_prices,
    )

    if used_constant:
        st.info(
            "Filled these tickers with a constant price series using your sheet's latest Origin P/P "
            "(because online data couldn't be fetched):\n\n"
            + ", ".join(used_constant)
        )

    if still_missing:
        st.warning(
            "These tickers could not be fetched from Stooq/Yahoo and could not be filled from Origin P/P:\n\n"
            + ", ".join(still_missing)
        )

    events = df.copy()
    events["Synthetic"] = False

    if enable_drip:
        divs = fetch_dividends_yf(portfolio_tickers, lookback_start, end_date)
        drip_raw = build_drip_events(portfolio_tickers, lookback_start, end_date, divs, close, events)
        events = apply_drip_to_events(events, drip_raw, close)

    tx_enriched, ending_states = compute_acb_realized(events, fx)

    fx_last = get_fx_for_date(fx, close.index.max() if not close.empty else end_date)
    rows = []
    total_value = total_acb = total_realized = 0.0

    for t, stt in ending_states.items():
        if stt.shares <= 1e-12:
            total_realized += stt.realized_pl_cad
            continue

        ccy = CAD
        if (events["Ticker"] == t).any():
            ccy = events.loc[events["Ticker"] == t, "Currency"].iloc[0]

        px = np.nan
        if not close.empty and t in close.columns:
            s = close[t].dropna()
            if not s.empty:
                px = float(s.iloc[-1])

        if not np.isfinite(px) or px <= 0:
            px = float(const_prices.get(t, np.nan))

        if not np.isfinite(px) or px <= 0:
            continue

        value_cad = stt.shares * px * (fx_last if ccy == USD else 1.0)
        unreal = value_cad - stt.acb_cad

        rows.append({
            "Ticker": t,
            "Currency": ccy,
            "Shares": stt.shares,
            "Price": px,
            "FX (USD->CAD)": fx_last if ccy == USD else 1.0,
            "Market Value (CAD)": value_cad,
            "ACB Remaining (CAD)": stt.acb_cad,
            "Unrealized P/L (CAD)": unreal,
            "Realized P/L (CAD)": stt.realized_pl_cad,
        })

        total_value += value_cad
        total_acb += stt.acb_cad
        total_realized += stt.realized_pl_cad

    holdings = pd.DataFrame(rows).sort_values("Market Value (CAD)", ascending=False) if rows else pd.DataFrame()

    port_value = build_portfolio_value_series(
        tx_df=events[events["TT"].isin(["BUY", "SELL"])].copy(),
        close_prices=close,
        fx_usdcad=fx,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value (CAD)", f"${total_value:,.2f}")
    c2.metric("ACB Remaining (CAD)", f"${total_acb:,.2f}")
    c3.metric("Unrealized P/L (CAD)", f"${(total_value - total_acb):,.2f}")
    c4.metric("Realized P/L (CAD)", f"${total_realized:,.2f}")

    st.subheader("Holdings (snapshot)")
    st.dataframe(holdings, use_container_width=True) if not holdings.empty else st.info("No holdings or no prices available.")

    with st.expander("Transactions (including synthetic DRIP buys if enabled)"):
        show_cols = ["Date", "Ticker", "Currency", "TT", "Shares", "Total Amount", "Amount_CAD", "Realized_PL_CAD", "Synthetic"]
        st.dataframe(tx_enriched[show_cols].sort_values("Date"), use_container_width=True)

    st.subheader("Portfolio history (CAD) + Benchmarks (cashflow-matched)")
    if port_value.empty:
        st.info("Not enough data to build portfolio history.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(port_value.index), y=port_value.values, mode="lines", name="Portfolio (CAD)"))

        index_days = list(port_value.index)
        ext_cashflows = build_external_cashflow_series(tx_enriched, index_days)

        for bt in bench_tickers:
            t = normalize_ticker(bt)
            if close.empty or t not in close.columns:
                continue

            s = close[t].copy().sort_index().ffill().bfill()

            is_usd = not t.endswith(".TO")
            if is_usd:
                fx_series = pd.Series({d: get_fx_for_date(fx, d) for d in s.index})
                s = s * fx_series

            s = s.reindex(index_days).ffill().bfill()
            bench_val = cashflow_matched_benchmark_value(s, ext_cashflows)

            if bench_val.dropna().empty:
                continue

            fig.add_trace(go.Scatter(
                x=list(bench_val.index),
                y=bench_val.values,
                mode="lines",
                name=f"{t} (cashflow-matched)"
            ))

        fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Value (CAD)", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monte Carlo forecast + Fan chart")

    nz_hist = port_value.replace(0.0, np.nan).dropna()
    if nz_hist.empty or len(nz_hist) < 15:
        st.info("Not enough non-zero history for Monte Carlo yet (need at least ~15 points).")
        return

    index_days = list(port_value.index)
    ext_cashflows = build_external_cashflow_series(tx_enriched, index_days)
    twr = time_weighted_returns(port_value, ext_cashflows)

    # ----- mu_hat/sigma_hat from portfolio TWR -----
    if len(twr) < 30:
        mu_hat, sig_hat = 0.07, 0.15
    else:
        lr_hat = np.log1p(twr)
        mu_hat, sig_hat = annualize_logret(lr_hat, winsorize=winsor)

    # ----- prior mu/sigma -----
    mu_prior, sig_prior = 0.07, 0.15  # fixed prior baseline

    if prior_mode == "Benchmark prior (recommended)":
        prior_end = end_date
        prior_start = max(dt.date(1990, 1, 1), prior_end - dt.timedelta(days=365 * int(prior_years)))

        t = normalize_ticker(prior_bench)

        if close.empty or t not in close.columns:
            extra_close, _, _ = fetch_close_prices_hybrid(
                [t],
                prior_start,
                prior_end,
                enable_yf_fallback=enable_yf_fallback,
                const_fallback_prices={},
            )
            bench = extra_close[t].dropna() if (not extra_close.empty and t in extra_close.columns) else pd.Series(dtype=float)
        else:
            bench = close[t].dropna()

        if not bench.empty:
            bench = bench[(bench.index >= prior_start) & (bench.index <= prior_end)].sort_index().ffill().bfill()

            is_usd = not t.endswith(".TO")
            if is_usd:
                fx_series = pd.Series({d: get_fx_for_date(fx, d) for d in bench.index})
                bench = bench * fx_series

            br = np.log(bench / bench.shift(1)).dropna()
            if len(br) >= 60:
                mu_prior, sig_prior = annualize_logret(br, winsorize=winsor)

    elif prior_mode == "Fixed prior":
        mu_prior, sig_prior = 0.07, 0.15

    elif prior_mode == "No prior (use only my data)":
        mu_prior, sig_prior = mu_hat, sig_hat

    # ----- shrinkage blend -----
    mu_annual = (1.0 - shrink_user) * mu_hat + shrink_user * mu_prior
    sigma_annual = (1.0 - shrink_user) * sig_hat + shrink_user * sig_prior
    if show_mc_debug:
        st.caption(
            f"MC inputs: mu_hat={mu_hat:.2%}/yr, sig_hat={sig_hat:.2%}/yr | "
            f"mu_prior={mu_prior:.2%}/yr, sig_prior={sig_prior:.2%}/yr | "
            f"mu_used={mu_annual:.2%}/yr, sig_used={sigma_annual:.2%}/yr | "
            f"n={len(twr)}"
        )

    target_value = st.sidebar.number_input("Target portfolio value (CAD)", value=100000.0, step=5000.0)
    target_date = st.sidebar.date_input("Target date", value=end_date + dt.timedelta(days=365))
    if target_date <= end_date:
        st.warning("Target date must be after valuation end date.")
        return

    n_sims = st.sidebar.slider("Simulations", 200, 5000, 1500, step=100)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    forecast_idx = list(pd.bdate_range(end_date, target_date).date)
    if len(forecast_idx) < 2:
        st.warning("Target date is too close to end date for Monte Carlo.")
        return

    n_steps = len(forecast_idx) - 1
    start_val = float(nz_hist.iloc[-1])

    if deposit_style.startswith("Smoothed"):
        dep_series = build_smoothed_daily_schedule(forecast_idx, float(monthly_deposit))
    else:
        dep_series = build_monthly_lump_schedule(forecast_idx, float(monthly_deposit), int(deposit_day))

    deposits_by_step = dep_series.values  # length n_steps+1

    paths = mc_paths_with_deposits(
        start_value=start_val,
        mu_annual=mu_annual,
        sigma_annual=sigma_annual,
        n_steps=int(n_steps),
        n_sims=int(n_sims),
        seed=int(seed),
        deposits_by_step=deposits_by_step,
    )

    ps = [5, 10, 50, 90, 95]
    bands = fan_percentiles(paths, ps)

    final_vals = paths[:, -1]
    prob = float(np.mean(final_vals >= float(target_value)))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"P(Value ≥ {target_value:,.0f} by {target_date.isoformat()})", f"{prob*100:.1f}%")
    m2.metric("P5", f"${np.percentile(final_vals, 5):,.0f}")
    m3.metric("P10", f"${np.percentile(final_vals, 10):,.0f}")
    m4.metric("P90", f"${np.percentile(final_vals, 90):,.0f}")
    m5.metric("P95", f"${np.percentile(final_vals, 95):,.0f}")

    fan = go.Figure()
    fan.add_trace(go.Scatter(x=forecast_idx, y=bands[95], mode="lines", name="P95", line=dict(width=0.6)))
    fan.add_trace(go.Scatter(x=forecast_idx, y=bands[5], mode="lines", name="P5", line=dict(width=0.6), fill="tonexty"))
    fan.add_trace(go.Scatter(x=forecast_idx, y=bands[90], mode="lines", name="P90", line=dict(width=0.6)))
    fan.add_trace(go.Scatter(x=forecast_idx, y=bands[10], mode="lines", name="P10", line=dict(width=0.6), fill="tonexty"))
    fan.add_trace(go.Scatter(x=forecast_idx, y=bands[50], mode="lines", name="Median (P50)", line=dict(width=3)))

    fan.update_layout(height=450, xaxis_title="Date", yaxis_title="Portfolio Value (CAD)", legend=dict(orientation="h"))
    st.plotly_chart(fan, use_container_width=True)

    st.divider()
    st.caption("Built by Gavin · Source: https://github.com/G-ScM124/Investment_Tracker_McSim")

if __name__ == "__main__":
    main()

