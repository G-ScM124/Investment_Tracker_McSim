# Interpretation Guide

This guide explains what the app is calculating and how to interpret the charts.

---

## 1) Holdings snapshot

For each ticker held at the valuation date, the app shows:

- **Shares**: net shares after all BUY/SELL and optional synthetic DRIP buys
- **Price**: latest close available from the price source (Stooq or fallback)
- **Market Value (CAD)**: shares × price × FX (if USD)
- **ACB Remaining (CAD)**: Adjusted cost base remaining (CAD)
- **Unrealized P/L (CAD)**: Market Value − ACB Remaining
- **Realized P/L (CAD)**: cumulative realized gains/losses from SELL transactions

### ACB logic (high level)
- BUY increases shares and increases ACB by the CAD-converted total amount.
- SELL reduces shares and reduces ACB using *average cost per share*.
- Realized P/L = (CAD proceeds) − (CAD cost basis of shares sold).

> This matches a common “average cost” approach. If you need specific jurisdictional rules or special cases, treat this as an approximation.

---

## 2) Portfolio history (CAD)

The history chart is built from:
- cumulative shares held by day (based on your transactions)
- daily closing prices (Stooq / fallback)
- daily FX (USD→CAD) applied to USD holdings

If you ever see sudden drops to zero, it usually indicates missing prices or missing FX for that date. The app forward-fills prices/FX to avoid single-day zeros.

---

## 3) Benchmarks: “cashflow-matched”

Benchmarks are not plotted as “$1 invested once”.

Instead, the app:
1. Takes your **external cashflows** (your BUYs minus SELL proceeds; excludes synthetic DRIP buys)
2. “Buys benchmark shares” on each cashflow date
3. Tracks the benchmark value over time

This makes the benchmark comparison fair when you add money over time.

---

## 4) Monte Carlo / Fan chart

### What it is
A geometric Brownian motion (GBM) simulation of portfolio value:

- Starts from your current portfolio value
- Applies daily random returns implied by an annual **mu** and **sigma**
- Adds future contributions (monthly deposits) as you specify

Outputs:
- Fan chart bands (P5/P10/Median/P90/P95)
- Probability of exceeding a target value by a target date

### Why “mu” can look weird
When you have short or noisy return history, the estimated drift can be unstable.
To reduce this, the app blends:
- **mu_hat** (estimated from your portfolio’s recent time-weighted returns)
- **mu_prior** (a long-run prior from a selected benchmark, or a fixed 7%/15%)

**Shrink strength** controls the blend:
- 0.0 = trust only your data
- 1.0 = rely almost entirely on the long-run prior

### Contributions
You can choose:
- **Smoothed daily**: spreads monthly contribution across business days (smooths stepwise jumps)
- **Monthly lump sum**: deposits on a chosen day each month (fan chart will look stepwise)

> Monte Carlo results are not predictions. They are scenario exploration under model assumptions.
