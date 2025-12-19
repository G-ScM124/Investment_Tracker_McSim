# FAQ

## Does this work for any ticker?
It works for many common US & Canadian symbols, but not everything.
- US symbols are mapped to Stooq as `SYMBOL.us`
- Canadian `.TO` symbols are mapped to Stooq as `SYMBOL.to`
Some instruments, OTC tickers, or unusual formats may not exist on Stooq.

If a price can’t be fetched:
1) (optional) the app tries yfinance
2) if still missing and you have `Origin P/P` in the sheet, it can fall back to a constant series

## Why am I rate-limited?
Yahoo (via yfinance) can rate-limit heavily. The app uses caching and only uses yfinance as a fallback for prices and for dividends.
If you hit limits, disable DRIP.

## Are dividends exact?
No. Dividends/DRIP are “best-effort”:
- It uses the dividend payment date and reinvests at that day’s close.
- Tax withholding, payment timing, and reinvestment mechanics are simplified.

## Why is my Monte Carlo probability “too low/high”?
Common causes:
- short history → unstable mu_hat
- high volatility → wide distribution of outcomes
- negative mu_used (if shrink strength is too low)

Increase shrink strength (closer to 1) to rely more on the long-run prior until you have more history.

## Should I commit my spreadsheet to GitHub?
No, never. Keep your tracker file private. Use the app’s uploader at runtime.

## Can I deploy this?
This project is designed to be run locally: you download the repo, run the Streamlit app, and upload your spreadsheet. I’m not providing a hosted/public instance.
If you choose to deploy it yourself (e.g., Streamlit Cloud or your own server), do so at your own risk and don’t upload sensitive spreadsheets to a public deployment. 
If you share the app or code publicly, please keep the attribution in the README/app UI.
