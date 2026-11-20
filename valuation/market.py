"""Market data from Yahoo Finance (via yfinance): prices, rates, beta, estimates, peer snapshots.

Two Yahoo quirks matter for a model built "as of" a past date:
- historical closes are back-adjusted for every later split and spin-off factor, so a
  close is multiplied back by those factors to get the price that actually traded
  (and that matches the share count in the filings of the time);
- consensus estimates and `info` fields are as of today, so they are only used when the
  valuation date is recent, and only for the fiscal year they actually refer to.
"""
from __future__ import annotations

import functools
import math
import threading
from datetime import date, datetime, timedelta
from typing import Optional

import yfinance as yf

RECENT_DAYS = 21  # a valuation date this close to today can use today's estimates
# A Yahoo request can hang on a dead socket past yfinance's own timeouts (a build waited 48 minutes on one,
# 2026-11-15): every call gets a hard deadline and the build carries on without that data point
YAHOO_DEADLINE = 60
TIMED_OUT: list = []  # calls that hit the deadline in this process (name, args)


def _deadline(fallback):
    """Run a Yahoo call on a daemon thread; past YAHOO_DEADLINE seconds return fallback() instead."""
    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kw):
            box = {}

            def target():
                try:
                    box["v"] = fn(*args, **kw)
                except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
                    box["e"] = e
            t = threading.Thread(target=target, daemon=True, name=f"yahoo-{fn.__name__}")
            t.start()
            t.join(YAHOO_DEADLINE)
            if t.is_alive():
                TIMED_OUT.append((fn.__name__, args))
                return fallback()
            if "e" in box:
                raise box["e"]
            return box["v"]
        return run
    return wrap


def _f(v) -> Optional[float]:
    try:
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def yahoo_symbol(ticker: str) -> str:
    """SEC/exchange tickers use dots for share classes (BRK.B); Yahoo uses dashes (BRK-B)."""
    return ticker.upper().replace(".", "-").replace("/", "-")


def is_recent(d: date) -> bool:
    return (date.today() - d).days <= RECENT_DAYS


def _ticker(ticker: str):
    return yf.Ticker(yahoo_symbol(ticker))


@_deadline(lambda: 1.0)
def split_factor_after(ticker: str, d: date) -> float:
    """Product of Yahoo split / spin-off factors dated after d (what back-adjustment divided by)."""
    try:
        sp = _ticker(ticker).splits
    except Exception:
        return 1.0
    f = 1.0
    for ts, ratio in (sp.items() if sp is not None else []):
        if ts.date() > d and _f(ratio):
            f *= float(ratio)
    return f


@_deadline(lambda: (None, None))
def close_on(ticker: str, d: date) -> tuple[Optional[float], Optional[date]]:
    """Close that actually traded on or before d (Yahoo's back-adjustment for later splits undone)."""
    try:
        h = _ticker(ticker).history(start=d - timedelta(days=10), end=d + timedelta(days=1), auto_adjust=False)
    except Exception:
        return None, None
    if h is None or h.empty:
        return None, None
    h = h.dropna(subset=["Close"])  # today's row has no close while the market is open
    if h.empty:
        return None, None
    px_date = h.index[-1].date()
    px = _f(h["Close"].iloc[-1])
    if px is None:
        return None, None
    return px * split_factor_after(ticker, px_date), px_date


def risk_free_on(d: date) -> Optional[float]:
    px, _ = close_on("^TNX", d)
    if px is None:
        return None
    return px / 1000 if px > 20 else px / 100  # quoted as % (older data: % x 10)


@_deadline(dict)
def info(ticker: str) -> dict:
    try:
        return _ticker(ticker).info or {}
    except Exception:
        return {}


@_deadline(lambda: (None, ""))
def beta_on(ticker: str, d: date, years: int = 5) -> tuple[Optional[float], str]:
    """Point-in-time beta: monthly total returns vs the S&P 500 over the `years` before d. Month-end closes come from
    daily prices up to d: Yahoo's monthly bar for d's month closes at the latest price, after the date (CRM: its
    Nov 17 drop moved a Nov 12 raw beta from 1.24 to 1.27)."""
    try:
        start = d - timedelta(days=365 * years + 40)
        kw = dict(start=start, end=d + timedelta(days=1), interval="1d", auto_adjust=True)

        def monthly(t):
            c = t.history(**kw)["Close"]
            c = c[c.index.date <= d]
            return c.groupby(c.index.strftime("%Y-%m")).last().pct_change().dropna()
        a, m = monthly(_ticker(ticker)), monthly(yf.Ticker("^GSPC"))
        j = a.to_frame("a").join(m.to_frame("m"), how="inner")
        if len(j) < 24:
            return None, ""
        b = float(j["a"].cov(j["m"]) / j["m"].var())
        return b, f"{len(j)} monthly returns vs S&P 500 to {d:%b %Y} (computed from Yahoo prices)"
    except Exception:
        return None, ""


def _ts_date(v) -> Optional[date]:
    try:
        return datetime.fromtimestamp(int(v)).date()
    except Exception:
        return None


@_deadline(dict)
def estimates(ticker: str, fy_end: date) -> dict:
    """Consensus EPS and revenue for the fiscal year ending near fy_end, if Yahoo has it.

    Yahoo labels the current fiscal year '0y' (ending at info['nextFiscalYearEnd']) and the
    next one '+1y'. Returns {} when neither matches fy_end within a month.
    """
    t = _ticker(ticker)
    i = info(ticker)
    nxt = _ts_date(i.get("nextFiscalYearEnd"))
    if not nxt:
        last = _ts_date(i.get("lastFiscalYearEnd"))
        nxt = last.replace(year=last.year + 1) if last else None
    if not nxt:
        return {}
    period = None
    for p, end in (("0y", nxt), ("+1y", nxt.replace(year=nxt.year + 1))):
        if abs((end - fy_end).days) <= 31:
            period = p
    if period is None:
        return {}
    out = {"period": period, "fy_end": fy_end}
    try:
        ee = t.earnings_estimate
        if ee is not None and period in ee.index:
            out["eps"] = _f(ee.loc[period, "avg"])
            out["eps_n"] = _f(ee.loc[period, "numberOfAnalysts"])
    except Exception:
        pass
    try:
        re_ = t.revenue_estimate
        if re_ is not None:
            for p in ("0y", "+1y"):
                if p in re_.index:
                    out[f"rev_{p}"] = _f(re_.loc[p, "avg"])
                    out[f"rev_{p}_growth"] = _f(re_.loc[p, "growth"])
                    out[f"rev_{p}_base"] = _f(re_.loc[p, "yearAgoRevenue"])
    except Exception:
        pass
    return out


def fy_eps_estimate(ticker: str, fy_end: Optional[date] = None, as_of: Optional[date] = None) -> tuple[Optional[float], str]:
    """Consensus EPS for the fiscal year ending at fy_end (default: the company's current fiscal
    year). Only used when the valuation date is recent; otherwise (None, reason)."""
    if as_of is not None and not is_recent(as_of):
        return None, "No point-in-time consensus for a past valuation date"
    i = info(ticker)
    if fy_end is None:
        fy_end = _ts_date(i.get("nextFiscalYearEnd")) or date.today()
    est = estimates(ticker, fy_end)
    if est.get("eps"):
        n = f", {est['eps_n']:.0f} analysts" if est.get("eps_n") else ""
        return est["eps"], f"Yahoo consensus FY ending {fy_end:%b %Y} ({est['period']}{n}; as of {date.today():%b %-d, %Y})"
    v = _f(i.get("forwardEps"))
    if v:
        return v, "Yahoo forward EPS (next 12 months; as of today)"
    v = _f(i.get("trailingEps"))
    return v, "TRAILING: Yahoo trailing GAAP EPS (no forward estimate; excluded from the forward P/E median)"


@_deadline(lambda: None)
def eps_next(ticker: str) -> Optional[float]:
    """Consensus EPS for the fiscal year after the current one ('+1y'), for growth-adjusted multiples."""
    try:
        ee = _ticker(ticker).earnings_estimate
        return _f(ee.loc["+1y", "avg"]) if ee is not None and "+1y" in ee.index else None
    except Exception:
        return None


@_deadline(lambda: None)
def eps_growth_next(ticker: str) -> Optional[float]:
    """Next-year consensus EPS growth from one source and one fiscal-year pair: '+1y' ÷ '0y' − 1."""
    try:
        ee = _ticker(ticker).earnings_estimate
        if ee is None or "+1y" not in ee.index or "0y" not in ee.index:
            return None
        e0, e1 = _f(ee.loc["0y", "avg"]), _f(ee.loc["+1y", "avg"])
        return e1 / e0 - 1 if e0 and e1 and e0 > 0 else None
    except Exception:
        return None


@_deadline(lambda: None)
def revenue_growth_next(ticker: str) -> Optional[float]:
    """Next-year consensus revenue growth: '+1y' ÷ '0y' − 1 (the enterprise-multiple growth axis)."""
    try:
        re_ = _ticker(ticker).revenue_estimate
        if re_ is None or "+1y" not in re_.index or "0y" not in re_.index:
            return None
        r0, r1 = _f(re_.loc["0y", "avg"]), _f(re_.loc["+1y", "avg"])
        return r1 / r0 - 1 if r0 and r1 and r0 > 0 else None
    except Exception:
        return None


def peer_snapshot(ticker: str, as_of: date) -> dict:
    """Price on the valuation date and the peer's own current-fiscal-year EPS.

    For a recent valuation date the EPS is Yahoo consensus. For a past date Yahoo has no
    point-in-time consensus, so `eps` is left for the SEC fallback (pipeline) or the analyst.
    """
    i = info(ticker)
    px, px_date = close_on(ticker, as_of)
    if is_recent(as_of):
        eps, eps_note = fy_eps_estimate(ticker)
    else:
        eps, eps_note = None, "Enter the peer's FY EPS guidance midpoint from its latest release"
    return {
        "ticker": ticker.upper(),
        "name": i.get("shortName") or i.get("longName") or ticker.upper(),
        "overlap": i.get("industry") or i.get("sector") or "",
        "price": round(px, 2) if px else px,
        "price_date": px_date,
        "eps": eps,
        "eps_basis": "trailing" if str(eps_note).startswith("TRAILING") else "forward",
        "ev_ebitda": _f(i.get("enterpriseToEbitda")) if is_recent(as_of) else None,
        "eps_next": eps_next(ticker) if is_recent(as_of) and eps else None,
        "g_next": eps_growth_next(ticker) if is_recent(as_of) and eps else None,
        "rev_g_next": revenue_growth_next(ticker) if is_recent(as_of) else None,
        "snapshot": str(date.today()) if is_recent(as_of) and eps else None,  # consensus date (look-ahead check)
        # the fiscal year the consensus is for (calendarized to the company's year: pipeline.calendarize_peer)
        "fy_end": str(_ts_date(i.get("nextFiscalYearEnd")) or "") if is_recent(as_of) and eps and
        str(eps_note).startswith("Yahoo consensus FY ending") else None,
        "source": f"{eps_note}; price = close {px_date:%b %-d, %Y} (Yahoo, unadjusted for later splits)"
                  if px_date else eps_note,
    }


@_deadline(list)
def corporate_actions(ticker: str, after: date, until: date) -> list[tuple[date, float]]:
    """Splits and spin-offs (Yahoo books spin-offs as fractional 'splits') in a date window."""
    try:
        sp = _ticker(ticker).splits
    except Exception:
        return []
    out = []
    for ts, ratio in (sp.items() if sp is not None else []):
        d = ts.date()
        if after < d <= until:
            out.append((d, float(ratio)))
    return out
