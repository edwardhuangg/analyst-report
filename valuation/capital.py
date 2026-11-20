"""Cost of capital, set once per valuation date and identical for every name (critique C1).

The per-name analyst never sets the market-wide inputs:
    risk-free rate   10-yr U.S. Treasury yield on the date (market data)
    ERP              one method: the implied ERP published at the start of the year (ERP_TABLE), or FA_ERP
    credit spreads   one table: the filer's agency rating (10-K / 10-Q) -> spread over the risk-free rate; the
                     synthetic rating from interest coverage only when the filings disclose none
and beta comes from a method, not judgment:
    bottom-up        median peer beta (Blume-adjusted, unlevered at each peer's market D/E), relevered at the
                     company's market D/E
    own              the company's 5-yr monthly regression beta, Blume-adjusted (0.67 x raw + 0.33)
    method beta      the average of the two (either alone when the other is missing)
An analyst may move beta by at most BETA_BAND (with a written reason) and the cost of debt by at most KD_BAND.
"""
from __future__ import annotations

import os
from datetime import date
from statistics import median
from typing import Optional

# Implied equity risk premium for the S&P 500 at the start of each year (Damodaran, FCFE-based), the most widely
# cited market-implied figure. It is measured against the 10-yr Treasury, the risk-free rate used here (Kroll's 5.0%
# is paired with the 20-yr yield instead). Update this table each January, or set FA_ERP (and FA_ERP_SOURCE) for a
# house view; every name on a date uses one value. Never back it out of the covered names' own prices: that assumes
# they are fairly priced and removes the variant view.
ERP_TABLE = [(date(2019, 1, 1), 0.0596), (date(2020, 1, 1), 0.0520), (date(2021, 1, 1), 0.0472),
             (date(2022, 1, 1), 0.0424), (date(2023, 1, 1), 0.0594), (date(2024, 1, 1), 0.0460),
             (date(2025, 1, 1), 0.0433), (date(2026, 1, 1), 0.0423)]

# Synthetic rating from EBIT / interest expense -> default spread (large non-financial firms, Damodaran table).
SPREADS = [(8.5, "AAA", 0.0059), (6.5, "AA", 0.0070), (5.5, "A+", 0.0092), (4.25, "A", 0.0107), (3.0, "A-", 0.0121),
           (2.5, "BBB", 0.0147), (2.25, "BB+", 0.0187), (2.0, "BB", 0.0225), (1.75, "B+", 0.0295), (1.5, "B", 0.0346),
           (1.25, "B-", 0.0425), (0.8, "CCC", 0.0600), (0.65, "CC", 0.0900), (0.2, "C", 0.1200), (-1e9, "D", 0.1500)]

# House long-run (terminal) growth: US nominal GDP, about 1.8% real growth (the Fed's longer-run projection) plus
# the 2% inflation target, below the 10-yr risk-free rate the ERP is measured against. The ERP source assumes the
# market grows at the risk-free rate after five years; 3% sat 2pt under today's rate and biased every DCF down.
# FA_TERMINAL_G overrides it for a house view.
TERMINAL_G = 0.0375

BETA_BAND = 0.10   # analyst override of the method beta, either way
KD_BAND = 0.0075   # analyst override of the rating-based cost of debt, either way
BETA_RANGE = (0.5, 2.0)


def terminal_growth() -> tuple[float, str]:
    """The one long-run growth rate for every name on a date (an industry in secular decline goes lower, with a
    reason), with its source."""
    env = os.environ.get("FA_TERMINAL_G")
    if env:
        try:
            return float(env), "House long-run growth (FA_TERMINAL_G)"
        except ValueError:
            pass
    return TERMINAL_G, ("House long-run growth: US nominal GDP (about 1.8% real + 2% inflation), below the 10-yr "
                        "risk-free rate")


def erp(as_of: date) -> tuple[float, str]:
    """The one ERP for every name on this date, with its source."""
    env = os.environ.get("FA_ERP")
    if env:
        try:
            return float(env), os.environ.get("FA_ERP_SOURCE") or "House ERP (FA_ERP)"
        except ValueError:
            pass
    d, v = max(((d, v) for d, v in ERP_TABLE if d <= as_of), default=ERP_TABLE[0])
    stale = (as_of - d).days > 400
    return v, (f"Implied ERP for the S&P 500 at the start of {d.year} (Damodaran, FCFE-based): the same for every "
               f"name on this date, measured against the 10-yr Treasury"
               f"{' — table not updated for this year: set FA_ERP' if stale else ''}")


# Agency ratings on the S&P scale -> the same default-spread table (notches between its points interpolated)
AGENCY_SPREADS = {"AAA": 0.0059, "AA+": 0.0065, "AA": 0.0070, "AA-": 0.0080, "A+": 0.0092, "A": 0.0107,
                  "A-": 0.0121, "BBB+": 0.0134, "BBB": 0.0147, "BBB-": 0.0167, "BB+": 0.0187, "BB": 0.0225,
                  "BB-": 0.0260, "B+": 0.0295, "B": 0.0346, "B-": 0.0425}
MOODYS_TO_SP = {"Aaa": "AAA", "Aa1": "AA+", "Aa2": "AA", "Aa3": "AA-", "A1": "A+", "A2": "A", "A3": "A-",
                "Baa1": "BBB+", "Baa2": "BBB", "Baa3": "BBB-", "Ba1": "BB+", "Ba2": "BB", "Ba3": "BB-",
                "B1": "B+", "B2": "B", "B3": "B-"}
SP_ORDER = list(AGENCY_SPREADS)


def agency_rating(ratings: dict) -> Optional[tuple[str, float, str]]:
    """(rating on the S&P scale, spread, how) from the filer's disclosed agency ratings: the middle of three, the
    lower of two (the market convention for split ratings). None when the filings name none."""
    sp = [(ag, r if ag != "Moody's" else MOODYS_TO_SP.get(r)) for ag, r in (ratings or {}).items()
          if ag in ("Moody's", "S&P", "Fitch")]
    sp = [(ag, r) for ag, r in sp if r in AGENCY_SPREADS]
    if not sp:
        return None
    ranked = sorted(sp, key=lambda x: SP_ORDER.index(x[1]))
    pick = ranked[1][1] if len(ranked) == 3 else ranked[-1][1]
    how = ", ".join(f"{ag} {ratings[ag]}" for ag, _ in sp)
    return pick, AGENCY_SPREADS[pick], how


def synthetic_rating(coverage: Optional[float]) -> tuple[str, float]:
    """Rating from interest coverage, capped at AA: AAA is an agency's verdict on two US industrial companies (MSFT,
    JNJ), not a coverage ratio (PM round 4: a synthetic AAA for all six names, actual AA to A)."""
    if coverage is None:
        return "A-", 0.0121
    for floor, rating, spread in SPREADS:
        if coverage > floor:
            return ("AA", 0.0070) if rating == "AAA" else (rating, spread)
    return "D", 0.15


def blume(raw: float) -> float:
    return 0.67 * raw + 0.33


def unlever(beta: float, de: float, tax: float) -> float:
    return beta / (1 + (1 - tax) * max(de, 0.0))


def relever(beta_u: float, de: float, tax: float) -> float:
    return beta_u * (1 + (1 - tax) * max(de, 0.0))


def method_beta(own_raw: Optional[float], peers: list, de: float, tax: float) -> tuple[float, str]:
    """peers: [{'ticker', 'beta_raw', 'de'}]. Returns (beta, how it was computed)."""
    parts, notes = [], []
    pu = [unlever(blume(p["beta_raw"]), p["de"], tax) for p in peers
          if p.get("beta_raw") is not None and p.get("de") is not None and 0 < p["beta_raw"] < 4]
    if len(pu) >= 3:
        bu = median(pu)
        b = relever(bu, de, tax)
        parts.append(b)
        notes.append(f"bottom-up {b:.2f} (median unlevered {bu:.2f} across {len(pu)} peers, relevered at D/E "
                     f"{de:.0%})")
    if own_raw is not None and 0 < own_raw < 4:
        parts.append(blume(own_raw))
        notes.append(f"own Blume-adjusted {blume(own_raw):.2f} (raw {own_raw:.2f})")
    if not parts:
        return 1.0, "No beta data: market beta 1.0"
    b = min(max(sum(parts) / len(parts), BETA_RANGE[0]), BETA_RANGE[1])
    return round(b, 2), "Method beta = " + (" and ".join(notes) if len(notes) == 1 else
                                            "average of " + "; ".join(notes))
