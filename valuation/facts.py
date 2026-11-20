"""Point-in-time fundamentals from SEC companyfacts (data.sec.gov/api/xbrl/companyfacts).

Used for peers: last-twelve-months EPS, EBITDA and net debt as they stood on the
valuation date (only facts *filed* on or before that date count), so a model dated in
the past never borrows numbers that were published later.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from . import sec

EPS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]
REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
OPINC = ["OperatingIncomeLoss"]
PRETAX = ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomesticAndForeign"]
INTEREST = ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"]
DA = ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortization",
      "DepreciationAmortizationAndOther"]
CASH = ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]
STI = ["ShortTermInvestments", "MarketableSecuritiesCurrent"]
DEBT_LT = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermNotesAndLoans",
           "LongTermNotesPayable", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"]
DEBT_CUR_TOTAL = ["DebtCurrent"]   # total current debt (incl. commercial paper / short-term borrowings)
DEBT_CUR = ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent", "NotesPayableCurrent"]
DEBT_ST = ["ShortTermBorrowings", "CommercialPaper"]
MAX_AGE = 550  # days: a tag the company stopped using is skipped rather than read stale


def company_facts(cik: str) -> dict:
    return sec.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")


def _rows(facts: dict, concept: str, as_of: date, unit_hint: str = "USD", ns: str = "us-gaap") -> list[dict]:
    node = facts.get("facts", {}).get(ns, {}).get(concept)
    if not node:
        return []
    units = node.get("units", {})
    rows = units.get(unit_hint) or units.get("USD/shares") or next(iter(units.values()), [])
    out = []
    for r in rows:
        try:
            if date.fromisoformat(r["filed"]) > as_of:
                continue
            r = dict(r, end_d=date.fromisoformat(r["end"]),
                     start_d=date.fromisoformat(r["start"]) if r.get("start") else None)
        except (KeyError, ValueError):
            continue
        out.append(r)
    return out


def _days(r) -> int:
    return (r["end_d"] - r["start_d"]).days if r.get("start_d") else 0


def ltm(facts: dict, concepts: list, as_of: date, unit: str = "USD") -> tuple[Optional[float], str]:
    """LTM flow = last fiscal year – prior-year YTD + current YTD, from filings dated <= as_of. A concept the
    company stopped tagging (GE's OperatingIncomeLoss ends in 2013) is skipped, not used stale."""
    for c in concepts:
        rows = _rows(facts, c, as_of, unit)
        annual = [r for r in rows if 350 <= _days(r) <= 380]
        if not annual:
            continue
        fy = max(annual, key=lambda r: (r["end_d"], r["filed"]))
        if (as_of - fy["end_d"]).days > MAX_AGE:
            continue
        ytd = [r for r in rows if r["end_d"] > fy["end_d"] and 80 <= _days(r) <= 290
               and abs((r["start_d"] - fy["end_d"]).days) <= 8]
        if not ytd:
            return fy["val"], f"FY ended {fy['end_d']:%b %Y}"
        cur = max(ytd, key=lambda r: (r["end_d"], _days(r)))
        prior = [r for r in rows if abs((r["end_d"] - (cur["end_d"] - timedelta(days=365))).days) <= 8
                 and abs(_days(r) - _days(cur)) <= 8]
        if not prior:
            return fy["val"], f"FY ended {fy['end_d']:%b %Y}"
        p = max(prior, key=lambda r: r["filed"])
        return fy["val"] - p["val"] + cur["val"], f"LTM to {cur['end_d']:%b %Y}"
    return None, ""


def share_change(facts: dict, as_of: date) -> Optional[float]:
    """Year-on-year change in the diluted share count: the latest quarter's weighted average against the same quarter
    a year earlier (a buyback shrinks it). Puts a peer's per-share growth on a net-income basis."""
    rows = [r for r in _rows(facts, "WeightedAverageNumberOfDilutedSharesOutstanding", as_of, "shares")
            if 80 <= _days(r) <= 100]
    if not rows:
        return None
    last = max(rows, key=lambda r: (r["end_d"], r["filed"]))
    prev = [r for r in rows if abs((last["end_d"] - r["end_d"]).days - 365) <= 20]
    if (as_of - last["end_d"]).days > MAX_AGE or not prev or not prev[0]["val"]:
        return None
    ch = last["val"] / max(prev, key=lambda r: r["filed"])["val"] - 1
    return ch if -0.15 <= ch <= 0.15 else None


def latest_instant(facts: dict, concepts: list, as_of: date) -> Optional[float]:
    for c in concepts:
        rows = [r for r in _rows(facts, c, as_of) if not r.get("start_d")]
        if rows:
            r = max(rows, key=lambda r: (r["end_d"], r["filed"]))
            if (as_of - r["end_d"]).days <= MAX_AGE:
                return r["val"]
    return None


def total_debt(facts: dict, as_of: date) -> float:
    lt = latest_instant(facts, DEBT_LT, as_of) or 0.0
    cur = latest_instant(facts, DEBT_CUR_TOTAL, as_of)
    if cur is None:
        cur = (latest_instant(facts, DEBT_CUR, as_of) or 0.0) + sum(latest_instant(facts, [c], as_of) or 0.0
                                                                   for c in DEBT_ST)
    return lt + cur


def peer_fundamentals(ticker: str, as_of: date) -> dict:
    """{'eps_ltm', 'eps_basis', 'ebitda_ltm', 'net_debt'} in $ (EPS) and $mm, as of the date."""
    cik, _ = sec.lookup_cik(ticker)
    f = company_facts(cik)
    eps, basis = ltm(f, EPS, as_of, "USD/shares")
    ebit, ebit_basis = ltm(f, OPINC, as_of)
    if ebit is None:  # no operating income tag: pre-tax income + interest expense
        pt, ebit_basis = ltm(f, PRETAX, as_of)
        ie, _ = ltm(f, INTEREST, as_of)
        ebit = pt + ie if pt is not None and ie is not None else None
    da, _ = ltm(f, DA, as_of)
    if da is None:
        dep, _ = ltm(f, ["Depreciation", "DepreciationNonproduction"], as_of)
        am, _ = ltm(f, ["AmortizationOfIntangibleAssets"], as_of)
        da = dep + (am or 0) if dep is not None else None
    cash = latest_instant(f, CASH, as_of) or 0.0
    sti = latest_instant(f, STI, as_of) or 0.0
    debt = total_debt(f, as_of)
    out = {"eps_ltm": eps, "eps_basis": basis, "net_debt": (debt - cash - sti) / 1e6, "debt": debt / 1e6}
    rev, _ = ltm(f, REV, as_of)
    if rev:
        out["revenue_ltm"] = rev / 1e6
    out["share_chg"] = share_change(f, as_of)
    if ebit is not None and da is not None:
        out["ebitda_ltm"] = (ebit + da) / 1e6
        out["ebitda_basis"] = ebit_basis
    # shares: 10-K/10-Q cover (dei), else the balance sheet count, else the latest diluted weighted average
    for concept, ns in (("EntityCommonStockSharesOutstanding", "dei"), ("CommonStockSharesOutstanding", "us-gaap")):
        sh = [r for r in _rows(f, concept, as_of, "shares", ns=ns) if not r.get("start_d")]
        if sh:  # several share classes on the same date are summed
            last = max(r["end_d"] for r in sh)
            latest = {}
            for r in sh:
                if r["end_d"] == last:
                    latest[r.get("frame") or r.get("accn") or len(latest)] = r["val"]
            out["shares"] = sum(latest.values()) / 1e6
            out["shares_date"] = last
            break
    if not out.get("shares"):
        wa = [r for r in _rows(f, "WeightedAverageNumberOfDilutedSharesOutstanding", as_of, "shares")
              if 80 <= _days(r) <= 100]
        if wa:
            r_ = max(wa, key=lambda r: (r["end_d"], r["filed"]))
            out["shares"], out["shares_date"] = r_["val"] / 1e6, r_["end_d"]
    return out


KPI_CONCEPTS = {"NumberOfStores": "Stores / warehouses (year-end)", "NumberOfRestaurants": "Restaurants (year-end)",
                "NumberOfRealEstateProperties": "Properties (year-end)"}


def kpi_history(cik: str, period_ends: list, as_of: date) -> dict:
    """Operating counts tagged in the company's own filings at each fiscal year end (filed by the date):
    {'Stores / warehouses (year-end)': [890, 914, 939]} — history for a volume x price KPI build."""
    try:
        f = company_facts(cik)
    except Exception:
        return {}
    out = {}
    for concept, label in KPI_CONCEPTS.items():
        rows = [r for r in _rows(f, concept, as_of, "pure") if not r.get("start_d")] or \
               [r for r in _rows(f, concept, as_of, "store") if not r.get("start_d")]
        vals = []
        for end in period_ends:
            m = [r for r in rows if abs((r["end_d"] - end).days) <= 7]
            vals.append(max(m, key=lambda r: r["filed"])["val"] if m else None)
        if sum(v is not None for v in vals) >= 2:
            out[label] = vals
    return out


def _annual_series(facts: dict, concepts: list, as_of: date, unit: str = "USD") -> dict:
    """{fiscal year end: value} for full-year facts filed by as_of. Concepts in priority order are merged by year
    (companies switch revenue tags over a decade); within a concept the latest filing wins (restatements)."""
    out = {}
    for c in concepts:
        best = {}
        for r in _rows(facts, c, as_of, unit):
            if 350 <= _days(r) <= 380:
                k = r["end_d"]
                if k not in best or r["filed"] >= best[k][1]:
                    best[k] = (r["val"], r["filed"])
        for k, (v, _) in best.items():
            if not any(abs((k - x).days) <= 8 for x in out):
                out[k] = v
    return out


def margin_history(cik: str, as_of: date, years: int = 10, interest_share: float = 1.0) -> list[tuple[int, float]]:
    """[(fiscal year, operating margin)] for up to `years` years: the 10-year range the terminal margin is tested
    against (operating income, else pre-tax income + interest expense for that year)."""
    f = company_facts(cik)
    rev = _annual_series(f, REV + ["SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense"], as_of)
    oi = _annual_series(f, OPINC, as_of)
    pt, ie = _annual_series(f, PRETAX, as_of), _annual_series(f, INTEREST, as_of)
    near = lambda d, end: next((d[k] for k in d if abs((k - end).days) <= 8), None)  # noqa: E731
    out = []
    for end in sorted(rev):
        r = rev[end]
        m = near(oi, end)
        if m is None and near(pt, end) is not None:
            m = near(pt, end) + (near(ie, end) or 0.0) * interest_share  # captive finance: its interest
            # is an operating cost in the model, so only the rest is added back
        if r and m is not None and r > 0:
            out.append((end.year, m / r))
    return out[-years:]


def ten_year_history(cik: str, as_of: date, years: int = 10) -> dict:
    """{'sales': [(fiscal year, revenue)], 'acq': [(fiscal year, cash paid for businesses)]}, up to `years` years
    filed by the date: how wrong a growth forecast can be (the scenario spread) and how much growth was bought
    (RONIC including goodwill for a serial acquirer)."""
    f = company_facts(cik)
    rev = _annual_series(f, REV + ["SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense"], as_of)
    acq = _annual_series(f, ["PaymentsToAcquireBusinessesNetOfCashAcquired", "PaymentsToAcquireBusinessesGross",
                             "PaymentsToAcquireBusinessesAndInterestInAffiliates"], as_of)
    sales = [(end.year, v) for end, v in sorted(rev.items()) if v and v > 0][-years:]
    yrs = {y for y, _ in sales}
    return {"sales": sales, "acq": [(end.year, v) for end, v in sorted(acq.items()) if end.year in yrs]}


def eps_history(cik: str, as_of: date, years: int = 10) -> list[tuple[date, float]]:
    """[(fiscal year end, diluted EPS)] for up to `years` years, filed by as_of."""
    f = company_facts(cik)
    s = _annual_series(f, EPS, as_of, "USD/shares")
    return sorted(s.items())[-years:]
