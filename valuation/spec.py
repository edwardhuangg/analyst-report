"""ModelSpec: everything the workbook needs, with defaults derived from the filings.

`build_spec(data)` turns fetched company data into a complete, editable spec.
The Streamlit app edits the spec (drivers, valuation inputs, peers, thesis) and
then hands it to `workbook.build_workbook`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from . import capital, industry
from .financials import History, Interim


def _nz(v):
    return 0.0 if v is None else v


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _r(v, step):
    return round(round(v / step) * step, 6)


def _add_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


@dataclass
class LineSpec:
    """A product / region line inside a segment: its own organic growth paths (scenario blocks)."""
    name: str
    sales: list                                  # 3 years of history ($mm)
    growth: dict = field(default_factory=dict)   # bear/base/bull -> 6 values
    note: str = ""
    kpi: Optional[dict] = None                   # volume x price build (KPIs tab), see overrides SCHEMA_DOC


@dataclass
class SegSpec:
    name: str
    sales: list
    profit: list
    organic_last: Optional[float] = None
    growth: dict = field(default_factory=dict)   # bear/base/bull -> 6 values
    margin: dict = field(default_factory=dict)
    mna: list = field(default_factory=lambda: [0.0] * 6)
    mna_note: str = ""
    sotp: tuple = (10.0, 12.0, 14.0)
    sotp_note: str = ""
    lines: list = field(default_factory=list)    # LineSpec: revenue split the forecast is built from
    use_lines: bool = False                      # True: segment sales = sum of its lines
    lines_explicit: set = field(default_factory=set)  # cases whose line paths the analyst set (not re-aligned)
    kpi: Optional[dict] = None                   # volume x price build for a segment without lines

    @property
    def units(self) -> list:
        """What carries the growth assumptions: the lines when they drive the segment, else the segment."""
        return self.lines if self.use_lines and self.lines else [self]


@dataclass
class ModelSpec:
    # identity & dates
    name: str
    short: str
    ticker: str
    exchange: str
    as_of: date
    fy_hist: list
    period_ends: list
    fy_end_cur: date
    exit_date: date
    hist: History
    interim: Interim
    segments: list
    segments_reported: bool
    # market
    price: float = 0.0
    price_note: str = ""
    basic_shares: float = 0.0
    basic_note: str = ""
    dilutive: float = 0.0
    dilutive_note: str = ""
    scenario: int = 2
    # drivers & valuation
    drivers: dict = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    corp_sales: list = field(default_factory=list)
    corp_cost_pct: list = field(default_factory=list)
    corp_sales_note: str = ""
    corp_cost_note: str = ""
    corp_multiples: tuple = (12.0, 12.0, 12.0)
    peers: list = field(default_factory=list)
    debt_like: dict = field(default_factory=dict)
    guidance: dict = field(default_factory=dict)
    guidance_date: str = ""
    adj_eps_hist: list = field(default_factory=lambda: [None, None, None])
    adj_eps_reported: bool = False
    organic_total_last: Optional[float] = None
    q_organic: dict = field(default_factory=dict)
    q_organic_total: Optional[float] = None
    thesis: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    filing_urls: dict = field(default_factory=dict)
    sic: str = ""
    eps_est: tuple = (None, "")
    # 1 when reported segment profit is already after amortization & stock comp (segment
    # profit + corporate = GAAP operating income, e.g. MSFT, AAPL, KO); 0 when the segments
    # exclude them (HON-style). Drives Assumptions!C14.
    seg_after_charges: int = 0
    seg_after_charges_note: str = ""
    adj_eps_ex_sbc: int = 0   # 1 when the company's adjusted EPS also excludes stock comp (Assumptions!C15)
    adj_eps_ex_repo: int = 0  # 1 when it also excludes non-segment charges: restructuring, deal costs (Assumptions!C5)
    repo_label: str = "   Less: repositioning & other non-segment items"
    # text the analyst layer can replace (Assumptions column L block notes, Comps / SOTP labels)
    notes: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    sotp_splits: dict = field(default_factory=dict)   # segment name -> [{name, ebitda|remainder, multiples, note}]
    consensus: dict = field(default_factory=dict)     # FY-matched Yahoo consensus (recent valuation dates only)
    provenance: list = field(default_factory=list)    # who set what: defaults / AI analyst / external agent
    last_review: dict = field(default_factory=dict)   # check.review() of the final AI-analyst build, if run
    ai_cost: str = ""                                 # "~$0.31 for 5 AI calls (...)": this build's OpenAI spend
    ai_usage: list = field(default_factory=list)      # per-call token usage
    profile: dict = field(default_factory=dict)       # industry profile (valuation/industry.py)
    market_rf: Optional[float] = None                 # 10-yr yield on the date from market data (a fact, not a view)
    extracted: dict = field(default_factory=dict)     # figures read from the releases (extraction step): reported
                                                      # numbers an analyst may not overwrite without a correction
    captive: dict = field(default_factory=dict)       # captive finance arm, if any (see financials.captive_finance)
    # "economic" (default): value-driver terminal value, cash flows from the balance-sheet date, acquisitions paid
    # out of value, stakes in every bridge, derived 3-year exit multiple. "reference" reproduces the mechanics of
    # the hand-built HON_Valuation_Model_2025-08-29.xlsx (regression test only).
    framework: str = "economic"
    adj_eps_ex_nonrec: float = 1.0    # 1: company adjusted EPS excludes one-off gains (0: CRM includes investment gains)
    adj_eps_amort_share: float = 1.0  # share of acquired-intangible amortization the company excludes (SHW: Valspar only)
    # the company's non-GAAP tax rate by history year when it taxes non-GAAP pre-tax income at one rate (CRM); the
    # forecast rate is drivers["adj_tax"]; None / 0 = the adjustments are taxed at the year's effective rate
    adj_eps_tax_hist: list = field(default_factory=lambda: [None, None, None])
    reported_eps: dict = field(default_factory=dict)  # adjusted EPS read from the release's reconciliation line
    post_bs_events: list = field(default_factory=list)  # 8-Ks filed between the balance sheet and the valuation date
    margin_basis: str = ""      # the margin guide's basis: "gaap", "adjusted" or "segment" ("" = matched)
    repo_structural: bool = False  # non-segment charges are a structural cost (segment profit above R&D / opex)
    asr_open: bool = False      # an accelerated share repurchase still to settle after the 10-Q share count
    audit_impact: list = field(default_factory=list)  # (group, keys, $/share): fair-value effect of each override group
    capital: dict = field(default_factory=dict)       # method beta, rating, ERP source (valuation/capital.py)
    val_source: dict = field(default_factory=dict)    # valuation key -> who set it last
    cf_share_bs: float = 1.0                          # share of FY1 cash flow after the balance-sheet date (seasonal)
    cf_share_note: str = ""
    ytd_capital: dict = field(default_factory=dict)   # year-to-date buybacks, acquisitions, debt flows (10-Q)
    deals: list = field(default_factory=list)         # pending / post-balance-sheet acquisitions and separations
    scenario_probs: Optional[list] = None             # bear / base / bull probabilities (default 25 / 50 / 25)
    margin_hist: list = field(default_factory=list)   # [(fiscal year, operating margin)] up to 10 years (companyfacts)
    hist10: dict = field(default_factory=dict)        # {'sales': [(year, revenue)], 'acq': [(year, paid)]} (companyfacts)
    own_actions: list = field(default_factory=list)   # [(date, Yahoo factor)] the company's splits / spin-offs after the 10-K
    pe_hist: list = field(default_factory=list)       # [(fiscal year, trailing P/E at the fiscal year end)]
    scenario_values: dict = field(default_factory=dict)  # bear / base / bull outputs (recalculated at build)
    rating: dict = field(default_factory=dict)        # rule-based rating (pipeline.rate)
    calibrated: dict = field(default_factory=dict)    # guidance metric -> how the engine calibrated it
    audit: list = field(default_factory=list)         # (source, key, before, after) for every assumption change
    audit_fv: list = field(default_factory=list)      # (stage, fair value) after each pass
    version_bridge: Optional[dict] = None             # bridge.version_bridge: the new version vs the previous one
    verification: list = field(default_factory=list)  # release figures checked against the source text
    rejected: dict = field(default_factory=dict)      # field -> (source, value, reason): analyst inputs a guard refused
    stated_adj_tax: list = field(default_factory=list)  # non-GAAP tax rates the releases state (history is locked)
    q_guide: list = field(default_factory=list)       # next quarter's non-GAAP EPS target [low, high] (Q3 releases)
    q_eps: list = field(default_factory=list)         # the year's reported quarterly adjusted EPS, oldest first
    short_interest: dict = field(default_factory=dict)  # Yahoo short % of float, days to cover (a short's crowding)

    def label(self, key: str, default: str) -> str:
        return self.labels.get(key) or default

    @property
    def growth_units(self) -> list:
        """(segment index, unit) for every row of the growth scenario blocks, in segment order."""
        return [(i, u) for i, sg in enumerate(self.segments) for u in sg.units]

    @property
    def kpi_units(self) -> list:
        return [(i, u) for i, u in self.growth_units if u.kpi]

    @property
    def has_nonrec(self) -> bool:
        """Show the non-recurring line (IS) and driver (Assumptions) when one-offs are material."""
        s = abs(_nz(self.hist.IS["sales"][2])) or 1.0
        hist = any(v is not None and abs(v) >= 0.005 * s for v in self.hist.IS.get("nonrec", []))
        return hist or any(abs(v) > 0 for v in self.drivers.get("nonrec", ([0], ""))[0])

    @property
    def nonrec_hist(self) -> list:
        """Material one-offs per history year (immaterial ones stay in pension & other)."""
        s = abs(_nz(self.hist.IS["sales"][2])) or 1.0
        return [v if v is not None and abs(v) >= 0.005 * s else 0.0 for v in self.hist.IS.get("nonrec", [None] * 3)]

    @property
    def stance_phrase(self) -> str:
        return stance_phrase(self.thesis.get("stance"))

    @property
    def data_notes(self) -> list:
        """Flags as of the final model: the pipeline's notes, what the revenue build uses, then the analyst's."""
        own = [f for f in self.flags if not f.startswith("Analyst (")]
        n = sum(len(sg.lines) for sg in self.segments if sg.use_lines and sg.lines)
        if n:
            own.append(f"Revenue built from {n} {self.hist.lines_axis or 'product'} lines tagged in the 10-K "
                       "(disaggregated revenue): each line has its own growth paths on Assumptions")
        kp = len(self.kpi_units)
        if kp:
            own.append(f"{kp} line(s) built as volume x price on the KPIs tab")
        return own + [f for f in self.flags if f.startswith("Analyst (")]

    def set_single_segment_basis(self, after: int):
        """Single-segment model: segment profit = EBIT when amortization and stock comp count as already inside it
        (Assumptions!C14 = 1), else EBIT + amortization + stock comp. History, the Interim columns and the note
        are restated on the same basis, so the history margin compares with the forecast's."""
        if self.segments_reported or not self.segments:
            return
        I, it = self.hist.IS, self.interim
        from .financials import _other_opex
        sg = self.segments[0]
        sg.profit = [None if I["opinc"][j] is None else
                     I["opinc"][j] + _other_opex({k: I[k][j] for k in I}) +
                     (0 if after else _nz(I["amort"][j]) + _nz(I["sbc"][j])) for j in range(3)]
        for d in (it.ytd_cur, it.ytd_prior):
            if d.get("opinc") is not None:
                d["seg_profit"] = d["opinc"] + _other_opex(d) + (0 if after else _nz(d.get("amort")) + _nz(d.get("sbc")))
        for r in it.q_segments:  # the quarter row of the Interim segment table
            for tag in ("cur", "prior"):
                if r.get("ebit_" + tag) is not None:
                    r["profit_" + tag] = r["ebit_" + tag] if after else r.get("full_" + tag)
        self.seg_after_charges_note = (
            "Single segment = GAAP operating income before other operating expense (IPR&D, impairments, "
            "restructuring): amortization and stock comp are inside it" if after else
            "Single segment = EBIT + amortization + stock comp + other operating expense, so all are deducted")

    def note(self, key: str, default: str) -> str:
        return self.notes.get(key) or default

    # ---------------------------------------------------------- derived labels
    @property
    def fy_fc(self):
        return [self.fy_hist[-1] + i for i in range(1, 7)]

    @property
    def hist_labels(self):
        return [f"{y}A" for y in self.fy_hist]

    @property
    def fc_labels(self):
        return [f"{y}E" for y in self.fy_fc]

    @property
    def hcols(self):
        return [c for c, v in zip("CDE", self.hist.IS["sales"]) if v is not None]

    @property
    def fye_text(self):
        return f"{self.period_ends[-1]:%b} {self.period_ends[-1].day}"

    @property
    def has_disc(self):
        return any(v is not None and abs(v) >= 0.5 for v in self.hist.IS.get("disc", []))

    @property
    def stub_start_text(self):
        d = self.as_of + timedelta(days=1)
        return f"{d:%b %Y}"

    @property
    def stub_text(self):
        start = self.as_of + timedelta(days=1)
        if start > self.fy_end_cur:
            return "(none: year already ended)"
        if start.year == self.fy_end_cur.year:
            return f"{start:%b}–{self.fy_end_cur:%b %Y}"
        return f"{start:%b %Y}–{self.fy_end_cur:%b %Y}"

    @property
    def q_label(self):
        return self.interim.quarter_label if self.interim.available else f"FY{self.fy_hist[-1]}"

    @property
    def built_from(self):
        s = f"Built from the FY{self.fy_hist[-1]} 10-K"
        return s + (f" and {self.interim.quarter_label} 10-Q" if self.interim.available else "")

    @property
    def interim_title(self):
        it = self.interim
        if it.available:
            return f"Latest reported period: {it.quarter_label} (10-Q, quarter ended {it.period_end:%B %-d, %Y})"
        return (f"Latest reported period: FY{self.fy_hist[-1]} (10-K, year ended {self.period_ends[-1]:%B %-d, %Y}); "
                "no 10-Q filed since")

    @property
    def interim_sub(self):
        it = self.interim
        if it.available:
            return (f"What a buy-side analyst had on {self.as_of:%b %-d, %Y}. "
                    f"LTM = FY{self.fy_hist[-1]} – {it.label_prior} + {it.label_cur}.")
        return f"What a buy-side analyst had on {self.as_of:%b %-d, %Y}. LTM = FY{self.fy_hist[-1]}."

    @property
    def interim_tab_text(self):
        it = self.interim
        if it.available:
            return (f"{it.quarter_label} 10-Q: {it.label_cur.split()[0]} results, LTM, "
                    f"{it.period_end:%B %-d} net debt, FY{self.fy_fc[0]} guidance vs model")
        return f"FY{self.fy_hist[-1]} results, LTM, year-end net debt, FY{self.fy_fc[0]} guidance vs model"

    @property
    def interim_source(self):
        fy = self.fy_hist[-1]
        if self.interim.available:
            return f"{self.interim.quarter_label} 10-Q ({self.interim.label_cur.split()[0]}); FY{fy} 10-K"
        return f"FY{fy} 10-K"

    @property
    def interim_bs_source(self):
        fy = self.fy_hist[-1]
        if self.interim.available:
            return f"{self.interim.quarter_label} 10-Q balance sheet; FY{fy} 10-K"
        return f"FY{fy} 10-K balance sheet"

    @property
    def as_of_note(self):
        return "Valuation (pitch) date"

    @property
    def is_source(self):
        return f"10-K FY{self.fy_hist[-1]} Consolidated Statement of Operations"

    @property
    def bs_source(self):
        return f"10-K FY{self.fy_hist[-1]} Consolidated Balance Sheet"

    @property
    def cf_source(self):
        return f"10-K FY{self.fy_hist[-1]} Consolidated Statement of Cash Flows"

    @property
    def seg_source(self):
        return (f"10-K FY{self.fy_hist[-1]} segment note (XBRL)" if self.segments_reported
                else "10-K: operating income + intangible amortization + stock comp + other operating expense "
                     "(no segment data)")

    @property
    def organic_note(self):
        if any(sg.organic_last is not None for sg in self.segments):
            return f"{self.fy_hist[-1]}: earnings release organic reconciliation; forecast: Assumptions (selected case)"
        return (f"{self.fy_hist[-1]}: organic not in XBRL, set to reported growth (enter from the release); "
                "forecast: Assumptions (selected case)")

    @property
    def organic_total_note(self):
        if self.organic_total_last is not None:
            return f"{self.fy_hist[-1]}: {self.organic_total_last:.0%} reported organic (earnings release)"
        return f"{self.fy_hist[-1]}: reported growth (organic not disclosed in XBRL)"

    @property
    def growth_band_text(self):
        g = self.guidance.get("organic")
        base = "Organic sales growth by segment (scenario blocks"
        if g:
            rng = f"{g[0]:.0%}" if round(g[0], 3) == round(g[1], 3) else f"{g[0]:.0%}–{g[1]:.0%}"
            return base + f"; {self.fc_labels[0]} anchored to guidance of {rng} organic)"
        return base + ")"

    @property
    def margin_band_text(self):
        g = self.guidance.get("margin")
        base = "Segment margin (segment profit / segment sales)"
        if g:
            rng = f"{g[0]:.1%}" if round(g[0], 4) == round(g[1], 4) else f"{g[0]:.1%}–{g[1]:.1%}"
            return base + f"; {self.fc_labels[0]} anchored to the {rng} total guide"
        return base

    @property
    def growth_note(self):
        if self.notes.get("growth"):
            return self.notes["growth"]
        hist = "; ".join(f"{sg.name}: {sg.sales[2] / sg.sales[1] - 1:+.1%}" for sg in self.segments
                         if sg.sales[1] and sg.sales[2])
        anchor = self.notes.get("growth_anchor", "")
        return (f"Default = recent growth{anchor} fading toward a long-run rate by {self.fc_labels[-1]}; "
                f"bear/bull ∓/± 2pt. FY{self.fy_hist[-1]} reported: {hist}")

    @property
    def margin_note(self):
        if self.notes.get("margin"):
            return self.notes["margin"]
        hist = "; ".join(f"{sg.name} {sg.profit[2] / sg.sales[2]:.1%}" for sg in self.segments
                         if sg.profit[2] is not None and sg.sales[2])
        return f"FY{self.fy_hist[-1]} actual: {hist}" + self.notes.get("margin_anchor", "")

    @property
    def corp_sales_hist(self):
        return self.hist.corp_sales

    @property
    def corp_profit_hist(self):
        return self.hist.corp_profit

    @property
    def corp_profit_note(self):
        if not self.segments_reported:
            return "Single-segment model: corporate costs sit inside segment profit"
        if all(v is None for v in self.hist.corp_profit):
            return "Corporate costs not tagged in XBRL: the gap to GAAP falls into 'repositioning & other' on the IS"
        return "10-K segment note: corporate & reconciling items"

    @property
    def seg_total_note(self):
        vals = [f"{sum(_nz(sg.profit[j]) for sg in self.segments) + _nz(self.hist.corp_profit[j]):,.0f}"
                for j in range(3) if all(sg.profit[j] is not None for sg in self.segments)]
        return "Segment profit + corporate: " + " / ".join(vals)

    @property
    def amort_note(self):
        return "10-K: amortization of intangible assets (XBRL AmortizationOfIntangibleAssets)"

    @property
    def interest_note(self):
        if not any(self.hist.IS["interest_exp"]):
            return "Interest expense not tagged separately (included in other income, net)"
        return self.is_source

    @property
    def shares_note(self):
        return "10-K diluted weighted average shares" + (" (split-adjusted)" if any("split" in f for f in self.flags) else "")

    @property
    def adj_eps_note(self):
        return ("Company-reported adjusted EPS (earnings releases)" if self.adj_eps_reported else
                "Estimated: GAAP EPS + after-tax amortization of acquired intangibles (company figure not extracted)")

    @property
    def adj_eps_hist_note(self):
        return ("History: company-reported adjusted EPS (earnings release); forecast: GAAP EPS + after-tax amortization"
                if self.adj_eps_reported else
                "GAAP EPS + after-tax amortization of acquired intangibles (history and forecast)")

    @property
    def is_hist(self):
        I = self.hist.IS
        d = {k: list(I[k]) for k in ("sales", "cogs", "rnd", "sga")}
        d["oth_opex"] = [None if I["sales"][j] is None or I["opinc"][j] is None else
                         I["sales"][j] - _nz(I["cogs"][j]) - _nz(I["rnd"][j]) - _nz(I["sga"][j]) - I["opinc"][j]
                         for j in range(3)]
        d["other_inc"] = [None if I["pretax"][j] is None or I["opinc"][j] is None else
                          I["pretax"][j] - I["opinc"][j] + _nz(I["interest_exp"][j]) for j in range(3)]
        return d

    @property
    def cf_hist(self):
        return self.hist.CF

    @property
    def memo_cogs_formula(self):
        out = []
        for j in range(3):
            if _nz(self.hist.IS["cogs"][j]) > 0:
                out.append("={c}{cogs}")
            else:  # no cost-of-sales line: use total operating costs as the base for inventory/payable days
                out.append("={c}{sales}-{c}{ebit}")
        return out

    @property
    def ending_shares_hist(self):
        """Shares at the last fiscal year end: the 10-K cover, unless it counts shares that are not outstanding
        for EPS (employee benefit trust), then the year's weighted basic count."""
        cover, wtd = self.hist.cover_shares, self.hist.IS["basic_shares"][2]
        return wtd if cover and wtd and cover > 1.025 * wtd else cover

    @property
    def ending_shares_note(self):
        if self.ending_shares_hist != self.hist.cover_shares:
            return f"{self.fy_hist[-1]}: weighted basic shares (the 10-K cover also counts shares not outstanding for EPS)"
        return f"{self.fy_hist[-1]}: shares outstanding on the 10-K cover"

    @property
    def fy_interim(self):
        I, j = self.hist.IS, 2
        cf = self.hist.CF
        seg_total = (sum(_nz(sg.profit[j]) for sg in self.segments) + _nz(self.hist.corp_profit[j])
                     if all(sg.profit[j] is not None for sg in self.segments) else None)
        return {"sales": I["sales"][j], "cogs": I["cogs"][j], "rnd": I["rnd"][j], "sga": I["sga"][j],
                "oth_opex": self.is_hist["oth_opex"][j], "other_inc": self.is_hist["other_inc"][j],
                "interest_exp": I["interest_exp"][j], "tax": I["tax"][j], "nci": I["nci"][j],
                "eps": I.get("eps_cont", I["eps_dil"])[j] or I["eps_dil"][j],
                "adj_eps": self.adj_eps_hist[j] if self.adj_eps_hist[j] is not None else self.adj_eps_estimate(j),
                "seg_profit": seg_total, "da": cf["da"][j], "cfo": cf["cfo"][j],
                "capex": -cf["capex"][j] if cf["capex"][j] is not None else None}

    def adj_eps_estimate(self, j: int):
        I = self.hist.IS
        if not I["dil_shares"][j] or I["ni_attr"][j] is None:
            return None
        etr = I["tax"][j] / I["pretax"][j] if I["pretax"][j] and I["tax"][j] is not None and I["pretax"][j] > 0 else 0.21
        addback = _nz(I["amort"][j]) + _nz(I["sbc"][j]) * self.adj_eps_ex_sbc - self.nonrec_hist[j]
        return round((I["ni_attr"][j] + addback * (1 - etr)) / I["dil_shares"][j], 2)

    @property
    def q_seg_title(self):
        return f"{self.q_label} segment results"

    @property
    def q_seg_headers(self):
        if self.interim.available:
            pe = self.interim.period_end
            cur, pri = f"{self.interim.fiscal_period}-{pe:%y}", f"{self.interim.fiscal_period}-{(pe.year - 1) % 100:02d}"
        else:
            cur, pri = f"FY{self.fy_hist[-1]}", f"FY{self.fy_hist[-2]}"
        return [f"{cur} sales", f"{pri} sales", "Organic", f"{cur} profit", f"{cur} margin", f"{pri} margin"]

    @property
    def q_segments(self):
        rows = []
        if self.interim.available and self.interim.q_segments:
            for r in self.interim.q_segments:
                mp = (r["profit_prior"] / r["sales_prior"]) if r.get("profit_prior") is not None and r.get("sales_prior") else None
                rows.append({**r, "organic": self.q_organic.get(r["name"]), "margin_prior": mp})
        elif not self.interim.available:
            for sg in self.segments:
                mp = sg.profit[1] / sg.sales[1] if sg.profit[1] is not None and sg.sales[1] else None
                rows.append({"name": sg.name, "sales_cur": sg.sales[2], "sales_prior": sg.sales[1],
                             "profit_cur": sg.profit[2], "organic": sg.organic_last, "margin_prior": mp})
        return rows

    @property
    def q_corp(self):
        if self.interim.available:
            return self.interim.q_corp or {}
        return {"sales_cur": self.hist.corp_sales[2], "sales_prior": self.hist.corp_sales[1],
                "profit_cur": self.hist.corp_profit[2]}

    @property
    def q_total(self):
        rows = self.q_segments
        sp = sum(_nz(r.get("sales_prior")) for r in rows) + _nz(self.q_corp.get("sales_prior"))
        pp = sum(_nz(r.get("profit_prior") if self.interim.available else
                     (self.segments[i].profit[1] if i < len(self.segments) else 0)) for i, r in enumerate(rows))
        pp += _nz(self.q_corp.get("profit_prior") if self.interim.available else self.hist.corp_profit[1])
        organic = self.q_organic_total if self.interim.available else self.organic_total_last
        return {"organic": organic, "margin_prior": pp / sp if sp else None}

    @property
    def guidance_title(self):
        when = f" ({self.guidance_date})" if self.guidance_date else ""
        return f"FY{self.fy_fc[0]} guidance{when} vs this model"

    @property
    def guidance_note(self):
        if self.guidance:
            return "Guidance ranges extracted from the latest earnings release (AI-assisted); verify against the release."
        return "Guidance not extracted: type the company's ranges into the blue cells (C:D) from the latest release."

    @property
    def hist_cols_available(self):
        return self.hcols


# ===================================================================== defaults
SPLIT_RATIOS = [1 / 20, 1 / 10, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 2, 2, 3, 4, 5, 10, 20]

LEGAL = r"\b(incorporated|inc|corporation|corp|company|co|ltd|limited|plc|holdings?|group|international|the|n\.?v|s\.?a|ag|se)\b\.?"


def short_name(name: str) -> str:
    s = re.sub(r"\.com\b", "", name, flags=re.I)
    s = re.sub(LEGAL, "", s, flags=re.I)
    s = re.sub(r"[,.]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(\s+(&|and|-))+$", "", s, flags=re.I).strip()  # "Deere & Company" -> "Deere", not "Deere &"
    return s or name


def stance_phrase(stance) -> str:
    """One wording for the thesis title / Cover: long, short, underweight or neutral on."""
    st = str(stance or "").strip().lower()
    if re.search(r"\b(short|sell)\b", st):
        return "short"
    if re.search(r"\bunderweight\b", st):
        return "underweight"
    if re.search(r"\b(long|buy|overweight)\b", st) and not re.search(r"\b(neutral|hold)\b", st):
        return "long"
    return "neutral on"


def display_name(sec_name: str, yahoo_name: Optional[str]) -> str:
    if yahoo_name:
        return yahoo_name
    return " ".join(w.capitalize() if w.isupper() and len(w) > 3 else w for w in sec_name.split()).replace(" Inc", " Inc.")


def build_spec(data) -> ModelSpec:
    """data: pipeline.CompanyData"""
    h: History = data.hist
    it: Interim = data.interim
    I, B, CF = h.IS, h.BS, h.CF
    sales_last = I["sales"][2]
    mkt = data.market
    rel = data.release or {}

    name = display_name(data.company.name, mkt.get("name"))
    fy_end_cur = _add_years(h.period_ends[-1], 1)
    segs = []
    for sg in h.segments:
        segs.append(SegSpec(name=sg.name, sales=list(sg.sales), profit=list(sg.profit),
                            lines=[LineSpec(name=n, sales=list(v)) for n, v in sg.lines]))
    short = short_name(name)
    spec = ModelSpec(name=name, short=short.split(" ")[0] if len(short.split()) > 3 else short,
                     ticker=data.company.ticker,
                     exchange=(data.company.exchange or "").upper().replace("NASDAQ", "NASDAQ") or "US",
                     as_of=data.as_of, fy_hist=list(h.years), period_ends=list(h.period_ends),
                     fy_end_cur=fy_end_cur, exit_date=_add_years(fy_end_cur, 3), hist=h, interim=it,
                     segments=segs, segments_reported=h.segments_reported, sic=data.company.sic)
    spec.flags = list(h.flags) + list(data.flags)
    spec.filing_urls = data.urls
    spec.captive = dict(h.captive)
    spec.profile = industry.profile(industry.detect(data.company.sic, mkt.get("industry"), mkt.get("sector"),
                                                    segs, h.captive))
    for sg in segs:
        sg.use_lines = bool(sg.lines) and (not h.segments_reported or spec.profile["lines_multi"])
    spec.adj_eps_ex_sbc = spec.profile["adj_eps_ex_sbc"]

    # ---------------------------------------------------------------- market
    spec.price = round(mkt.get("price") or 0.0, 2)
    spec.price_note = mkt.get("price_note", "")
    cover = it.cover_shares if it.available and it.cover_shares else h.cover_shares
    src_doc = f"{it.quarter_label} 10-Q cover" if it.available and it.cover_shares else f"FY{h.years[-1]} 10-K cover"
    wtd = it.q_basic if it.available and it.q_basic else I["basic_shares"][2]
    if cover and wtd and cover > 1.025 * wtd:
        # LLY: ~50mm shares in an employee benefit trust are on the cover but not outstanding for EPS
        spec.flags.append(f"{src_doc} shows {cover:,.1f}mm shares, {cover / wtd - 1:.1%} more than the {wtd:,.1f}mm "
                          "weighted basic shares used for EPS (shares held in an employee benefit trust or otherwise "
                          "not outstanding for EPS): the model uses the EPS count — check for a share issuance after "
                          "the quarter")
        cover, src_doc = wtd, f"{it.quarter_label if it.available and it.q_basic else f'FY{h.years[-1]}'} weighted basic shares"
    if cover is not None and cover < 1 and it.cover_shares and it.cover_shares > 1:
        cover, src_doc = it.cover_shares, f"{it.quarter_label} 10-Q cover"  # never a share count below 1mm
    spec.basic_shares = round(cover or (I["basic_shares"][2] or I["dil_shares"][2] or 0), 1)
    spec.basic_note = f"{src_doc}: {spec.basic_shares:,.1f}mm shares outstanding"
    if it.available and it.q_diluted and it.q_basic:
        spec.dilutive = round(max(0.0, it.q_diluted - it.q_basic), 1)
        spec.dilutive_note = (f"{it.quarter_label} diluted {it.q_diluted:,.1f}mm less basic {it.q_basic:,.1f}mm (10-Q)")
    else:
        dil = max(0.0, _nz(I["dil_shares"][2]) - _nz(I["basic_shares"][2]))
        spec.dilutive = round(dil, 1)
        spec.dilutive_note = f"FY{h.years[-1]} diluted less basic weighted shares (10-K)"

    # ---------------------------------------------------------- release data
    adj = rel.get("adjusted_eps") or {}
    bad = adj_eps_problem(spec, adj, guide=(rel.get("guidance") or {}).get("adj_eps")) if adj else ""
    if bad:
        spec.flags.append(f"Adjusted EPS read from the releases ({eps_txt(adj)}) looks wrong ({bad}): the model's "
                          "definition is used")
        adj = {}
    if adj.get("fy"):
        spec.adj_eps_hist = [None, adj.get("fy_prior"), adj.get("fy")]
        spec.adj_eps_reported = True
    org_fy = rel.get("organic_fy_by_segment") or {}
    org_q = rel.get("organic_q_by_segment") or {}
    for sg in spec.segments:
        sg.organic_last = _match(org_fy, sg.name)
        v = _match(org_q, sg.name)
        if v is not None:
            spec.q_organic[sg.name] = v
    spec.organic_total_last = rel.get("organic_fy_total")
    spec.q_organic_total = rel.get("organic_q_total")
    spec.guidance = {}
    for k, v in (rel.get("guidance") or {}).items():
        if v and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            bad = guidance_problem(spec, k, *v)
            if bad:
                spec.flags.append(f"Guidance {k} {guide_txt(k, v)} read from the release looks wrong ({bad}): not used")
            else:
                spec.guidance[k] = tuple(v)
    spec.guidance_date = rel.get("release_date", "")

    # ------------------------------------------------- interim derived lines
    def complete(d: dict):
        if not d:
            return
        if d.get("sales") is not None and d.get("opinc") is not None:
            d["oth_opex"] = d["sales"] - _nz(d.get("cogs")) - _nz(d.get("rnd")) - _nz(d.get("sga")) - d["opinc"]
        if d.get("pretax") is not None and d.get("opinc") is not None:
            d["other_inc"] = d["pretax"] - d["opinc"] + _nz(d.get("interest_exp"))
        if d.get("capex") is not None:
            d["capex"] = abs(d["capex"])
    complete(it.ytd_cur)
    complete(it.ytd_prior)
    etr_last = (I["tax"][2] / I["pretax"][2]) if I["pretax"][2] and I["tax"][2] is not None and I["pretax"][2] > 0 else 0.21
    if it.available and (adj.get("ytd_cur") is not None or adj.get("ytd_prior") is not None):
        bad = ytd_adj_problem(spec, adj.get("ytd_cur"), adj.get("ytd_prior"), adj.get("fy"))
        if bad:
            spec.flags.append(f"Year-to-date adjusted EPS read from the release looks wrong ({bad}): estimated instead")
            adj = {k: v for k, v in adj.items() if not k.startswith("ytd_")}
    spec.extracted = {"adjusted_eps": {k: adj.get(k) for k in ("fy", "fy_prior", "ytd_cur", "ytd_prior")},
                      "guidance": dict(spec.guidance)}
    for tag, d in (("cur", it.ytd_cur), ("prior", it.ytd_prior)):
        if not it.available:
            d["adj_eps"] = None
            continue
        v = adj.get("ytd_" + tag)
        if v is None and d.get("eps") and d.get("ni_attr"):
            nr = _nz(d.get("nonrec")) if abs(_nz(d.get("nonrec"))) >= 0.005 * _nz(d.get("sales")) else 0.0
            addback = _nz(d.get("amort")) + _nz(d.get("sbc")) * spec.adj_eps_ex_sbc - nr
            v = (d["ni_attr"] + addback * (1 - etr_last)) / (d["ni_attr"] / d["eps"])
            d["adj_eps_est"] = (etr_last, d["ni_attr"] / d["eps"])  # (tax rate, shares) used for the estimate
            d["adj_eps_oneoff"] = nr
        d["adj_eps"] = round(v, 2) if v is not None else None
        d["adj_eps_model"] = d["adj_eps"]  # restored if an analyst resets adjusted EPS to the model definition
    if not spec.adj_eps_reported:
        spec.adj_eps_hist = [None, None, None]  # workbook writes the formula

    # ---------------------------------------------------------------- drivers
    spec.consensus = mkt.get("consensus") or {}
    verify_release_figures(spec, data)
    post_balance_sheet_events(spec, data)
    spec.margin_hist = list(getattr(data, "margin_hist", []) or [])
    spec.pe_hist = list(getattr(data, "pe_hist", []) or [])
    spec.hist10 = dict(getattr(data, "hist10", {}) or {})
    spec.own_actions = list(getattr(data, "own_actions", []) or [])
    set_balance_sheet_timing(spec)
    detect_segment_measure(spec)
    defaults_drivers(spec, data)
    defaults_valuation(spec, data)
    defaults_peers_sotp(spec, data)
    spec.debt_like = default_debt_like(spec)
    spec.thesis = data.thesis or template_thesis(spec)
    spec.sources = sources(spec, data)
    spec.provenance = ["Defaults: derived from the filings and market data (rules in valuation/spec.py)"]
    return spec


def guidance_problem(spec: ModelSpec, key: str, lo: float, hi: float) -> str:
    """Why a guidance range cannot be right for this company (empty, reversed, or far from last year)."""
    I = spec.hist.IS
    if lo > hi:
        return "low above high"
    mid = (lo + hi) / 2
    if key == "sales":
        last = I["sales"][2]
        return "" if last and 0.6 * last <= mid <= 2.0 * last else f"far from last year's {last:,.0f}mm"
    if key == "fcf":
        cf = spec.hist.CF
        last = _nz(cf["cfo"][2]) + _nz(cf["capex"][2])
        if hi <= 0 < last:
            return "zero or negative while the company generates cash"
        return "" if last <= 0 or 0.25 * last <= mid <= 3 * last else f"far from last year's {last:,.0f}mm"
    if key == "adj_eps":
        last = spec.adj_eps_hist[2] or I["eps_dil"][2]
        return "" if mid > 0 and (not last or last <= 0 or 0.4 * last <= mid <= 3 * last) else \
            f"far from last year's {last}"
    if key == "organic":
        return "" if -0.3 <= lo and hi <= 0.6 else "outside -30% / +60%"
    if key == "margin":
        return "" if 0 < lo and hi < 0.9 else "not a margin"
    return ""


def eps_txt(d: dict) -> str:
    """Adjusted-EPS figures as a reader sees them in a data note: 'FY 9.64, prior FY 8.74'."""
    names = {"fy": "FY", "fy_prior": "prior FY", "ytd_cur": "year to date", "ytd_prior": "prior year to date"}
    return ", ".join(f"{names[k]} {d[k]:g}" for k in names
                     if isinstance((d or {}).get(k), (int, float)) and not isinstance(d[k], bool)) or "none"


def guide_txt(k: str, v) -> str:
    """A guidance range as text: '10.0%–12.0%' for rates, '40,800–41,300' for amounts."""
    if not isinstance(v, (list, tuple)):
        return str(v)
    pct = k in ("organic", "margin") or (k != "adj_eps" and
                                         all(isinstance(x, (int, float)) and abs(x) < 1.5 for x in v))
    return "–".join(f"{x:.1%}" if pct else f"{x:,g}" for x in v)


def adj_eps_problem(spec: ModelSpec, adj: dict, guide=None) -> str:
    """Company adjusted EPS must be for the right periods, not a GAAP copy, and near GAAP: extraction and analyst
    passes have put year-to-date figures in the full-year slot (GOOGL fy = 9M GAAP EPS 16.33, fy_prior = FY2025
    GAAP EPS), GAAP EPS where the company reports no adjusted EPS (NFLX), and this year's guide in last year's slot
    (ADBE: fy = the FY26 guide's $24.50 against an actual $20.94; the analyst then cut the FY25 tax rate to 4.8% to
    reproduce it)."""
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
    g, it = spec.hist.IS["eps_dil"], spec.interim
    fy, fp, yc = (adj.get(k) if num(adj.get(k)) else None for k in ("fy", "fy_prior", "ytd_cur"))
    guide = [x for x in (guide or (spec.guidance or {}).get("adj_eps") or ()) if num(x)]
    if fy is not None and guide and any(abs(fy - x) < 0.006 for x in guide + [sum(guide) / len(guide)]):
        return f"fy equals the FY{spec.fy_hist[-1] + 1} adjusted EPS guide ({fy}): guidance in the actuals slot"
    if fy is not None and yc is not None and abs(fy - yc) < 0.006 and it.available:
        return "fy equals ytd_cur: a year-to-date figure in the full-year slot"
    if fy is not None and it.available and it.ytd_cur.get("eps") and abs(fy - it.ytd_cur["eps"]) < 0.006:
        return f"fy equals {it.label_cur} GAAP EPS: a year-to-date figure in the full-year slot"
    if fp is not None and g[2] and abs(fp - g[2]) < 0.006 and fy is not None and abs(fy - g[2]) > 0.006:
        return f"fy_prior equals FY{spec.fy_hist[-1]} GAAP EPS: periods shifted by a year"
    for v, j in ((fy, 2), (fp, 1)):
        if v is not None and g[j] and abs(v - g[j]) < 0.006 and spec.nonrec_hist[j]:
            return f"FY{spec.fy_hist[j]} equals GAAP EPS although that year had one-offs: a GAAP copy"
        if v is not None and g[j] and g[j] > 0 and not 0.6 * g[j] <= v <= 2.5 * g[j]:
            return f"{v} is implausible against GAAP EPS {g[j]:.2f} for FY{spec.fy_hist[j]}"
    return ""


def ytd_adj_problem(spec: ModelSpec, cur, prior, fy) -> str:
    """Year-to-date adjusted EPS must be a sensible fraction of the full year (CRM: H1 3.77 vs FY 13.19 = 29%)
    and close to GAAP year to date (LLY: 9M 15.56 was the GAAP figure, the non-GAAP one is 16.67)."""
    it = spec.interim
    frac = {"Q1": 0.25, "Q2": 0.5, "Q3": 0.75}.get(it.fiscal_period)
    fy = fy or spec.adj_eps_hist[2] or spec.hist.IS["eps_dil"][2]
    if frac and prior and fy and fy > 0 and not frac - 0.2 <= prior / fy <= frac + 0.2:
        return f"{it.label_prior} {prior} is {prior / fy:.0%} of FY {fy}"
    reports = spec.adj_eps_hist[2] and spec.hist.IS["eps_dil"][2] and \
        abs(spec.adj_eps_hist[2] - spec.hist.IS["eps_dil"][2]) > 0.01 * abs(spec.hist.IS["eps_dil"][2])
    for v, d in ((cur, it.ytd_cur), (prior, it.ytd_prior)):
        g = d.get("eps")
        if v is not None and g and g > 0 and not 0.6 * g <= v <= 2.5 * g:
            return f"{v} vs GAAP {g:.2f}"
        if v is not None and g and reports and abs(v - g) < 0.006:
            # the company's adjusted EPS differs from GAAP for the full year, so an identical YTD figure is a copy
            return f"{v} equals GAAP EPS year to date although the company's adjusted EPS differs from GAAP"
    return ""


def default_debt_like(spec: ModelSpec) -> dict:
    """Loss-contingency accruals (asbestos, litigation) net of insurance recoveries, from XBRL."""
    it, h = spec.interim, spec.hist
    cur = it.debt_like if it.available and it.debt_like else {}
    fy = h.debt_like or {}
    liab_fy, off_fy = (fy.get("liab") or [0, 0])[-1], (fy.get("offset") or [0, 0])[-1]
    liab, off = ((cur.get("liab") or [liab_fy])[0], (cur.get("offset") or [off_fy])[0]) if cur else (liab_fy, off_fy)
    if not (liab or liab_fy):
        return {"liab": 0.0, "offset": 0.0, "liab_fy": 0.0, "offset_fy": 0.0,
                "note": "None tagged in XBRL; enter debt-like items not in net debt (e.g. asbestos, environmental, "
                        "litigation settlements) from the 10-Q"}
    concept = cur.get("liab_concept") or fy.get("liab_concept")
    spec.labels.setdefault("debt_like", "Loss contingency accruals (litigation / asbestos, noncurrent)")
    spec.labels.setdefault("debt_like_offset", "Insurance recoveries / receivables against them")
    return {"liab": round(liab), "offset": round(off), "liab_fy": round(liab_fy), "offset_fy": round(off_fy),
            "note": f"XBRL {concept} less LossContingencyReceivable (verify in the contingencies note)"}


def _other_opex_hist(spec: ModelSpec) -> list:
    from .financials import _other_opex
    I = spec.hist.IS
    return [_other_opex({k: I[k][j] for k in I}) for j in range(3)]


def detect_segment_measure(spec: ModelSpec):
    """Does reported segment profit already deduct amortization and stock comp?

    HON-style segments exclude them (segment profit + corporate – amortization – stock comp
    – other = GAAP EBIT). Many filers (MSFT, AAPL, KO) report segment operating income that
    already sums to GAAP EBIT; deducting amortization and stock comp again would understate
    the forecast EBIT by both.
    """
    h, I = spec.hist, spec.hist.IS
    if not spec.segments_reported:
        spec.seg_after_charges = 0
        spec.seg_after_charges_note = ("Single segment = EBIT + amortization + stock comp + other operating expense, "
                                       "so all are deducted below segment profit")
        if any(_other_opex_hist(spec)):
            spec.repo_label = "   Less: other operating expense (IPR&D, impairments, restructuring)"
        return
    ratios = []
    for j in (1, 2):
        if not all(sg.profit[j] is not None for sg in spec.segments) or I["opinc"][j] is None:
            continue
        plug = sum(sg.profit[j] for sg in spec.segments) + _nz(h.corp_profit[j]) - I["opinc"][j]
        charges = _nz(I["amort"][j]) + _nz(I["sbc"][j])
        if charges > 0:
            ratios.append(plug / charges)
    after = bool(ratios) and max(ratios) < 0.35
    spec.seg_after_charges = 1 if after else 0
    txt = ", ".join(f"{r:.2f}" for r in ratios)
    spec.seg_after_charges_note = (
        f"(Segment profit + corporate – EBIT) / (amortization + stock comp) = {txt}: segment profit already "
        "deducts both, so the IS does not deduct them again" if after else
        f"(Segment profit + corporate – EBIT) / (amortization + stock comp) = {txt or 'n/a'}: segments exclude "
        "them (HON-style); the IS deducts both below segment profit")


def _match(d: dict, name: str):
    if not d:
        return None
    n = re.sub(r"[^a-z]", "", name.lower())
    for k, v in d.items():
        kk = re.sub(r"[^a-z]", "", str(k).lower())
        if kk and (kk in n or n in kk) and isinstance(v, (int, float)):
            return float(v)
    return None


def _match_text(d: dict, name: str) -> str:
    n = re.sub(r"[^a-z]", "", name.lower())
    for k, v in (d or {}).items():
        kk = re.sub(r"[^a-z]", "", str(k).lower())
        if kk and (kk in n or n in kk):
            return str(v)
    return ""


def etr_ytd_default(spec: ModelSpec) -> float:
    I = spec.hist.IS
    return (I["tax"][2] / I["pretax"][2]) if I["pretax"][2] and I["tax"][2] is not None and I["pretax"][2] > 0 else 0.21


def ytd_growth(spec: ModelSpec) -> Optional[float]:
    """Reported year-to-date sales growth, once half the year or more is reported (H1 / 9M)."""
    it = spec.interim
    if not it.available or it.fiscal_period not in ("Q2", "Q3"):
        return None
    cur, pri = it.ytd_cur.get("sales"), it.ytd_prior.get("sales")
    return _clip(cur / pri - 1, -0.5, 1.5) if cur and pri else None


def ytd_margin_change(spec: ModelSpec) -> Optional[float]:
    """Change in total segment margin, year to date vs the same period last year (H1 / 9M)."""
    it = spec.interim
    if not it.available or it.fiscal_period not in ("Q2", "Q3"):
        return None
    c, p = it.ytd_cur, it.ytd_prior
    if not (c.get("sales") and p.get("sales")) or c.get("seg_profit") is None or p.get("seg_profit") is None:
        return None
    return _clip(c["seg_profit"] / c["sales"] - p["seg_profit"] / p["sales"], -0.08, 0.08)


def align_lines(sg: SegSpec, cases=("bear", "base", "bull")):
    """Default line paths whose sales-weighted growth equals the segment's organic path every year.

    Each line starts from its own recent growth relative to the segment (spread capped at ±20pt), the
    spread halves every year, and a common shift keeps the weighted total on the segment path. These are
    placeholders for the analyst's line paths; cases the analyst set explicitly are left alone."""
    if not sg.lines or not all(len(sg.growth.get(c, [])) == 6 for c in cases):
        return
    last = [_nz(ln.sales[2]) for ln in sg.lines]
    prior = sum(_nz(ln.sales[1]) for ln in sg.lines)
    if sum(last) <= 0:
        return
    g_seg = sum(last) / prior - 1 if prior else 0.0
    spreads = [_clip((ln.sales[2] / ln.sales[1] - 1 if ln.sales[1] else g_seg) - g_seg, -0.20, 0.20)
               for ln in sg.lines]
    for case in cases:
        if case in sg.lines_explicit:
            continue
        path, sales = sg.growth[case], list(last)
        out = [[0.0] * 6 for _ in sg.lines]
        for t in range(6):
            raw = [path[t] + sp * 0.5 ** t for sp in spreads]  # momentum halves every year
            w = sum(sales)
            shift = path[t] - sum(x * r for x, r in zip(sales, raw)) / w if w else 0.0
            for k in range(len(sg.lines)):
                out[k][t] = round(max(raw[k] + shift, -0.9), 6)
            sales = [x * (1 + out[k][t]) for k, x in enumerate(sales)]
        for k, ln in enumerate(sg.lines):
            ln.growth[case] = out[k]


def kpi_growth(kpi: dict, case: str) -> Optional[list]:
    """Revenue growth implied by a volume x price build: (1 + volume growth) x (1 + price growth) - 1."""
    try:
        v, p = kpi["volume"]["growth"][case], kpi["price"]["growth"][case]
        return [round((1 + a) * (1 + b) - 1, 6) for a, b in zip(v, p)]
    except (KeyError, TypeError):
        return None


def defaults_drivers(spec: ModelSpec, data):
    h = spec.hist
    I, B, CF = h.IS, h.BS, h.CF
    S = I["sales"]
    s2 = S[2]
    yrs = range(6)

    def flat(v):
        return [v] * 6

    # --- segment growth and margins
    # year 1 (and 2) anchored to company guidance or FY-matched consensus when available, then a
    # linear fade toward a long-run rate (part-way to nominal GDP) by the last forecast year
    g_org = spec.guidance.get("organic")
    m_guide = spec.guidance.get("margin")
    anchors = {}
    if g_org:
        anchors[0] = (sum(g_org) / 2, f"{spec.fc_labels[0]}: organic guidance midpoint {sum(g_org) / 2:.1%}")
    con = spec.consensus or {}
    con_growth = [con.get("g_fy1"), con.get("g_fy2")]
    for t, g in enumerate(con_growth):
        if g is not None and t not in anchors:
            anchors[t] = (g, f"{spec.fc_labels[t]}: Yahoo consensus revenue growth {g:.1%}")
    ytd = ytd_growth(spec)
    if ytd is not None and 0 not in anchors:
        anchors[0] = (ytd, f"{spec.fc_labels[0]}: {spec.interim.label_cur} reported growth {ytd:+.1%} year to date "
                           "(10-Q), carried to the full year")
    w = sum(_nz(sg.sales[2]) for sg in spec.segments)
    g0s = {}
    for sg in spec.segments:
        gl = sg.organic_last if sg.organic_last is not None else (
            sg.sales[2] / sg.sales[1] - 1 if sg.sales[1] and sg.sales[2] else 0.03)
        gp = sg.sales[1] / sg.sales[0] - 1 if sg.sales[0] and sg.sales[1] else gl
        g0s[sg.name] = _clip(0.6 * gl + 0.4 * gp, -0.05, 0.25)
    cur = sum(_nz(sg.sales[2]) * g0s[sg.name] for sg in spec.segments) / w if w else 0
    last_anchor = max(anchors) if anchors else 0
    for sg in spec.segments:
        path = [g0s[sg.name]] * 6
        for t, (target, _) in anchors.items():
            path[t] = g0s[sg.name] + (target - cur)
        start = path[last_anchor]
        lt = _clip(0.4 * start + 0.6 * capital.terminal_growth()[0], 0.015, 0.06)  # fades toward the house rate
        span = 5 - last_anchor
        for t in range(last_anchor + 1, 6):
            if start > 0.15:  # hyper-growth halves its excess each year (AVGO: a linear fade from 65% went 7x)
                path[t] = lt + (start - lt) * 0.5 ** (t - last_anchor)
            else:
                path[t] = start + (lt - start) * (t - last_anchor) / span
        sg.growth["base"] = [_r(v, 0.0025) for v in path]
    if anchors:
        spec.notes.setdefault("growth_anchor", " (" + "; ".join(anchors[t][1] for t in sorted(anchors)) + ")")
    # scenario spread = how wrong a growth forecast has been for this company (PM round 3): 0.75 standard deviation
    # of 10-year revenue growth, 2-5pt, half that in the guided first year
    from .gates import growth_sigma
    sig = growth_sigma(spec)
    d_ = _clip(0.75 * sig, 0.02, 0.05) if sig is not None else 0.02
    spec.notes.setdefault("scenario_spread", (f"Bear / bull growth ±{d_:.1%} from FY2 (0.75 × the {sig:.1%} standard "
                                              "deviation of 10-year revenue growth, 2-5pt), half in FY1")
                          if sig is not None else "Bear / bull growth ±2pt from FY2, ±1pt in FY1")
    for sg in spec.segments:
        b = sg.growth["base"]
        sg.growth["bear"] = [_r(v - (d_ / 2 if t == 0 else d_), 0.0025) for t, v in enumerate(b)]
        sg.growth["bull"] = [_r(v + (d_ / 2 if t == 0 else d_), 0.0025) for t, v in enumerate(b)]
        sg.mna = [0.0] * 6
        sg.mna_note = "Enter announced acquisitions / divestitures and FX (contribution to growth)"
        align_lines(sg)
    # sales guidance usually includes M&A and FX: back-solve the year-1 contribution so that
    # modelled sales land on the guidance midpoint (organic stays on the organic guide)
    g_sales = spec.guidance.get("sales")
    if g_sales:
        seg_next = sum(_nz(sg.sales[2]) * (1 + sg.growth["base"][0]) for sg in spec.segments)
        target = sum(g_sales) / 2 - _nz(h.corp_sales[2])
        mna = target / seg_next - 1 if seg_next else 0
        if abs(mna) < 0.15:
            for sg in spec.segments:
                sg.mna[0] = _r(mna, 0.0025)
                sg.mna_note = (f"{spec.fc_labels[0]}: {mna:+.1%} calibrated so sales ≈ the guidance midpoint "
                               f"({sum(g_sales) / 2:,.0f}mm); split by segment if the release gives detail")
    total_margin_last = None
    if all(sg.profit[2] is not None for sg in spec.segments) and s2:
        total_margin_last = (sum(sg.profit[2] for sg in spec.segments) + _nz(h.corp_profit[2])) / s2
    delta = 0.0
    if m_guide and total_margin_last is not None:
        # the guide moves margins by its change on its own basis (PM, ADBE: the 45% non-GAAP operating margin guide
        # was applied to gross-profit segment margins of 89% and cut every segment by 44pt) — the basis is the
        # analyst's when given, else the one nearest the guide, as on the Interim tab
        mid = sum(m_guide) / 2
        bases = {"segment": total_margin_last}
        if I["opinc"][2] is not None and s2:
            bases["gaap"] = I["opinc"][2] / s2
            bases["adjusted"] = (I["opinc"][2] + _nz(I["amort"][2]) + _nz(I["sbc"][2])) / s2
        b = str(spec.margin_basis or "")[:3]
        key = next((k for k in bases if b and k.startswith(b)), None) or min(bases, key=lambda k: abs(bases[k] - mid))
        delta = mid - bases[key]
        if key != "segment":
            spec.notes.setdefault("margin_anchor", f"; the {mid:.1%} margin guide is on the {key} operating basis "
                                                   f"(FY{spec.fy_hist[-1]} {bases[key]:.1%}; segment basis "
                                                   f"{total_margin_last:.1%}): segment margins move by its "
                                                   f"{delta:+.1%} change")
    elif (ym := ytd_margin_change(spec)) is not None:
        delta = ym
        spec.notes.setdefault("margin_anchor", f"; base = FY{spec.fy_hist[-1]} margin {delta:+.1%} "
                                               f"({spec.interim.label_cur} vs {spec.interim.label_prior} segment "
                                               "margin change, 10-Q)")
    # a cyclical away from its 10-year median operating margin returns to mid-cycle by the last forecast year, from
    # a trough (UPS, NKE ended below it) or a peak alike (PM round 4: CAT at 22% against a 15% median was never cut)
    lift = 0.0
    mh = sorted(m_ for _, m_ in (spec.margin_hist or []))
    if spec.profile.get("cyclical") and len(mh) >= 5 and s2 and I["opinc"][2] is not None and \
            total_margin_last is not None and w:
        med = mh[len(mh) // 2]
        basis_gap = total_margin_last - I["opinc"][2] / s2  # segment basis above GAAP operating margin
        seg6 = sum(_nz(sg.sales[2]) * ((sg.profit[2] / sg.sales[2] if sg.profit[2] is not None and sg.sales[2] else
                                        0.15) + delta + 0.010) for sg in spec.segments) / w
        lift = med - (seg6 + _nz(h.corp_profit[2]) / s2 - basis_gap)
        if abs(lift) > 0.01:
            spec.notes["margin_anchor"] = (spec.notes.get("margin_anchor", "") +
                                           f"; cyclical: margins {'rise' if lift > 0 else 'fall'} {abs(lift):.1%} by "
                                           f"{spec.fc_labels[-1]} to the 10-year median operating margin ({med:.1%}, "
                                           "mid-cycle)")
        else:
            lift = 0.0
    for sg in spec.segments:
        m = sg.profit[2] / sg.sales[2] if sg.profit[2] is not None and sg.sales[2] else 0.15
        base = [_r(m + delta + 0.002 * t + lift * (t + 1) / 6, 0.001) for t in yrs]
        sg.margin["base"] = base
        sg.margin["bear"] = [_r(v - min(0.02, 0.005 * (t + 1)), 0.001) for t, v in enumerate(base)]
        sg.margin["bull"] = [_r(v + min(0.02, 0.005 * (t + 1)), 0.001) for t, v in enumerate(base)]

    cs = _nz(h.corp_sales[2])
    spec.corp_sales = flat(round(cs))
    spec.corp_sales_note = f"FY{spec.fy_hist[-1]}: {cs:,.0f}mm (total sales less segments)"
    cp = h.corp_profit[2]
    pct = (cp / s2) if cp is not None and s2 else 0.0
    spec.corp_cost_pct = flat(_r(pct, 0.0005))
    spec.corp_cost_note = (f"FY{spec.fy_hist[-1]}: {cp:,.0f}mm ({pct:.1%})" if cp is not None
                           else "Not disclosed in XBRL; set to 0 (corporate costs sit in 'repositioning & other')")

    D = {}
    fy = f"FY{spec.fy_hist[-1]}"
    # amortization: hold flat but never amortize more than the remaining intangibles
    am = _nz(I["amort"][2])
    rem = _nz(B["intang"][2])
    vals = []
    for _ in yrs:
        a = min(am, max(rem, 0) * 0.5) if am else 0.0
        vals.append(_r(a, 5))
        rem -= a
    D["amort"] = (vals, f"{fy}: {am:,.0f}mm; held flat (capped by remaining intangibles)")
    sbc = _nz(I["sbc"][2]) / s2 if s2 else 0
    D["sbc_pct"] = (flat(_r(sbc, 0.0005)), f"{fy}: {_nz(I['sbc'][2]):,.0f}mm = {sbc:.1%} of sales")
    if spec.profile.get("key") == "software" and sbc > 0.065:
        # stock comp fades toward a mature software level as the business scales (critique D1)
        path = [_r(sbc + (0.055 - sbc) * k / 6, 0.0005) for k in range(1, 7)]
        D["sbc_pct"] = (path, f"{fy}: {sbc:.1%} of sales, fading to a mature 5.5% by {spec.fc_labels[-1]} "
                              "(dilution modeled through shares issued)")
    repo = []
    deduct = 1 - spec.seg_after_charges
    for j in (1, 2):
        sp = (sum(_nz(sg.profit[j]) for sg in spec.segments) + _nz(h.corp_profit[j])
              if all(sg.profit[j] is not None for sg in spec.segments) else None)
        if sp is not None and I["opinc"][j] is not None and S[j]:
            repo.append((sp - I["opinc"][j] - deduct * (_nz(I["amort"][j]) + _nz(I["sbc"][j]))) / S[j])
    structural = len(repo) == 2 and min(repo) > 0.03  # e.g. gross-profit segments: opex sits below them
    spec.repo_structural = structural  # SOTP must capitalize them (MRK: R&D outside segment profit, SOTP $519)
    if structural:
        rp = sum(repo) / 2
        spec.repo_label = "   Less: operating expenses not in segment profit"
        how = "; default = two-year average (structural: segment measure is above operating expenses)"
    else:
        rp = _clip(min(repo), 0.0, 0.03) if repo else 0.0
        how = "; default = lower of the last two years, floored at 0, capped at 3%"
    if h.seg_measure == "gross profit":
        spec.repo_label = "   Less: operating expenses not in segment profit"
    what = "segment profit – EBIT" + (" – amortization – stock comp" if deduct else "")
    D["repo_pct"] = (flat(_r(rp, 0.0005)), f"History plug ({what}): " + ", ".join(f"{v:.1%}" for v in repo) + how)
    other_inc = spec.is_hist["other_inc"][2]
    nr_last = spec.nonrec_hist[2]
    nonop = _nz(other_inc) - _nz(I["interest_inc"][2]) - nr_last
    D["other_nonop"] = (flat(_r(nonop, 5)), f"{fy}: other income {_nz(other_inc):,.0f}mm less interest income "
                        f"{_nz(I['interest_inc'][2]):,.0f}mm" + (f" less one-offs {nr_last:,.0f}mm" if nr_last else "")
                        + "; treated as non-cash (pension & other)")
    # FY1 carries the one-offs already booked this year, so GAAP FY1 fits the year to date: tagged securities
    # gains / debt extinguishment, plus any other income above last year's same period (untagged one-offs)
    it_ = spec.interim
    mat = lambda v: v if abs(v) >= 0.005 * abs(_nz(s2)) * 0.75 else 0.0  # noqa: E731
    nr_ytd = mat(_nz(it_.ytd_cur.get("nonrec"))) if it_.available else 0.0
    extra = 0.0
    if it_.available and it_.ytd_cur.get("other_inc") is not None and it_.ytd_prior.get("other_inc") is not None:
        extra = ((it_.ytd_cur["other_inc"] - nr_ytd) -
                 (it_.ytd_prior["other_inc"] - mat(_nz(it_.ytd_prior.get("nonrec")))))
        extra = extra if abs(extra) >= 0.01 * abs(_nz(s2)) else 0.0
    parts = ([f"{nr_ytd:,.0f}mm tagged securities gains / debt extinguishment"] if nr_ytd else []) + \
        ([f"{extra:,.0f}mm more other income than in {it_.label_prior} (untagged: treated as non-recurring; move it "
          "to pension & other if it recurs)"] if extra else [])
    for tag, d in (("cur", it_.ytd_cur), ("prior", it_.ytd_prior)):  # estimated YTD adjusted EPS: same exclusion
        x = extra if tag == "cur" else 0.0
        booked = x + (nr_ytd if tag == "cur" else mat(_nz(d.get("nonrec"))))
        if booked and not d.get("adj_eps_est") and d.get("eps") and d.get("ni_attr") and d.get("adj_eps") is not None \
                and abs(d["adj_eps"] - d["eps"]) < 0.006:
            # a "company adjusted EPS" equal to GAAP although the period has one-offs: a GAAP copy, re-estimate
            sh, t_ = d["ni_attr"] / d["eps"], etr_ytd_default(spec)
            d["adj_eps"] = round((d["ni_attr"] + (_nz(d.get("amort")) - booked) * (1 - t_)) / sh, 2)
            d["adj_eps_model"] = d["adj_eps"]
            (spec.extracted.get("adjusted_eps") or {})["ytd_" + tag] = None
            continue
        if x and d.get("adj_eps_est") and d.get("adj_eps") is not None:
            t_, sh = d["adj_eps_est"]
            d["adj_eps"] = round(d["adj_eps"] - x * (1 - t_) / sh, 2)
            d["adj_eps_model"] = d["adj_eps"]
            d["adj_eps_oneoff"] = _nz(d.get("adj_eps_oneoff")) + x
    D["nonrec"] = ([_r(nr_ytd + extra, 1)] + [0.0] * 5,
                   (f"{spec.fc_labels[0]} = one-offs already booked in {it_.label_cur}: " + " and ".join(parts) +
                    "; none forecast after. " if parts else "") +
                   "Non-recurring gains (+) / losses (–) in pre-tax income: in GAAP EPS, excluded from adjusted EPS "
                   "and treated as non-cash")
    beg_cash = _nz(B["cash"][1]) + _nz(B["sti"][1])
    if _nz(I["interest_inc"][2]) <= 0 and beg_cash > 0.02 * _nz(s2):
        # interest income not reported on its own line (NFLX, LLY: inside "interest and other income"): cash still
        # earns money. Estimate it at the risk-free rate less 1pt and take the same amount out of the recurring
        # other income, so FY1 other income starts from last year's total
        y = _clip((data.market.get("rf") or 0.0425) - 0.01, 0.01, 0.05)
        est = y * beg_cash
        vals, note = D["other_nonop"]
        D["other_nonop"] = ([_r(v - est, 5) for v in vals],
                            note + f"; less ~{est:,.0f}mm estimated interest income (not reported separately)")
        D["int_yield"] = (flat(_r(y, 0.0025)), f"Interest income not reported separately: estimated at the risk-free "
                                               f"rate less 1pt ({y:.1%}) on cash & ST investments")
    else:
        y = _clip(_nz(I["interest_inc"][2]) / beg_cash, 0.0, 0.06) if beg_cash else 0.03
        D["int_yield"] = (flat(_r(y, 0.0025)), f"{fy}: {_nz(I['interest_inc'][2]):,.0f}mm on {beg_cash:,.0f}mm "
                                               "beginning cash & ST investments")
    debt = [_nz(B["st_debt"][j]) + _nz(B["cur_ltd"][j]) + _nz(B["ltd"][j]) for j in (1, 2)]
    avg_debt = sum(debt) / 2 * (1 - (spec.captive.get("share") or 0.0))
    ir = _nz(I["interest_exp"][2]) / avg_debt if avg_debt else 0.0
    if ir == 0 and avg_debt:
        ir = (data.market.get("rf") or 0.0425) + 0.01
        irn = "Interest expense not tagged; default = risk-free + 1.0pt"
    else:
        irn = f"{fy}: {_nz(I['interest_exp'][2]):,.0f}mm / {avg_debt:,.0f}mm average debt = {ir:.1%}"
    ir = _clip(ir, 0.0, 0.12)
    D["int_rate"] = ([_r(ir + 0.001 * t, 0.0005) for t in yrs], irn + "; drifts up as debt refinances")
    # the forecast opens from the 10-Q (critique B4): debt raised early in the year (CRM's ASR financing) already
    # costs its run-rate, which last year's rate on last year's average debt misses
    it_ = spec.interim
    frac_ = {"Q1": 0.25, "Q2": 0.5, "Q3": 0.75}.get(it_.fiscal_period) if it_.available else None
    ie_ytd = _nz(it_.ytd_cur.get("interest_exp")) if frac_ else 0.0
    q_debt = (_nz(it_.bs_cur.get("st_debt")) + _nz(it_.bs_cur.get("cur_ltd")) + _nz(it_.bs_cur.get("ltd"))
              if frac_ else 0.0)
    q_debt *= 1 - ((getattr(it_, "captive_share", None) or spec.captive.get("share") or 0.0) if spec.captive else 0.0)
    if ie_ytd > 0 and q_debt > 0 and (not spec.captive or spec.captive.get("on_face")):
        run = ie_ytd / frac_
        nc_ = 1 - (spec.captive.get("share") or 0.0) if spec.captive else 1.0  # industrial debt only (CAT)
        avg1 = (debt[1] * nc_ + q_debt) / 2  # the first year ends at the 10-Q debt (debt_iss)
        if avg1 > 0 and 0.01 <= run / q_debt <= 0.12 and abs(ir * avg1 - run) > 0.15 * run:
            r1, r_on = _clip(run / avg1, 0.0, 0.12), _clip(run / q_debt, 0.0, 0.12)
            D["int_rate"] = ([_r(r1, 0.0005)] + [_r(r_on + 0.001 * t, 0.0005) for t in yrs[1:]],
                             f"{spec.fc_labels[0]}: {run:,.0f}mm annualized from {it_.label_cur} on average debt "
                             f"{avg1:,.0f}mm = {r1:.1%} (debt raised or repaid early in the year); then {r_on:.1%} on "
                             f"the {q_debt:,.0f}mm outstanding at {it_.period_end:%b %-d, %Y}, drifting up as debt "
                             f"refinances ({irn})")
    etr = (I["tax"][2] / I["pretax"][2]) if I["pretax"][2] and I["pretax"][2] > 0 and I["tax"][2] is not None else 0.21
    etr_prev = (I["tax"][1] / I["pretax"][1]) if I["pretax"][1] and I["pretax"][1] > 0 and I["tax"][1] is not None else etr
    etr_d = _clip(_r((etr + etr_prev) / 2, 0.005), 0.10, 0.30)
    D["etr"] = (flat(etr_d), f"{fy}: {etr:.1%}; FY{spec.fy_hist[-2]}: {etr_prev:.1%}")
    D["adj_tax"] = (flat(0.0), "0 = adjusted EPS taxes its adjustments at the year's effective rate (most companies). "
                               "Set the company's non-GAAP tax rate when its reconciliation taxes non-GAAP pre-tax "
                               "income at one long-term rate (CRM, most software)")
    nci = _nz(I["nci"][2])
    D["nci_ni"] = (flat(_r(nci, 1)), f"{fy}: {nci:,.0f}mm")
    cogs_pct = _nz(I["cogs"][2]) / s2 if s2 and _nz(I["cogs"][2]) > 0 else (
        (s2 - _nz(I["opinc"][2])) / s2 if s2 else 0.6)
    D["cogs_pct"] = (flat(_r(cogs_pct, 0.001)), f"{fy}: {cogs_pct:.1%}")
    da_other = max((_nz(CF["da"][2]) - _nz(I["amort"][2])) / s2 if s2 else 0.03, 0)
    capex = -_nz(CF["capex"][2]) / s2 if s2 else 0.03
    da_note = (f"{fy}: ({_nz(CF['da'][2]):,.0f} D&A – {_nz(I['amort'][2]):,.0f} acquired-intangible amortization) "
               f"/ sales = {da_other:.1%}")
    dep = _nz(I.get("dep", [None] * 3)[2])
    ppe0, ppe1 = _nz(B["ppe"][1]), _nz(B["ppe"][2])
    rf_dep = ppe0 - _nz(CF["capex"][2]) - ppe1  # beginning + capex – ending PP&E: what depreciation ran through it
    if s2 and 0 < dep < 0.85 * da_other * s2:
        # the rest of D&A amortizes leases / capitalized costs (CRM): not PP&E, and paid for in operating cash flow,
        # so it must not run PP&E down (CRM's PP&E went negative on total D&A)
        da_note = (f"{fy}: PP&E depreciation {dep:,.0f}mm / sales = {dep / s2:.1%} (D&A {_nz(CF['da'][2]):,.0f} also "
                   f"amortizes leases / capitalized costs: {da_other * s2 - dep:,.0f}mm, not PP&E)")
        da_other = dep / s2
    # only where capex runs below depreciation (CRM); FX and acquisitions distort the roll-forward (LIN's D&A
    # fell 36% when the cap applied to a company investing above depreciation)
    if s2 and ppe0 > 0 and capex < da_other and 0.4 * da_other * s2 < rf_dep < 0.8 * da_other * s2:
        # the PP&E roll-forward shows what depreciation actually ran through PP&E; CRM's depreciation tag (1.2bn)
        # also covers assets outside PP&E (roll-forward 0.7bn). Only that much may run PP&E down (critique: PP&E
        # rises with growth), the rest amortizes leases / capitalized costs paid for in operating cash flow
        da_note = (f"{fy}: PP&E depreciation {rf_dep:,.0f}mm from the roll-forward ({ppe0:,.0f} + "
                   f"{-_nz(CF['capex'][2]):,.0f} capex – {ppe1:,.0f}) / sales = {rf_dep / s2:.1%}; the rest of the "
                   f"{_nz(CF['da'][2]):,.0f}mm D&A amortizes acquired intangibles, leases and capitalized costs, "
                   "not PP&E")
        da_other = rf_dep / s2
    capex_note = f"{fy}: {-_nz(CF['capex'][2]):,.0f}mm = {capex:.1%} of sales"
    if da_other > 0 and capex > 1.5 * da_other and spec.profile.get("key") != "utility":
        # investment phase: holding capex at today's % of sales forever would starve the terminal
        # cash flow. D&A catches up with the asset base and capex settles at 1.3x D&A by the last year.
        d_t = da_other + 0.5 * (capex - da_other)
        c_t = min(capex, 1.3 * d_t)
        da_path = [_r(da_other + (d_t - da_other) * t / 5, 0.001) for t in yrs]
        cx_path = [_r(capex + (c_t - capex) * t / 5, 0.001) for t in yrs]
        D["da_pct"] = (da_path, da_note + f"; rises to {d_t:.1%} by {spec.fc_labels[-1]} as the capex build depreciates")
        D["capex_pct"] = (cx_path, capex_note + f" ({capex / da_other:.1f}x D&A: investment phase); fades to "
                          f"1.3x D&A ({c_t:.1%}) by {spec.fc_labels[-1]} so the terminal year reinvests at a "
                          "steady-state rate")
    elif 0 < capex < da_other:
        # capex below depreciation runs PP&E down while sales grow (critique gate 4): replacement at today's prices
        # plus some growth capex is at least 1.1x depreciation
        D["da_pct"] = (flat(_r(da_other, 0.001)), da_note)
        D["capex_pct"] = (flat(_r(1.1 * da_other, 0.001)),
                          capex_note + f", below depreciation ({da_other:.1%}): set to 1.1x depreciation so PP&E "
                          "rises with sales")
    else:
        D["da_pct"] = (flat(_r(da_other, 0.001)), da_note)
        D["capex_pct"] = (flat(_r(capex, 0.001)), capex_note)
    memo = s2 * cogs_pct
    dso = _nz(B["ar"][2]) / s2 * 365 if s2 else 60
    dio = _nz(B["inv"][2]) / memo * 365 if memo else 0
    oca = _nz(B["oca"][2]) / s2 if s2 else 0
    dpo = _nz(B["ap"][2]) / memo * 365 if memo else 0
    acc = _nz(B["accrued"][2]) / s2 if s2 else 0
    D["dso"] = (flat(round(dso)), f"{fy}: {dso:.1f} days")
    D["dio"] = (flat(round(dio)), f"{fy}: {dio:.1f} days")
    D["oca_pct"] = (flat(_r(oca, 0.0005)), f"{fy}: {oca:.1%}")
    D["dpo"] = (flat(round(dpo)), f"{fy}: {dpo:.1f} days")
    D["accrued_pct"] = (flat(_r(acc, 0.001)), f"{fy}: {acc:.1%} (accrued & other current liabilities)")
    oo = [v for v in CF["other_op"] if v is not None]
    ct = spec.hist.content if hasattr(spec.hist, "content") else {}
    if ct and spec.profile.get("key") == "media" and s2 and ct["amort"][2] and ct["additions"][2]:
        # content schedule: cash spend = amortization x a ratio trending to 1.05; the excess is operating cash out
        am_pct = ct["amort"][2] / s2
        r0 = ct["additions"][2] / ct["amort"][2]
        sales_path, x = [], s2
        g0 = spec.consensus.get("g_fy1") if spec.consensus.get("g_fy1") is not None else 0.08
        for k in range(6):
            x *= 1 + (g0 if k == 0 else max(0.03, g0 - 0.01 * k))
            sales_path.append(x)
        ratios = [r0 + (1.05 - r0) * k / 5 for k in range(6)]
        D["other_op"] = ([round(-(rt - 1) * am_pct * sv) for rt, sv in zip(ratios, sales_path)],
                         f"Content schedule: cash content spend = amortization ({am_pct:.0%} of sales in {fy}) x "
                         f"{r0:.2f} trending to 1.05; the excess over amortization is an operating cash outflow "
                         f"({fy}: additions {ct['additions'][2]:,.0f} vs amortization {ct['amort'][2]:,.0f}mm)")
    else:
        D["other_op"] = (flat(0.0), "History: " + ", ".join(f"{v:,.0f}" for v in oo) +
                         " (restructuring, legal, deferred tax); default 0 — not a calibration line (FCF guidance "
                         "is met through working capital)")
    D["acq"] = (flat(0.0), "Enter announced deals (cash paid)")
    ish = (spec.profile or {}).get("intang_share", 0.3)
    D["intang_share"] = (flat(ish), f"Remainder to goodwill; typical for "
                                    f"{(spec.profile or {}).get('label', 'industrial').split(' /')[0].lower()} deals")
    it_ = spec.interim
    q_debt = (_nz(it_.bs_cur.get("st_debt")) + _nz(it_.bs_cur.get("cur_ltd")) + _nz(it_.bs_cur.get("ltd"))
              if it_.available else None)
    fy_debt = _nz(B["st_debt"][2]) + _nz(B["cur_ltd"][2]) + _nz(B["ltd"][2])
    if q_debt is not None and abs(q_debt - fy_debt) > 0.02 * max(fy_debt, 1.0):
        # the forecast opens from the latest 10-Q: debt raised or repaid this year is already done (CRM's ASR debt)
        D["debt_iss"] = ([round(q_debt - fy_debt)] + [0.0] * 5,
                         f"{spec.fc_labels[0]}: {q_debt - fy_debt:+,.0f}mm already issued (repaid) by "
                         f"{it_.period_end:%b %-d, %Y} (10-Q debt {q_debt:,.0f}mm vs {fy_debt:,.0f}mm at FY{fy[2:]} end); "
                         "later years held flat")
    else:
        D["debt_iss"] = (flat(0.0), f"{fy} net issuance: {_nz(CF['debt'][2]):,.0f}mm; default holds debt flat")
    bb = -_nz(CF["buyback"][2])
    D["buyback"] = (flat(_r(bb, 50)), f"{fy}: {bb:,.0f}mm")
    px = spec.price or 100.0
    D["buyback_px"] = ([round(px * 1.08 ** t, 2) for t in yrs], "Assumes ~8% annual share price appreciation")
    # dilution tied to stock comp: the shares granted are worth the expense at the price they are granted at (PM
    # round 6: ADBE issued 1.6mm shares a year against $2.2bn of stock comp, ~8mm at $250-300); EPS growth and the
    # 3-year return otherwise count buybacks without the grants that offset them
    sales_t, x = [], _nz(s2)
    w0 = sum(_nz(sg.sales[2]) for sg in spec.segments) or 1.0
    for t in yrs:
        x *= 1 + sum(_nz(sg.sales[2]) * (sg.growth["base"][t] + sg.mna[t]) for sg in spec.segments) / w0
        sales_t.append(x)
    sbc_mm = [D["sbc_pct"][0][t] * sales_t[t] for t in yrs]
    iss_ = [round(max(spec.basic_shares * 0.004, sbc_mm[t] / D["buyback_px"][0][t]), 1) for t in yrs]
    D["shares_issued"] = (iss_, f"Stock comp ÷ the repurchase price: {sbc_mm[0]:,.0f}mm ÷ ${D['buyback_px'][0][0]:,.2f}"
                                f" = {iss_[0]:,.1f}mm shares in {spec.fc_labels[0]} (grants are dilution at the price "
                                "they are worth); at least 0.4% of basic shares")
    sp_ = _nz(CF["stock"][2])
    D["stock_proc"] = (flat(_r(sp_, 5)), f"{fy}: {sp_:,.0f}mm")
    it = spec.interim
    dps_hist = [v for v in h.dps if v]
    if it.available and it.q_dps:
        dps0 = round(it.q_dps * 4, 2)
        dn = f"${it.q_dps:.2f} quarterly ({it.quarter_label}) × 4"
    elif dps_hist:
        dps0 = round(dps_hist[-1] * 1.05, 2)
        dn = f"{fy} dividends declared ${dps_hist[-1]:.2f} × 1.05"
    else:
        dps0 = 0.0
        dn = "No dividend"
    D["dps0"] = ([dps0], dn)
    dg = 0.05
    if len(dps_hist) >= 2 and dps_hist[-2]:
        dg = _clip(_r(dps_hist[-1] / dps_hist[-2] - 1, 0.005), 0.0, 0.10)
    D["div_growth"] = ([dg] * 5 if dps0 else [0.0] * 5,
                       f"Recent annual increase {dg:.1%}" if dps0 else "No dividend")
    spec.drivers = D
    floor_booked_charges(spec)
    spec.flags.extend(balance_guards(spec))


def defaults_valuation(spec: ModelSpec, data):
    m = data.market
    h, it = spec.hist, spec.interim
    I, B = h.IS, h.BS
    V = {}
    rf = m.get("rf") or 0.0425
    V["rf"] = (_r(rf, 0.0005), m.get("rf_note") or "Approx. 10-yr Treasury yield (check FRED DGS10)")
    spec.market_rf = m.get("rf")
    V["tax_wacc"] = (spec.drivers["etr"][0][-1], "Long-run effective rate")
    eq = spec.price * (spec.basic_shares + spec.dilutive)
    debt = _nz(it.bs_cur.get("st_debt")) + _nz(it.bs_cur.get("cur_ltd")) + _nz(it.bs_cur.get("ltd"))
    debt *= 1 - (spec.captive.get("share") or 0.0)  # captive finance debt is operating, not capital
    # cost of capital by method (valuation/capital.py): one ERP for every name on the date, beta from the method,
    # cost of debt from the agency rating the filings disclose, else a synthetic rating
    erp, erp_note = capital.erp(spec.as_of)
    V["erp"] = (erp, erp_note)
    de = debt / eq if eq else 0.0
    beta, beta_how = capital.method_beta(m.get("beta"), data.peers or [], de, V["tax_wacc"][0])
    V["beta"] = (beta, beta_how + (f"; raw regression: {m['beta_note']}" if m.get("beta_note") else ""))
    fy = spec.fy_interim
    ytd_c, ytd_p = it.ytd_cur, it.ytd_prior
    ltm_ebit = _nz(I["opinc"][2]) + (_nz(ytd_c.get("opinc")) - _nz(ytd_p.get("opinc")) if it.available else 0)
    ltm_int = _nz(I["interest_exp"][2]) + (_nz(ytd_c.get("interest_exp")) - _nz(ytd_p.get("interest_exp"))
                                           if it.available else 0)
    cover = ltm_ebit / ltm_int if ltm_int > 0 else None
    rating, spread = capital.synthetic_rating(cover if cover is not None else (99 if ltm_ebit > 0 else None))
    syn = (f"synthetic {rating} from EBIT / interest {cover:.1f}x LTM" if cover is not None else
           f"synthetic {rating} (no interest expense)")
    ag = capital.agency_rating(data.market.get("ratings") or {})
    if ag:  # the filer's actual ratings (PM round 4: a synthetic AAA for all six names, actual AA to A)
        rating, spread, how = ag
        V["kd"] = (_r(rf + spread, 0.0005), f"Risk-free + {spread:.2%} spread for the {rating} rating ({how}; "
                                            f"{data.market.get('ratings_src', 'filings')}); {syn} as a cross-check")
    else:
        V["kd"] = (_r(rf + spread, 0.0005), f"Risk-free + {spread:.2%} spread for a {syn} (no agency rating in the "
                                            "filings)")
    spec.capital = {"beta": beta, "beta_how": beta_how, "erp": erp, "rating": rating, "spread": spread,
                    "rating_src": "agency" if ag else "synthetic",
                    "coverage": cover, "kd": V["kd"][0], "de": de, "exit_derived": True, "own_raw": m.get("beta")}
    ke = rf + beta * erp
    kdat = V["kd"][0] * (1 - V["tax_wacc"][0])
    wacc = (eq * ke + debt * kdat) / (eq + debt) if eq + debt else ke
    g_house, g_note = capital.terminal_growth()
    # never above the forecast's own last-year growth (a 2.7% grower does not speed up forever: VZ), floor 2%
    w6 = sum(_nz(sg.sales[2]) for sg in spec.segments) or 1.0
    g_last = sum(_nz(sg.sales[2]) * (sg.growth["base"][5] + sg.mna[5]) for sg in spec.segments) / w6
    g = min(g_house, _r(wacc - 0.03, 0.0025), _r(max(g_last, 0.02), 0.0025))
    V["g"] = (g, g_note + ("" if g == g_house else
                           f"; held 3pt below WACC {wacc:.1%}" if g == _r(wacc - 0.03, 0.0025) else
                           f"; held at the {spec.fc_labels[-1]} growth rate ({g_last:.1%}): growth does not "
                           "re-accelerate into perpetuity"))
    # return on new invested capital for the value-driver terminal value: reinvestment = g / RONIC
    roic = current_roic(spec)
    spec.capital.update({"roic": roic, "roic_gw": current_roic(spec, with_goodwill=True), "wacc": wacc})
    if roic is not None:
        roic = min(roic, 0.60)  # a tiny capital base makes ROIC explode: cap what new capital can earn
    if roic is None or roic <= wacc or spec.profile.get("ronic_rule") == "wacc":
        ronic = wacc
        why = ("about WACC: growth adds little value (commoditizing / capital-intensive industry)"
               if spec.profile.get("ronic_rule") == "wacc" else
               f"= WACC: current ROIC {roic:.1%} does not exceed it" if roic is not None else "= WACC (no ROIC)")
    else:
        ronic = wacc + 0.5 * (roic - wacc)
        why = (f"halfway between WACC {wacc:.1%} and current operating ROIC {roic:.1%} (working capital + PP&E; "
               "wide-moat default, never above ROIC)")
    V["ronic"] = (_r(ronic, 0.0025), f"Return on new invested capital {why}")
    V["stake_haircut"] = (0.25, "Tax and liquidity haircut on stakes outside EBITDA (carrying value or last round)")
    # current EV / LTM EBITDA
    net_debt = debt - _nz(it.bs_cur.get("cash")) - _nz(it.bs_cur.get("sti"))  # excl. captive finance debt
    ltm_da = _nz(fy["da"]) + (_nz(ytd_c.get("da")) - _nz(ytd_p.get("da")) if it.available else 0)
    nci_v = _nz(B["nci_bs"][2]) + _nz(B["redeem_nci"][2])
    ev = eq + net_debt + nci_v
    ebitda = ltm_ebit + ltm_da
    fwd = fy1_ebitda(spec)
    # forward basis, as a buy-side model quotes it: LTM multiples of fast growers embed today's growth
    # and push the exit-multiple DCF far above the perpetuity DCF
    basis, base_ebitda = ("FY1E", fwd) if fwd and fwd > 0 else ("LTM", ebitda)
    if base_ebitda > 0:
        raw = ev / base_ebitda
        mult = _clip(_r(raw, 0.5), 6.0, 30.0)
        ltm_txt = f"; LTM {ev / ebitda:.1f}x" if basis == "FY1E" and ebitda > 0 else ""
        V["exit_mult"] = (mult, f"Current EV / {spec.fc_labels[0] if basis == 'FY1E' else 'LTM'} EBITDA ≈ {raw:.1f}x "
                                f"at the valuation date (computed{ltm_txt}){' (capped at 30x)' if raw > 30 else ''}; "
                                "a modest discount to it when it reflects peak growth or sentiment; cross-check "
                                "the peers' EV/EBITDA (Comps) and the implied perpetual growth (DCF!D53)")
    else:
        V["exit_mult"] = (12.0, "EBITDA not positive; analyst judgment")
    single = not spec.segments_reported or len(spec.segments) < 2
    V["w_pg"] = (0.25, "")
    V["w_exit"] = (0.25, "")
    V["w_sotp"] = (0.30 if not single else 0.0,
                   "Breakup / mix value" if not single else "Single segment: no sum of the parts (it would be another "
                                                             "EV/EBITDA comp); its weight goes to the DCF")
    V["w_comps"] = (0.20, "")
    if single:
        V["w_pg"], V["w_exit"] = (0.45, ""), (0.35, "")
    if spec.profile.get("key") == "utility":  # NEE, DUK: capex above cash flow for years, funded by debt and equity
        why = ("Regulated utility: unlevered free cash flow is negative while the rate base grows, so P/E vs peers "
               "carries the value and the DCF is a cross-check")
        V["w_pg"], V["w_exit"], V["w_sotp"], V["w_comps"] = (0.25, why), (0.15, why), (0.0, why), (0.60, why)
    V["peer_discount"] = (0.0, "Positive = discount to the peer median (e.g. conglomerate discount); negative = premium")
    eps_base = spec.adj_eps_hist[2] or spec.adj_eps_estimate(2)
    pe = spec.price / eps_base if eps_base and eps_base > 0 else 18.0
    V["exit_pe"] = (_clip(round(pe), 8, 40), f"Near the current P/E on FY{spec.fy_hist[-1]} adjusted EPS ({pe:.1f}x)")
    # stakes and long-term investments sit outside EBITDA and outside invested capital (current_roic), so they
    # are worth something only in the equity bridge (CRM: $11.3bn of strategic investments were left at 0)
    lt_q = _nz(it.bs_cur.get("lt_inv")) if it.available else 0.0
    lt_inv = lt_q or _nz(B["lt_inv"][2])
    when = it.period_end if lt_q else spec.period_ends[-1]
    V["other_stake"] = ((float(round(lt_inv)), f"Long-term investments on the {when:%b %-d, %Y} balance sheet "
                         f"({lt_inv:,.0f}mm: strategic / equity-method stakes, long-term securities): outside EBITDA "
                         "and invested capital, so added to equity value after the haircut")
                        if lt_inv > 0.005 * eq else
                        (0.0, "Equity stakes / investments not captured in segment EBITDA (none material on the "
                              "balance sheet)"))
    V["nci_value"] = (round(nci_v), f"{spec.period_ends[-1]:%b %-d, %Y} BS: {_nz(B['nci_bs'][2]):,.0f}mm + "
                                    f"{_nz(B['redeem_nci'][2]):,.0f}mm redeemable")
    # three-stage DCF: after the six explicit years growth fades to the terminal rate over N more years, so
    # a business still growing well above GDP in year six is not cut to terminal growth overnight
    w_s = sum(_nz(sg.sales[2]) for sg in spec.segments) or 1.0
    g6 = sum(_nz(sg.sales[2]) * (sg.growth["base"][5] + sg.mna[5]) for sg in spec.segments) / w_s
    fade = 0 if g6 <= g + 0.01 else int(min(spec.profile.get("fade_max", 4), round((g6 - g) / 0.01)))
    V["fade_years"] = (fade, f"Base-case growth in {spec.fc_labels[-1]} is {g6:.1%} vs terminal {g:.1%}: "
                             + (f"fades linearly over {fade} more years (three-stage DCF)" if fade else
                                "close enough to go straight to terminal growth"))
    V["comps_pe_weight"] = (spec.profile.get("comps_pe_weight", 0.5),
                            f"{spec.profile.get('label', 'Industry')} convention: P/E vs EV / LTM EBITDA in the comps "
                            "value (EV/EBITDA is skipped automatically when peers' EBITDA is not available)")
    cs = spec.captive.get("share") or 0.0
    if spec.captive.get("on_face"):
        q_cs = getattr(it, "captive_share", None) if it.available else None
        cs = q_cs or cs
        V["captive_share"] = (cs, f"{spec.captive['segment']} debt = {cs:.1%} of total debt on the "
                                  f"{(it.period_end if q_cs else spec.period_ends[-1]):%b %-d, %Y} balance sheet (the "
                                  "filer's own columns): it funds financing receivables, so it is excluded from net "
                                  "debt, and its interest is an operating cost inside segment profit")
    else:
        V["captive_share"] = (cs, (f"{spec.captive['segment']} interest = {cs:.0%} of total interest "
                                   f"(FY{spec.fy_hist[-1]} segment note): that share of debt funds financing "
                                   "receivables, so it is excluded from net debt and its interest is inside segment "
                                   "profit; replace with the finance arm's debt from the 10-K supplemental "
                                   "consolidating data") if cs else "No captive finance arm")
    spec.valuation = V


def _number_forms(x: float, kind: str) -> list:
    """How a figure can be written in a press release."""
    if kind == "pct":
        p = x * 100
        return [f"{p:.1f}%", f"{p:.0f}%", f"{p:g}%", f"{p:.1f} %", f"{p:.0f} percent", f"{p:g} percent"]
    if kind == "mm":
        b = x / 1000
        return [f"{x:.0f}", f"{b:.1f} billion", f"{b:.2f} billion", f"{b:g} billion", f"{b:.1f}b", f"{b:.2f}b",
                f"{x:,.0f}".replace(",", "")]
    return [f"{x:.2f}", f"{x:g}", f"{x:.1f}"]


_EPS_LABEL = re.compile(r"(?:non-gaap|adjusted)\s+(?:diluted\s+)?(?:net\s+income|earnings|eps)"
                        r"(?:\s+per\s+(?:diluted\s+)?(?:common\s+)?share)?(?:\s*\(\d\))?", re.I)
_EPS_SEQ = re.compile(r"[\s|]*((?:\$?[\s|]*\(?\d{1,3}\.\d{2}\)?[\s|]*){2,5})")  # table cells: '| $ | 3.81 | |'


def _in_range(text: str, start: int, end: int) -> bool:
    """A figure written as one end of a range ('$13.11 - $13.19', '46.1 to 46.4 billion') is guidance."""
    return bool(re.search(r"(?:-|–|—|\bto)\s*\$?\s*$", text[max(0, start - 8):start]) or
                re.match(r"\s*(?:-|–|—|to\b)\s*\$?\s*\d", text[end:end + 8]))


def _reported_eps_line(text: str) -> tuple:
    """The release's own reconciliation line, e.g. 'Non-GAAP diluted net income per share $ 3.81 $ 2.78 $ 12.52
    $ 10.20' (quarter, prior-year quarter, year or year to date, prior year). A line holding a range is guidance and
    is skipped; the reading that repeats most wins."""
    reads = []
    for m in _EPS_LABEL.finditer(text):
        seq = _EPS_SEQ.match(text, m.end())
        if not seq or re.match(r"[\s|]*(?:-|–|—|to\b)\s*\$?\s*\d", text[seq.end(1):seq.end(1) + 10]):
            continue
        reads.append(tuple(-float(x.strip("()")) if x.startswith("(") else float(x)
                           for x in re.findall(r"\(?\d{1,3}\.\d{2}\)?", seq.group(1))))
    for n in (4, 5, 2):  # five: quarter, prior-year quarter, prior quarter, year, prior year (ADBE)
        cands = [r for r in reads if len(r) == n]
        if cands:
            return max(set(cands), key=cands.count)
    return ()


def refresh_ytd_adj_eps(spec: ModelSpec) -> None:
    """An estimated year-to-date adjusted EPS follows the definition switches as they stand, not as they were when
    the defaults were built (PM round 6, ADBE: 9M estimated at $17.00 with stock comp still deducted after the
    analyst excluded it; the release's guides imply $18.15)."""
    it = spec.interim
    if not it.available:
        return
    q_g, fy_g, q_eps = spec.q_guide, (spec.guidance or {}).get("adj_eps"), getattr(spec, "q_eps", None) or []
    fixed = None
    if q_eps and it.ytd_cur.get("adj_eps_est"):
        # the quarters the company reported, each from its own release: a fact, and gate 47 tests it against the
        # guides independently
        fixed = round(sum(q_eps), 2)
        note = (f"{it.label_cur} adjusted EPS ${fixed:.2f} = the {len(q_eps)} reported quarters "
                f"({' + '.join(f'${x:.2f}' for x in q_eps)}, each quarter's release)")
    elif q_g and fy_g and it.ytd_cur.get("adj_eps_est"):
        # else the full-year guide less the guided last quarter (ADBE: $24.45-24.50 less $6.30-6.35 = $18.15); gate 47
        # cannot test a figure derived from the guides it would test against
        fixed = round(sum(fy_g) / 2 - sum(q_g) / 2, 2)
        note = (f"{it.label_cur} adjusted EPS ${fixed:.2f} = the full-year guide midpoint ${sum(fy_g) / 2:.2f} less "
                f"the Q4 guide midpoint ${sum(q_g) / 2:.2f} (no release gives the year to date or every quarter)")
    if fixed is not None:
        it.ytd_cur["adj_eps"] = it.ytd_cur["adj_eps_model"] = fixed
        spec.flags = [f for f in spec.flags if not f.startswith(f"{it.label_cur} adjusted EPS $")] + [note]
    for tag, d in (("cur", it.ytd_cur), ("prior", it.ytd_prior)):
        est = d.get("adj_eps_est")
        if not est or d.get("ni_attr") is None or (tag == "cur" and fixed is not None):
            continue
        t_, sh = est
        sp, oi = d.get("seg_profit"), d.get("opinc")
        repo = (sp - oi - _nz(d.get("amort")) - _nz(d.get("sbc"))) if sp is not None and oi is not None and \
            not spec.seg_after_charges else _nz(d.get("oth_opex"))
        items = (_nz(d.get("amort")) * spec.adj_eps_amort_share + _nz(d.get("sbc")) * spec.adj_eps_ex_sbc +
                 max(repo, 0.0) * spec.adj_eps_ex_repo - _nz(d.get("adj_eps_oneoff")) * spec.adj_eps_ex_nonrec)
        ngt = _nz((spec.drivers.get("adj_tax") or ([0.0], ""))[0][0]) if tag == "cur" else \
            _nz(spec.adj_eps_tax_hist[2])
        if ngt > 0 and d.get("pretax") is not None:
            v = ((d["pretax"] + items) * (1 - ngt) - _nz(d.get("nci"))) / sh
        else:
            v = (d["ni_attr"] + items * (1 - t_)) / sh
        d["adj_eps"] = d["adj_eps_model"] = round(v, 2)


def _quarter_eps(text: str):
    """The quarter's adjusted EPS from a release: the first figure of a reconciliation line with three or more
    columns (quarter, prior-year quarter, ...); two-figure lines are guide ranges (ADBE's '$6.30 | $6.35')."""
    reads = []
    for m in _EPS_LABEL.finditer(text or ""):
        seq = _EPS_SEQ.match(text, m.end())
        if seq:
            r = [float(x) for x in re.findall(r"\d{1,3}\.\d{2}", seq.group(1))]
            if len(r) >= 3:
                reads.append(r[0])
    return max(set(reads), key=reads.count) if reads else None


def _next_quarter_eps_guide(text: str) -> tuple:
    """(low, high) non-GAAP EPS target for the next quarter from a release's targets table ('fourth quarter FY2026
    targets ... Non-GAAP $6.30 to $6.35'), or ()."""
    m = re.search(r"(?i)(first|second|third|fourth) quarter[^\n]{0,60}(targets?|outlook|guidance)|"
                  r"\bQ[1-4]\b[^\n]{0,30}(targets?|outlook|guidance)", text)
    if not m:
        return ()
    blk = text[m.end():m.end() + 1500]
    stop = re.search(r"(?i)(fiscal year|full[- ]year|FY ?20\d\d)[^\n]{0,30}(targets?|outlook|guidance)", blk)
    blk = blk[:stop.start()] if stop else blk
    g = re.search(r"(?i)non-gaap[^$\n]{0,60}\$\s?(\d{1,3}\.\d{2})\s*(?:to|-|–)\s*\$?\s?(\d{1,3}\.\d{2})", blk)
    return (float(g.group(1)), float(g.group(2))) if g else ()


def verify_release_figures(spec: ModelSpec, data) -> None:
    """'Verify' means verify (critique C13): every figure the extraction step read from the releases is searched
    for in the release text itself, as a whole number. A reported figure found only as one end of a range is
    guidance, not a result (CRM: the FY27 guide '$13.11 - $13.19' was read as FY26 and FY25 adjusted EPS); the
    release's own reconciliation line then replaces it."""
    texts = {k: t for k, (_, t) in (getattr(data, "release_texts", None) or {}).items() if t}
    if not texts:
        return
    norm = re.sub(r"(?<=\d),(?=\d{3})", "", " ".join(texts.values())).lower()

    def status(x, kind, reported=False):
        forms = [f"{x:.2f}"] if reported else list(dict.fromkeys(_number_forms(x, kind)))
        tail = r"(?![\d%])" if kind == "eps" else r"(?!\d)"
        hits = [m for f in forms for m in re.finditer(r"(?<![\d.])" + re.escape(f.lower()) + tail, norm)]
        if not hits:
            return False
        if reported and all(_in_range(norm, m.start(), m.end()) for m in hits):
            return "range"
        return True

    kinds = {"sales": "mm", "fcf": "mm", "organic": "pct", "margin": "pct", "adj_eps": "eps"}
    out = []
    for k, (lo, hi) in (spec.guidance or {}).items():
        kind = kinds.get(k, "eps")
        for label, x in (("low", lo), ("high", hi)):
            if isinstance(x, (int, float)):
                out.append((f"Guidance {k} ({label})", x, status(x, kind)))
    it = spec.interim
    rec, line = {}, {}
    if texts.get("fy_release"):
        r = _reported_eps_line(texts["fy_release"])
        if len(r) in (4, 5):
            rec.update(fy=r[-2], fy_prior=r[-1])
            line.update(fy=r, fy_prior=r)
    if texts.get("q_release") and it.available:
        r = _reported_eps_line(texts["q_release"])
        if len(r) in (4, 5):
            rec.update(ytd_cur=r[-2], ytd_prior=r[-1])
            line.update(ytd_cur=r, ytd_prior=r)
        elif len(r) == 2 and it.fiscal_period == "Q1":
            rec.update(ytd_cur=r[0], ytd_prior=r[1])
    if rec.get("fy") and adj_eps_problem(spec, {k: rec.get(k) for k in ("fy", "fy_prior")}):
        rec = {k: v for k, v in rec.items() if k.startswith("ytd_")}  # implausible against GAAP: not used
    spec.reported_eps = rec
    adj = dict((spec.extracted or {}).get("adjusted_eps") or {})
    fixed = {}
    for k in ("fy", "fy_prior", "ytd_cur", "ytd_prior"):
        v = adj.get(k)
        st = status(v, "eps", reported=True) if isinstance(v, (int, float)) else False
        # another column of the same line (CRM: the Q2 quarter's $5.90 read as the six months) is period-shifted
        shifted = isinstance(v, (int, float)) and any(abs(v - x) < 0.005 for x in line.get(k, ()))
        if k in rec and (v is None or abs(v - rec[k]) > 0.005) and (st is not True or shifted):
            fixed[k] = (v, "shifted" if st is True else st, rec[k])
            out.append((f"Adjusted EPS {k}", rec[k], "replaced"))
        elif isinstance(v, (int, float)):
            out.append((f"Adjusted EPS {k}", v, st))
    if fixed:
        adj.update({k: new for k, (_, _, new) in fixed.items()})
        spec.extracted = {**(spec.extracted or {}), "adjusted_eps": adj}
        if adj.get("fy"):
            spec.adj_eps_hist = [None, adj.get("fy_prior"), adj.get("fy")]
            spec.adj_eps_reported = True
        for tag, d in (("cur", it.ytd_cur), ("prior", it.ytd_prior)):
            if f"ytd_{tag}" in fixed and it.available:
                d["adj_eps"] = d["adj_eps_model"] = fixed[f"ytd_{tag}"][2]
                d.pop("adj_eps_est", None)
        spec.flags.append("Adjusted EPS replaced with the release's own reconciliation line: " + "; ".join(
            f"{k} {new:.2f} (extracted {old if old is not None else 'none'}"
            f"{', found only as one end of a range: guidance' if st == 'range' else ', another column of that line: period-shifted' if st == 'shifted' else ', not in the text' if old is not None else ''})"
            for k, (old, st, new) in fixed.items()))
    rates = list(dict.fromkeys(m.group(1) for m in re.finditer(
        r"non-gaap tax (?:rate|provision)[^.%]{0,80}?(\d{1,2}(?:\.\d)?)%", norm)))
    spec.stated_adj_tax = [float(r) / 100 for r in rates]
    # the year to date + the guided last quarter = the full-year guide (PM round 6: ADBE's 9M estimated at $17.00;
    # the $24.45-24.50 year less the $6.30-6.35 Q4 is $18.15): a release with no year-to-date figure gives it this way
    q_g = _next_quarter_eps_guide(texts.get("q_release") or "") if it.available and it.fiscal_period == "Q3" else ()
    spec.q_guide = list(q_g)
    # the year to date as the quarters the company reported (PM round 7: guide less guide is circular)
    n_q = {"Q1": 1, "Q2": 2, "Q3": 3}.get(it.fiscal_period) if it.available else None
    qs = [_quarter_eps(t) for t in list(getattr(data, "prior_q_texts", None) or []) + [texts.get("q_release") or ""]]
    spec.q_eps = qs if n_q and len(qs) == n_q and all(isinstance(x, float) for x in qs) else []
    if rates:  # CRM: "for fiscal 2025 and 2026 ... a projected non-GAAP tax rate of 22.0%; for fiscal 2027 ... 20.5%"
        spec.flags.append(f"The releases state a non-GAAP tax rate ({', '.join(r + '%' for r in rates)}): if adjusted "
                          "EPS taxes non-GAAP pre-tax income at that rate, set adj_eps_tax_hist (history years) and "
                          "drivers.adj_tax (forecast)")
    spec.verification = out
    missing = [f"{i} {v:g}" + (" (only inside a range: guidance?)" if ok == "range" else "")
               for i, v, ok in out if ok is False or ok == "range"]
    if missing:
        spec.flags.append("Release figures not found in the release text as results (verify before relying on "
                          "them): " + ", ".join(missing))


ITEMS_8K = {"1.01": "material definitive agreement (a pending deal?)",
            "2.01": "completion of an acquisition or disposal", "2.03": "new debt obligation",
            "2.05": "exit / restructuring costs", "2.06": "material impairment", "3.02": "unregistered equity sale"}


def post_balance_sheet_events(spec: ModelSpec, data) -> None:
    """8-Ks filed between the balance sheet the forecast opens from and the valuation date (CRM: Contentful and
    Fin closed and Braze was signed in that window), and an accelerated share repurchase still to settle (CRM's
    October ASR settlement changed the share count after the 10-Q). The analyst must enter or dismiss each."""
    it = spec.interim
    bs = it.period_end if it.available else spec.period_ends[-1]
    ev = []
    for f in getattr(getattr(data, "company", None), "filings", None) or []:
        if not str(f.form).startswith("8-K") or not (bs < f.filing_date <= spec.as_of):
            continue
        items = [x.strip() for x in str(f.items or "").split(",") if x.strip()]
        what = [ITEMS_8K[x] for x in items if x in ITEMS_8K]
        if what:
            ev.append({"date": str(f.filing_date), "items": ", ".join(items), "what": "; ".join(what),
                       "url": f.folder + f.primary_doc})
    spec.post_bs_events = ev
    txt = " ".join(t for _, t in (getattr(data, "release_texts", None) or {}).values() if t).lower()
    spec.asr_open = bool(re.search(r"accelerated share repurchase|\basr\b", txt) and
                         re.search(r"final settlement|remaining shares|expected to (?:be )?(?:complete|settle)|"
                                   r"upon (?:final )?settlement", txt))


def set_balance_sheet_timing(spec: ModelSpec) -> None:
    """The forecast's cash flows count from the latest balance sheet (the net-debt date), not the valuation date:
    the share of FY1 cash flow after the 10-Q, by last year's seasonality (DE: Q4 is the big equipment quarter;
    COST: the holiday quarter). Year-to-date buybacks, acquisitions and dividends are already in that balance sheet."""
    it, CF = spec.interim, spec.hist.CF
    if not it.available:
        spec.cf_share_bs = 1.0
        spec.cf_share_note = (f"No 10-Q since the 10-K: all of {spec.fc_labels[0]} is after the "
                              f"{spec.period_ends[-1]:%b %-d, %Y} balance sheet")
        spec.ytd_capital = {}
        return
    days = (spec.fy_end_cur - spec.period_ends[-1]).days or 365
    cal = max(0.0, min(1.0, (spec.fy_end_cur - it.period_end).days / days))
    p = it.ytd_prior
    fcf_fy = _nz(CF["cfo"][2]) + _nz(CF["capex"][2])  # capex is negative in the CF history
    fcf_ytd = None if p.get("cfo") is None else p["cfo"] - abs(_nz(p.get("capex")))
    share, how = cal, "calendar share (no usable seasonality)"
    if fcf_ytd is not None and fcf_fy > 0 and 0.05 < fcf_ytd / fcf_fy < 1.2:
        share = max(0.0, min(1.0, 1 - fcf_ytd / fcf_fy))
        how = (f"last year's seasonality: {it.label_prior} was {fcf_ytd / fcf_fy:.0%} of FY{spec.fy_hist[-1]} "
               f"free cash flow")
    spec.cf_share_bs = round(share, 4)
    spec.cf_share_note = (f"{share:.0%} of {spec.fc_labels[0]} cash flow falls after the {it.period_end:%b %-d, %Y} "
                          f"balance sheet ({how}; calendar share {cal:.0%})")
    c = it.ytd_cur
    spec.ytd_capital = {"buyback": abs(_nz(c.get("buyback"))), "acq": abs(_nz(c.get("acq"))),
                        "divs": abs(_nz(c.get("divs"))), "label": it.label_cur}


def current_roic(spec: ModelSpec, with_goodwill: bool = False) -> Optional[float]:
    """FY0 NOPAT / invested capital, capital measured from the financing side (equity + debt - cash - investments)
    so every operating asset counts (NFLX's content library, COST's negative working capital). Operating ROIC
    excludes goodwill and acquired intangibles (what new organic capital earns, the basis for RONIC);
    with_goodwill includes them (the return on everything paid)."""
    I, B = spec.hist.IS, spec.hist.BS
    try:
        captive = (spec.captive.get("share") or 0.0) if spec.captive else 0.0
        debt = (_nz(B["st_debt"][2]) + _nz(B["cur_ltd"][2]) + _nz(B["ltd"][2])) * (1 - captive)
        ic = (_nz(B["equity_total"][2]) + _nz(B["redeem_nci"][2]) + debt - _nz(B["cash"][2]) - _nz(B["sti"][2])
              - _nz(B["lt_inv"][2]))
        if not with_goodwill:
            ic -= _nz(B["goodwill"][2]) + _nz(B["intang"][2])
        etr = I["tax"][2] / I["pretax"][2] if I["pretax"][2] and I["pretax"][2] > 0 else 0.21
        nopat = _nz(I["opinc"][2]) * (1 - _clip(etr, 0.0, 0.4))
    except (KeyError, IndexError, TypeError):
        return None
    if nopat <= 0:
        return None
    if ic <= 0:  # buybacks above book or negative operating capital: new capital earns more than any cap
        return 0.60
    return nopat / ic


def fy1_sales(spec: ModelSpec) -> Optional[float]:
    """Base-case FY1E sales, computed the way the workbook does (lines and KPIs included)."""
    try:
        tot = 0.0
        for sg in spec.segments:
            for u in sg.units:
                tot += _nz(u.sales[2]) * (1 + u.growth["base"][0] + sg.mna[0])
        return tot + spec.corp_sales[0]
    except (KeyError, IndexError, TypeError):
        return None


def booked_charges(spec: ModelSpec) -> float:
    """Non-segment charges already booked year to date (10-Q): segment profit - EBIT - amortization & stock comp."""
    it = spec.interim
    c = it.ytd_cur if it.available else {}
    if c.get("seg_profit") is None or c.get("opinc") is None:
        return 0.0
    return c["seg_profit"] - c["opinc"] - (_nz(c.get("amort")) + _nz(c.get("sbc"))) * (1 - spec.seg_after_charges)


def floor_booked_charges(spec: ModelSpec) -> bool:
    """FY1 non-segment charges cannot be smaller than what is already booked this year (LLY: $8.8bn of acquired
    IPR&D in 9M 2026 against a full-year assumption of $4.2bn implied a 57% Q4 operating margin)."""
    booked, sales = booked_charges(spec), fy1_sales(spec)
    vals, note = spec.drivers["repo_pct"]
    if not sales or booked <= 0 or vals[0] * sales >= 0.98 * booked:
        return False
    pct = round(booked / sales, 4)
    spec.drivers["repo_pct"] = ([pct] + list(vals[1:]), f"{note}; {spec.fc_labels[0]} floored at the {booked:,.0f}mm "
                                f"of non-segment charges already booked in {spec.interim.label_cur} (10-Q)")
    return True


def balance_guards(spec: ModelSpec) -> list[str]:
    """Capital inputs the balance sheet cannot absorb, whoever set them: acquisitions are cash paid (LLY's analyst
    entered -4.2bn of expensed IPR&D, a phantom inflow that also cut goodwill), and amortization cannot exceed the
    intangibles left (CRM's path ran the balance to -1.0bn). Returns what was changed, for the data notes."""
    D, done = spec.drivers, []
    if "acq" in D:
        vals, note = D["acq"]
        if any(_nz(x) < 0 for x in vals):
            D["acq"] = ([max(_nz(x), 0.0) for x in vals],
                        f"{note}; negative entries set to 0 (cash paid for acquisitions: divestitures and expensed "
                        "acquired IPR&D are not acquisitions)")
            done.append("Negative acquisitions set to 0: the line is cash paid for businesses (divestitures and "
                        "expensed acquired IPR&D do not belong there)")
    if "amort" in D and "acq" in D and spec.hist.BS.get("intang"):
        vals, note = D["amort"]
        share = (D.get("intang_share") or ([0.3] * 6, ""))[0]
        left, new, cut = _nz(spec.hist.BS["intang"][2]), [], False
        for t, a in enumerate(vals):
            left += _nz(D["acq"][0][t]) * _nz(share[t] if t < len(share) else share[-1])
            x = min(_nz(a), max(left, 0.0))
            cut |= x < _nz(a) - 0.5
            new.append(round(x, 1))
            left -= x
        if cut:
            zero = next((spec.fc_labels[t] for t, x in enumerate(new) if x <= 0.5), None)
            last_acq = max((t for t, a in enumerate(D["acq"][0]) if _nz(a) > 0), default=None)
            note = re.sub(r"(?i)[^.;]*\b(retain\w*|recurring|held flat|ongoing)\b[^.;]*(amortization)?[^.;]*[.;]?\s*",
                          "", note).strip(" ;") or note.split(";")[0]
            D["amort"] = (new, f"{note}; capped at the intangibles left (Schedules), so the balance never turns "
                               "negative" + (f": amortization reaches 0 in {zero} because no acquisitions are forecast "
                                             f"after {spec.fc_labels[last_acq] if last_acq is not None else 'the 10-K'} "
                                             "(recurring deal amortization needs recurring drivers.acq)" if zero else ""))
            done.append("Amortization capped at the acquired intangibles left on the balance sheet")
    # what the 10-Q shows was already done this year sets a floor under the full year (CRM's $25bn ASR)
    ytd = spec.ytd_capital or {}
    for key, what in (("buyback", "buybacks"), ("acq", "acquisitions")):
        booked = _nz(ytd.get(key))
        if key in D and booked > 0 and _nz(D[key][0][0]) < 0.98 * booked:
            vals, note = D[key]
            D[key] = ([round(booked)] + list(vals[1:]),
                      f"{note}; {spec.fc_labels[0]} floored at the {booked:,.0f}mm of {what} already made in "
                      f"{ytd.get('label', 'the year to date')} (10-Q)")
            done.append(f"{spec.fc_labels[0]} {what} raised to the {booked:,.0f}mm already made year to date")
    if "debt_iss" in D:  # repayment cannot exceed the debt outstanding (LLY: $50bn repaid against $48.9bn)
        B = spec.hist.BS
        held = _nz(B["st_debt"][2]) + _nz(B["cur_ltd"][2])  # held flat on the BS: long-term debt takes repayments
        debt = held + _nz(B["ltd"][2])
        vals, note = D["debt_iss"]
        new, cut = [], False
        for x in vals:
            y = max(_nz(x), held - debt)
            cut |= y > _nz(x) + 0.5
            new.append(round(y, 1))
            debt += y
        if cut:
            D["debt_iss"] = (new, f"{note}; repayment capped at the long-term debt outstanding")
            done.append("Debt repayment capped at the long-term debt outstanding")
    return done


def fy1_ebitda(spec: ModelSpec):
    """Base-case FY1E EBITDA, computed from the drivers the same way the workbook does."""
    D = spec.drivers
    try:
        seg_sales = [_nz(sg.sales[2]) * (1 + sg.growth["base"][0] + sg.mna[0]) for sg in spec.segments]
        sales = sum(seg_sales) + spec.corp_sales[0]
        segp = sum(x * sg.margin["base"][0] for x, sg in zip(seg_sales, spec.segments)) + sales * spec.corp_cost_pct[0]
        amort = D["amort"][0][0]
        ebit = segp - (amort + sales * D["sbc_pct"][0][0]) * (1 - spec.seg_after_charges) - sales * D["repo_pct"][0][0]
        return ebit + sales * D["da_pct"][0][0] + amort
    except (KeyError, IndexError, TypeError):
        return None


def no_peer_fallback(spec: ModelSpec, eps_est: tuple = (None, "")):
    """Without peers the Comps tab shows the company's own consensus P/E and gets no weight."""
    eps, note = eps_est or (None, "")
    if not eps:
        eps = spec.adj_eps_hist[2] or spec.adj_eps_estimate(2)
        note = f"FY{spec.fy_hist[-1]} adjusted EPS (no consensus available)"
    spec.peers = [{"ticker": spec.ticker, "name": f"{spec.name} (self — add peers)", "overlap": "n/a",
                   "price": spec.price, "eps": eps,
                   "source": (note or "No estimate") + "; placeholder until peers are added"}]
    w = spec.valuation
    moved = w["w_comps"][0]
    w["w_pg"] = (round(w["w_pg"][0] + moved / 2, 4), w["w_pg"][1])
    w["w_exit"] = (round(w["w_exit"][0] + moved / 2, 4), w["w_exit"][1])
    w["w_comps"] = (0.0, "No peers selected: weight moved to the two DCF methods")


def defaults_peers_sotp(spec: ModelSpec, data):
    spec.peers = list(data.peers or [])
    spec.eps_est = data.market.get("eps_est") or (None, "")
    spec.short_interest = dict(data.market.get("short") or {})
    if not spec.peers:
        no_peer_fallback(spec, data.market.get("eps_est"))
    evs = sorted(p["ev_ebitda"] for p in spec.peers if p.get("ev_ebitda") and 3 < p["ev_ebitda"] < 60)
    peer_med = evs[len(evs) // 2] if evs else None
    # anchor on the company's own multiple so the default SOTP starts near market value;
    # the analyst then differentiates segments against their peer groups
    base = _clip(_r(spec.valuation["exit_mult"][0], 0.5), 6.0, 30.0)
    ai_mult = data.sotp_multiples or {}
    for sg in spec.segments:
        m = _match(ai_mult, sg.name)
        b = _clip(_r(m, 0.5), 4.0, 40.0) if m else base
        sg.sotp = (b - 2, b, b + 2)
        sg.sotp_note = _match_text(data.sotp_notes, sg.name) or (
            f"Company EV/EBITDA ± 2x; peer median today ≈ {peer_med:.1f}x (Yahoo)" if peer_med
            else "Company EV/EBITDA ± 2x")
    spec.corp_multiples = (base, base, base)


def sources(spec: ModelSpec, data) -> list:
    u = data.urls
    fy = spec.fy_hist[-1]
    out = [f"10-K FY{fy}: {u['annual']}"]
    st = data.statements
    pick = []
    for label, rx in (("operations", r"OPERATIONS|INCOME|EARNINGS"), ("balance sheet", r"BALANCE|FINANCIAL POSITION|CONDITION"),
                      ("cash flows", r"CASH FLOW")):
        for rpt in st:
            if rpt["category"] == "Statements" and re.search(rx, rpt["name"].upper()) and \
                    not re.search(r"COMPREHENSIVE|PARENTHETICAL", rpt["name"].upper()):
                pick.append(f"{rpt['file']} ({label})")
                break
    if pick:
        out.append("Statements (XBRL): " + ", ".join(pick) + " in the same filing folder")
    if u.get("fy_release"):
        out.append(f"FY{fy} earnings release: {u['fy_release']}")
    if u.get("q_release"):
        out.append(f"{spec.interim.quarter_label} earnings release: {u['q_release']}")
    if u.get("quarterly"):
        out.append(f"{spec.interim.quarter_label} 10-Q: {u['quarterly']}")
    out.append("Market data: Yahoo Finance via yfinance (share price, 10-yr Treasury yield, beta, peer prices and "
               "consensus EPS)")
    out.append("Model structure: Anthropic financial-services skills (3-statement model → DCF, comps, sum of the parts)")
    out.append("Approximate, not from filings: risk-free rate, beta, ERP, cost of debt, exit and SOTP multiples, "
               "peer estimates — flagged on Assumptions")
    if data.ai_used:
        out.append("AI-assisted (verify): guidance, adjusted EPS and organic growth read from the earnings releases; "
                   "peer list and thesis draft")
    return out


def template_thesis(spec: ModelSpec) -> dict:
    h = spec.hist
    I = h.IS
    s = spec.short
    g = I["sales"][2] / I["sales"][1] - 1 if I["sales"][1] and I["sales"][2] else 0
    m = _nz(I["opinc"][2]) / I["sales"][2] if I["sales"][2] else 0
    fcf = _nz(h.CF["cfo"][2]) + _nz(h.CF["capex"][2])
    conv = fcf / I["ni_attr"][2] if I["ni_attr"][2] else 0
    seg_txt = ", ".join(sg.name for sg in spec.segments)
    return {
        "stance": "neutral on",
        "recommendation": f"[Analyst to complete] State the call on {s} at ${spec.price:,.2f}: long / short / "
                          "avoid, sizing, and the time horizon.",
        "points": [
            f"1. Growth: FY{spec.fy_hist[-1]} sales grew {g:+.1%}; the model's base case is set on the Assumptions tab.",
            f"2. Quality: GAAP operating margin {m:.1%} and FCF conversion {conv:.0%} of net income in FY{spec.fy_hist[-1]}.",
            f"3. Mix: segments modelled — {seg_txt}.",
            "4. [Analyst to complete] The catalyst or variant view the market is missing.",
        ],
        "model_says": [
            "• Compare the DCF value with the price: the gap is what the thesis has to explain.",
            "• Check which driver moves value most (DCF sensitivity grids) and stress it in the Bear case.",
            "• [Analyst to complete] What the SOTP and comps say versus the DCF.",
        ],
        "risks": [
            "• [Analyst to complete] Execution / margin risk.",
            "• Leverage and capital allocation: see net debt / EBITDA on the Summary tab.",
            "• Multiple risk: peer multiples can compress; the comps and SOTP borrow them.",
            "• [Analyst to complete] Macro, regulatory or competitive risk.",
        ],
        "monitor": [
            "• Next quarterly results vs guidance (Interim tab).",
            "• [Analyst to complete] Company-specific events and dates.",
        ],
        "about": "Template draft (no AI key configured): replace the bracketed text with the analyst's view.",
    }
