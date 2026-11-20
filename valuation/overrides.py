"""Analyst overrides: one JSON format for every judgment layered on top of the filings.

The deterministic pipeline (SEC XBRL + market data) fills a ModelSpec with defaults. The
judgment a buy-side analyst adds (scenario paths and why, announced M&A, guidance,
debt-like items, hidden stakes, SOTP splits, peer guidance, the thesis) is expressed as
an overrides document:

    export(spec)          -> dict   the current spec in override format (edit it, don't start blank)
    apply(spec, doc)      -> list   merge a (partial) document into the spec; returns warnings
    SCHEMA_DOC                      human/agent-readable description of every field

The in-app AI analyst, an external agent (Claude Code, ChatGPT), or a person editing JSON
all produce the same format, so a model can be rebuilt reproducibly from
(ticker, date, overrides.json).
"""
from __future__ import annotations

import json
import re
from datetime import date

from . import capital

from .spec import (LineSpec, ModelSpec, SegSpec, adj_eps_problem, align_lines, balance_guards, guidance_problem,
                   kpi_growth, ytd_adj_problem)
from .spec import eps_txt as _eps_txt, guide_txt as _guide_txt

N = 6  # forecast years

DRIVER_KEYS = ["amort", "sbc_pct", "repo_pct", "other_nonop", "int_yield", "int_rate", "etr", "nci_ni",
               "cogs_pct", "da_pct", "capex_pct", "dso", "dio", "oca_pct", "dpo", "accrued_pct", "other_op",
               "acq", "intang_share", "debt_iss", "buyback", "buyback_px", "shares_issued", "stock_proc",
               "dps0", "div_growth", "nonrec"]
VAL_KEYS = ["rf", "beta", "erp", "kd", "tax_wacc", "g", "exit_mult", "w_pg", "w_exit", "w_sotp", "w_comps",
            "peer_discount", "exit_pe", "other_stake", "nci_value", "fade_years", "comps_pe_weight", "captive_share",
            "ronic", "stake_haircut"]
LABEL_KEYS = ["peer_discount", "other_stake", "debt_like", "debt_like_offset", "debt_like_net", "repo"]
GUIDE_KEYS = ["sales", "organic", "margin", "adj_eps", "fcf", "net_income"]
THESIS_KEYS = ["stance", "recommendation", "points", "model_says", "risks", "monitor", "about",
               # Part F: what the price implies, variant view, dated catalysts, the thesis-breaking risk
               "price_implies", "variant_view", "catalysts", "breaker", "sensitivities", "consensus_notes",
               # a short (PM round 4): when the premium last compressed and on what; the long leg of a pair
               "short_evidence", "pair"]
# every warning gate and open check is resolved or explained before a rating is issued (PM round 2)
EXPLAIN_KEY = "explanations"
DEAL_KEYS = {"name": str, "kind": str, "status": str, "close": str, "note": str, "in_forecast": bool,
             **{k: float for k in ("probability", "consideration", "ebitda", "multiple", "synergies",
                                   "integration_cost", "stake", "cash_received", "separation_cost", "stranded_cost",
                                   "break_fee", "financing_debt", "revenue", "margin", "premium")}}
SOURCE_PREFIX = "Analyst source: "
FLAG_RX = re.compile(r"^Analyst \((?:[^()]|\([^()]*\))*\): ")  # "Analyst (AI analyst (gpt-5.5) + review (gpt-5.5)): "

SCHEMA_DOC = """\
Overrides JSON (every key optional; percentages as decimals, money in $ millions, per-share in $).
Arrays of 6 numbers run over the six forecast years (first = the current fiscal year, FY1E).

{
  "meta": {"author": "who/what produced this", "summary": "2-3 sentence view", "sources": ["url", ...]},
  "market": {"price": {"value": 219.50, "note": "close on the valuation date"},
             "basic_shares": {"value": 635.0, "note": "..."}, "dilutive": {"value": 3.4, "note": "..."}},
  "scenario": 2,                                  # 1 bear, 2 base, 3 bull (Assumptions!C4)
  "seg_after_charges": 0,                         # 1 if segment profit already deducts amortization & SBC
  "adj_eps_ex_sbc": 0,                            # 1 if the company's adjusted / non-GAAP EPS also excludes stock
                                                  # comp (most software); SBC stays a cost in EBIT and the DCF
  "adj_eps_ex_repo": 0,                           # 1 if it also excludes non-segment charges (restructuring,
                                                  # acquisition / divestiture costs: ETN); they stay in EBIT
  "segments": {
    "<segment name exactly as in the model>": {
      "rename": "optional display name",
      "organic_last": 0.11,                       # last fiscal year's organic growth (earnings release)
      "growth": {"bear": [6], "base": [6], "bull": [6]},   # ORGANIC growth by year
      "margin": {"bear": [6], "base": [6], "bull": [6]},   # segment profit / segment sales
      "mna": {"values": [6], "note": "deal names, close dates, annualization"},  # M&A + FX contribution
      "sotp": {"multiples": [bear, base, bull], "note": "peer group the multiple is set against"},
      "sotp_split": [                             # value a segment in pieces (e.g. a pending spin-off)
        {"name": "Solstice Advanced Materials", "ebitda": 1000, "multiples": [8, 10, 13], "note": "Form 10"},
        {"name": "UOP & process technology", "remainder": true, "multiples": [12, 14, 16], "note": "stays"}],
      "use_lines": true,                          # build the segment's sales from its product / region lines
      "lines": {                                  # lines listed in the brief (XBRL); growth replaces the
        "<line name>": {                          # segment's organic path, which becomes their weighted sum
          "growth": {"bear": [6], "base": [6], "bull": [6]}, "note": "why (launch curve, LOE, pricing)",
          "kpi": {...}},                          # optional volume x price build, see below
        "<new line>": {"history": [3], ...}},     # define lines yourself (history must add up to the
                                                  # segment's reported sales within 3%, e.g. from the release)
      "kpi": {                                    # volume x price build for a segment without lines:
        "volume": {"name": "Warehouses (year-end)", "unit": "#", "history": [3],
                   "growth": {"bear": [6], "base": [6], "bull": [6]}},
        "price": {"name": "Net sales per warehouse", "unit": "$mm", "history": [3],
                  "growth": {"bear": [6], "base": [6], "bull": [6]}},
        "note": "source of the KPI history"}       # revenue growth = (1 + volume g) x (1 + price g) - 1
    }
  },
  "notes": {"growth": "why the organic paths look like this (Assumptions column L)",
            "margin": "why the margin paths look like this"},
  "corporate": {"sales": [6], "sales_note": "...", "cost_pct": [6], "cost_note": "...",
                "multiples": [12, 12, 12], "multiple_note": "..."},
  "drivers": {"<key>": {"values": [6], "note": "basis"}},   # keys below; dps0 takes 1 value, div_growth 5
  "valuation": {"<key>": {"value": 0.0425, "note": "basis"}},
  "labels": {"peer_discount": "Conglomerate discount", "other_stake": "Quantinuum stake (SOTP only)",
             "debt_like": "Asbestos-related liabilities", "debt_like_offset": "Insurance recoveries for asbestos"},
  "debt_like": {"liab": 1243, "offset": 166, "liab_fy": 1325, "offset_fy": 171, "note": "source"},
  "guidance": {"date": "July 24, 2025", "sales": [lo, hi], "organic": [lo, hi], "margin": [lo, hi],
               "margin_basis": "gaap | adjusted | segment (what the margin guide measures)",
               "adj_eps": [lo, hi], "fcf": [lo, hi]},
  "adjusted_eps": {"fy": 9.89, "fy_prior": 9.52, "ytd_cur": 5.26, "ytd_prior": 4.74},   # company-reported
                                                  # non-GAAP EPS only; null if it reports none (the model computes it)
  "organic": {"fy_total": 0.03, "q_total": 0.05, "q_by_segment": {"<segment>": 0.06}},
  "peers": [{"ticker": "GE", "name": "GE Aerospace", "overlap": "Aerospace", "price": 275.20,
             "eps": 5.70, "source": "Q2 2025 release: FY2025 adj. EPS guide $5.60-5.80"}],
  "adj_eps_ex_nonrec": 1,                        # 0 if company non-GAAP EPS INCLUDES one-off gains (CRM)
  "adj_eps_amort_share": 1.0,                     # share of acquired amortization the company excludes (SHW < 1)
  "adj_eps_tax_hist": [null, 0.22, 0.215],        # company non-GAAP tax rate by history year when it taxes non-GAAP
                                                  # income at one rate (CRM); forecast rate in drivers.adj_tax
  "deals": [{"name": "<target>", "kind": "acquisition|separation", "status": "pending|closed", "close": "YYYY-MM-DD",
             "probability": 0.8, "consideration": 1000, "ebitda": null, "revenue": 200, "margin": 0.1,
             "premium": 0.35, "multiple": 12.0, "synergies": 10,
             "integration_cost": 30, "break_fee": -40, "in_forecast": false,
             "stake": 0.501, "cash_received": 0, "separation_cost": 0, "stranded_cost": 0,
             "note": "every figure from the deal 8-K / proxy / target's filings, with its source; a public "
                     "target's premium is the offer over its undisturbed price (its standalone value)"}],
                                                  # (illustrative numbers: never copy them)
  "scenario_probs": [0.25, 0.5, 0.25],            # bear / base / bull
  "thesis": {"recommendation": "one sentence supporting the call; the call itself (Long / Short / Neutral, bias) is "
                               "generated from the rule — never state a different one", "points": ["1. ...", ...], "model_says": ["• ..."], "risks": ["• ..."],
             "monitor": ["• ..."], "about": "...",
             "price_implies": "one sentence from the reverse DCF", "variant_view": "We differ on X because Y; we "
             "will know by Z", "catalysts": ["Feb 2027 Q4 print: cRPO organic growth >= 12%"],
             "breaker": "the one risk that breaks the thesis, quantified", "sensitivities": "the two inputs",
             "consensus_notes": "why model vs consensus differs by > 3%",
             "explanations": {"gate 8": "why each warning / open check left is acceptable (key from the checker)"},
             # model numbers in any thesis text as placeholders: {fv} {dcf_pg} {dcf_exit} {comps} {sotp} {bear}
             # {base} {bull} {pw} {etr} {alpha} {irr} {wacc} {ke} {pe} {exit_pe} {ret3y} {req_margin} {req_g}
             # {gap} {irr_spread} {p_bear_implied} {p_bear_long} {peer_discount_pe} {ret3y_pw} {g} {beta};
             # {irr} = long-run IRR at the price (reverse DCF), {ret3y} = 3-year return (base), {ret3y_pw} its
             # probability-weighted version, {p_bear_implied} = the bear probability the price implies
             "rating_override": {"rating": "Long", "reason": "falsifiable reason, only to override the rule"},
             "short_evidence": "a short only: when the premium last compressed, by how much, on what (dated)",
             "pair": "a short of a low-volatility compounder only: the long leg (ticker or ETF) and why"},
  "flags": ["anything the PM must know that the model cannot show"]
}

Driver keys: nonrec ($mm non-recurring gains (+) / losses (-) in pre-tax income, e.g. mark-to-market gains on
equity stakes or debt extinguishment: in GAAP EPS, excluded from adjusted EPS; FY1 = what is already booked
year to date), amort ($mm acquired-intangible amortization), sbc_pct, repo_pct (non-segment charges or,
for gross-profit segments, opex below segment profit), other_nonop ($mm pension & other), int_yield,
int_rate, etr, nci_ni ($mm), cogs_pct, da_pct (D&A excl. acquired amortization), capex_pct, dso, dio,
oca_pct, dpo, accrued_pct, other_op ($mm other operating cash items), acq ($mm cash paid for acquisitions, >= 0:
never divestitures or expensed acquired IPR&D, which is a charge in repo_pct), intang_share,
debt_iss ($mm net issuance), buyback ($mm), buyback_px ($/share), shares_issued (mm), stock_proc ($mm),
dps0 ($ DPS in FY1E, 1 value), div_growth (5 values, FY2E-FY6E).
Valuation keys: rf, beta, erp, kd (pre-tax), tax_wacc, g (terminal growth), exit_mult (EV/EBITDA),
w_pg, w_exit, w_sotp, w_comps (weights, sum to 1), peer_discount (positive = discount to peer median
P/E), exit_pe (3-year exit P/E), other_stake ($mm stakes outside EBITDA), nci_value ($mm), fade_years
(0-10: years after FY6E over which growth fades to the terminal rate - a three-stage DCF for businesses
still growing well above GDP), comps_pe_weight (0-1: weight of P/E vs EV / LTM EBITDA in the comps value),
captive_share (0-0.99: share of total debt that funds a captive finance arm's receivables - excluded from
net debt, its interest inside segment profit), ronic (return on new invested capital: terminal reinvestment =
g / ronic; between WACC and current ROIC for a moat, about WACC when commoditizing, never above ROIC without an
argument naming ROIC), stake_haircut (0-0.9, default 0.25). A decline still running in FY6E (a drug's loss of
exclusivity, a product run-off) is finished in the fade, not proxied by a lower terminal growth: top-level
"fade_path": {"values": [growth by fade year, up to 10], "note": "source"} with fade_years at least that long;
terminal growth stays at the house rate. rf and erp are set once per valuation date for
every name and cannot be changed; beta moves at most ±0.10 from the method value (with a reason); kd at most
±0.75pt from the rating-based value. exit_mult is derived (DCF!C53) unless you set it; if you do, keep it
within ±20% of the perpetuity-implied multiple. exit_pe is not used: the 3-year exit P/E is derived.
"""


# plausible ranges: values outside are rejected with a warning (e.g. 5 meant as 5% instead of 0.05)
BOUNDS = {
    "growth": (-0.6, 1.5), "margin": (-1.0, 0.95), "mna": (-0.6, 1.0), "multiples": (-5.0, 80.0),
    "sbc_pct": (0, 0.4), "repo_pct": (-0.3, 0.5), "adj_tax": (0, 0.4), "int_yield": (0, 0.15), "int_rate": (0, 0.2), "etr": (-0.1, 0.5),
    "cogs_pct": (0, 1.5), "da_pct": (0, 0.6), "capex_pct": (0, 0.8), "dso": (0, 400), "dio": (0, 600),
    "oca_pct": (0, 1.0), "dpo": (0, 600), "accrued_pct": (0, 1.5), "intang_share": (0, 1), "div_growth": (-0.95, 1.0),  # special dividends
    "rf": (0, 0.15), "beta": (0.1, 3.5), "erp": (0.02, 0.1), "kd": (0, 0.2), "tax_wacc": (0, 0.5), "g": (-0.02, 0.06),
    "fade_years": (0, 10), "comps_pe_weight": (0, 1), "captive_share": (0, 0.99), "kpi_growth": (-0.6, 3.0),
    "exit_mult": (1, 80), "w_pg": (0, 1), "w_exit": (0, 1), "w_sotp": (0, 1), "w_comps": (0, 1),
    "peer_discount": (-1.0, 0.9), "exit_pe": (1, 150), "ronic": (0.02, 0.6), "stake_haircut": (0, 0.9),
}
# a review pass may not move valuation inputs toward the share price (critique C13: no goal-seeking)
PRICE_RX = re.compile(r"(?i)\b(closer to|toward[s]?|narrow\w*|reduc\w* the gap|clos\w* the gap|in line with) "
                      r"(the )?(gap|price|market|share price|current price)|market[- ]implied|implied by the price|"
                      r"to the (share |current )?price")
JUDGMENT_VAL = {"beta", "erp", "kd", "g", "exit_mult", "ronic", "fade_years", "peer_discount", "w_pg", "w_exit",
                "w_sotp", "w_comps", "comps_pe_weight"}
# an input set to produce an output: the skew, the gap, the rating (PM round 6: the bear "made harsher to reduce the
# upside/downside skew")
TARGET_RX = re.compile(r"(?i)\b(skew|upside[/ -]downside|(reduce|narrow|shrink|cut|lower)\w* (the )?(upside|gap|"
                       r"spread)|to (keep|hold|reach|get|land|move) (the |a )?(rating|call)|so (that )?the (rating|call)|"
                       r"(clear|meet|stay (within|inside)) the (±?15%|band|threshold)|whatever gap remains|"
                       r"(residual|remaining) (gap )?is the (explicit )?variant view|"
                       r"to make (the )?(terminal )?reinvestment( rate)? consistent|"
                       r"so (that )?(the )?(terminal )?reinvestment (fits|matches|is consistent))")
# scenario risk named as the reason for a non-scenario input
RISK_ONCE = {"beta", "fade_years"}  # terminal growth has its own rule (the house rate)
RISK_RX = re.compile(r"(?i)\b(disrupt\w*|cannibali[sz]\w*|AI[- ](risk|substitution|native|threat|competition)|"
                     r"structural (decline|risk|threat)|decline (risk|case)|competitive (threat|risk|pressure)|"
                     r"uncertain\w*|bear case|downside risk|risk premium|execution risk|obsolescen\w*|"
                     r"substitution|durability risk|(pricing|competitive|AI|freemium|margin) pressure|saturation|"
                     r"commoditi[sz]\w*)")
MEASURE_RX = re.compile(r"(?i)\b(levera\w*|capital structure|business mix|mix shift|regression|standard error|"
                        r"listing|trading history|spin-?off|peer set|unlever\w*|relever\w*)")
POST_BS_RX = re.compile(r"(?i)\b(after the balance sheet|post-balance|clos(es|ed|ing) (on|in) (q[1-4]|the fourth|"
                        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*)|expected to close)")


def _inside(key: str, vals) -> bool:
    lo, hi = BOUNDS.get(key, (float("-inf"), float("inf")))
    return all(lo <= x <= hi for x in vals)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _vec(v, n=N):
    if isinstance(v, list) and len(v) == n and all(_num(x) for x in v):
        return [float(x) for x in v]
    return None


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_segment(spec: ModelSpec, name: str):
    k = _key(name)
    for sg in spec.segments:
        if _key(sg.name) == k:
            return sg
    for sg in spec.segments:
        sk = _key(sg.name)
        if k and (k in sk or sk in k):
            return sg
    return None


# ---------------------------------------------------------------- export
def export(spec: ModelSpec) -> dict:
    """The spec's current judgment layer in override format (round-trips through apply)."""
    segs = {}
    for sg in spec.segments:
        d = {"organic_last": sg.organic_last,
             "growth": {k: list(v) for k, v in sg.growth.items()},
             "margin": {k: list(v) for k, v in sg.margin.items()},
             "mna": {"values": list(sg.mna), "note": sg.mna_note},
             "sotp": {"multiples": list(sg.sotp), "note": sg.sotp_note},
             "history": {"sales": sg.sales, "profit": sg.profit}}
        if sg.lines:
            d["use_lines"] = sg.use_lines
            d["lines"] = {ln.name: {"growth": {k: list(v) for k, v in ln.growth.items()}, "note": ln.note,
                                    "history": ln.sales, **({"kpi": ln.kpi} if ln.kpi else {})} for ln in sg.lines}
        if sg.kpi:
            d["kpi"] = sg.kpi
        if spec.sotp_splits.get(sg.name):
            d["sotp_split"] = spec.sotp_splits[sg.name]
        segs[sg.name] = d
    # who set the judgment layer travels with the file, so a rebuild from it keeps the attribution,
    # the analyst's summary, sources and data notes (the Cover would otherwise say "defaults")
    authors = list(dict.fromkeys(p.split(":", 1)[0] for p in spec.provenance[1:] if ":" in p))
    meta = {"ticker": spec.ticker, "as_of": str(spec.as_of), "author": " + ".join(authors) or "defaults",
            "forecast_years": spec.fc_labels, "history_years": spec.hist_labels}
    if spec.notes.get("analyst_summary"):
        meta["summary"] = spec.notes["analyst_summary"]
    src = [x[len(SOURCE_PREFIX):] for x in spec.sources if x.startswith(SOURCE_PREFIX)]
    if src:
        meta["sources"] = src
    return {
        "meta": meta,
        "market": {"price": {"value": spec.price, "note": spec.price_note},
                   "basic_shares": {"value": spec.basic_shares, "note": spec.basic_note},
                   "dilutive": {"value": spec.dilutive, "note": spec.dilutive_note}},
        "scenario": spec.scenario,
        "seg_after_charges": spec.seg_after_charges,
        "adj_eps_ex_sbc": spec.adj_eps_ex_sbc,
        "adj_eps_ex_repo": spec.adj_eps_ex_repo,
        "adj_eps_ex_nonrec": spec.adj_eps_ex_nonrec,
        "adj_eps_amort_share": spec.adj_eps_amort_share,
        "adj_eps_tax_hist": spec.adj_eps_tax_hist,
        "framework": spec.framework,
        "segments": segs,
        "notes": {"growth": spec.growth_note, "margin": spec.margin_note},
        "corporate": {"sales": list(spec.corp_sales), "sales_note": spec.corp_sales_note,
                      "cost_pct": list(spec.corp_cost_pct), "cost_note": spec.corp_cost_note,
                      "multiples": list(spec.corp_multiples)},
        # every driver the spec carries, not a fixed list (round 4: drivers.adj_tax was dropped, so the saved file
        # rebuilt CRM at $251.98 against the $256.58 it was saved from)
        "drivers": {k: {"values": list(spec.drivers[k][0]), "note": spec.drivers[k][1]}
                    for k in DRIVER_KEYS + sorted(set(spec.drivers) - set(DRIVER_KEYS))},
        "valuation": {k: {"value": spec.valuation[k][0], "note": spec.valuation[k][1]} for k in VAL_KEYS
                      if k in spec.valuation and not (k == "exit_mult" and spec.capital.get("exit_derived"))},
        "deals": list(spec.deals),
        "scenario_probs": spec.scenario_probs,
        "labels": dict(spec.labels),
        "debt_like": dict(spec.debt_like),
        "guidance": {"date": spec.guidance_date, **{k: list(v) for k, v in spec.guidance.items()}},
        # company-reported figures only; null = the model's own definition (re-derived on rebuild)
        "adjusted_eps": ({"fy": spec.adj_eps_hist[2], "fy_prior": spec.adj_eps_hist[1],
                          "ytd_cur": spec.interim.ytd_cur.get("adj_eps"),
                          "ytd_prior": spec.interim.ytd_prior.get("adj_eps")} if spec.adj_eps_reported else None),
        "organic": {"fy_total": spec.organic_total_last, "q_total": spec.q_organic_total,
                    "q_by_segment": dict(spec.q_organic)},
        "peers": [{k: p.get(k) for k in ("ticker", "name", "overlap", "price", "eps", "source", "ev", "ebitda",
                                         "ev_source", "eps_next", "g_next", "rev_g_next", "beta_raw", "de", "snapshot", "pro_forma",
                                         "fy_end", "eps_fy", "ebitda_margin", "share_chg")
                   if p.get(k) is not None or k in ("price", "eps")}
                  for p in spec.peers],
        # the recommendation's call is generated from the rule at build (pipeline.rate): the analyst's own sentence
        # is what round-trips
        "market_data": {"as_of": str(spec.as_of), "own_beta_raw": (spec.capital or {}).get("own_raw"),
                        "rf": spec.market_rf, "pe_hist": [list(x) for x in spec.pe_hist],
                        "short_interest": spec.short_interest or None},
        **({"credit_rating": (spec.capital or {}).get("ratings_input")}
           if (spec.capital or {}).get("ratings_input") else {}),
        "thesis": {k: (spec.thesis.get("recommendation_text") or v if k == "recommendation" else v)
                   for k, v in spec.thesis.items() if k != "recommendation_text"},
        "flags": [FLAG_RX.sub("", f) for f in spec.flags if FLAG_RX.match(f)],
        **({"fade_path": {"values": list(spec.fade_path), "note": spec.fade_path_note}} if spec.fade_path else {}),
        **({"audit_fv": [list(x) for x in spec.audit_fv]} if spec.audit_fv else {}),
        **({"rejected": {f: {"source": r[0], "value": r[1], "why": r[2], "kept": r[3]}
                         for f, r in spec.rejected.items()}} if spec.rejected else {}),
    }


def dumps(spec: ModelSpec) -> str:
    return json.dumps(export(spec), indent=1, default=str)


# ----------------------------------------------------------------- apply
def _market_data(spec: ModelSpec, doc: dict, changed: list) -> None:
    """Market series the saved model was built on, used when today's refetch comes back empty (round 5: a Yahoo
    rate limit dropped COST's own beta and P/E history in the final build, and its comps fell from $611 to $343)."""
    md = doc.get("market_data")
    if not isinstance(md, dict):
        return
    used = []
    cap = spec.capital if spec.capital is not None else {}
    if cap is not None and cap.get("own_raw") is None and isinstance(md.get("own_beta_raw"), (int, float)):
        cap["own_raw"] = float(md["own_beta_raw"])
        used.append("own beta")
    if not spec.pe_hist and isinstance(md.get("pe_hist"), list) and md["pe_hist"]:
        spec.pe_hist = [(int(y), float(v)) for y, v in md["pe_hist"]]
        used.append("own P/E history")
    if spec.market_rf is None and isinstance(md.get("rf"), (int, float)):
        spec.market_rf = float(md["rf"])
        if cap.get("spread") is not None:
            cap["kd"] = round(spec.market_rf + cap["spread"], 4)
        used.append("risk-free rate")
    if not spec.short_interest and isinstance(md.get("short_interest"), dict):
        spec.short_interest = dict(md["short_interest"])
    if used:
        spec.flags.append(f"Market data not returned today ({', '.join(used)}): the values saved with the model are "
                          f"used (as of {md.get('as_of', 'the save')})")
        changed.append("saved market data")


def _credit_rating(spec: ModelSpec, doc: dict, changed: list) -> None:
    """The analyst's agency ratings (most filers name none: CRM's 10-K never mentions an agency), on the same spread
    table; replaces a synthetic rating, never a rating read from the filings."""
    cr = doc.get("credit_rating")
    cap = spec.capital
    if not isinstance(cr, dict) or not cap or cap.get("rating_src") == "agency":
        return
    from . import capital
    ag = capital.agency_rating({k: v for k, v in cr.items() if k in ("S&P", "Moody's", "Fitch")})
    if not ag:
        return
    rating, spread, how = ag
    rf = spec.market_rf if spec.market_rf is not None else spec.valuation["rf"][0]
    cap.update(rating=rating, spread=spread, rating_src="agency (analyst)", kd=round(rf + spread, 4),
               ratings_input=dict(cr))
    spec.valuation["kd"] = (round(round((rf + spread) / 0.0005) * 0.0005, 4),
                            f"Risk-free + {spread:.2%} spread for the {rating} rating ({how}; {cr.get('source', 'analyst')})")
    changed.append("credit rating")


def _reject(spec: ModelSpec, warn: list, field_: str, source: str, value, why: str) -> None:
    """A guard refused an analyst input: a warning, and a failed gate (45) until the field is set to a value the guard
    accepts (PM, ADBE: seven explanations claimed a margin fix the guard had silently refused)."""
    warn.append(f"{field_}: {value} {why}, skipped")
    spec.rejected[field_] = (source, value, why, _field_value(spec, field_))


def _field_value(spec: ModelSpec, field_: str):
    """The value a field holds now (what the model kept after a rejection)."""
    parts = field_.split(".")
    if parts[0] == "segments" and parts[-1] == "mna":
        sg = _find_segment(spec, ".".join(parts[1:-1]))
        return list(sg.mna) if sg else None
    if parts[0] == "segments":
        sg = _find_segment(spec, ".".join(parts[1:-2]))
        return list(getattr(sg, parts[-2])[parts[-1]]) if sg else None
    if parts[0] == "drivers":
        return list(spec.drivers.get(parts[1], ([],))[0])
    if parts[0] == "valuation":
        return spec.valuation.get(parts[1], (None,))[0]
    if parts[0] == "guidance":
        return list(spec.guidance.get(parts[1]) or [])
    return getattr(spec, field_, None)


def _restore_rejected(spec: ModelSpec, rj) -> None:
    """Rejections carried in an overrides file hold while the field still holds what the model kept (a round trip
    reproduces gate 45); once the field is changed, the analyst has dealt with it."""
    for f, r in (rj or {}).items() if isinstance(rj, dict) else ():
        if isinstance(r, dict) and r.get("kept") == _field_value(spec, f):
            spec.rejected[f] = (str(r.get("source") or ""), r.get("value"), str(r.get("why") or ""), r.get("kept"))


def _margin_cap(sg) -> float:
    """A segment whose measure is gross profit legitimately runs above 95% (ADBE Digital Media: 93% in FY25)."""
    p, s_ = (sg.profit[2] if sg.profit else None), (sg.sales[2] if sg.sales else None)
    return 0.995 if p is not None and s_ and p / s_ > 0.85 else BOUNDS["margin"][1]


def _bs_date(spec: ModelSpec):
    return spec.interim.period_end if spec.interim.available else spec.period_ends[-1]


def _closed_before_bs(spec: ModelSpec, d: dict) -> bool:
    try:
        close = date.fromisoformat(str(d.get("close") or "")[:10])
    except ValueError:
        return False
    return str(d.get("status") or "").lower() == "closed" and close <= _bs_date(spec)


def apply(spec: ModelSpec, doc: dict, source: str = "overrides") -> list[str]:
    """Merge an overrides document into the spec. Unknown or malformed fields are skipped
    with a warning; nothing is applied half-way within a field."""
    if isinstance(doc, str):
        doc = json.loads(doc)
    warn: list[str] = []
    changed: list[str] = []

    by_ai = "AI analyst" in f"{(doc.get('meta') or {}).get('author') or ''} {source}"
    spec.audit_impact = []  # recomputed in the next deliverable build
    _market_data(spec, doc, changed)
    _credit_rating(spec, doc, changed)
    if doc.get("framework") in ("economic", "reference"):
        spec.framework = doc["framework"]
    m = doc.get("market") or {}
    # a file built for another date carries that date's prices: market facts of the build date replace them (PM round
    # 7: the ADBE file rolled back to Nov 17 at $235.55 when the close was $226.20)
    doc_date = str((doc.get("meta") or {}).get("as_of") or "")[:10]
    redated = bool(doc_date) and doc_date != str(spec.as_of)
    if redated:
        warn.append(f"Overrides dated {doc_date} applied on {spec.as_of}: the share price and peer prices are the "
                    f"{spec.as_of} closes, not the file's")
    for key, attr, note_attr in (("price", "price", "price_note"), ("basic_shares", "basic_shares", "basic_note"),
                                 ("dilutive", "dilutive", "dilutive_note")):
        if key == "price" and redated:
            continue
        v = m.get(key)
        if isinstance(v, dict) and _num(v.get("value")):
            setattr(spec, attr, float(v["value"]))
            if v.get("note"):
                setattr(spec, note_attr, str(v["note"]))
            changed.append(f"market.{key}")
    if doc.get("scenario") in (1, 2, 3):
        spec.scenario = int(doc["scenario"])
    if doc.get("seg_after_charges") in (0, 1):
        if int(doc["seg_after_charges"]) != spec.seg_after_charges:
            changed.append("seg_after_charges")
            spec.audit.append((source, "seg_after_charges", spec.seg_after_charges, int(doc["seg_after_charges"])))
            spec.set_single_segment_basis(int(doc["seg_after_charges"]))
        spec.seg_after_charges = int(doc["seg_after_charges"])
    for key in ("adj_eps_ex_sbc", "adj_eps_ex_repo", "adj_eps_ex_nonrec", "adj_eps_amort_share"):
        # the adjusted-EPS definition contract: switches reproduce the company's reconciliation (fractions allowed
        # where the company excludes only part: LLY keeps acquired IPR&D, SHW excludes only Valspar amortization)
        v = doc.get(key)
        if _num(v) and 0 <= v <= 1:
            v = float(v) if key in ("adj_eps_ex_repo", "adj_eps_amort_share") else int(round(v))
            if v != getattr(spec, key):
                changed.append("adjusted EPS definition")
                spec.audit.append((source, key, getattr(spec, key), v))
            setattr(spec, key, v)
    th = doc.get("adj_eps_tax_hist")  # the company's stated non-GAAP tax rate by history year (CRM)
    stated = spec.stated_adj_tax or []
    off = [x for x in th or [] if _num(x) and stated and min(abs(x - r) for r in stated) > 0.03] \
        if isinstance(th, list) else []
    if isinstance(th, list) and len(th) == 3 and not stated:  # no stated rate: near that year's GAAP rate, never ~0
        I_ = spec.hist.IS
        for j, x in enumerate(th):
            pt, tx = I_["pretax"][j], I_["tax"][j]
            etr = tx / pt if pt and tx is not None and pt > 0 else None
            if _num(x) and (x < 0.08 or (etr is not None and 0 < etr < 0.4 and abs(x - etr) > 0.08)):
                off.append(x)
        stated = stated or ["that year's GAAP rate ±8pt, at least 8%"]
    if off:
        # history is locked (PM, ADBE: the FY25 non-GAAP tax rate went to 4.8% against a stated 18% so the adjusted
        # EPS tie-out would pass; moving history to pass a check is worse than failing it)
        _reject(spec, warn, "adj_eps_tax_hist", source, th,
                f"{', '.join(f'{x:.1%}' for x in off)} is off the " +
                (f"{', '.join(f'{r:.1%}' for r in stated)} non-GAAP tax rate the releases state (±3pt)"
                 if all(_num(r) for r in stated) else str(stated[0])) +
                ": history is locked — if adjusted EPS still does not tie, the reported figure or the definition "
                "switches are wrong, not the history")
    elif isinstance(th, list) and len(th) == 3 and all(x is None or (_num(x) and 0 <= x <= 0.4) for x in th):
        th = [float(x) if x else None for x in th]
        spec.rejected.pop("adj_eps_tax_hist", None)
        if th != spec.adj_eps_tax_hist:
            changed.append("adjusted EPS definition")
            spec.audit.append((source, "adj_eps_tax_hist", spec.adj_eps_tax_hist, th))
        spec.adj_eps_tax_hist = th
    elif th is not None:
        warn.append(f"adj_eps_tax_hist: {th} needs three rates (or null) between 0 and 0.4, skipped")

    for name, sd in (doc.get("segments") or {}).items():
        sg: SegSpec = _find_segment(spec, name)
        if sg is None:
            warn.append(f"segments.{name}: no such segment in the model ({', '.join(s.name for s in spec.segments)})")
            continue
        if not isinstance(sd, dict):
            continue
        if _num(sd.get("organic_last")):
            sg.organic_last = float(sd["organic_last"])
        for kind in ("growth", "margin"):
            block = sd.get(kind) or {}
            for case in ("bear", "base", "bull"):
                if case in block:
                    v = _vec(block[case])
                    hi = _margin_cap(sg) if kind == "margin" else BOUNDS[kind][1]
                    if v is None:
                        _reject(spec, warn, f"segments.{sg.name}.{kind}.{case}", source, block[case],
                                f"needs {N} numbers")
                    elif not _inside(kind, v) and not (kind == "margin" and all(BOUNDS[kind][0] <= x <= hi
                                                                                  for x in v)):
                        _reject(spec, warn, f"segments.{sg.name}.{kind}.{case}", source, v,
                                f"outside {(BOUNDS[kind][0], hi)} (decimals?)")
                    else:
                        spec.rejected.pop(f"segments.{sg.name}.{kind}.{case}", None)
                        if case == "base" and list(getattr(sg, kind)[case]) != list(v):
                            spec.audit.append((source, f"segments.{sg.name}.{kind}.base",
                                               list(getattr(sg, kind)[case]), list(v)))
                        getattr(sg, kind)[case] = v
                        changed.append(f"{sg.name} {kind} {case}")
        mna = sd.get("mna")
        if isinstance(mna, dict):
            v = _vec(mna.get("values"))
            if v is not None and not _inside("mna", v):
                _reject(spec, warn, f"segments.{sg.name}.mna", source, v,
                        f"outside {BOUNDS['mna']} (a contribution to growth as a decimal, not $mm)")
            elif v is not None:
                spec.rejected.pop(f"segments.{sg.name}.mna", None)
                if list(sg.mna) != list(v):
                    spec.audit.append((source, f"segments.{sg.name}.mna", list(sg.mna), list(v)))
                sg.mna = v
                changed.append(f"{sg.name} M&A")
            elif "values" in mna:
                _reject(spec, warn, f"segments.{sg.name}.mna", source, mna.get("values"), f"needs {N} numbers")
            if mna.get("note"):
                sg.mna_note = str(mna["note"])
        so = sd.get("sotp")
        if isinstance(so, dict):
            v = _vec(so.get("multiples"), 3)
            if v is not None and _inside("multiples", v):
                sg.sotp = tuple(v)
                changed.append(f"{sg.name} SOTP multiples")
            if so.get("note"):
                sg.sotp_note = str(so["note"])
        split = sd.get("sotp_split")
        if isinstance(split, list) and split:
            parts, ok = [], True
            for p in split:
                mult = _vec(p.get("multiples"), 3) if isinstance(p, dict) else None
                if mult is None or not p.get("name") or not (_num(p.get("ebitda")) or p.get("remainder")):
                    ok = False
                    break
                parts.append({"name": str(p["name"]), "ebitda": float(p["ebitda"]) if _num(p.get("ebitda")) else None,
                              "remainder": bool(p.get("remainder")), "multiples": mult, "note": str(p.get("note", ""))})
            if ok and sum(p["remainder"] for p in parts) <= 1:
                spec.sotp_splits[sg.name] = parts
                changed.append(f"{sg.name} SOTP split")
            else:
                warn.append(f"segments.{name}.sotp_split: each part needs name, multiples[3] and ebitda (or one "
                            "remainder part)")
        warn += _apply_lines(sg, sd, name, changed)
        if sd.get("rename"):
            new = str(sd["rename"]).strip()
            if sg.name in spec.sotp_splits:
                spec.sotp_splits[new] = spec.sotp_splits.pop(sg.name)
            sg.name = new

    notes = doc.get("notes") or {}
    for k in ("growth", "margin"):
        if notes.get(k):
            spec.notes[k] = str(notes[k])

    corp = doc.get("corporate") or {}
    if _vec(corp.get("sales")):
        spec.corp_sales = _vec(corp["sales"])
        changed.append("corporate sales")
    if _vec(corp.get("cost_pct")):
        spec.corp_cost_pct = _vec(corp["cost_pct"])
        changed.append("corporate costs")
    if corp.get("sales_note"):
        spec.corp_sales_note = str(corp["sales_note"])
    if corp.get("cost_note"):
        spec.corp_cost_note = str(corp["cost_note"])
    if _vec(corp.get("multiples"), 3):
        spec.corp_multiples = tuple(_vec(corp["multiples"], 3))
    if corp.get("multiple_note"):
        spec.notes["corp_multiple"] = str(corp["multiple_note"])

    for k, d in (doc.get("drivers") or {}).items():
        if k not in spec.drivers:
            warn.append(f"drivers.{k}: unknown driver")
            continue
        if not isinstance(d, dict):
            continue
        n = 1 if k == "dps0" else 5 if k == "div_growth" else N
        if "values" in d:
            v = _vec(d["values"], n)
            if v is None:
                _reject(spec, warn, f"drivers.{k}", source, d["values"], f"needs {n} numbers")
                continue
            if not _inside(k, v):
                _reject(spec, warn, f"drivers.{k}", source, v, f"outside {BOUNDS[k]} (decimals?)")
                continue
            spec.rejected.pop(f"drivers.{k}", None)
            if by_ai and TARGET_RX.search(str(d.get("note") or "")):  # an input bent to fit another (PM round 8, DE:
                # capex ramped from 3% to 8% "to make terminal reinvestment consistent with" a broken RONIC)
                _reject(spec, warn, f"drivers.{k}", source, v, "its reason is to fit another input or the output "
                        f"('{TARGET_RX.search(str(d.get('note'))).group(0)}'), not the business")
                continue
            if k == "shares_issued" and str(d.get("note") or "").startswith("~0.4% of basic shares") and \
                    max(v) - min(v) < 0.06:
                continue  # an earlier engine's default, not a judgment: today's default (stock comp ÷ price) stands
            ytd_acq = _num((spec.ytd_capital or {}).get("acq")) and float(spec.ytd_capital["acq"]) or 0.0
            if k == "acq" and by_ai and ytd_acq > 0 and v[0] > ytd_acq * 1.01 + 5 and not spec.post_bs_events and \
                    not POST_BS_RX.search(str(d.get("note") or "")):
                warn.append(f"drivers.acq {spec.fc_labels[0]}: {v[0]:,.0f}mm is above the {ytd_acq:,.0f}mm paid by the "
                            "balance-sheet date with no closing after it named: the cash paid is a fact, held at "
                            f"{ytd_acq:,.0f}mm (a deal closing later goes in 'deals' or names its date)")
                v = [ytd_acq] + list(v[1:])
                d = {**d, "note": f"{spec.fc_labels[0]}: the {ytd_acq:,.0f}mm paid by the balance-sheet date (10-Q cash "
                                  "flow); later years as the analyst set them"}  # the old note described the add-on
            if k == "da_pct" and spec.drivers[k][0][0] > 0.002 and min(v) < 0.25 * spec.drivers[k][0][0]:
                warn.append(f"Depreciation {min(v):.1%} of sales from the analyst is below a quarter of history "
                            f"({spec.drivers[k][0][0]:.1%}): depreciation cannot vanish while PP&E is in use; kept "
                            "the history-based path")
                continue
            if k == "sbc_pct" and max(v) < 0.25 * spec.drivers[k][0][0]:
                warn.append(f"drivers.sbc_pct: {v} would remove a real cost (history {spec.drivers[k][0][0]:.1%} of "
                            "sales); kept the history-based path. Use adj_eps_ex_sbc = 1 for non-GAAP EPS instead")
                continue
            if list(spec.drivers[k][0]) != list(v):
                spec.audit.append((source, f"drivers.{k}", list(spec.drivers[k][0]), list(v)))
            spec.drivers[k] = (v, str(d.get("note") or spec.drivers[k][1]))
            changed.append(f"driver {k}")
        elif d.get("note"):
            spec.drivers[k] = (spec.drivers[k][0], str(d["note"]))

    for k, d in (doc.get("valuation") or {}).items():
        if k not in spec.valuation:
            warn.append(f"valuation.{k}: unknown input")
            continue
        if isinstance(d, dict) and _num(d.get("value")):
            if k == "ronic" and d["value"] > BOUNDS[k][1] and not d["value"] > 5:
                # above the cap is an intent ("as high as the policy allows"), not a typo: held at the cap, which is
                # also gate 34's reference (PM, ADBE: 0.70 and 1.0 were both rejected while gate 34 asked for 194%)
                warn.append(f"valuation.ronic: {d['value']} held at the {BOUNDS[k][1]:.0%} cap")
                d = {**d, "value": BOUNDS[k][1], "note": f"{d.get('note') or ''} (held at the {BOUNDS[k][1]:.0%} "
                                                         "cap)".strip()}
            if not _inside(k, [d["value"]]):
                _reject(spec, warn, f"valuation.{k}", source, d["value"], f"outside {BOUNDS[k]} (decimals?)")
                continue
            spec.rejected.pop(f"valuation.{k}", None)
            v, note = float(d["value"]), str(d.get("note") or "")
            cap = spec.capital or {}
            if k == "rf" and by_ai and spec.market_rf and abs(v - spec.market_rf) > 0.0025:
                warn.append(f"Risk-free rate {v:.2%} from the analyst differs from the market 10-yr yield on the date "
                            f"({spec.market_rf:.2%}): a market fact, kept")
                continue
            if k == "erp" and by_ai and cap.get("erp") is not None and abs(v - cap["erp"]) > 1e-9:
                warn.append(f"ERP {v:.2%} from the analyst not used: the ERP is set once per valuation date for every "
                            f"name ({cap['erp']:.2%}, valuation/capital.py)")
                continue
            if by_ai and k in JUDGMENT_VAL and TARGET_RX.search(note) and abs(v - spec.valuation[k][0]) > 1e-9:
                warn.append(f"Change to {k} ({spec.valuation[k][0]:g} -> {v:g}) not used: its reason is the output "
                            f"('{TARGET_RX.search(note).group(0)}'); inputs are set on the business, and the rating "
                            "follows from them")
                continue
            house = capital.terminal_growth()[0]
            if by_ai and k == "g" and abs(v - house) > 1e-9 and abs(v - spec.valuation[k][0]) > 1e-9 and \
                    (RISK_RX.search(note) or not re.search(r"\b(19|20)\d\d\b", note)):
                # terminal growth is the house rate unless a dated reason the scenarios do not carry says otherwise
                # (PM round 7; a loss of exclusivity with its year, as gate 34 asks of RONIC): a decline is the bear
                # case, whose terminal growth follows its own last-year growth (DCF C74)
                warn.append(f"Terminal growth {v:.2%} from the analyst not used: the house rate ({house:.2%}) applies "
                            "unless a dated reason the scenarios do not carry (a loss of exclusivity with its year) says "
                            "otherwise; a decline belongs in the bear case and its probability")
                v, note = house, (f"House long-run growth {house:.2%} (the analyst's {float(d['value']):.2%} held: no "
                                  "dated reason the scenarios do not carry)")
            if by_ai and k in RISK_ONCE and RISK_RX.search(note) and not MEASURE_RX.search(note) and \
                    abs(v - spec.valuation[k][0]) > 1e-9:
                # count each risk once (PM round 6, ADBE: the same AI-disruption risk sat in the bear, base growth,
                # base margin, beta and terminal growth): scenario risk is the bear case and its probability
                warn.append(f"Change to {k} ({spec.valuation[k][0]:g} -> {v:g}) not used: its reason "
                            f"('{RISK_RX.search(note).group(0)}') is scenario risk, which the bear case and its "
                            "probability carry; beta, terminal growth and fade years move only for a measurement "
                            "reason (leverage, business mix, a regression or listing issue)")
                continue
            if by_ai and "review" in source.lower() and k in JUDGMENT_VAL and PRICE_RX.search(note):
                warn.append(f"Review change to {k} ({spec.valuation[k][0]:g} -> {v:g}) not used: its reason refers to "
                            "the share price; inputs are set on fundamentals, the price is what the model tests")
                continue
            if k == "beta" and by_ai:
                # judged against the method beta of the FINAL peer set (pipeline.apply_capital, after the peers are
                # in): COST v4 echoed the brief's 6-peer method beta, switched to 4 peers, and a load-time clamp
                # against a fetch whose peer betas had failed set it to 0.80
                cap["beta_ai"] = (v, note)
            if k == "beta" and by_ai and cap.get("beta") and False:
                lo, hi = cap["beta"] - capital.BETA_BAND, cap["beta"] + capital.BETA_BAND
                if not lo - 1e-9 <= v <= hi + 1e-9 or not note:
                    nv = min(max(v, lo), hi) if note else cap["beta"]
                    warn.append(f"Beta {v:.2f} from the analyst {'outside' if note else 'without a reason, not used; '}"
                                f"the method's {cap['beta']:.2f} ± {capital.BETA_BAND:.2f}: set to {nv:.2f}")
                    v = nv
            if k == "kd" and str(cap.get("rating_src", "")).startswith("agency") and \
                    not re.search(r"(?i)\d[\d.]*%\s*(senior\s+)?(unsecured\s+)?(notes|bonds|debentures)|notes? due 20\d\d|"
                                  r"yield to maturity|(trad(e|es|ing)|priced) (at|to yield)", note) and \
                    abs(v - spec.valuation["kd"][0]) > 1e-9:
                continue  # an agency rating sets kd; only market evidence (a bond's yield or coupon) moves it
            if k == "kd" and by_ai and cap.get("kd"):
                lo, hi = cap["kd"] - capital.KD_BAND, cap["kd"] + capital.KD_BAND
                if not lo - 1e-9 <= v <= hi + 1e-9:
                    nv = min(max(v, lo), hi)
                    warn.append(f"Cost of debt {v:.2%} from the analyst outside the rating-based {cap['kd']:.2%} ± "
                                f"{capital.KD_BAND:.2%}: set to {nv:.2%}")
                    v = nv
            if k == "ronic" and by_ai and cap.get("roic") and v > max(cap["roic"], cap.get("wacc", 0)) + 0.005 \
                    and "roic" not in note.lower():
                nv = max(cap["roic"], cap.get("wacc", 0))
                warn.append(f"RONIC {v:.1%} above the current ROIC {cap['roic']:.1%} without a written argument: "
                            f"set to {nv:.1%}")
                v = nv
            if k == "exit_mult":
                cap["exit_derived"] = False
            before = spec.valuation[k][0]
            spec.valuation[k] = (v, note or spec.valuation[k][1])
            spec.val_source[k] = source
            if before != v:
                spec.audit.append((source, f"valuation.{k}", before, v))
            changed.append(f"valuation {k}")

    for k, v in (doc.get("labels") or {}).items():
        if k in LABEL_KEYS and v:
            spec.labels[k] = str(v)

    dl = doc.get("debt_like")
    if isinstance(dl, dict):
        for k in ("liab", "offset", "liab_fy", "offset_fy"):
            if _num(dl.get(k)):
                spec.debt_like[k] = float(dl[k])
        if dl.get("note"):
            spec.debt_like["note"] = str(dl["note"])
        changed.append("debt-like items")

    g = doc.get("guidance")
    ext_g = (spec.extracted or {}).get("guidance") or {}
    if isinstance(g, dict) and str(g.get("margin_basis") or "").lower().startswith(("gaap", "adj", "non", "seg")):
        spec.margin_basis = str(g["margin_basis"]).lower()
    if isinstance(g, dict):
        for k in GUIDE_KEYS:
            if k in g:
                v = _vec(g[k], 2)
                if v is not None:
                    bad = guidance_problem(spec, k, *v)
                    if bad:
                        _reject(spec, warn, f"guidance.{k}", source, list(v), f"{bad}; not used")
                    elif by_ai and k in ext_g and any(abs(a - b) > 0.01 * max(abs(b), 1e-9) for a, b in zip(v, ext_g[k])):
                        warn.append(f"Guidance {k}: the analyst's {_guide_txt(k, v)} differs from the release's "
                                    f"{_guide_txt(k, ext_g[k])}; the release figure is kept")
                    else:
                        spec.guidance[k] = tuple(v)
                elif g[k] is None and not (by_ai and k in ext_g):
                    spec.guidance.pop(k, None)
        if g.get("date"):
            spec.guidance_date = str(g["date"])
        changed.append("guidance")

    adj = doc.get("adjusted_eps")
    if "adjusted_eps" in doc and adj is None:  # explicit null: the company reports no adjusted EPS
        spec.adj_eps_hist, spec.adj_eps_reported = [None, None, None], False
        for d in (spec.interim.ytd_cur, spec.interim.ytd_prior):
            if "adj_eps_model" in d:
                d["adj_eps"] = d["adj_eps_model"]
        changed.append("adjusted EPS = model definition")
    if isinstance(adj, dict):
        bad = adj_eps_problem(spec, adj) or ytd_adj_problem(spec, adj.get("ytd_cur") if _num(adj.get("ytd_cur")) else None,
                                                              adj.get("ytd_prior") if _num(adj.get("ytd_prior")) else None,
                                                              adj.get("fy") if _num(adj.get("fy")) else None)
        if bad:
            warn.append(f"Adjusted EPS from the analyst ({_eps_txt(adj)}) not used ({bad}); the model's definition "
                        "applies")
            adj = None
    ext = (spec.extracted or {}).get("adjusted_eps") or {}
    if isinstance(adj, dict) and by_ai and any(
            _num(ext.get(a)) and _num(adj.get(b)) and abs(ext[a] - adj[b]) < 0.006
            for a, b in (("fy", "ytd_cur"), ("fy_prior", "ytd_prior"))):
        warn.append(f"Adjusted EPS read from the release ({_eps_txt(ext)}) is period-shifted: its full-year figures "
                    f"are the year-to-date ones, so the analyst's ({_eps_txt(adj)}) are used")
        spec.extracted["adjusted_eps"] = {k: adj.get(k) for k in ("fy", "fy_prior", "ytd_cur", "ytd_prior")}
        spec.flags = [f for f in spec.flags if not f.startswith("Year-to-date adjusted EPS read from the release")]
        ext = {}
    if isinstance(adj, dict) and by_ai and any(_num(ext.get(k)) for k in ext):
        # reported figures come from the release extraction; an analyst pass has copied the wrong period / GAAP
        kept = {k: ext[k] for k in ext if _num(ext.get(k)) and _num(adj.get(k)) and abs(adj[k] - ext[k]) > 0.01 * abs(ext[k])}
        if kept:
            warn.append(f"Adjusted EPS: kept the release figures ({_eps_txt(kept)}) over the analyst's "
                        f"({_eps_txt(adj)})")
            adj = {**adj, **kept}
    if isinstance(adj, dict) and _num(adj.get("fy")):
        # a company without non-GAAP EPS: an analyst may copy GAAP EPS here although the year had one-offs the
        # model excludes; the model's own figure (GAAP + amortization - one-offs) is then the consistent one
        gaap = spec.hist.IS["eps_dil"]
        for k, j in (("fy", 2), ("fy_prior", 1)):
            if _num(adj.get(k)) and gaap[j] and abs(adj[k] - gaap[j]) < 0.006 and spec.nonrec_hist[j]:
                warn.append(f"adjusted_eps.{k} = GAAP EPS {gaap[j]:.2f} although FY{spec.fy_hist[j]} had "
                            f"{spec.nonrec_hist[j]:,.0f}mm of one-offs: kept the model's adjusted EPS")
                adj = {**adj, "fy": None, "fy_prior": None}
                break
    if isinstance(adj, dict) and _num(adj.get("ytd_cur")) and spec.interim.available:
        ytd_eps, booked = spec.interim.ytd_cur.get("eps"), spec.drivers["nonrec"][0][0]
        if ytd_eps and abs(adj["ytd_cur"] - ytd_eps) < 0.006 and abs(booked) >= 0.005 * abs(spec.hist.IS["sales"][2] or 1):
            warn.append(f"adjusted_eps.ytd_cur = GAAP EPS {ytd_eps:.2f} although {spec.interim.label_cur} includes "
                        f"{booked:,.0f}mm of one-offs: kept the model's year-to-date adjusted EPS")
            adj = {k: v for k, v in adj.items() if k not in ("ytd_cur", "ytd_prior")}
    if isinstance(adj, dict):
        if _num(adj.get("fy")):
            spec.adj_eps_hist = [None, adj.get("fy_prior") if _num(adj.get("fy_prior")) else None, float(adj["fy"])]
            spec.adj_eps_reported = True
        for tag in ("cur", "prior"):
            if spec.interim.available and _num(adj.get("ytd_" + tag)):
                getattr(spec.interim, "ytd_" + tag)["adj_eps"] = float(adj["ytd_" + tag])

    org = doc.get("organic")
    if isinstance(org, dict):
        if _num(org.get("fy_total")):
            spec.organic_total_last = float(org["fy_total"])
        if _num(org.get("q_total")):
            spec.q_organic_total = float(org["q_total"])
        for name, v in (org.get("q_by_segment") or {}).items():
            sg = _find_segment(spec, name)
            if sg and _num(v):
                spec.q_organic[sg.name] = float(v)

    peers = doc.get("peers")
    if isinstance(peers, list) and peers:
        clean = []
        for p in peers:
            if isinstance(p, dict) and p.get("ticker"):
                clean.append({"ticker": str(p["ticker"]).upper(), "name": str(p.get("name") or p["ticker"]),
                              "overlap": str(p.get("overlap") or ""),
                              "price": float(p["price"]) if _num(p.get("price")) else None,
                              "eps": float(p["eps"]) if _num(p.get("eps")) else None,
                              "source": str(p.get("source") or source),
                              "ev": float(p["ev"]) if _num(p.get("ev")) else None,
                              "ebitda": float(p["ebitda"]) if _num(p.get("ebitda")) else None,
                              "ev_source": str(p.get("ev_source") or ""),
                              **{k: p[k] for k in ("eps_next", "g_next", "rev_g_next", "beta_raw", "de", "snapshot", "pro_forma",
                                                   "fy_end", "eps_fy", "ebitda_margin", "share_chg")
                                 if p.get(k) is not None}})
        if clean:
            if "AI analyst" in f"{(doc.get('meta') or {}).get('author') or ''} {source}" or redated:
                for p in clean:  # a model's prices are re-checked against the market close (pipeline.fill_peer_ev)
                    p["_reprice"] = True
                    if redated:  # EV / EBITDA, margins and share counts point in time at the new date
                        p["ev"] = p["ebitda"] = None
            spec.peers = clean
            changed.append(f"{len(clean)} peers")

    th = doc.get("thesis")
    if isinstance(th, dict) and th.get("points") and not th.get("stance"):
        # the template's placeholder stance must not survive an analyst thesis that omits it
        txt = f"{th.get('recommendation', '')} {(doc.get('meta') or {}).get('summary', '')}".lower()
        th["stance"] = ("short" if re.search(r"\b(short|sell)\b", txt) else
                        "neutral on" if re.search(r"\b(neutral|hold|market[- ]weight|underweight|avoid)\b", txt) else
                        "long" if re.search(r"\b(long|overweight|buy)\b", txt) else "neutral on")
    if isinstance(th, dict):
        if isinstance(th.get("recommendation"), str) and th["recommendation"].strip():
            spec.thesis.pop("recommendation_text", None)  # the doc's sentence replaces the last one
        for k in THESIS_KEYS:
            v = th.get(k)
            if isinstance(v, list) and v:
                spec.thesis[k] = [str(x) for x in v]
            elif isinstance(v, str) and v.strip():
                spec.thesis[k] = v.strip()
        ex = th.get(EXPLAIN_KEY)
        if isinstance(ex, dict) and ex:  # {"gate 8": "why", "<start of an open check>": "why"}
            spec.thesis[EXPLAIN_KEY] = {**(spec.thesis.get(EXPLAIN_KEY) or {}),
                                        **{str(k): str(v) for k, v in ex.items() if str(v).strip()}}
        ov = th.get("rating_override")
        if isinstance(ov, dict) and ov.get("rating") and ov.get("reason"):
            spec.thesis["rating_override"] = {"rating": str(ov["rating"]), "reason": str(ov["reason"])}
        changed.append("thesis")

    deals = doc.get("deals")
    if isinstance(deals, list):
        clean = []
        for d in deals:
            if not isinstance(d, dict) or not d.get("name"):
                continue
            c = {}
            for k, typ in DEAL_KEYS.items():
                if d.get(k) is None:
                    continue
                if typ is float and _num(d[k]):
                    c[k] = float(d[k])
                elif typ is bool:
                    c[k] = bool(d[k])
                elif typ is str:
                    c[k] = str(d[k])
            if _closed_before_bs(spec, c):
                # PM, ADBE: Semrush closed in April and was paid before the August balance sheet; on the Deals tab
                # its price came off a second time (-$4/share)
                warn.append(f"deals.{c['name']}: closed {c.get('close') or ''} before the "
                            f"{_bs_date(spec):%b %-d, %Y} balance sheet — its price is in net debt and its revenue in "
                            "the forecast, so it is not a Deals-tab item; removed")
                continue
            if 0 <= c.get("probability", 0.5) <= 1:
                clean.append(c)
        spec.deals = clean
        if clean:
            changed.append(f"{len(clean)} deal(s)")
    sp = doc.get("scenario_probs")
    if isinstance(sp, list) and len(sp) == 3 and all(_num(x) and x >= 0 for x in sp) and sum(sp) > 0:
        spec.scenario_probs = [float(x) for x in sp]
        changed.append("scenario probabilities")

    meta = doc.get("meta") or {}
    author = str(meta.get("author") or "")
    who = author if author and author != "defaults" else source
    if author and author != "defaults" and source not in ("AI analyst", "AI analyst review", "overrides"):
        who = f"{author}, applied from an {source}"
    for f in doc.get("flags") or []:
        if any(FLAG_RX.sub("", x) == str(f) for x in spec.flags):  # the review pass often repeats a flag
            continue
        spec.flags.append(f"Analyst ({author if author and author != 'defaults' else source}): {f}")

    fp = doc.get("fade_path")
    if isinstance(fp, dict) and isinstance(fp.get("values"), list):
        v = [float(x) for x in fp["values"] if _num(x)]
        if v and len(v) <= 10 and all(-0.5 <= x <= 0.3 for x in v):
            spec.fade_path, spec.fade_path_note = v, str(fp.get("note") or "")
            spec.rejected.pop("fade_path", None)
            changed.append("fade path")
            if len(v) > spec.valuation["fade_years"][0]:
                spec.valuation["fade_years"] = (float(len(v)), spec.valuation["fade_years"][1] +
                                                f"; {len(v)} years to carry the fade path")
        elif fp["values"]:
            _reject(spec, warn, "fade_path", source, fp["values"], "needs up to 10 growth rates between -50% and +30%")
    _restore_rejected(spec, doc.get("rejected"))
    if not spec.audit_fv and isinstance(doc.get("audit_fv"), list):  # the passes as run (PM round 6: a rebuild lost them)
        spec.audit_fv = [(str(a_[0]), float(a_[1])) for a_ in doc["audit_fv"]
                         if isinstance(a_, list) and len(a_) == 2 and _num(a_[1])]
    w = sum(spec.valuation[k][0] for k in ("w_pg", "w_exit", "w_sotp", "w_comps"))
    if abs(w - 1) > 0.01:
        warn.append(f"Method weights sum to {w:.0%}; the Summary rescales them")
    if spec.valuation["g"][0] >= 0.8 * _wacc_guess(spec):
        warn.append("Terminal growth is close to the cost of capital: the perpetuity value will be unstable")
    before = {k: list(spec.drivers[k][0]) for k in (doc.get("drivers") or {}) if k in spec.drivers}
    changes = balance_guards(spec)
    warn += changes
    for k, b in before.items():  # a floor or cap that changes what the analyst wrote is a refusal (PM round 8: the
        # $647mm acquisitions figure was floored back to $8,779mm three times while gate 45 passed)
        if list(spec.drivers[k][0]) != b:
            spec.rejected[f"drivers.{k}"] = (source, b, "changed by an engine guard: " + "; ".join(changes)[:300],
                                             list(spec.drivers[k][0]))
    spec.provenance.append(f"{who}: {_summary(changed)}")
    if meta.get("summary"):
        spec.notes["analyst_summary"] = str(meta["summary"])
    for u in (meta.get("sources") or [])[:12]:
        if f"{SOURCE_PREFIX}{u}" not in spec.sources:
            spec.sources.append(f"{SOURCE_PREFIX}{u}")
    return warn


def _kpi(d, where: str, warn: list) -> dict | None:
    """Validate a volume x price block; returns the normalized dict or None (with a warning)."""
    if not isinstance(d, dict):
        return None
    out = {}
    for part in ("volume", "price"):
        x = d.get(part) or {}
        hist = x.get("history") if isinstance(x, dict) else None
        hist = [float(v) if _num(v) else None for v in hist] if isinstance(hist, list) and len(hist) == 3 else None
        g = {c: _vec((x.get("growth") or {}).get(c)) for c in ("bear", "base", "bull")} if isinstance(x, dict) else {}
        if not x.get("name") or not hist or hist[-1] is None or not g.get("base"):
            warn.append(f"{where}.kpi.{part}: needs name, history [3] and growth.base [6]")
            return None
        for c in ("bear", "bull"):
            g[c] = g[c] or g["base"]
        if not all(_inside("kpi_growth", v) for v in g.values()):
            warn.append(f"{where}.kpi.{part}.growth: outside {BOUNDS['kpi_growth']} (decimals?)")
            return None
        out[part] = {"name": str(x["name"]), "unit": str(x.get("unit") or ""), "history": hist, "growth": g}
    out["note"] = str(d.get("note") or "")
    return out


def _attach_kpi(unit, kpi: dict, sales_last, changed: list, label: str):
    v, p = kpi["volume"]["history"][-1], kpi["price"]["history"][-1]
    kpi["scale"] = (sales_last / (v * p)) if v and p and sales_last else 1.0
    unit.kpi = kpi
    for c in ("bear", "base", "bull"):
        unit.growth[c] = kpi_growth(kpi, c)
    changed.append(f"{label} KPI build")


def _apply_lines(sg: SegSpec, sd: dict, name: str, changed: list) -> list:
    """segments.<name>.use_lines / lines / kpi. Lines the analyst does not set are re-aligned to the
    segment's (possibly new) organic paths, so the segment total still follows the segment assumptions."""
    warn = []
    lines = sd.get("lines")
    if isinstance(lines, dict) and lines:
        new = []
        for lname, ld in lines.items():
            if not isinstance(ld, dict):
                continue
            ln = next((x for x in sg.lines if _key(x.name) == _key(lname)), None)
            hist = _vec(ld.get("history"), 3)
            if ln is None and hist:  # renamed line (names are cleaned up between versions): match on its history
                ln = next((x for x in sg.lines if x.sales[2] is not None and abs(x.sales[2] - hist[2]) < 0.5), None)
            if ln is None:
                hist = _vec(ld.get("history"), 3)
                if hist is None:
                    warn.append(f"segments.{name}.lines.{lname}: not a line of this segment "
                                f"({', '.join(x.name for x in sg.lines) or 'none tagged'}); to define it give history [3]")
                    continue
                ln = LineSpec(name=str(lname), sales=hist)
                new.append(ln)
            for case in ("bear", "base", "bull"):
                v = _vec((ld.get("growth") or {}).get(case))
                if v is not None and _inside("growth", v):
                    ln.growth[case] = v
                    sg.lines_explicit.add(case)
                elif (ld.get("growth") or {}).get(case) is not None:
                    warn.append(f"segments.{name}.lines.{lname}.growth.{case}: need 6 numbers in {BOUNDS['growth']}")
            if ld.get("note"):
                ln.note = str(ld["note"])
            k = _kpi(ld.get("kpi"), f"segments.{name}.lines.{lname}", warn)
            if k:
                _attach_kpi(ln, k, ln.sales[2], changed, f"{sg.name} / {ln.name}")
                sg.lines_explicit.update(("bear", "base", "bull"))
        if new:
            hist_sum = [sum(x.sales[j] or 0 for x in new) for j in range(3)]
            if all(sg.sales[j] is None or abs(hist_sum[j] - sg.sales[j]) <= 0.03 * abs(sg.sales[j]) for j in range(3)):
                sg.lines = new
            else:
                warn.append(f"segments.{name}.lines: defined lines add up to {hist_sum} vs reported {sg.sales}; "
                            "not used")
        for ln in sg.lines:  # a partial set of explicit lines: the others still need every case
            for case in ("bear", "base", "bull"):
                if case not in ln.growth and case in sg.lines_explicit:
                    ln.growth[case] = list(sg.growth.get(case) or [0.0] * N)
        if sg.lines:
            sg.use_lines = True
            changed.append(f"{sg.name} lines")
    if isinstance(sd.get("use_lines"), bool):
        if sd["use_lines"] and not sg.lines:
            warn.append(f"segments.{name}.use_lines: no lines tagged for this segment (define them under lines)")
        else:
            sg.use_lines = sd["use_lines"]
    k = _kpi(sd.get("kpi"), f"segments.{name}", warn)
    if k:
        if sg.use_lines and sg.lines:
            warn.append(f"segments.{name}.kpi: the segment is built from lines; attach the KPI to a line")
        else:
            _attach_kpi(sg, k, sg.sales[2], changed, sg.name)
    align_lines(sg)  # cases not set line by line follow the segment's organic paths
    return warn


def _summary(changed: list) -> str:
    """'4 segments (growth, margin, M&A, SOTP), 26 drivers, 15 valuation inputs, guidance, 6 peers, thesis'."""
    if not changed:
        return "no changes"
    changed = list(dict.fromkeys(changed))
    segs, kinds, drivers, vals, rest = set(), [], 0, 0, []
    for c in changed:
        m = re.match(r"(.+) (growth|margin) (bear|base|bull)$|(.+) (M&A|SOTP multiples|SOTP split)$", c)
        if m:
            segs.add(m.group(1) or m.group(4))
            k = m.group(2) or m.group(5).split()[0]
            if k not in kinds:
                kinds.append(k)
        elif c.startswith("driver "):
            drivers += 1
        elif c.startswith("valuation "):
            vals += 1
        else:
            rest.append(c)
    out = [f"{len(segs)} segment{'s' if len(segs) != 1 else ''} ({', '.join(kinds)})"] if segs else []
    out += [f"{drivers} drivers"] if drivers else []
    out += [f"{vals} valuation inputs"] if vals else []
    return ", ".join(out + rest)


def _wacc_guess(spec: ModelSpec) -> float:
    V = spec.valuation
    return V["rf"][0] + V["beta"][0] * V["erp"][0]


def parse(txt: str) -> dict:
    """JSON text, or Markdown with the JSON in a fenced block (agents often reply that way)."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, re.S)
    return json.loads(m.group(1) if m else txt)


def load(path: str) -> dict:
    with open(path) as fh:
        return parse(fh.read())
