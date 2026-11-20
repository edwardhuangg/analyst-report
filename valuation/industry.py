"""Industry profiles: what a buy-side analyst in each industry models, and the valuation conventions.

The HON template is a segment-driven industrial model. Other industries are valued off different
drivers (memberships x price, product franchises and patent expiries, warehouses x sales per
warehouse, cloud mix and stock comp) and different conventions (longer growth runways, EV/EBITDA vs
P/E, captive finance debt that is operating). A profile carries:

    name / label        how the model and the brief refer to it
    lines_multi         split multi-segment revenue into product lines by default (single-segment
                        companies always use their product / regional lines when the 10-K tags them)
    adj_eps_ex_sbc      default for "adjusted EPS also excludes stock comp" (most software / internet)
    comps_pe_weight     weight of P/E in the comps value (the rest is EV / LTM EBITDA)
    fade_max            longest growth fade after the six explicit years (three-stage DCF)
    playbook            the industry-specific checklist for the analyst (brief + AI prompt)
    kpis                operating metrics that build revenue in this industry (volume x price)
"""
from __future__ import annotations

import re

BASE = {"lines_multi": False, "adj_eps_ex_sbc": 0, "comps_pe_weight": 0.5, "fade_max": 6, "kpis": [],
        "intang_share": 0.3,   # share of an acquisition's price booked as identified intangibles
        "cyclical": False,     # terminal year at mid-cycle; comps and SOTP on mid-cycle earnings, never the trough
        "ronic_rule": "mid"}   # RONIC default: "mid" = halfway WACC..ROIC (moat), "wacc" = commoditizing

PROFILES = {
    "industrial": {
        "label": "Diversified industrial / multi-segment", "comps_pe_weight": 0.5, "fade_max": 4,
        "cyclical": True,
        "playbook": [
            'Orders engine: orders, backlog and book-to-bill by segment; revenue = backlog conversion + book-and-ship, capped by capacity additions; incremental margins 30-35% on volume less acquisition dilution and start-up costs.',
            'Short-cycle names: distributor inventory (destocking / restocking swings reported sales against end demand), ISM PMI and industrial production; build a destocking scenario. Data-center capex exposure: a digestion scenario (orders -20-30% for 2-3 quarters, decremental margins).',
            'Convert total-company guidance (divestiture headwinds, FX) into segment impacts in dollars, not by applying the rate to one segment; run off temporary items (tariff refunds) when the program ends.',
            "Coatings / owned retail (SHW): stores x productivity, price vs volume, raw-material basket with the pricing lag, end-market mix (repaint, new residential, commercial); corporate cost ex interest; the persistent premium is measured against the company's own 10-year P/E band.",
            "Segment organic growth by end market (short- vs long-cycle, backlog, book-to-bill, pricing vs volume).",
            "Segment margins: price/cost, productivity, mix, restructuring savings; incremental margins in up/down cycles.",
            "Portfolio actions (acquisitions, divestitures, spin-offs) with close dates; SOTP splits for pending spins.",
            "Debt-like items (asbestos, environmental, pensions) and stakes outside segment EBITDA.",
            "Cycle: if the latest year is a trough or a peak, the last forecast years and the terminal value must "
            "sit at mid-cycle volumes and margins (the last cycle's average), or the DCF capitalizes the extreme.",
        ],
    },
    "captive_finance": {
        "label": "Industrial with a captive finance arm", "comps_pe_weight": 1.0, "fade_max": 4,
        "cyclical": True,
        "playbook": [
            'Mid-cycle is the core: mid-cycle industry units from the replacement cycle (installed fleet / average fleet age), net farm income, crop prices, dealer new and used inventory; 10-15 years of segment sales and margins set mid-cycle levels (adjusted for precision-tech mix); the terminal year sits at mid-cycle.',
            'Split equipment operations and financial services with the 10-K supplemental consolidating data: equipment ops on EV / mid-cycle EBIT ex-captive; financial services on P/B justified by (ROE - g) / (CoE - g), with receivables growth funded by retained equity at the target leverage. Comps EV ex-captive (CAT includes Cat Financial debt). Short theses need a dated catalyst (early order programs, FY outlook).',
            "Equipment operations and the finance arm are different businesses: equipment ops on EV/EBITDA and FCF; "
            "the finance arm on P/E or P/B of its own equity. Its debt is funded by financing receivables, so it is "
            "excluded from net debt (Assumptions captive_share) and its interest stays inside its segment profit.",
            "Cycle position: farm income / construction activity, dealer inventory, order book, pricing vs volume; "
            "incremental margins down-cycle vs up-cycle. From a trough, the outer years and the terminal value must "
            "reach mid-cycle sales and segment margins (the last full cycle's average and the company's "
            "through-cycle targets), not stay near the trough.",
            "Finance arm: receivables growth, provision for credit losses, spread, ROE; supplemental consolidating "
            "statements in the 10-K give its debt and equity.",
            "Comps on P/E only (peers' EV/EBITDA is distorted by their own captive debt).",
        ],
    },
    "software": {
        "label": "Software / SaaS", "lines_multi": True, "adj_eps_ex_sbc": 1, "comps_pe_weight": 0.5, "fade_max": 8,
        "intang_share": 0.35,
        "playbook": [
            'cRPO / RPO growth leads revenue by 2-3 quarters: drive near-term revenue from it; net revenue retention, new bookings; AI-agent ARR as the swing factor including seat cannibalization.',
            "Stock comp is a real cost: fade it toward a mature 5-6% of revenue over 5-7 years and model dilution net of buybacks; value on FCF - SBC and GAAP EBIT. A strategic-investment portfolio is valued separately (other_stake at carrying value less 20-30%) and its gains are not operating EPS (adj_eps_ex_nonrec = 0 if the company's non-GAAP EPS includes them). Separate true organic growth from acquired growth.",
            "Revenue by product / cloud (lines): subscription vs services; growth from new logos vs expansion; "
            "cRPO / RPO growth as the leading indicator; FX.",
            "Non-GAAP operating margin vs GAAP: stock comp stays a real cost in EBIT and the DCF; set "
            "adj_eps_ex_sbc = 1 to match the company's non-GAAP EPS for comps.",
            "Free cash flow margin and the gap to non-GAAP income (deferred revenue, capitalized commissions); "
            "share count: buybacks vs dilution from stock comp.",
            "Rule of 40 (growth + FCF margin) and the growth runway: a longer fade (fade_years) for durable "
            "growers; terminal growth 3-4%.",
            "Strategic-investment gains/losses are non-recurring (drivers.nonrec), never run-rate.",
        ],
        "kpis": ["Remaining performance obligation / cRPO", "Subscription revenue per customer"],
    },
    "internet": {
        "label": "Internet platform / digital advertising", "lines_multi": True, "adj_eps_ex_sbc": 0,
        "comps_pe_weight": 0.6, "fade_max": 8,
        "playbook": [
            'Capex ROIC is the central variable: model AI infrastructure as an investment program (capex -> capacity -> Cloud revenue and Services monetization) with incremental ROIC by capex cohort; bear = ROIC below WACC.',
            'Depreciation lags capex by useful life (servers ~6 years): D&A rises toward capex; terminal capex / D&A converges to maintenance plus growth. Non-marketable stakes (and venture marks) go in the equity bridge (other_stake); Other Bets valued separately, their losses out of capitalized corporate cost; search economics (queries x monetization x TAC) and regulatory remedies as scenarios.',
            "Revenue by product line (search, video, network, subscriptions, cloud): growth by line, ad pricing vs "
            "volume (paid clicks / impressions), cloud backlog.",
            "Capex intensity (AI / data centers) and the depreciation it will bring: capex and D&A paths must be "
            "consistent with each other and with the terminal year (capex settles near D&A x 1.1-1.3).",
            "Mark-to-market gains on equity stakes are non-recurring: put them in drivers.nonrec (FY1 = the "
            "year-to-date gains already booked) and value the stakes in the SOTP (other_stake).",
            "Segment margins: core vs investment businesses (Other Bets, cloud scaling); regulatory fines are one-offs.",
            "Long growth runway: fade_years 6-8; method weights lean on DCF and SOTP.",
        ],
        "kpis": ["Paid clicks / impressions", "Daily active users"],
    },
    "media": {
        "label": "Media / streaming / subscription", "lines_multi": True, "comps_pe_weight": 1.0, "fade_max": 6,
        "playbook": [
            'Content is the business: cash content spend = content amortization x the guided ratio (~1.1x trending to 1.0-1.05x at maturity); the excess is an operating cash outflow (drivers.other_op), and content amortization as % of revenue is the main margin lever. Members x average revenue per member by region plus ads; engagement (view hours per member) caps price increases in the bear case.',
            "Value on P/E and EV / FCF (EBITDA is meaningless when content amortization is the main cost); peers are global consumer-subscription platforms plus the company's own multiple history. A pending large deal needs standalone, pro forma (price, financing, synergies, library economics) and probability-weighted cases; buybacks constrained by the financing plan.",
            "Revenue by region (lines) and its drivers: paid memberships x average revenue per membership (KPIs, if "
            "disclosed), price increases vs member growth, advertising tier.",
            "Content: cash content spend vs content amortization (the main gap between net income and FCF); "
            "operating margin targets.",
            "FX: reported vs constant-currency growth; guidance is usually given in both.",
            "M&A (pending transactions): model standalone unless closed, but show deal financing and leverage "
            "in the thesis and flags.",
        ],
        "kpis": ["Paid memberships (mm)", "Average revenue per membership ($ / month)"],
    },
    "retail": {
        "label": "Retail / consumer membership", "lines_multi": True, "comps_pe_weight": 0.5, "fade_max": 6,
        "playbook": [
            "Membership models (COST): paid members x fee x renewal rate, fee increases on the historical 5-6 year cadence as discrete events, executive penetration and the reward cost; deferred membership income growth explicit (not an unexplained 'other' cash line); special dividends separate from regular; interest income on average (not beginning) cash when a special dividend leaves.",
            'Merchandise: warehouses / stores x sales per unit with a new-unit ramp, comparable sales, e-commerce; gross margin discipline; reverse DCF first: what growth duration and IRR the price implies.',
            "Revenue build: store / warehouse count x sales per unit (KPIs) — new units vs comparable sales "
            "(traffic x ticket), e-commerce, membership / fee income where reported.",
            "Gross margin and SG&A leverage; fuel / mix effects; shrink; wage inflation.",
            "Capex per new unit and remodels; inventory turns and payables (negative working capital).",
            "Special dividends are one-offs (div_growth path), not run-rate yield.",
        ],
        "kpis": ["Stores / warehouses (year-end)", "Net sales per store / warehouse ($mm)"],
    },
    "housing": {
        "label": "Home improvement / housing-linked retail", "lines_multi": True, "comps_pe_weight": 0.5,
        "fade_max": 6, "cyclical": True,
        "playbook": [
            "Housing cycle: existing-home turnover, home-price appreciation, mortgage rates and home-equity "
            "extraction drive big-ticket and Pro demand. When turnover sits near a multi-decade low, the base case "
            "carries the recovery to mid-cycle turnover over the forecast (or the scenarios carry it with an "
            "explicit probability); the terminal year is at mid-cycle comps and margins, not the trough.",
            "Revenue build: stores x sales per store (comparable sales = transactions x ticket), Pro vs DIY, "
            "acquired distribution (SRS, GMS) as M&A until annualized; the KPI build must tie to modeled revenue.",
            "Gross margin and operating leverage on comps (deleverage below ~2-3% comps); SG&A per store.",
            "Capital return: dividends plus buybacks at the stated leverage target (net debt / EBITDA); cash "
            "above the operating minimum is returned unless a deal or debt paydown is named.",
        ],
        "kpis": ["Stores (year-end)", "Net sales per store ($mm)"],
    },
    "staples": {
        "label": "Consumer staples / brands", "comps_pe_weight": 0.6, "fade_max": 4,
        "playbook": [
            "Organic growth = price/mix + volume by segment / region; FX translation; pricing elasticity.",
            "Gross margin: commodity and input-cost inflation vs pricing and productivity.",
            "Brand portfolio changes (divestitures, acquisitions) and their dilution; emerging-markets exposure.",
            "Dividend policy and payout; leverage targets.",
        ],
    },
    "pharma": {
        "label": "Pharma / biotech", "lines_multi": True, "comps_pe_weight": 0.7, "fade_max": 6,
        "intang_share": 0.7,  # product rights and IPR&D dominate drug-deal purchase prices
        "playbook": [
            "Revenue by product / franchise (lines): launch curves for new drugs, price vs volume, US vs ex-US, "
            "net pricing (rebates, IRA negotiation).",
            "Loss of exclusivity: patent / exclusivity dates of the main products and their erosion curves inside "
            "the six-year forecast; terminal growth must reflect the portfolio after expiries.",
            "Pipeline: late-stage readouts and launches dated before or after the valuation date; acquired IPR&D "
            "charges are one-offs.",
            "Gross margin, R&D and SG&A as % of sales; manufacturing capex for capacity build-outs.",
            "Company non-GAAP EPS excludes amortization and IPR&D: match it for comps.",
        ],
    },
    "semis": {
        "label": "Semiconductors / hardware", "lines_multi": True, "comps_pe_weight": 0.6, "fade_max": 6,
        "cyclical": True,
        "playbook": [
            "Revenue by end market / product line (data center, PC, mobile, auto, industrial); cycle position and "
            "inventory in the channel; customer concentration.",
            "Gross margin vs utilization and mix; opex intensity; stock comp.",
            "Capex (fabs) and its depreciation; supply constraints; export controls.",
            "Normalized (mid-cycle) margins and volumes for the terminal year, from a trough or a peak alike.",
        ],
    },
    "energy": {
        "label": "Energy / commodities", "comps_pe_weight": 0.4, "fade_max": 0,
        "ronic_rule": "wacc",
        "cyclical": True,
        "playbook": [
            "Revenue build: production volumes x realized price (KPIs); commodity price deck by scenario "
            "(bear / base / bull) rather than growth rates.",
            "Unit costs (lifting, refining margins), capex to sustain production, reserve life.",
            "Shareholder returns (buybacks, variable dividends) tied to free cash flow; leverage.",
            "Terminal value on mid-cycle prices and sustaining capex; exit EV/EBITDA on mid-cycle EBITDA.",
        ],
        "kpis": ["Production (mboe/d)", "Realized price ($/boe)"],
    },
    "utility": {
        "label": "Regulated utility", "comps_pe_weight": 0.8, "fade_max": 0,
        "ronic_rule": "wacc",
        "playbook": [
            "Rate base growth x allowed ROE drives earnings; regulatory lag and rate cases.",
            "Capex plan and its financing (equity issuance, debt); dividend payout.",
            "Value on P/E and dividend yield vs peers; DCF with low terminal growth.",
        ],
    },
}


def _match(text: str, *words) -> bool:
    return any(re.search(w, text or "", re.I) for w in words)


def detect(sic: str, industry: str = "", sector: str = "", segments: list = (), captive: dict = None) -> str:
    """Profile key from SIC code, Yahoo industry / sector and the segment structure."""
    sic = str(sic or "")
    ind = f"{industry or ''} {sector or ''}"
    if captive:
        return "captive_finance"
    if _match(ind, r"drug manufacturers", r"biotechnology") or sic[:4] in ("2834", "2835", "2836"):
        return "pharma"
    if _match(ind, r"internet content", r"internet retail") or sic[:4] in ("7370",):
        return "internet"
    if _match(ind, r"software") or sic[:3] == "737":
        return "software"
    if _match(ind, r"entertainment", r"broadcasting", r"publishing") or sic[:3] in ("781", "484", "483"):
        return "media"
    if _match(ind, r"semiconductor", r"computer hardware", r"consumer electronics") or sic[:4] in ("3674", "3571",
                                                                                                    "3572", "3577"):
        return "semis"
    if _match(ind, r"home improvement") or sic[:4] in ("5211",):
        return "housing"
    if _match(ind, r"discount stores", r"retail", r"grocery", r"home improvement") or sic[:2] in ("53", "54", "56",
                                                                                                    "57", "59"):
        return "retail"
    if _match(ind, r"oil & gas", r"\boil\b", r"coal", r"uranium") or sic[:2] in ("13", "29"):
        return "energy"
    if _match(ind, r"utilities") or sic[:2] == "49":
        return "utility"
    if _match(ind, r"beverages", r"household", r"packaged foods", r"tobacco", r"personal products",
              r"confectioners") or sic[:2] in ("20", "21"):
        return "staples"
    return "industrial"


def profile(key: str) -> dict:
    p = dict(BASE)
    p.update(PROFILES.get(key, PROFILES["industrial"]))
    p["key"] = key if key in PROFILES else "industrial"
    return p
