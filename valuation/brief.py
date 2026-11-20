"""Research brief: everything an analyst agent needs to add the judgment layer.

`build(spec, data)` returns Markdown with the company's facts (as parsed from the
filings), the source documents, the checklist of judgments the HON-quality model
contains, the current assumptions as an editable overrides JSON, and the schema.
The same text is the prompt for the in-app AI analyst and the hand-off to an
external agent (Claude Code, ChatGPT), which returns an overrides JSON.
"""
from __future__ import annotations

import json

from . import overrides


def _fmt(v, pct=False, eps=False):
    if v is None:
        return "–"
    if pct:
        return f"{v:.1%}"
    if eps:
        return f"{v:.2f}"
    return f"{v:,.0f}"


CHECKLIST = """\
Work through these, using only information public on or before the valuation date. Each judgment
goes in the overrides JSON with a note that cites its basis (document + figure), the way a PM expects.

1. Guidance. From the latest earnings release before the date: FY sales, organic growth, segment /
   adjusted operating margin, adjusted EPS, free cash flow (low, high). Release date.
2. Reported non-GAAP history and the adjusted-EPS definition contract. Company adjusted EPS for the last
   two fiscal years and the YTD periods; organic growth by segment and in total. Read the company's
   reconciliation table and set the switches so the model's definition reproduces reported adjusted EPS
   within $0.05 (gate 1): adj_eps_ex_sbc, adj_eps_ex_repo (share of non-segment charges excluded: LLY keeps
   acquired IPR&D, so less than 1), adj_eps_ex_nonrec (0 when the company's non-GAAP EPS includes one-off
   gains — CRM's strategic-investment gains; multiples then use operating EPS without them),
   adj_eps_amort_share (SHW excludes only Valspar amortization). When the reconciliation taxes non-GAAP
   pre-tax income at one long-term rate (CRM, most software) set it: drivers.adj_tax for the forecast and
   adj_eps_tax_hist for the history years (gate 1 prints the rates that would tie). Reported figures come
   from the results columns of the reconciliation, never from a guidance range. Use null when the company
   reports no adjusted EPS, never GAAP or year-to-date figures in its place.
3. Perimeter and deals (value conservation). Search the 8-Ks, press releases and the 10-Q subsequent-events
   and acquisitions notes before calling a price "undisclosed". Closed acquisitions: growth in
   segments.<name>.mna, cash in drivers.acq (it is deducted from value in the year paid), funding in
   drivers.debt_iss, intangibles amortized on a stated life; a price that is not disclosed is estimated (the
   target's revenue x a peer EV / sales multiple) and said so: acquired revenue is never free. Signed pending
   deals of any size (a small deal still moves value by its premium; gate 32): keep them
   OUT of the segment forecast (standalone) and enter a "deals" item (probability, consideration, financing,
   the target's own EBITDA and margin, synergies phased, integration cost, break fee). Separations (spins,
   RMTs, splits): forecast RemainCo from the close (remove the business through its growth / mna) and enter
   a "deals" item of kind "separation" (stake distributed, the piece's EBITDA and its own multiple, cash
   received, separation and stranded costs). A pending acquisition is worth its standalone value less the
   price: enter its revenue and margin (or EBITDA) and a peer multiple — without them the engine values it
   at the price less a 30% control premium, never at zero. Every 8-K filed after the balance sheet the
   forecast opens from is listed on the Checks tab (gate 10): enter each closing's price in drivers.acq
   (FY1, above the year-to-date amount) or as a closed deal, or dismiss it in thesis.explanations; an
   accelerated share repurchase that settles after the 10-Q changes the share count (market.basic_shares,
   from the 8-K or the next filing).
4. Segment paths. Organic growth and margin, bear / base / bull, for six years. Base anchored to
   guidance in year 1; explain the shape (cycle, backlog, pricing, mix, restructuring) in notes.
   Check: the base case should land inside the guidance ranges (Interim tab "Within range?").
5. Below segment profit. Acquired-intangible amortization (company guide if given), repositioning /
   restructuring run-rate, pension & other income, interest rate on debt, tax rate.
6. Cash flow and the opening balance sheet. The forecast opens from the latest 10-Q: FY1 debt, buybacks
   (an ASR) and acquisitions already done this year are in FY1 (the engine floors them at the year to date);
   first-year interest within ±15% of the annualized run-rate. Capex (guide), working capital (year-end
   days and ratios from the 10-K's year-end, never the 10-Q's seasonal balance: every forecast year ends on the
   fiscal year-end), buybacks by policy (the engine returns cash above FY0's share of sales as buybacks from
   FY2; to hold it for a deal or debt repayment, explain under thesis.explanations['cash build']), dividends (regular and special separately),
   debt issuance / repayment. Never calibrate to guidance through tax, other income or a generic "other"
   line: move the operating driver that explains the gap (segment margin, price / mix, share count timing,
   working-capital days). Every first-year line away from its normal level shows on the Checks tab plug table;
   a non-operating plug fails gate 11, and FY2 must not jump because a plug unwinds (gate 12).
7. Debt-like items outside net debt (asbestos, environmental, litigation, nuclear decommissioning)
   with offsetting recoveries; give the label. Stakes or assets outside segment EBITDA (equity stakes,
   venture holdings) with a value and label.
8. Valuation. The risk-free rate, the ERP and the credit-spread table are set once per valuation date for
   every name (you cannot change them); beta comes from the method (bottom-up peers, Blume-adjusted) and
   may move at most ±0.10 with a written reason; the cost of debt follows the agency rating the filings disclose (else a synthetic
   rating from coverage; ±0.75pt for a cited bond yield). Never set an input by reference to the share price. Terminal value is NOPAT ×
   (1 − g ÷ RONIC) ÷ (WACC − g) — the fade moves RONIC from the return the last forecast year implies (DCF
   C89) to terminal RONIC, and gate 27 fails if free cash flow drops more than 5% into the fade; terminal
   growth defaults to the house 3.75% (long-run US nominal GDP; go
   lower only for an industry reason such as secular decline or a regulated return); set RONIC (between
   WACC and the current ROIC for a moat, about WACC for a commoditizing business, never above ROIC without
   an argument). The exit multiple
   is derived from the same framework by default; if you set one it must sit within ±20% of the perpetuity-
   implied multiple and imply growth within 1pt of g (gate 15). The terminal margin must sit inside the 10–
   90th percentile of the 10-year history in the brief unless the thesis argues a structural change; a
   cyclical ends at mid-cycle (gate 16). Method weights and why.
9. Comps. 5-8 economic peers (similar ROIC, growth, margins and business model, regardless of industry
   label: COST vs WMT and BJ, not dollar stores): price on the date and each peer's own adjusted EPS on a
   consistent basis (company-adjusted, defined SBC treatment), calendarized to this company's fiscal year.
   The engine fits P/E against growth (regression or PEG) and blends the company's own 10-year P/E band;
   a flat "quality premium" needs evidence. Use the multiple the sector is priced on: cyclicals on mid-cycle
   EPS (never trough), captive finance on P/E (equipment ex-captive), content businesses on P/E and EV/FCF
   (not EV/EBITDA), software on EV/(FCF − SBC) alongside non-GAAP P/E.
10. SOTP. A base multiple per segment set against named peers (bear / base / bull), on the same earnings
    basis as those peers (EBITDA after stock comp); corporate costs are capitalized (never 0x: the engine
    uses the blended segment multiple); ventures and stakes valued separately (other_stake: carrying value or
    last round, with a haircut) and their losses kept out of capitalized corporate cost; cyclicals on mid-
    cycle EBITDA; sub-splits for pending separations net of separation and stranded costs. Single-segment
    companies get no SOTP (its weight goes to the DCF).
11. Thesis (the seven questions a PM answers in two minutes). price_implies: one sentence from the reverse
    DCF (implied IRR, years of growth or terminal margin the price needs). variant_view: "we differ from the
    Street on X because Y; we will know by Z (date / event)" — without one the rating is Neutral. Scenarios:
    bear and bull must move the thesis-critical variables (DE: recovery timing and mid-cycle margin; CRM:
    organic reacceleration; NFLX: deal closes or not; GOOGL: AI capex return), not ±1pt on everything; set
    scenario_probs if not 25/50/25. catalysts: dated, each with the datapoint to watch ("Q4 cRPO organic
    growth ≥12%", "used large-ag inventory down another 10%"). breaker: the one risk that breaks the thesis,
    quantified. sensitivities: the two inputs value is most sensitive to. consensus_notes: why the model
    differs from consensus by more than 3%. The rating is computed by rule: Long when the probability-weighted
    value is 15%+ above the price (the model's error band) AND the price offers a long-run IRR 2pt+ above WACC
    with 2:1 skew and the price inside bear–bull; Short at −15% and −2pt with a dated catalyst, bull ≤ +25%,
    thesis.short_evidence (when the premium last compressed, by how much, on what), a crowding check
    (short interest) and, for a low-volatility compounder, a pair (thesis.pair); else Neutral (with a bias);
    Under review while a gate fails or any warning / open check is unexplained. The expected 12-month return
    assumes a third of the gap closes a year. The recommendation's call is generated from the rule:
    thesis.recommendation is one supporting sentence with no call in it, and no thesis text may state a
    call or bias the rule does not (gate 25). Override only with rating_override {rating, reason},
    falsifiable and in the thesis: it is labeled "Override" and logged. The price must sit inside
    bear–bull (gate 41): a bull below the price of a liquid large cap means the scenarios are
    miscalibrated — build it from what the reverse DCF says the price requires. Base case at consensus unless the variant view says why and when it is
    tested (management targets go in the bull); the bear is the breaker made numerical, 15%+ below base.
    Write model numbers as placeholders — {fv} {price} {upside}
    {dcf_pg} {dcf_exit} {comps} {sotp} {bear} {base} {bull} {pw} {etr} {alpha} {irr} (long-run IRR at the
    price, not a 3-year return) {wacc} {ke} {pe} {exit_pe} {ret3y} (3-year return) {gap} {irr_spread}
    {req_margin} {req_g} {p_bear_implied} (bear probability the price implies) {p_bear_long} (bear probability
    where the call changes) {peer_discount_pe} {ret3y_pw} (3-year return, probability-weighted) {g} {beta}
    — the file shows the live value; a typed valuation figure that no
    longer matches the model fails gate 28. thesis.explanations: every warning gate and open check you leave
    gets one entry under its key ("gate 8": "why the run-rate does not apply", or the first words of the
    check): an unexplained one holds the rating. A working-capital path away from history (deferred revenue,
    receivable days) or a RONIC far from the forecast's own incremental return is a thesis-level view: state
    it in variant_view and consensus_notes. No sentence restates a model number without interpreting it;
    "expensive" only relative to what the price implies; nothing dated after the valuation date.
12. The year so far. The current fiscal year must fit what is already reported: the Interim tab shows
    the rest of the year the model implies vs the year-to-date trend. One-offs already booked (gains on
    stakes, settlements, untagged items in other income) go in drivers.nonrec for FY1 so GAAP EPS is
    right and adjusted EPS excludes them; history one-offs are already separated where tagged.
13. Revenue build and runway. Use the revenue lines / KPIs the industry is driven by (launch curves,
    patent expiries, regions, units x productivity). For a business still growing well above GDP in
    the last forecast year, set fade_years (three-stage DCF) rather than forcing terminal growth up.
    Keep capex and D&A consistent in the terminal year. If a method still disagrees badly with the
    others, say which and why in model_says, and weight it accordingly.
14. Comps quality. Peers must share the business model and growth / margin profile; exclude companies
    being acquired (or acquiring the target) and pure outliers. comps_pe_weight sets P/E vs EV/EBITDA
    (peers' LTM EBITDA and net debt come from their filings).
"""


def build(spec, data=None, review: dict = None) -> str:
    """review: check.review() of the model as currently set up (what the defaults compute)."""
    h, it, I = spec.hist, spec.interim, spec.hist.IS
    L = []
    L.append(f"# Research brief: {spec.name} ({spec.ticker}) as of {spec.as_of:%B %-d, %Y}")
    L.append(f"Share price {spec.price:,.2f} · basic shares {spec.basic_shares:,.1f}mm · dilutive {spec.dilutive:,.1f}mm"
             f" · fiscal year ends {spec.fye_text} · history {spec.hist_labels[0]}–{spec.hist_labels[-1]} · forecast "
             f"{spec.fc_labels[0]}–{spec.fc_labels[-1]}")
    L.append("\n## Source documents (read these)")
    for k, v in (spec.filing_urls or {}).items():
        if v:
            L.append(f"- {k}: {v}")
    L.append("- EDGAR filing index: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
             f"{data.company.cik if data else spec.ticker}&type=8-K&dateb={spec.as_of:%Y%m%d}")

    L.append("\n## Income statement (10-K, $mm)")
    L.append("| | " + " | ".join(spec.hist_labels) + " |")
    L.append("|---|" + "---|" * 3)
    for k, lab, kw in [("sales", "Net sales", {}), ("opinc", "Operating income (GAAP)", {}),
                       ("amort", "Acquired-intangible amortization", {}), ("sbc", "Stock compensation", {}),
                       ("interest_exp", "Interest expense", {}), ("pretax", "Pre-tax income", {}),
                       ("tax", "Tax", {}), ("ni_attr", "Net income attributable", {}),
                       ("dil_shares", "Diluted shares (mm)", {}), ("eps_dil", "Diluted EPS", {"eps": True})]:
        L.append(f"| {lab} | " + " | ".join(_fmt(v, **kw) for v in I[k]) + " |")
    L.append("\n## Segments (10-K, $mm)")
    L.append(f"Segment measure: {h.seg_measure} ({h.seg_profit_concept or 'single segment'}); "
             f"seg_after_charges = {spec.seg_after_charges} — {spec.seg_after_charges_note}")
    for sg in spec.segments:
        L.append(f"- {sg.name}: sales " + " / ".join(_fmt(v) for v in sg.sales) + "; profit "
                 + " / ".join(_fmt(v) for v in sg.profit))
    if not spec.segments_reported:
        L.append("- Single-segment model: segment profit = GAAP operating income + acquired-intangible amortization + "
                 "stock comp (or = GAAP operating income when seg_after_charges = 1), so it still contains acquired "
                 "IPR&D, restructuring, impairments and litigation. A company margin guide that excludes those items "
                 "is on a different basis: either set the segment margin on the model's basis, or set it to the "
                 "company's basis and put every excluded charge — including those already booked this year — in "
                 "repo_pct. Check: the rest of the year the model implies must have a plausible operating margin.")
    L.append("- Corporate & all other: sales " + " / ".join(_fmt(v) for v in h.corp_sales) + "; profit "
             + " / ".join(_fmt(v) for v in h.corp_profit))
    if any(sg.lines for sg in spec.segments):
        L.append(f"\n## Revenue lines (10-K disaggregated revenue by {h.lines_axis or 'product'}, $mm)")
        L.append("Lines marked ON build their segment's sales (set their growth under segments.<name>.lines); OFF "
                 "lines can be switched on with use_lines.")
        for sg in spec.segments:
            for ln in sg.lines:
                g = (f"{ln.sales[2] / ln.sales[1] - 1:+.0%}" if ln.sales[1] else "new")
                L.append(f"- {sg.name} / {ln.name} [{'ON' if sg.use_lines else 'OFF'}]: "
                         + " / ".join(_fmt(v) for v in ln.sales) + f" (last year {g})")
    if spec.captive:
        c = spec.captive
        L.append(f"\n## Captive finance arm: {c['segment']}")
        L.append(f"Its interest ({' / '.join(_fmt(v) for v in c['interest'])}mm) is reclassified above EBIT (inside its "
                 f"segment profit); valuation.captive_share = {spec.valuation['captive_share'][0]:.0%} of total debt is "
                 "treated as funding its receivables and excluded from net debt. Replace the share with the finance "
                 "arm's debt / total debt from the 10-K supplemental consolidating data.")
    C = h.CF
    L.append("\n## Cash flow (10-K, $mm)")
    for k, lab in [("cfo", "Cash from operations"), ("capex", "Capex"), ("acq", "Acquisitions"),
                   ("buyback", "Buybacks"), ("divs", "Dividends"), ("debt", "Net debt issuance")]:
        L.append(f"- {lab}: " + " / ".join(_fmt(v) for v in C[k]))
    L.append("\n## Latest period")
    L.append(spec.interim_title)
    if it.available:
        for k, lab in [("sales", "Net sales"), ("opinc", "Operating income"), ("seg_profit", "Segment profit"),
                       ("eps", "Diluted EPS"), ("cfo", "CFO"), ("capex", "Capex")]:
            L.append(f"- {lab}: {it.label_prior} {_fmt(it.ytd_prior.get(k), eps=k == 'eps')} → {it.label_cur} "
                     f"{_fmt(it.ytd_cur.get(k), eps=k == 'eps')}")
    b = it.bs_cur
    debt = sum(b.get(k, 0) or 0 for k in ("st_debt", "cur_ltd", "ltd"))
    L.append(f"- Debt {debt:,.0f}mm, cash & ST investments {(b.get('cash') or 0) + (b.get('sti') or 0):,.0f}mm at "
             f"{it.period_end:%b %-d, %Y}")
    if it.available:
        oc, op_ = it.ytd_cur.get("other_inc"), it.ytd_prior.get("other_inc")
        L.append(f"- Other income (pre-tax income – EBIT + interest): {it.label_prior} {_fmt(op_)} → {it.label_cur} "
                 f"{_fmt(oc)}; tagged one-offs (securities gains, debt extinguishment) {it.label_cur} "
                 f"{_fmt(it.ytd_cur.get('nonrec'))}")
        L.append(f"- Default {spec.fc_labels[0]} one-offs (drivers.nonrec): {spec.drivers['nonrec'][0][0]:,.0f}mm — "
                 f"{spec.drivers['nonrec'][1]}")
    if spec.consensus:
        L.append(f"- Consensus (Yahoo, FY-matched): {json.dumps(spec.consensus, default=str)}")
    if spec.eps_est and spec.eps_est[0]:
        L.append(f"- Consensus EPS FY1: {spec.eps_est[0]:.2f} ({spec.eps_est[1]})")
    from . import gates as G
    h10 = spec.hist10 or {}
    if h10.get("sales"):
        sal = h10["sales"]
        sig = G.growth_sigma(spec)
        L.append("- 10-year revenue (companyfacts): " + ", ".join(f"FY{y} {v / 1e3 if v > 1e5 else v:,.0f}"
                                                                   for y, v in sal[-10:]) +
                 (f"; standard deviation of annual growth {sig:.1%} (the bear's growth sits about one sigma below "
                  "base: gate 33)" if sig is not None else ""))
        sl = dict(sal)
        acq = [(y, a) for y, a in h10.get("acq") or [] if sl.get(y)]
        if acq:
            pct = sum(a for _, a in acq) / sum(sl[y] for y, _ in acq)
            L.append(f"- Cash paid for acquisitions over {len(acq)} years: {sum(a for _, a in acq):,.0f} "
                     f"({pct:.1%} of sales; stock-funded deals not included). Goodwill + intangibles FY0: "
                     f"{(spec.hist.BS['goodwill'][2] or 0) + (spec.hist.BS['intang'][2] or 0):,.0f}mm; ROIC including "
                     f"goodwill FY0 {(spec.capital or {}).get('roic_gw') or 0:.1%} vs operating ROIC "
                     f"{(spec.capital or {}).get('roic') or 0:.1%} (gate 34: RONIC when growth was bought)")
    # what a short needs (PM round 4): the own multiple band, crowding, the agency rating
    if spec.pe_hist:
        pes = sorted(pe for _, pe in spec.pe_hist if 0 < pe < 200)
        if pes:
            L.append(f"- Own trailing P/E at fiscal year-ends ({len(pes)} years): " +
                     ", ".join(f"FY{y} {pe:.0f}x" for y, pe in spec.pe_hist) +
                     f"; median {pes[len(pes) // 2]:.0f}x (a premium that has lasted is a valuation view; a short "
                     "needs thesis.short_evidence of when it last compressed)")
    si = spec.short_interest or {}
    if si:
        L.append(f"- Short interest (Yahoo, current): " + ", ".join(f"{k} {v}" for k, v in si.items()))
    cap_ = spec.capital or {}
    if cap_.get("rating"):
        L.append(f"- Credit rating for the cost of debt: {cap_['rating']} ({cap_.get('rating_src', '')}); "
                 f"{spec.valuation['kd'][1]}")

    if review:
        from . import check
        L.append("\n## What the model computes with the current assumptions")
        L.append("```\n" + check.review_text(review) + "\n```")
    if spec.data_notes:
        L.append("\n## Data notes from the pipeline (resolve these)")
        L.extend(f"- {f}" for f in spec.data_notes)

    if data is not None:
        from . import facts
        kh = facts.kpi_history(data.company.cik, spec.period_ends, spec.as_of)
        if kh:
            L.append("\n## Operating KPIs tagged in the filings (fiscal year ends " +
                     " / ".join(f"{d:%b %Y}" for d in spec.period_ends) + ")")
            for k, vals in kh.items():
                L.append(f"- {k}: " + " / ".join("–" if v is None else f"{v:,.0f}" for v in vals) +
                         " — use for a volume x price KPI build (price = segment or line sales per unit)")
    pr = spec.profile or {}
    L.append(f"\n## Industry: {pr.get('label', 'Industrial')} — what matters for this company")
    L.extend(f"- {x}" for x in pr.get("playbook", []))
    if pr.get("kpis"):
        L.append(f"- Typical operating KPIs (use segments.<name>.kpi or lines.<line>.kpi when the company discloses "
                 f"their history): {', '.join(pr['kpis'])}")
    L.append("\n## What to decide")
    L.append(CHECKLIST)
    L.append("## Current assumptions (edit this JSON and return it; keep keys you don't change or drop them)")
    doc = overrides.export(spec)
    for sg in doc["segments"].values():
        sg.pop("history", None)
        for ln in (sg.get("lines") or {}).values():
            ln.pop("history", None)
    L.append("```json\n" + json.dumps(doc, indent=1, default=str) + "\n```")
    L.append("\n## Overrides format")
    L.append("```\n" + overrides.SCHEMA_DOC + "```")
    L.append("\nReturn ONLY the overrides JSON object.")
    return "\n".join(L)
