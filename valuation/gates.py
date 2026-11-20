"""Pre-publication gates (critique Part E): machine checks a model must pass before it reaches a PM.

Each gate returns PASS, WARN, FAIL or N/A with the evidence. A FAIL is fed to the AI review as a hard problem
and holds the rating at "Under review" until it is resolved (pipeline.rate). `plugs` lists every line that was
moved away from its normal level in the first forecast year, so guidance calibration is visible, not hidden.
"""
from __future__ import annotations

import re
from datetime import date
from statistics import median

FC = "FGHIJK"
# a driver whose own note says it was set to hit a guide is a plug, wherever it sits (PM round 2: interest cut
# "to tie to the EPS guide"; receivable days moved "so FY27 FCF meets the guide")
CALIB_RX = re.compile(r"(?i)\b(calibrat\w*|to (?:tie|match|hit|meet|reach|reconcile)\b[^.;]{0,90}\b(?:guid\w*|guide|"
                      r"eps|fcf|free cash flow|consensus)|so (?:that )?[^.;]{0,40}\b(?:eps|fcf|free cash flow)\b"
                      r"[^.;]{0,20}\b(?:meets?|ties?|matches|hits?)\b|"
                      # PM round 4: CRM's FY27 interest "raised to bring adjusted EPS back inside guidance"
                      r"\b(?:bring|brings|keep|keeps|put|puts|land|lands|place|places|get|gets|pull|pulls)\b[^.;]{0,60}"
                      r"\b(?:inside|within|into|in line with|back to)\b[^.;]{0,30}\b(?:guid\w*|guide|consensus))")
def calibrated_note(note) -> bool:
    """The note says the driver was set to hit a guide (not merely that it was not: CRM round 5, interest "reset to
    the run-rate estimate rather than calibrated to adjusted-EPS guidance" read as a plug)."""
    t = str(note or "")
    for m in CALIB_RX.finditer(t):
        if not re.search(r"(?i)\b(rather than|instead of|not|never|no longer|without being|isn't|is not)\b[^.;]{0,12}$",
                         t[max(0, m.start() - 30):m.start()]) and "not a calibration" not in t.lower():
            return True
    return False


NONOP_DRIVERS = ("int_rate", "int_yield", "etr", "other_nonop", "nonrec", "nci_ni", "adj_tax")
WC_DRIVERS = {"dso": (-1, "sales"), "dio": (-1, "cogs"), "dpo": (1, "cogs"), "accrued_pct": (1, "pct"),
              "oca_pct": (-1, "pct")}
PASS, WARN, FAIL, NA = "PASS", "WARN", "FAIL", "N/A"


RONIC_CAP = 0.60  # terminal RONIC ceiling: the default (pipeline.default_ronic), the analyst bound and gate 34's reference


def _n(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _nz(x):
    return _n(x) or 0.0


def plugs(spec, v: dict, IS: dict) -> list[dict]:
    """First-forecast-year lines set away from their normal level (the years after, and history)."""
    D = spec.drivers
    sales = _n(v.get(("IS", f"F{IS['sales_b']}"))) or 1.0
    shares = _n(v.get(("IS", f"F{IS['shares']}"))) or 1.0
    eps = abs(_n(v.get(("IS", f"F{IS['op_eps']}"))) or 1.0)
    pretax = _n(v.get(("IS", f"F{IS['pretax']}"))) or 0.0
    etr = D["etr"][0]
    out = []

    def add(line, kind, fy1, normal, delta_mm, after_tax, why, force=None):
        pct_rev = abs(delta_mm) / abs(sales)
        pct_eps = abs(after_tax / shares) / eps if eps else 0.0
        if force:
            status = force
        elif kind == "non-operating":
            status = FAIL if pct_eps > 0.02 or pct_rev > 0.005 else PASS
        else:
            status = WARN if pct_rev > 0.02 or pct_eps > 0.05 else PASS
        out.append({"line": line, "kind": kind, "fy1": fy1, "normal": normal, "delta_mm": delta_mm,
                    "pct_rev": pct_rev, "pct_eps": pct_eps, "status": status, "why": why})

    normal_etr = median(etr[1:]) if len(etr) > 1 else etr[0]
    if abs(etr[0] - normal_etr) > 0.02:
        d = (etr[0] - normal_etr) * pretax
        add("Effective tax rate", "non-operating", etr[0], normal_etr, -d, -d, D["etr"][1])
    # the company's non-GAAP tax rate drives operating EPS, so every multiple: FY1 off its stated history is a plug
    # (CRM v4: 24.3% "needed to reconcile ... to the adjusted EPS guide" vs the 20.5-22% the releases state)
    ngt_h = [x for x in (getattr(spec, "adj_eps_tax_hist", None) or []) if _n(x)]
    ngt = (D.get("adj_tax") or ([0.0], ""))[0]
    if ngt_h and _n(ngt[0]) and abs(ngt[0] - median(ngt_h)) > 0.015:
        d = ngt[0] - median(ngt_h)
        at = -d * eps * shares / max(1 - ngt[0], 0.5)
        add("Non-GAAP tax rate in adjusted / operating EPS", "non-operating", ngt[0], median(ngt_h), at, at,
            D["adj_tax"][1])
    hist_on = [x for x in spec.hist.IS.get("pension", []) if _n(x) is not None] if hasattr(spec.hist, "IS") else []
    on = D["other_nonop"][0]
    normal_on = median(list(on[1:3]) + hist_on[-1:]) if len(on) > 2 else on[0]
    if abs(on[0] - normal_on) > 0.002 * abs(sales):
        d = on[0] - normal_on
        add("Pension & other non-operating income", "non-operating", on[0], normal_on, d, d * (1 - etr[0]),
            D["other_nonop"][1])
    oo = D["other_op"][0]
    normal_oo = median(oo[1:]) if len(oo) > 1 else 0.0
    if abs(oo[0] - normal_oo) > 0.005 * abs(sales):
        add("Other operating cash items", "operating (cash)", oo[0], normal_oo, oo[0] - normal_oo, 0.0,
            D["other_op"][1])
    for sg in spec.segments:
        if sg.mna and sg.mna[0] < -0.01 and not any(w in (sg.mna_note or "").lower()
                                                     for w in ("divest", "sold", "sale of", "fx", "currency", "exit")):
            seg_sales = _nz(sg.sales[2])
            add(f"{sg.name}: M&A / FX contribution", "operating (revenue)", sg.mna[0], 0.0, sg.mna[0] * seg_sales,
                sg.mna[0] * seg_sales * 0.2 * (1 - etr[0]), sg.mna_note)
        m = sg.margin.get("base") or []
        hist_m = [p / s_ for p, s_ in zip(sg.profit, sg.sales) if _n(p) is not None and _n(s_)]
        if len(m) > 1 and hist_m and m[0] > hist_m[-1] + 0.01 and m[1] < m[0] - 0.005:
            d = (m[0] - m[1]) * _nz(sg.sales[2])
            add(f"{sg.name}: one-year margin step (FY1 above both FY0 and FY2)", "operating (margin)", m[0], m[1], d,
                d * (1 - etr[0]), "")
    # drivers set to hit a guide, by their own note: a non-operating one is a plug, an operating one a disclosed
    # calibration that the thesis has to explain
    first = {}
    for _, key, before, _after in spec.audit:
        first.setdefault(key, before)
    for k, (vals, note) in D.items():
        if not calibrated_note(note):
            continue
        b0 = first.get(f"drivers.{k}")
        d = (vals[0] - b0[0]) if isinstance(b0, list) and b0 and _n(vals[0]) is not None and _n(b0[0]) is not None \
            else 0.0
        nonop = k in NONOP_DRIVERS
        add(f"{k} set to meet guidance (its note: '{str(note)[:80]}')", "non-operating" if nonop else
            "operating (calibration)", vals[0], b0[0] if isinstance(b0, list) and b0 else vals[0], d, 0.0, note,
            force=FAIL if nonop else WARN)
    # working-capital paths moved away from the defaults: a thesis-level view, with its cash cost by the last year
    s6 = _n(v.get(("IS", f"K{IS['sales_b']}"))) or sales
    cogs6 = s6 * (D["cogs_pct"][0][-1] if D.get("cogs_pct") else 0.6)
    moved, cash = [], 0.0
    for k, (sign, base) in WC_DRIVERS.items():
        b0 = first.get(f"drivers.{k}")
        if not isinstance(b0, list) or not b0 or k not in D:
            continue
        dl = D[k][0][-1] - b0[-1]
        c = sign * dl * (s6 if base == "pct" else (s6 if base == "sales" else cogs6) / 365)
        if abs(c) > 0.002 * s6:
            moved.append(f"{k} {b0[-1]:.3g} → {D[k][0][-1]:.3g}")
            cash += c
    if moved:
        add("Working capital away from the default path (" + "; ".join(moved) + ")", "operating (working capital)",
            0.0, 0.0, cash, 0.0, "cumulative cash effect by the last forecast year; state the view in the thesis",
            force=WARN if abs(cash) > 0.02 * s6 else PASS)
    return out


def growth_sigma(spec) -> float | None:
    """Standard deviation of annual revenue growth over up to 10 years (companyfacts): how far a growth forecast
    has been wrong for this company, the yardstick for the bear / bull spread."""
    sales = [v for _, v in (getattr(spec, "hist10", {}) or {}).get("sales") or []]
    gs = [b / a - 1 for a, b in zip(sales, sales[1:]) if a and b and a > 0]
    gs = [x for x in gs if -0.5 < x < 1.0]
    if len(gs) < 4:
        return None
    m = sum(gs) / len(gs)
    return (sum((x - m) ** 2 for x in gs) / (len(gs) - 1)) ** 0.5


def scen_growth(spec, case: str, years: tuple = (1, 3)) -> float:
    """Sales-weighted average organic growth of a scenario over forecast years 2-3 (the first is anchored to
    guidance)."""
    tot, w = 0.0, 0.0
    for sg in spec.segments:
        for u in sg.units:
            s0 = _nz(u.sales[2])
            path = (u.growth.get(case) or u.growth.get("base") or [0.0])[years[0]:years[1]]
            if s0 > 0 and path:
                tot += s0 * sum(path) / len(path)
                w += s0
    return tot / w if w else 0.0


def scenario_gate(spec, bear: float, base: float, bull: float, px: float) -> dict:
    """Gate 33 (PM round 3: CRM's bear grew 7.5% and sat 6% below the price, so the 7.3:1 skew was an artifact).
    The bear is built around the thesis breaker and is at least 15% below base; its growth sits about one
    historical standard deviation below base in years 2-3 (year 1 is anchored to guidance); skew above 3:1 is a
    sign the bear is not a bear."""
    probs = []
    if bull < base * 0.99 or bear > base * 1.01:  # NEE: bull $42 < bear $66 < base $67 (growth funded below its cost)
        return {"id": 33, "gate": "Bear case is a real bear; scenarios in order (bear ≤ base ≤ bull)", "status": FAIL,
                "detail": f"scenario values out of order: bear ${bear:,.2f}, base ${base:,.2f}, bull ${bull:,.2f} — more "
                          "growth is worth less, so the valuation method is broken for this business (a regulated "
                          "utility's unlevered DCF, capex-funded growth below the cost of capital): fix the method "
                          "before the scenarios mean anything"}
    down = 1 - bear / base if base else 0.0
    if down < 0.15:
        probs.append(f"bear ${bear:,.2f} is only {down:.0%} below base ${base:,.2f} (a 2-year forecast is routinely "
                     "20%+ wrong)")
    if bear >= px:
        probs.append(f"the bear (${bear:,.2f}) is above the price: no downside case")
    elif bull > px and (bull - px) / (px - bear) > 3:
        probs.append(f"skew {(bull - px) / (px - bear):.1f}:1 (above 3:1 the bear is usually not a bear)")
    sig = growth_sigma(spec)
    gb, gr = scen_growth(spec, "base"), scen_growth(spec, "bear")
    if sig is not None and sig >= 0.02 and gb - gr < min(0.75 * sig, 0.05) - 1e-9:
        probs.append(f"bear growth {gr:.1%} vs base {gb:.1%} in forecast years 2-3 ({(gb - gr) * 100:.1f}pt apart) "
                     f"against a {sig:.1%} standard deviation of 10-year revenue growth: about {min(0.75 * sig, 0.05) * 100:.1f}pt "
                     "is the historical error")
    detail = ("; ".join(probs) + ". Build the bear around the thesis breaker (its growth, margin and stock-comp "
              "levels), not ±1pt on the base") if probs else \
        (f"bear ${bear:,.2f} ({down:.0%} below base), bull ${bull:,.2f}" +
         (f"; bear growth {gb - gr:.1%} below base vs 10-year σ {sig:.1%}" if sig is not None else ""))
    return {"id": 33, "gate": "Bear case is a real bear: built on the breaker, 15%+ below base, growth about one "
                              "historical σ below; skew ≤ 3:1", "status": WARN if probs else PASS, "detail": detail}


def kpi_mna_gap(spec, i_, u_) -> list:
    """Expected KPI-vs-model gap by forecast year from acquired / FX revenue outside the organic KPI build."""
    sg = spec.segments[i_] if isinstance(i_, int) and 0 <= i_ < len(spec.segments) else \
        next((x for x in spec.segments if u_ in x.units), None)
    if sg is None or len(sg.units) != 1:
        return [0.0] * 6
    kd = getattr(u_, "kpi", None) or {}
    try:
        gv, gp = kd["volume"]["growth"]["base"], kd["price"]["growth"]["base"]
    except (KeyError, TypeError):
        return [0.0] * 6
    out, f = [], 1.0
    for o_v, o_p, m in zip(gv, gp, sg.mna):
        o = (1 + o_v) * (1 + o_p) - 1
        f *= (1 + o + _nz(m)) / (1 + o)
        out.append(1 / f - 1)
    return out


def round3_gates(spec, v: dict, out: dict, IS: dict) -> list:
    """Gates added after the round-3 PM critique: (gate id, name, status, detail)."""
    get = lambda sh, c: _n(v.get((sh, c)))  # noqa: E731
    res = []
    B, I = spec.hist.BS, spec.hist.IS
    s0 = _nz(I["sales"][2])
    sales_fc = [get("IS", f"{c}{IS['sales_b']}") for c in FC]
    # 34: RONIC for a company whose growth has been bought (goodwill + intangibles a large share of capital)
    ron = spec.valuation.get("ronic", (None,))[0]
    wacc, roic_t = get("DCF", "C17"), get("DCF", "C93")
    try:
        gw = _nz(B["goodwill"][2]) + _nz(B["intang"][2])
        cap_sh = (spec.valuation.get("captive_share", (0.0,))[0] or 0.0) if spec.captive else 0.0
        ic_gw = (_nz(B["equity_total"][2]) + _nz(B["redeem_nci"][2]) +
                 (_nz(B["st_debt"][2]) + _nz(B["cur_ltd"][2]) + _nz(B["ltd"][2])) * (1 - cap_sh) -
                 _nz(B["cash"][2]) - _nz(B["sti"][2]) - _nz(B["lt_inv"][2]))
    except (KeyError, IndexError, TypeError):
        gw, ic_gw = 0.0, 0.0
    gw_share = gw / ic_gw if ic_gw > 0 else 0.0
    h10 = getattr(spec, "hist10", {}) or {}
    sl = dict(h10.get("sales") or [])
    acq10 = [(y, a) for y, a in h10.get("acq") or [] if sl.get(y)]
    acq_pct = sum(a for _, a in acq10) / sum(sl[y] for y, _ in acq10) if acq10 else 0.0
    acq_fc = list(spec.drivers.get("acq", ([0.0] * 6, ""))[0])
    recurring = all(_nz(a) >= 0.005 * _nz(sv) for a, sv in zip(acq_fc[3:], sales_fc[3:]) if sv)
    if ron is not None and wacc and roic_t is not None:
        bought = gw_share >= 0.20 or acq_pct >= 0.03  # DE: 7% goodwill, 2.0% of sales on deals is not bought growth
        cap = max(wacc, roic_t)
        st = FAIL if bought and not recurring and ron > cap + 0.005 else PASS
        res.append((34, "RONIC within the return including goodwill when growth has been bought (or recurring "
                        "acquisitions forecast)", st,
                    f"RONIC {ron:.1%} vs {spec.fc_labels[-1]} ROIC including goodwill {roic_t:.1%} (DCF C93); goodwill + "
                    f"intangibles {gw_share:.0%} of invested capital, acquisitions {acq_pct:.1%} of sales over "
                    f"{len(acq10)} years" +
                    (": the forecast's own return excludes the acquisitions that bought the growth — set RONIC at or "
                     "below the return including goodwill, or forecast recurring acquisition spend (drivers.acq, "
                     "the 10-year average share of sales) with the growth it buys (segments mna)" if st == FAIL else
                     "; recurring acquisitions are forecast" if bought and recurring else "")))
        # two-sided (PM round 4): RONIC set well below the return including goodwill is a bearish judgment the
        # analyst dates and explains (LLY: cut from 46% to 22% for deal-enabled growth already expensed in EBIT)
        # the reference is the return including goodwill capped where the RONIC guard caps it (PM, ADBE: gate 34
        # asked for 194% while the guard refused anything above 60%)
        ref = min(max(wacc, roic_t), RONIC_CAP)
        if st == PASS and ron < ref - 0.02 and spec.profile.get("ronic_rule") != "wacc":
            res.append((34, "RONIC below the return including goodwill only for a dated sector reason", WARN,
                        f"RONIC {ron:.1%} vs {spec.fc_labels[-1]} ROIC including goodwill {roic_t:.1%} (DCF C93)" +
                        (f", capped at {RONIC_CAP:.0%}" if roic_t > RONIC_CAP else "") + ": new capital earning less "
                        "than the business earns on everything paid for it needs a dated sector reason — loss of "
                        "exclusivity (product, year), competitive entry, or recurring acquisitions not otherwise "
                        f"modeled — in thesis.explanations['gate 34'] with the year; otherwise RONIC = {ref:.1%}",
                        r"(19|20)\d\d"))
    # scenario probabilities are a judgment with a basis, not a template (PM round 8: MRK carried ADBE's 35/50/15)
    ex_ = {norm(k): str(w) for k, w in ((spec.thesis or {}).get("explanations") or {}).items()}
    basis = ex_.get("scenario probabilities") or ex_.get("probabilities") or ""
    if not re.search(r"\d", basis):  # (round3_gates runs in the economic framework only)
        pr_ = spec.scenario_probs or [0.25, 0.5, 0.25]
        res.append((51, "Scenario probabilities have a stated basis", WARN,
                    f"bear / base / bull at {' / '.join(f'{p:.0%}' for p in pr_)} with no basis in "
                    "thesis.explanations['scenario probabilities']: say what sets each, with numbers (pipeline "
                    "probabilities of success, how often past cycles reached the bear, the evidence for the breaker)",
                    r"\d"))
    # a decline still running when the forecast ends is finished in the fade (PM round 8, MRK: Keytruda at 34% of
    # sales falling 15% a year in 2031; the straight-line fade took growth back up and terminal growth was cut instead)
    s0_ = sum(_nz(sg.sales[2]) for sg in spec.segments) or 1.0
    falling = [(u.name, u.growth["base"][-1], _nz(u.sales[2]) / s0_) for sg in spec.segments for u in sg.units
               if u.growth.get("base") and u.growth["base"][-1] <= -0.08 and _nz(u.sales[2]) >= 0.10 * s0_]
    if falling and not getattr(spec, "fade_path", None):
        res.append((50, "A decline still running in the last forecast year is finished in the fade (fade_path)", WARN,
                    "; ".join(f"{n} {g:+.0%} in {spec.fc_labels[-1]} ({w:.0%} of FY{spec.fy_hist[-1]} sales)"
                              for n, g, w in falling[:3]) +
                    ": the straight-line fade takes growth back up to terminal before the decline ends — set "
                    "fade_path (growth by fade year to the trough, with its source) and keep terminal growth at the "
                    "house rate, or explain with the trough year", r"(19|20)\d\d"))
    # growth that destroys value means the capital base is wrong, not the business (PM round 8, DE: RONIC 7.5% under
    # an 8.9% WACC from captive-finance capital; value fell as terminal growth rose and capex was bent to fit)
    if ron is not None and wacc and ron < wacc - 0.0005 and spec.profile.get("ronic_rule") != "wacc":
        res.append((49, "Terminal RONIC at or above WACC (else growth destroys value: the capital base is wrong)", FAIL,
                    f"RONIC {ron:.1%} under the {wacc:.1%} WACC: value falls as terminal growth rises. Check what sits "
                    "in invested capital (a captive arm's receivables and debt, buyback-shrunk equity, expensed "
                    "R&D) before using it; a business that truly earns below its cost of capital is the bear case"))
    # 35: base case vs consensus (PM round 3: FY28 at 10.5% vs 9.7% consensus, pinned to a management target)
    cons = spec.consensus or {}
    gaps = []
    for k, c, lbl in (("g_fy1", "F", spec.fc_labels[0]), ("g_fy2", "G", spec.fc_labels[1])):
        m, cv = get("IS", f"{c}{IS['growth']}"), cons.get(k)
        if m is not None and isinstance(cv, (int, float)) and abs(m - cv) > 0.0075:
            gaps.append(f"{lbl} revenue growth {m:.1%} vs consensus {cv:.1%} ({(m - cv) * 100:+.1f}pt)")
    e1, ce = get("IS", f"F{IS['adj_eps']}"), cons.get("eps")
    if e1 and isinstance(ce, (int, float)) and ce > 0 and abs(e1 / ce - 1) > 0.03:
        gaps.append(f"{spec.fc_labels[0]} adjusted EPS ${e1:,.2f} vs consensus ${ce:,.2f} ({e1 / ce - 1:+.1%})")
    notes = " ".join([str(spec.notes.get("growth") or "")] + [str(sg.growth_note or "") for sg in spec.segments
                                                              if hasattr(sg, "growth_note")] +
                     [str(getattr(u, "note", "") or "") for sg in spec.segments for u in sg.units])
    tgt = next((m for m in re.finditer(r"(?i)[^.;]{0,60}\b(long[- ]term|multi[- ]year|fy\s?'?20\d\d|20\d\d)\b[^.;]{0,30}"
                                       r"\b(revenue |sales )?(target|goal|ambition|aspiration)s?\b[^.;]{0,60}", notes)
                if not re.search(r"(?i)\bbull|\bnot (the |a )?(base|anchor)|rather than", m.group(0))), None)
    if gaps or (tgt and cons):
        res.append((35, "Base case at consensus, or the gap is a stated variant view (a management target belongs "
                        "in the bull case)", WARN,
                    "; ".join(gaps) + (f"{'; ' if gaps else ''}the growth notes anchor to a management target "
                                       f"('…{tgt.group(0).strip()[:110]}…')" if tgt else "") +
                    ": a base above (below) consensus needs the variant view that justifies it and the date it is "
                    "tested; otherwise base at consensus and the target in the bull"))
    elif cons.get("g_fy1") is not None:
        res.append((35, "Base case at consensus, or the gap is a stated variant view", PASS,
                    "Within 0.75pt of consensus revenue growth (and 3% of EPS)"))
    # 36: acquisitions paid in FY0 annualize as M&A in FY1, not organic growth
    acq0 = abs(_nz((spec.hist.CF.get("acq") or [0, 0, 0])[2])) if hasattr(spec.hist, "CF") else 0.0
    ev = get("SUMMARY", "C8")
    if s0 > 0 and acq0 > 0.02 * s0 and ev and ev > 0:
        evs = max(ev / s0, 1.0)
        est = acq0 / evs  # revenue bought at the acquirer's own EV / sales
        ann = 0.5 * est  # a mid-year close leaves half a year to annualize
        mna1 = sum(_nz(sg.mna[0]) * _nz(sg.sales[2]) for sg in spec.segments if sg.mna)
        st = WARN if ann >= 0.01 * s0 and mna1 < 0.8 * ann else PASS  # PM round 4: CRM passed at 581 vs 929mm
        res.append((36, "Acquired revenue annualizes as M&A in the first forecast year, not as organic growth", st,
                    f"FY{spec.fy_hist[-1]} paid {acq0:,.0f}mm for acquisitions ({acq0 / s0:.0%} of sales): at the "
                    f"company's own {evs:.1f}x EV / sales that buys about {est:,.0f}mm of revenue, ~{ann:,.0f}mm "
                    f"({ann / s0:.1%} growth) still to annualize in {spec.fc_labels[0]} if the deals closed mid-year; "
                    f"the model's {spec.fc_labels[0]} M&A contribution is {mna1:,.0f}mm" +
                    (": the rest sits in 'organic' growth — move it to segments mna (the target's revenue and close "
                     "date from the 8-K / 10-K) or say why the estimate does not apply" if st == WARN else "")))
    # 37: growth-adjusted comps are sane
    from .workbook import ev_dispersed, ev_growth_fit, peer_eps_issue, peer_growth_fit
    fit, evf = peer_growth_fit(spec.peers, bool(spec.profile.get("cyclical"))), ev_growth_fit(spec.peers)
    one_off = [f"{p_.get('ticker')} ({peer_eps_issue(p_, spec.peers).split(':')[0]})" for p_ in spec.peers
               if peer_eps_issue(p_, spec.peers)]
    disp = ev_dispersed(spec.peers)
    name37 = ("Growth-adjusted comps: each regression has slope > 0, R² ≥ 0.2 and 6+ peers, else a median (P/E on "
              "EPS growth, EV / EBITDA on revenue growth)")
    txt = (f"P/E: {fit['how'] or 'peer median'}" + (f" ({fit['rejected']})" if fit["rejected"] and
                                                     fit["rejected"] not in fit["how"] else "") +
           f"; EV / EBITDA: {evf['how'] or 'peer median'}" + (f" ({evf['rejected']})" if evf["rejected"] else "") +
           (f"; {disp}" if disp else "") +
           (f"; excluded from the peer P/E (one-offs or a basis error): {', '.join(one_off)}" if one_off else ""))
    # a rejected EV / EBITDA fit the engine already set aside (P/E only) is handled, not an open question
    if (fit["rejected"] and fit["n"] >= 6 and "pays less" in fit["rejected"]) or \
            (evf["rejected"] and evf["n"] >= 6 and "pays less" in evf["rejected"] and not disp):
        res.append((37, name37, WARN, txt + " — a fit that pays less for more growth points at the peer set or the "
                                            "peers' growth inputs (Comps columns L and M)"))
    elif spec.peers:
        res.append((37, name37, PASS, txt))
    # 40: the company in the model is the one the price is for (PM test: HON valued on its 10-K perimeter, Aerospace
    # included, against the price after the Jun 29, 2026 Aerospace spin: +105% "upside")
    spins = [(d_, f_) for d_, f_ in getattr(spec, "own_actions", []) or []
             if 0 < f_ < 1 and abs(1 / f_ - round(1 / f_)) > 0.02]  # a fractional factor is a spin-off, not a split
    cons_flag = next((f_ for f_ in spec.flags if "implies a perimeter change" in f_), None)
    modeled = any(str(d_.get("kind") or "").lower() == "separation" for d_ in spec.deals or []) or \
        bool(re.search(r"(?i)remain(ing)?co|continuing operations|post-spin|after the (spin|separation)",
                       " ".join(str(x) for x in (spec.notes or {}).values()) + " " +
                       " ".join(str(sg.mna_note or "") for sg in spec.segments)))
    if spins or cons_flag:
        st = PASS if modeled else FAIL
        res.append((40, "The model is the company the price is for (no separation since the 10-K left in the "
                        "perimeter)", st,
                    ("; ".join(f"spin-off on {d_:%b %-d, %Y} (Yahoo factor {f_:g})" for d_, f_ in spins) +
                     ("; " if spins and cons_flag else "") + (cons_flag[:160] if cons_flag else "") +
                     (": the 10-K perimeter includes a business the price no longer owns — rebuild on the remaining "
                      "company (10-Q continuing operations; drop the separated segments), or the value is not "
                      "comparable to the price" if st == FAIL else ": modeled on the remaining company"))))
    # 38: a deal's standalone value: market-based and EBITDA-based estimates within 50% of each other
    bad = []
    for d in spec.deals or []:
        c, pr, eb, mu = (_n(d.get(k)) for k in ("consideration", "premium", "ebitda", "multiple"))
        if c and pr is not None and eb and mu and eb > 0:
            sa_m, sa_e = c / (1 + pr), eb * mu
            if not 0.67 <= sa_m / sa_e <= 1.5:
                bad.append(f"{d.get('name')}: standalone {sa_m:,.0f}mm from the price ÷ (1 + {pr:.0%} premium) vs "
                           f"{sa_e:,.0f}mm on {eb:,.0f}mm EBITDA × {mu:g}x ({max(sa_m, sa_e) / min(sa_m, sa_e):.1f}x "
                           "apart)")
    # an input backed out so the cross-check ties is not a source (CRM v4: Braze's margin "set so standalone EBITDA x
    # the peer multiple reconciles to the consideration")
    solve_rx = re.compile(r"(?i)\b(set|solved|chosen|backed out|implied|calibrated)\b[^.;]{0,60}\bso\b[^.;]{0,90}"
                          r"\b(reconcil\w*|ties?|tied|match\w*|equals?)\b")
    for d in spec.deals or []:
        if solve_rx.search(str(d.get("note") or "")) and str(d.get("kind", "acquisition")).lower().startswith("acq"):
            bad.append(f"{d.get('name')}: an input is solved so the cross-check ties ('…"
                       f"{solve_rx.search(str(d.get('note'))).group(0)[:90]}…'), not sourced")
    for d in spec.deals or []:  # a deal on the tab with no price moves no value (LLY v4: Merida "closed after the
        # quarter, no consideration provided": what was paid was never deducted)
        if str(d.get("kind", "acquisition")).lower().startswith("acq") and not _n(d.get("consideration")):
            bad.append(f"{d.get('name')}: no price entered — an acquisition moves value by what is paid; enter it "
                       "(estimated from the release or the IPR&D charge if undisclosed, and paid in drivers.acq or "
                       "drivers.nonrec when closed) or remove the item")
    acqs = [i for i, d in enumerate(spec.deals or []) if str(d.get("kind", "acquisition")).lower().startswith("acq")]
    created = [_n(v.get(("DEALS", f"E{5 + i}"))) for i in acqs]
    if len(acqs) >= 2 and all(x is not None and x > 0 for x in created):
        # PM round 4: every CRM deal came out value-creating; acquirers on average create none at announcement
        bad.append("every acquisition creates value (" + ", ".join(
            f"{spec.deals[i].get('name')} {x:+,.0f}mm" for i, x in zip(acqs, created)) + "): acquirers on average "
            "create none at announcement — check each standalone value and synergy input against its source")
    if bad:
        res.append((38, "Deal inputs agree: standalone value from the price and from EBITDA × multiple within 50%",
                    WARN, "; ".join(bad) + ": one of the EBITDA, the multiple or the premium is wrong — source each "
                                           "(8-K, proxy, the target's own filings) or label it an estimate"))
    # 39: buybacks above free cash flow lift leverage: a capital-allocation call the thesis must state
    hits = []
    for j, c in enumerate(FC):
        bb, fcf = -_nz(get("CF", f"{c}28")), get("CF", f"{c}39")
        e_ = get("IS", f"{c}{IS['ebitda']}")
        debt = sum(_nz(get("BS", f"{c}{r}")) for r in (24, 25, 29))
        nd = debt - _nz(get("BS", f"{c}6")) - _nz(get("BS", f"{c}7"))
        if fcf is not None and bb > 1.1 * max(fcf, 0.0) + 1 and e_ and e_ > 0 and nd / e_ > 1.0:
            hits.append(f"{spec.fc_labels[j]} buybacks {bb:,.0f}mm vs free cash flow {fcf:,.0f}mm "
                        f"(net debt / EBITDA {nd / e_:.1f}x)")
    if hits:
        res.append((39, "Buybacks above free cash flow are stated in the thesis (they lift EPS and the 3-year return "
                        "with debt)", WARN, "; ".join(hits[:3]) + ": say in the thesis that the buyback is debt-funded "
                                                                  "and what it adds to EPS"))
    # 46: dilution tied to stock comp (PM round 6, ADBE: 1.6mm shares a year issued against $2.2bn of stock comp,
    # ~8mm at $250-300; EPS growth, the PEG comps and the 3-year return counted buybacks without the grants)
    thin = []
    iss, bpx = spec.drivers.get("shares_issued", ([], ""))[0], spec.drivers.get("buyback_px", ([], ""))[0]
    for j, c in enumerate(FC):
        sbc_ = abs(_nz(get("IS", f"{c}{IS['sbc']}")))
        sv = _nz(sales_fc[j]) if j < len(sales_fc) else 0.0
        if j < len(iss) and j < len(bpx) and sv and sbc_ > 0.01 * sv and _nz(iss[j]) * _nz(bpx[j]) < 0.5 * sbc_:
            thin.append(f"{spec.fc_labels[j]} {_nz(iss[j]):,.1f}mm shares × ${_nz(bpx[j]):,.0f} = "
                        f"{_nz(iss[j]) * _nz(bpx[j]):,.0f}mm against {sbc_:,.0f}mm of stock comp")
    if thin:
        res.append((46, "Shares issued match stock comp (grants are dilution at the price they are worth)", WARN,
                    "; ".join(thin[:2]) + ": set drivers.shares_issued to stock comp ÷ the repurchase price, or explain "
                                         "the gap with its number (net settlement paid in cash shows as a financing "
                                         "outflow)", r"\d"))
    # 47: the year to date + the guided last quarter = the full-year guide
    q_g, fy_g = getattr(spec, "q_guide", None) or [], (spec.guidance or {}).get("adj_eps")
    q_eps = getattr(spec, "q_eps", None) or []
    y_adj = (spec.interim.ytd_cur.get("adj_eps") if spec.interim.available else None)
    if q_g and fy_g and q_eps:  # independent: the reported quarters against the guides (PM round 7)
        tot = sum(q_eps) + sum(q_g) / 2
        off = tot / (sum(fy_g) / 2) - 1
        res.append((47, "Reported quarters + the guided last quarter = the full-year guide (±1.5%)",
                    PASS if abs(off) <= 0.015 else FAIL,
                    f"{' + '.join(f'${x:.2f}' for x in q_eps)} reported + Q4 guide ${sum(q_g) / 2:.2f} = ${tot:.2f} vs "
                    f"the ${sum(fy_g) / 2:.2f} full-year guide ({off:+.1%})" +
                    ("" if abs(off) <= 0.015 else ": one of them is misread — check each release's reconciliation "
                                                  "table before explaining the rest of the year")))
    elif q_g and fy_g and y_adj:
        res.append((47, "Reported quarters + the guided last quarter = the full-year guide", NA,
                    f"{spec.interim.label_cur} ${y_adj:.2f} is the full-year guide less the Q4 guide (the earlier "
                    "quarters' releases were not read), so it cannot be tested against them"))
    # 43: the rest of the first forecast year implied by the forecast less the year to date looks like last year's
    # (PM round 4: LLY's forecast implied Q4 free cash flow of -$2bn — year-end working-capital ratios, inventory at
    # 454 days, built $5.2bn with no basis in the run-rate)
    it = spec.interim
    frac = {"Q1": 0.25, "Q2": 0.5, "Q3": 0.75}.get(it.fiscal_period) if it.available else None
    f1, f0 = get("CF", "F39"), get("CF", "E39")
    yc, yp = it.ytd_cur if it.available else {}, it.ytd_prior if it.available else {}
    if frac and f1 is not None and f0 is not None and yc.get("cfo") is not None and yp.get("cfo") is not None:
        y_now = _nz(yc.get("cfo")) - abs(_nz(yc.get("capex")))
        y_pri = _nz(yp.get("cfo")) - abs(_nz(yp.get("capex")))
        rest, rest0 = f1 - y_now, f0 - y_pri
        s1 = _nz(sales_fc[0]) if sales_fc else 0.0
        bad43 = (rest < 0 <= rest0 or rest < 0.5 * rest0 - 0.01 * s1) and abs(rest - rest0) > 0.01 * s1
        rest_lbl = {"Q1": "Q2–Q4", "Q2": "H2", "Q3": "Q4"}[it.fiscal_period]
        res.append((43, f"The rest of {spec.fc_labels[0]} (forecast less year to date) is in line with last year's",
                    WARN if bad43 else PASS,
                    f"free cash flow: {spec.fc_labels[0]} {f1:,.0f}mm less {y_now:,.0f}mm in {it.label_cur} implies "
                    f"{rest:,.0f}mm for {rest_lbl}, against {rest0:,.0f}mm in the same stretch last year" +
                    (": a year-end working-capital ratio (inventory, receivable days) applied to a fast-growing year "
                     "builds working capital the run-rate does not show — set the FY1 drivers from the 10-Q balance "
                     "sheet, or explain the build with its number" if bad43 else "")))
    # 44: a KPI revenue build ties to the modeled revenue (PM round 4: HD's was -2.8% every year)
    from .workbook import kpi_layout
    try:
        KP = kpi_layout(spec)
    except Exception:
        KP = {}
    off = []
    for b_, (i_, u_) in enumerate(spec.kpi_units or []):
        row = (KP.get(b_) or {}).get("diff")
        ds = [_n(v.get(("KPIS", f"{c}{row}"))) for c in FC] if row else []
        ds = [d_ for d_ in ds if d_ is not None]
        # the volume x price build is organic: acquired / FX revenue (segments mna) sits outside it, so the gap it
        # explains is not a tie-out error (HD: GMS, +2.9% in FY26, showed as -2.8% every year)
        ds = [(1 + d_) / (1 + e_) - 1 for d_, e_ in zip(ds, kpi_mna_gap(spec, i_, u_))]
        if ds and max(abs(d_) for d_ in ds) > 0.01:
            off.append(f"{getattr(u_, 'name', '') or i_}: implied vs model revenue " +
                       ", ".join(f"{d_:+.1%}" for d_ in ds[:6]))
    if spec.kpi_units:
        res.append((44, "KPI revenue builds tie to modeled revenue (±1%)", WARN if off else PASS,
                    "; ".join(off) + (": the volume × price path and the segment growth disagree — make one follow "
                                      "the other (kpi growth = the segment's organic growth)" if off else "")
                    if off else f"{len(spec.kpi_units)} KPI build(s) within 1% in every forecast year"))
    # 42: what the explanations say was done is in the data (PM round 4: LLY's thesis said NVO's and MRK's EPS were
    # fixed; the Comps table still showed 1.8x and 53.2x)
    from .workbook import _pe
    th = spec.thesis or {}
    said = [str(x) for x in list((th.get("explanations") or {}).values()) +
            [y for k in ("model_says", "points", "consensus_notes") for y in (th.get(k) or [])
             if isinstance(th.get(k), list)] + [th.get("consensus_notes") or ""]]
    claim_rx = re.compile(r"(?i)\b(fixed|corrected|reset|replaced|normali[sz]ed|restated|now (on|uses?)|re-?based)\b")
    excl_rx = re.compile(r"(?i)\b(excluded|removed|dropped)\b")
    pes = sorted(x for x in (_pe(q) for q in spec.peers) if x)
    med = pes[len(pes) // 2] if pes else None
    wrong = []
    for p_ in spec.peers:
        t_ = str(p_.get("ticker") or "")
        if not t_ or not med:
            continue
        pe = _pe(p_)
        for txt in said:
            for sent in re.split(r"(?<=[.;])\s+", txt):
                if not re.search(rf"\b{re.escape(t_)}\b", sent):
                    continue
                if claim_rx.search(sent) and pe and not med / 2.5 <= pe <= med * 2.5:
                    wrong.append(f"{t_}: the text says '…{claim_rx.search(sent).group(0)}…' but its P/E is still "
                                 f"{pe:.1f}x against a {med:.1f}x median (EPS {p_.get('eps')}, source "
                                 f"'{str(p_.get('source') or '')[:60]}')")
                    break
                if excl_rx.search(sent) and pe and not peer_eps_issue(p_, spec.peers) and \
                        re.search(r"(?i)peer|comps|p/e|eps", sent):
                    wrong.append(f"{t_}: the text says it was {excl_rx.search(sent).group(0)}, but it is still in the "
                                 f"peer P/E at {pe:.1f}x")
                    break
    if wrong:
        res.append((42, "What the explanations say was fixed is in the data", FAIL,
                    "; ".join(dict.fromkeys(wrong)) + ": change the input (peers' EPS / source), or correct the text"))
    return res


# a call is stated, not mentioned: "Long, sized for…", "we rate it Long", "warrants ownership", "Underweight" — not
# "hurdles for a Long", "not a Long", "Long-term"
_CALL_LEAD = r"(?:^|[.;:!?]\s+|\n)\s*(?:\*\*)?"
_CALL_VERB = r"\b(?:rate[sd]?|rating(?: is)?|we are|we're|stay|remain|initiat\w*(?: at)?|upgrade\w* to|downgrade\w* to|move to)\b" \
             r"[^.;]{0,25}?"
RATING_WORDS = {
    "long": re.compile(_CALL_LEAD + r"Long(?=\s*[,:/—–-]\s|\s+(with|sized|the|on|at|here)\b)|" + _CALL_VERB +
                       r"\bLong\b(?![\w-])|\b(Overweight|Outperform|Accumulate)\b|\bBuy\b(?![\w-])|"
                       r"\b(go(ing)? long|long position|warrants? (ownership|a position|owning)|own the (stock|shares))\b"),
    "short": re.compile(_CALL_LEAD + r"Short(?=\s*[,:/—–-]\s|\s+(with|sized|the|on|at|here)\b)|" + _CALL_VERB +
                        r"\bShort\b(?![\w-])|\b(Underweight|Underperform)\b|\bSell\b(?![\w-])|"
                        r"\b(go(ing)? short|short position|avoid (the (stock|shares)|owning))\b", re.I),
    "positive": re.compile(r"\bpositive bias\b", re.I), "negative": re.compile(r"\bnegative bias\b", re.I)}
LEAD_RX = re.compile(r"(?i)^\s*(\*\*)?(long|short|neutral|underweight|overweight|equal[- ]weight|market[- ]weight|buy|"
                     r"sell|hold|outperform|underperform|avoid|accumulate|under review|override)\b[^:.]{0,110}[:.]\s*")


def recommendation_text(text: str) -> str:
    """The analyst's supporting sentence, without a leading call ('Long, sized for…', 'Underweight / avoid adding:',
    or the line this function composed last time)."""
    t = str(text or "").strip()
    for _ in range(3):
        t2 = LEAD_RX.sub("", t, count=1).strip()
        if t2 == t:
            break
        t = t2
    return t[:1].upper() + t[1:] if t else ""


def stated_calls(spec) -> list:
    """(field, call, phrase) for every rating word in the thesis text."""
    th = spec.thesis or {}
    out = []
    for k in ("recommendation_text", "points", "model_says", "variant_view", "catalysts", "price_implies"):
        x = th.get(k)
        for t in (x if isinstance(x, list) else [x]):
            t = str(t or "")
            if k == "recommendation_text":
                t = recommendation_text(t)
            for call, rx in RATING_WORDS.items():
                m = rx.search(t)
                if m and not re.search(r"(?i)\b(not|no|nor|never|rather than|instead of|isn't|is not)\b[^.;]{0,30}$",
                                       t[max(0, m.start() - 40):m.start()]):
                    out.append((k, call, t[max(0, m.start() - 30):m.end() + 30].strip()))
    return out


def decision_gates(spec, out: dict) -> list:
    """Gates 41 and 25 on the review's bear / base / bull (provisional: the final rating also weighs open items), so
    the analyst sees them before the deliverable is built."""
    bear, base, bull, px = (_n(out.get(k)) for k in ("bear_fv", "fair_value", "bull_fv", "price"))
    if None in (bear, base, bull, px) or not px:
        return []
    res = []
    inside = bear < px < bull
    res.append({"id": 41, "gate": "The price lies inside bear–bull (else the scenarios are miscalibrated, not the "
                                  "market)", "status": PASS if inside else WARN,
                "detail": (f"bear ${bear:,.2f} < price ${px:,.2f} < bull ${bull:,.2f}" if inside else
                           f"price ${px:,.2f} outside bear ${bear:,.2f} – bull ${bull:,.2f}: build the bull from what "
                           "the reverse DCF says the price requires (terminal margin, years of growth) or the Street "
                           "bull; the engine rebuilds a bull below the price to the reverse-DCF margin at build")})
    pr = spec.scenario_probs or [0.25, 0.5, 0.25]
    gap = (pr[0] * bear + pr[1] * base + pr[2] * bull) / sum(pr) / px - 1
    irr = out.get("implied_irr")
    irr = 0.20 if isinstance(irr, str) and irr.strip().startswith(">") else 0.03 if isinstance(irr, str) else _n(irr)
    wacc = _n(out.get("wacc"))
    spread = irr - wacc if irr is not None and wacc is not None else None
    skew = (bull - px) / (px - bear) if inside else None
    th = spec.thesis or {}
    # a short also needs written evidence the gap closes (and a pair for a low-volatility compounder): without it
    # the rule's call is Neutral, so short language contradicts it
    call = ("long" if gap >= 0.15 and spread is not None and spread >= 0.02 and skew is not None and skew >= 2 else
            "short?" if gap <= -0.15 and spread is not None and spread <= -0.02 and inside and
            re.search(r"\d", str(th.get("short_evidence") or "")) else "")
    bias = "" if call or abs(gap) < 0.05 else ("positive" if gap > 0 else "negative")  # noqa: E501
    contra = []
    for fld, c, phrase in stated_calls(spec):
        if c == "long" and call != "long":
            contra.append(f"{fld}: '…{phrase}…' states a Long call")
        elif c == "short" and call != "short?":
            contra.append(f"{fld}: '…{phrase}…' states a Short / underweight call")
        elif c in ("positive", "negative") and c != bias:
            contra.append(f"{fld}: '…{phrase}…' states a {c} bias")
    ov = (spec.thesis or {}).get("rating_override") or {}
    if contra and not (isinstance(ov, dict) and ov.get("rating") and ov.get("reason")):
        res.append({"id": 25, "gate": "Rating and recommendation text follow the rule", "status": FAIL,
                    "detail": "; ".join(contra[:3]) + f" — the rule (probability-weighted gap {gap:+.1%}, IRR spread "
                              + (f"{spread * 100:+.1f}pt" if spread is not None else "n/a") +
                              f") says {'Long' if call == 'long' else 'Short if its other tests pass' if call else 'Neutral'}"
                              + (f" with a {bias} bias" if bias else "") + ": write thesis.recommendation as one "
                              "supporting sentence with no call, remove the call from the other text, or write a "
                              "rating_override with its reason"})
    return res


THESIS_FIELDS = ("recommendation", "points", "model_says", "price_implies", "variant_view", "breaker",
                 "sensitivities", "consensus_notes", "risks", "monitor", "catalysts")
VALUE_WORDS = [(re.compile(r"(?i)bear"), ("C49",)), (re.compile(r"(?i)bull"), ("C51",)),
               (re.compile(r"(?i)\bbase\b"), ("C50", "G19")),
               (re.compile(r"(?i)sum of the parts|sotp|break-?up"), ("D17",)),
               (re.compile(r"(?i)\bexit\b"), ("D16",)),
               (re.compile(r"(?i)comps|peer|multiple"), ("D18",)),
               (re.compile(r"(?i)dcf|perpetuity|intrinsic"), ("D15",)),
               (re.compile(r"(?i)fair value|weighted|target|worth|valued? at|probability"), ("G19", "C52", "C50"))]


def thesis_number_problems(spec, v: dict) -> list[str]:
    """Typed dollar values in the thesis next to a valuation word must equal the model's (±1.5%); typed rates,
    premium / discount claims and buyback-funding claims must match the model; every placeholder must render (PM
    round 6: '3.75% terminal growth' against 3.25%, a 'P/E premium' at a 45% discount, buybacks 'within FCF' above it,
    an unrendered {gap})."""
    out = []
    px = _n(v.get(("SUMMARY", "C5"))) or spec.price
    th = spec.thesis or {}
    texts = [str(t or "") for k in THESIS_FIELDS for t in (th.get(k) if isinstance(th.get(k), list) else [th.get(k)])]
    texts += [str(w) for w in (th.get("explanations") or {}).values()]
    from .workbook import LIVE, _pe
    for t in texts:
        for m in re.finditer(r"\{(\w+)\}", t):
            if m.group(1) not in LIVE:
                out.append(f"unrendered placeholder {{{m.group(1)}}} in '{t[max(0, m.start() - 30):m.end()]}'")
    # (keyword, model value, name, window): RONIC and WACC are quoted right next to the word (HD: a 17% reinvestment
    # rate "of g / RONIC" is not a RONIC)
    rates = [(r"terminal growth(?: rate)?", _n(v.get(("DCF", "C74"))) or spec.valuation["g"][0], "terminal growth", 30),
             (r"\bWACC\b", _n(v.get(("DCF", "C17"))), "WACC", 12),
             (r"\bRONIC\b", spec.valuation["ronic"][0], "RONIC", 12)]
    for t in texts:
        for sent in re.split(r"(?<=[.;])\s+", t):
            for kw, val, name, win in rates:
                km = re.search(r"(?i)" + kw, sent)
                if not km or val is None or re.search(r"(?i)\b(bear|bull|requires?|implie[sd]|implied|price|sensitiv\w*|"
                                                      r"grid|range|history|historical|consensus)\b", sent):
                    continue  # a scenario, the reverse DCF or a range quotes other rates by design
                near = sent[max(0, km.start() - win):km.end() + win]
                if re.search(r"(?i)\bfrom\b[^;]{0,40}?\bto\b", sent):  # a change quotes the old and the new value
                    near = sent
                typed = [float(x) / 100 for x in re.findall(r"(\d{1,2}(?:\.\d{1,2})?)%", near)]
                if typed and all(abs(q - val) > 0.0015 for q in typed):
                    out.append(f"'{near.strip()}' but the model's {name} is {val:.2%}")
            if re.search(r"(?i)\bcomps\b|trading comparables", sent):  # a typed comps weight (PM round 7: "a 20%
                # cross-check" with comps at 10%)
                for m_ in re.finditer(r"(?i)(\d{1,3})% (?:cross-check|weight)|weight(?:ed)? (?:of|at|to) (\d{1,3})%", sent):
                    q = float(m_.group(1) or m_.group(2)) / 100
                    w_c = spec.valuation["w_comps"][0]
                    if abs(q - w_c) > 0.005:
                        out.append(f"'{m_.group(0)}' for comps but their weight is {w_c:.0%}")
            bm = re.search(r"(?i)\bbeta(?: of| at| is| =|:)?\s*(\d\.\d{1,2})\b", sent)
            if bm and abs(float(bm.group(1)) - spec.valuation["beta"][0]) > 0.015:
                out.append(f"'{bm.group(0)}' but the model's beta is {spec.valuation['beta'][0]:.2f}")
    # a premium claim against a discount (and the reverse)
    pes = sorted(x for x in (_pe(q) for q in spec.peers) if x)
    own = _n(v.get(("SUMMARY", "C9")))
    if pes and own:
        med = pes[len(pes) // 2]
        for t in texts:
            if own < 0.85 * med and re.search(r"(?i)\b(P/E|multiple|valuation) premium|premium (P/E|multiple|"
                                              r"valuation|to (the )?peers)|trades? at a premium", t):
                out.append(f"the text speaks of a premium, but the P/E is {own:.1f}x against the peers' {med:.1f}x "
                           f"({own / med - 1:+.0%})")
                break
            if own > 1.15 * med and re.search(r"(?i)\b(P/E|multiple|valuation) discount|trades? at a discount", t):
                out.append(f"the text speaks of a discount, but the P/E is {own:.1f}x against the peers' {med:.1f}x "
                           f"({own / med - 1:+.0%})")
                break
    # buybacks said to be funded by free cash flow
    over = [(spec.fc_labels[j], -_nz(_n(v.get(("CF", f"{c}28")))), _n(v.get(("CF", f"{c}39"))))
            for j, c in enumerate("FGHIJK") if j >= 1]
    over = [(y, bb, f) for y, bb, f in over if f is not None and bb > 1.03 * max(f, 0.0) + 1]
    if over:
        rx_ = re.compile(r"(?i)(buybacks?|repurchases?)[^.;]{0,80}(within|below|covered by|funded (by|from|out of)|"
                         r"paid (from|out of)|sized (to|within))[^.;]{0,25}(free cash flow|FCF)")
        if any(rx_.search(t) for t in texts):
            out.append("the text says buybacks stay within free cash flow, but " + ", ".join(
                f"{y} buybacks {bb:,.0f}mm exceed {f:,.0f}mm" for y, bb, f in over[:3]))
    for k in THESIS_FIELDS:
        x = (spec.thesis or {}).get(k)
        for t in (x if isinstance(x, list) else [x]):
            t = str(t or "")
            for m in re.finditer(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?![\d.,]*\s*(?:bn|tn|mm|m\b|b\b|k\b|trillion|billion|"
                                 r"million|x\b|%))", t):
                q = float(m.group(1).replace(",", ""))
                if q < 5 or (px and abs(q / px - 1) < 0.005):
                    continue
                ctx = t[max(0, m.start() - 45):m.start()]
                for rx, cells in VALUE_WORDS:
                    if rx.search(ctx):
                        vals = [_n(v.get(("SUMMARY", c))) for c in cells]
                        vals = [x_ for x_ in vals if x_]
                        if vals and all(abs(q / x_ - 1) > 0.015 for x_ in vals):
                            out.append(f"'{ctx.strip()[-30:]} ${q:,.2f}' but the model shows ${vals[0]:,.2f}")
                        break
    return out


from .overrides import TARGET_RX  # noqa: E402  (an input or explanation set to produce an output)

# an explanation that reports a fix: the check it explains still fires, so the fix did not happen
FIXED_RX = re.compile(r"(?i)(?<!not )(?<!n't )(?<!never )\b(fixed|corrected|resolved|restored|removed the|"
                      r"(?:now|no longer) (?:pass|passes|align|aligns|ties|reconciles|inside|within|implies?)|brought (?:back )?(?:inside|within|into)|"
                      r"should (?:now )?(?:no longer|restore|align|pass|bracket|fall inside))\b")


def open_items(spec, gate_list: list, problems: list, plug_list: list | None = None) -> list:
    """Warnings and open checks a rating cannot ship with unless the thesis explains each:
    [(key for thesis.explanations, text, explanation or None)]. A plug-table warning worth more than 5% of revenue
    is its own item and its explanation must carry the number (PM round 3: a 15%-of-revenue working-capital move
    passed on gate 11's one-line explanation)."""
    ex = {norm(k): str(w) for k, w in ((spec.thesis or {}).get("explanations") or {}).items()}

    def words(t):  # numbers move between builds ("fair value is -55%" -> "-54%"): match on the words only
        return " ".join(w for w in norm(t).split() if not re.fullmatch(r"\d+[a-z]?", w))

    def find(key, text):
        nk, nt = norm(key), norm(text)
        wk, wt = words(key), words(text)
        for k, w in ex.items():
            kw = words(k)
            if k == nk or (len(k) >= 12 and k in nt) or (nk.startswith("gate ") and k in (nk[5:], "g" + nk[5:])) or \
                    (not nk.startswith("gate ") and len(kw) >= 12 and (kw == wk or kw in wt)):
                return w
        return None
    claimed = " [its explanation claims a fix, but this check still fires: fix it, or say what remains and why]"

    def claims_fix(w):  # the item is listed, so it still fires: a fix it claims did not happen (PM, ADBE)
        if w and TARGET_RX.search(w):  # reasoning back from the rating (PM round 6: "to reduce the skew")
            return None, (f" [its explanation argues from the output ('{TARGET_RX.search(w).group(0)}'): explain the "
                          "input on the business instead]")
        return (w, "") if not w or not FIXED_RX.search(w) else (None, claimed)

    out = []
    for g in gate_list:
        if g.get("status") == WARN:
            key = f"gate {g['id']}"
            w, tag = claims_fix(find(key, g["gate"]))
            if w and g.get("needs") and not re.search(g["needs"], w):
                w = None  # the explanation must carry what the gate asks for (a date, a number)
            out.append((key, f"{g['gate']}" + (f" — {g['detail']}" if g.get("detail") else "") + tag, w))
    for p in problems:
        if p.startswith("Gate "):
            continue
        key = " ".join(words(p).split()[:8])
        w, tag = claims_fix(find(key, p))
        out.append((key, p + tag, w))
    for p in plug_list or []:
        if p.get("status") != WARN or (p.get("pct_rev") or 0) <= 0.05:
            continue
        key = "working capital" if "working capital" in str(p.get("kind")) else " ".join(norm(p["line"]).split()[:5])
        w, tag = claims_fix(find(key, p["line"]))
        if w and not re.search(r"\d", w):
            w = None  # an explanation of a number this size quotes its cash or value effect
        out.append((key, f"Plug table: {p['line']} — {p['delta_mm']:,.0f}mm ({p['pct_rev']:.0%} of revenue): state "
                         "the view and its cash / value effect" + tag, w))
    return out


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


def gates(spec, v: dict, out: dict, IS: dict, guidance: list, plug_list: list) -> list[dict]:
    """The Part E gate list for a built, recalculated model."""
    g = []
    econ = spec.framework != "reference"

    def add(gid, name, status, detail="", needs=None):
        g.append({"id": gid, "gate": name, "status": status, "detail": detail, **({"needs": needs} if needs else {})})

    get = lambda sh, c: _n(v.get((sh, c)))  # noqa: E731
    # ---------------------------------------------------------------- data and definitions
    ties = []
    for j, c in enumerate("CDE"):
        t = get("IS", f"{c}{IS['adj_tie']}")
        rep = spec.adj_eps_hist[j] if j < len(spec.adj_eps_hist) else None
        if t is not None and rep:
            ties.append((spec.fy_hist[j], t, rep))
    if not ties:
        add(1, "Adjusted EPS definition reproduces reported history", NA,
            "The company reports no adjusted EPS: the model's definition is shown")
    else:
        worst = max(ties, key=lambda x: abs(x[1]))
        tol = max(0.05, 0.02 * abs(worst[2]))
        st = PASS if abs(worst[1]) <= 0.05 else (WARN if abs(worst[1]) <= tol else FAIL)
        hint = ""
        if st != PASS:
            # the non-GAAP tax rate that would tie each year with today's switches (companies that tax non-GAAP
            # pre-tax income at one rate: CRM, most software)
            sw = [get("ASSUMPTIONS", c) for c in ("C17", "C15", "C5", "C16")]
            rates = []
            for j, c in enumerate("CDE"):
                rep = spec.adj_eps_hist[j] if j < len(spec.adj_eps_hist) else None
                vals = [get("IS", f"{c}{IS[k]}") for k in ("pretax", "amort", "sbc", "repo", "nci", "shares")]
                if not rep or None in vals[:4] or not vals[5] or None in sw[:3]:
                    continue
                nr = get("IS", f"{c}{IS['nonrec']}") if "nonrec" in IS else 0.0
                items = vals[1] * sw[0] + vals[2] * sw[1] + vals[3] * sw[2] - _nz(nr) * _nz(sw[3])
                base = vals[0] + items
                if base > 0:
                    rates.append(f"FY{spec.fy_hist[j]} {1 - (rep * vals[5] + _nz(vals[4])) / base:.1%}")
            hint = (": set adj_eps_ex_sbc / adj_eps_ex_repo / adj_eps_ex_nonrec / adj_eps_amort_share to the company's "
                    "reconciliation" + (f"; if the company taxes non-GAAP income at one rate, these non-GAAP tax rates "
                                        f"would tie: {', '.join(rates)} (adj_eps_tax_hist, drivers.adj_tax)"
                                        if rates else ""))
        add(1, "Adjusted EPS definition reproduces reported history (±$0.05)", st,
            "; ".join(f"FY{y}: model {rep + t:.2f} vs reported {rep:.2f}" for y, t, rep in ties) + hint)
    it = spec.interim
    if it.available and it.ytd_cur.get("sales") and it.ytd_prior.get("sales") and spec.hist.IS["sales"][2]:
        fy1 = get("IS", f"F{IS['sales_b']}")
        sh_now = it.ytd_cur["sales"] / fy1 if fy1 else None
        sh_prior = it.ytd_prior["sales"] / spec.hist.IS["sales"][2]
        st = PASS if sh_now is not None and abs(sh_now - sh_prior) <= 0.10 else WARN
        add(2, "Year to date is a normal share of the full year (±10pt vs last year)", st,
            f"{it.label_cur} = {sh_now:.0%} of the {spec.fc_labels[0]} forecast vs {sh_prior:.0%} last year"
            if sh_now is not None else "")
    else:
        add(2, "Year to date is a normal share of the full year", NA, "No 10-Q year to date")
    da = [get("IS", f"{c}{IS['da']}") for c in FC]
    am = [get("IS", f"{c}{IS['amort']}") for c in FC]
    dep_bad = [spec.fc_labels[k] for k, (d, a) in enumerate(zip(da, am)) if d is not None and d - _nz(a) <= 0]
    sbc_hist = any(_nz(x) > 0 for x in spec.hist.IS.get("sbc", []))
    sbc_f = get("IS", f"F{IS['sbc']}")
    probs = []
    if dep_bad:
        probs.append(f"depreciation ≤ 0 in {', '.join(dep_bad)}")
    if sbc_hist and not _nz(sbc_f):
        probs.append("stock comp is 0 although the company grants equity")
    add(3, "Depreciation > 0, D&A ≥ amortization, stock comp present", FAIL if probs else PASS, "; ".join(probs))
    ppe0, ppe6 = get("BS", "E14"), get("BS", "K14")
    s0, s6 = get("IS", f"E{IS['sales_b']}"), get("IS", f"K{IS['sales_b']}")
    if ppe0 and ppe6 is not None and s0 and s6:
        cagr = (s6 / s0) ** (1 / 6) - 1
        st = FAIL if cagr > 0.05 and ppe6 < ppe0 else PASS
        add(4, "PP&E does not shrink while sales grow > 5% a year", st,
            f"sales {cagr:.1%} a year; PP&E {ppe0:,.0f} → {ppe6:,.0f}mm" +
            (": depreciation likely includes amortization, or capex is too low" if st == FAIL else ""))
    snaps = [p for p in spec.peers if p.get("snapshot") and str(p["snapshot"]) > str(spec.as_of)
             and re.search(r"(?i)Yahoo (consensus|forward|trailing)|TRAILING|consensus", str(p.get("source") or ""))]
    # a snapshot a few days late stands for the date's consensus only if the peer released no earnings in between
    # (checked against its 8-K item 2.02 filings: the data, not the label — PM round 3)
    days = lambda p: (date.fromisoformat(str(p["snapshot"])[:10]) - spec.as_of).days  # noqa: E731
    clean = [p for p in snaps if p.get("pit") == "" and days(p) <= 10]
    late = [p for p in snaps if p not in clean]
    add(5, "No market data dated after the valuation date", WARN if late else PASS,
        (f"Peer consensus EPS for {', '.join(p['ticker'] for p in late)} is a snapshot taken after "
         f"{spec.as_of:%b %-d, %Y} (look-ahead" +
         (f"; earnings released in between: {', '.join(p['ticker'] + ' ' + p['pit'] for p in late if p.get('pit'))}"
          if any(p.get("pit") for p in late) else "") +
         "): replace with each peer's guidance from releases dated by the valuation date") if late else
        (f"Consensus snapshots for {', '.join(p['ticker'] for p in clean)} taken ≤10 days after the date with no "
         "earnings release (8-K item 2.02) in between: they stand for the date's consensus" if clean else ""))
    excl = [p.get("ticker") for p in spec.peers if "Corporate action" in str(p.get("ev_source") or "")]
    add(6, "Peers with a spin, split or merger in 18 months are pro forma or excluded", PASS,
        f"EV / EBITDA excluded for {', '.join(excl)}" if excl else "None in the peer set")
    # ---------------------------------------------------------------- balance sheet and timing
    d_q = get("INTERIM", "C37")
    d_fy1 = get("SCHEDULES", "F24")
    d_k = sum(_nz(spec.hist.BS[k][2]) for k in ("st_debt", "cur_ltd", "ltd"))
    if econ and it.available and not d_q and d_k > 0.01 * max(_nz(spec.hist.IS["sales"][2]), 1.0):
        # missing data fails (PM round 4: CAT's 10-Q debt read as $0, so its bridge deducted no debt and this gate
        # went N/A)
        add(7, "Forecast opens from the latest 10-Q (FY1 debt vs 10-Q debt)", FAIL,
            f"10-Q debt is {_nz(d_q):,.0f}mm on {it.period_end:%b %-d, %Y} while the 10-K shows {d_k:,.0f}mm: the "
            "10-Q debt was not read (tags by business or segment only?), so the equity bridge deducts no debt — "
            "fix the data before use")
    elif econ and d_q and d_fy1 is not None and it.available:
        cap_share = spec.valuation.get("captive_share", (0.0,))[0] if spec.captive else 0.0
        gap = abs(d_fy1 - d_q) / d_q
        st = PASS if gap <= 0.10 else (WARN if gap <= 0.25 else FAIL)
        add(7, "Forecast opens from the latest 10-Q (FY1 debt vs 10-Q debt)", st,
            f"{spec.fc_labels[0]} ending debt {d_fy1:,.0f} vs {d_q:,.0f}mm on {it.period_end:%b %-d, %Y}"
            + (" (incl. captive)" if cap_share else "") +
            ("" if st == PASS else ": set FY1 drivers.debt_iss from the 10-Q (issuance already done this year)"))
    else:
        add(7, "Forecast opens from the latest 10-Q", NA, "No 10-Q since the 10-K" if not it.available else "")
    frac = {"Q1": 0.25, "Q2": 0.5, "Q3": 0.75}.get(it.fiscal_period)
    ie_ytd = it.ytd_cur.get("interest_exp") if it.available else None
    ie_fy1 = get("IS", f"F{IS['interest']}")
    cap_sh = (spec.valuation.get("captive_share", (0.0,))[0] or 0.0) if spec.captive else 0.0
    b_q = it.bs_cur if it.available else {}
    debt_q0 = sum(_nz(b_q.get(k)) for k in ("st_debt", "cur_ltd", "ltd")) * (1 - cap_sh)
    if econ and frac and not ie_ytd and debt_q0 > 0.01 * max(_nz(spec.hist.IS["sales"][2]), 1.0):
        # missing data is never N/A (PM round 4); a 10-Q that does not tag interest at all (LLY) leaves the analyst to
        # quote the figure from its text
        add(8, "First-year interest within ±10% of the recent run-rate", WARN,
            f"No year-to-date interest expense in the {it.label_cur} 10-Q XBRL against {debt_q0:,.0f}mm of debt"
            + (" outside the captive arm" if cap_sh else "") + ": the forecast's interest has no anchor — quote the "
                                                               "10-Q's year-to-date interest expense and compare it in "
                                                               "thesis.explanations['gate 8']", r"\d")
    elif frac and ie_ytd and ie_fy1 is not None:
        run = ie_ytd / frac
        gap = abs(abs(ie_fy1) - run) / run if run else 0
        # debt raised during the year to date: the year to date carries it for part of the period only, so the
        # benchmark is the year to date plus the rest of the year on the 10-Q debt at the forecast rate (CRM: the
        # $25bn March notes were in H1 for 4.5 months; $790mm + 1/2 x $39.3bn x 5.2% = $1,811mm, not $1,580mm)
        debt_q = debt_q0  # outside the captive arm: its interest is an operating cost
        r2 = (spec.drivers.get("int_rate") or ([0, 0], ""))[0]
        rest = ie_ytd + (1 - frac) * debt_q * _nz(r2[1] if len(r2) > 1 else r2[0])
        gap2 = abs(abs(ie_fy1) - rest) / rest if rest else gap
        use2 = gap2 < gap and debt_q > 0
        # interest that costs more than twice the market rate on the debt read means debt is missing (PM, ADBE: $194mm
        # of nine-month interest on $1.6bn of debt; $4.6bn of senior notes were not read, and the analyst "fixed" the
        # gap with a 16% rate instead of finding the notes)
        debt_k = sum(_nz(spec.hist.BS[k][2]) for k in ("st_debt", "cur_ltd", "ltd")) * (1 - cap_sh)
        avg_d = (debt_k + debt_q) / 2 if debt_k else debt_q
        kd = _nz(spec.valuation.get("kd", (0.05,))[0]) or 0.05
        implied = run / avg_d if avg_d > 0 else None
        if econ and (avg_d <= 0 or implied > max(2 * kd, 0.08)):
            add(8, "Debt and interest agree: year-to-date interest ÷ average debt within twice the cost of debt", FAIL,
                f"{ie_ytd:,.0f}mm of {it.label_cur} interest ({run:,.0f}mm annualized) on "
                f"{avg_d:,.0f}mm average debt" + (f" = {implied:.1%}, against a {kd:.1%} cost of debt" if implied
                                                  else "") +
                ": debt is missing from the balance sheet read (senior notes under a tag the parser does not "
                "know?). Find the notes in the 10-Q debt table and fix the data; a higher interest rate is not a fix")
            frac = None  # reported once
        if frac:
            add(8, "First-year interest within ±10% of the recent run-rate (year to date + the rest of the year on "
                   "the 10-Q debt)", PASS if min(gap, gap2) <= 0.10 else WARN,
                f"{spec.fc_labels[0]} {abs(ie_fy1):,.0f} vs {run:,.0f}mm annualized from {it.label_cur}" +
                (f" and {rest:,.0f}mm = {ie_ytd:,.0f}mm year to date + the rest of the year on {debt_q:,.0f}mm of "
                 f"10-Q debt at the {spec.fc_labels[1]} rate" if use2 else "") +
                (f" (industrial debt only: the captive arm's {cap_sh:.0%} is funded inside segment profit)" if cap_sh
                 else ""))
    else:
        add(8, "First-year interest within ±10% of the run-rate", NA,
            "No 10-Q" if not frac else "No debt")
    add(9, "Cash flows from the balance-sheet date counted, seasonally weighted", PASS if econ else NA,
        spec.cf_share_note if econ else "Reference framework")
    ev = spec.post_bs_events or []
    closings = [e for e in ev if "2.01" in e["items"]]
    acq1 = spec.drivers["acq"][0][0] if spec.drivers.get("acq") else 0.0
    ytd_acq = (spec.ytd_capital or {}).get("acq") or 0.0
    listed = "; ".join(f"8-K {e['date']} ({e['what']})" for e in ev[:6])
    if closings and not spec.deals and acq1 <= ytd_acq * 1.02 + 1:
        st, why = FAIL, (f"{listed}: a closing after the balance sheet is not paid for (drivers.acq above the "
                         f"{ytd_acq:,.0f}mm year to date, or a closed deal)")
    elif ev or spec.asr_open:
        shares_set = bool(re.search(r"(?i)asr|accelerated|settle", str(spec.basic_note or "")))
        st = PASS if spec.deals and (shares_set or not spec.asr_open) else WARN
        why = ((listed + "; " if listed else "") +
               ("an accelerated share repurchase settles after the 10-Q: update market.basic_shares; "
                if spec.asr_open else "") +
               (f"{len(spec.deals)} deal(s) on the Deals tab" if spec.deals else "none entered: enter or dismiss each"))
    else:
        st, why = (PASS, f"{len(spec.deals)} deal(s) modeled on the Deals tab") if spec.deals else \
            (WARN, "No 8-K with a deal, debt or impairment item since the balance sheet; check press releases: none "
                   "entered (deals)")
    add(10, "Post-balance-sheet events (closings, signings, issuance, share-count settlements)", st, why)
    said =[str(x) for x in list(spec.flags) + list((spec.notes or {}).values())] + \
           [str(d_[1]) for d_ in spec.drivers.values() if isinstance(d_, tuple) and len(d_) > 1] + \
           [str(sg.mna_note or "") for sg in spec.segments]
    hits = []
    for t_ in said:
        for m_ in re.finditer(r"(?i)\b(signed|pending)\b[^.;]{0,40}?\b(acquisition|merger|deal)s?\b|"
                              r"\b(acquisition|merger|deal)s?\b[^.;]{0,25}?\b(signed|pending)\b", t_):
            ctx_ = t_[max(0, m_.start() - 60): m_.end() + 40]
            if not re.search(r"(?i)\bno (signed |pending )|none (signed|pending)|not pending", ctx_):
                hits.append(re.sub(r"\s+", " ", ctx_).strip())
    # each signed / pending deal named in the notes must be one on the Deals tab (PM round 3: Braze was modeled, so
    # "pending Contentful and Fin acquisitions" in the M&A note passed unpriced)
    names = [str(d_.get("name") or "").lower() for d_ in spec.deals or [] if d_.get("name")]
    own = {w.lower() for w in re.findall(r"[A-Za-z]+", f"{spec.name} {spec.short}")}
    generic = {"signed", "pending", "the", "no", "a", "an", "fy", "deal", "deals", "acquisition", "acquisitions", "merger",
               "q1", "q2", "q3", "q4", "post", "inc", "corp", "company", "this", "these", "any", "all", "gaap", "sec"}

    def named(h):  # a capitalized word that is not the company itself or a generic term: a target's name
        return any(w.lower() not in own | generic and not re.fullmatch(r"(FY|Q[1-4])\d*", w)
                   for w in re.findall(r"\b[A-Z][A-Za-z0-9&.-]+\b", h[1:]))
    # a deal is named by any distinctive word of its name ("pending Topaz deal" for "Topaz Labs Inc.", ADBE round 6)
    words_ = {w for n in names for w in re.findall(r"[a-z][a-z0-9&-]{2,}", n)} - generic - \
        {"labs", "holdings", "group", "corporation", "limited", "llc", "plc", "ltd", "technologies", "systems"}
    hits = [h for h in hits if not any(n and n in h.lower() for n in names) and
            not any(re.search(rf"\b{re.escape(w)}\b", h.lower()) for w in words_) and named(h) and
            not re.search(r"(?i)deals tab|in 'deals'|deals items?\b", h)]  # a sentence about the tab itself
    if econ and hits:
        add(32, "Signed pending deals are on the Deals tab (any size: the price leaves, the business arrives)", WARN,
            f"'…{hits[0]}…' names a deal that is not on the Deals tab: enter it (probability, price — estimated at "
            "its revenue × a peer EV / sales if undisclosed — its revenue and margin or EBITDA, a peer multiple) or "
            "dismiss it in thesis.explanations")
    # ---------------------------------------------------------------- plugs and calibration
    bad = [p for p in plug_list if p["status"] == FAIL]
    warn = [p for p in plug_list if p["status"] == WARN]
    add(11, "No plug in tax or other non-operating lines; operating plugs > 2% revenue / 5% EPS explained",
        FAIL if bad else (WARN if warn else PASS),
        "; ".join(f"{p['line']} {p['fy1']:.3g} vs normal {p['normal']:.3g} ({p['pct_eps']:.0%} of EPS)"
                  for p in bad + warn))
    e0, e1, e2 = (get("IS", f"{c}{IS['op_eps']}") for c in "EFG")
    if e0 and e1 and e2 and e0 > 0 and e1 > 0:
        g1, g2 = e1 / e0 - 1, e2 / e1 - 1
        unwind = [p["line"] for p in plug_list if p["kind"] == "non-operating" and p["status"] != PASS]
        st = FAIL if (g2 - g1 > 0.05 and unwind) else (WARN if abs(g2 - g1) > 0.15 else PASS)
        f0, f1, f2 = (get("CF", f"{c}39") for c in "EFG")
        s1, s2 = get("IS", f"F{IS['sales_b']}"), get("IS", f"G{IS['sales_b']}")
        fcf_txt = ""
        if f0 and f1 and f2 and f0 > 0 and f1 > 0:
            h1, h2 = f1 / f0 - 1, f2 / f1 - 1
            fcf_txt = f"; free cash flow {spec.fc_labels[0]} {h1:+.1%}, {spec.fc_labels[1]} {h2:+.1%}"
            if st == PASS and (h2 - h1 < -0.15 or (h2 < -0.05 and s1 and s2 and s2 > s1)):
                st = WARN
                lines = {(6, 9): "net income after non-cash items & one-offs", (7,): "D&A", (11,): "receivables",
                         (12,): "inventories", (13,): "other current assets", (14,): "payables",
                         (15,): "accrued & deferred revenue", (19,): "capex"}
                cf_ = lambda rs, c: sum(_nz(get("CF", f"{c}{r}")) for r in rs)  # noqa: E731
                sw = sorted((cf_(rs, "G") - cf_(rs, "F"), rs) for rs in lines)[:2]
                fcf_txt += (" (falls while sales grow: " + ", ".join(
                    f"{lines[rs]} {cf_(rs, 'F'):,.0f} → {cf_(rs, 'G'):,.0f}mm" for d_, rs in sw if d_ < 0) +
                    ". A working-capital ratio that dips in FY1 and returns later is a seasonal argument applied to "
                    "a year-end balance: every forecast year ends on the fiscal year-end. Fix it, or name the "
                    "driver in the thesis)")
        add(12, "FY2 growth vs FY1: a gap over 5pt has drivers that are not plug unwinds (EPS and FCF)", st,
            f"operating EPS growth {spec.fc_labels[0]} {g1:+.1%}, {spec.fc_labels[1]} {g2:+.1%}" + fcf_txt +
            (f": unwinding {', '.join(unwind)}" if unwind and st == FAIL else ""))
    calib = ", ".join(f"{k} ({how})" for k, how in spec.calibrated.items())
    add(13, "Guidance checks separate 'calibrated to' from 'independently consistent with'", PASS,
        f"Calibrated by the engine: {calib}" if calib else "No engine calibration; Interim shows how each metric "
                                                           "compares")
    # ---------------------------------------------------------------- valuation integrity
    cap = spec.capital or {}
    if econ and cap:
        issues = []
        if spec.market_rf and abs(spec.valuation["rf"][0] - round(spec.market_rf / 0.0005) * 0.0005) > 0.0006:
            issues.append(f"risk-free {spec.valuation['rf'][0]:.2%} vs market {spec.market_rf:.2%}")
        if cap.get("erp") is not None and abs(spec.valuation["erp"][0] - cap["erp"]) > 1e-6:
            issues.append(f"ERP {spec.valuation['erp'][0]:.2%} vs the date's {cap['erp']:.2%}")
        if cap.get("beta") and abs(spec.valuation["beta"][0] - cap["beta"]) > 0.10 + 1e-9:
            issues.append(f"beta {spec.valuation['beta'][0]:.2f} vs method {cap['beta']:.2f} ± 0.10")
        add(14, "Global parameters (risk-free, ERP, spreads) are the date's; beta from the method ±0.10",
            FAIL if issues else PASS, "; ".join(issues) or
            f"ERP {cap.get('erp', 0):.2%}; beta {spec.valuation['beta'][0]:.2f} (method {cap.get('beta', 0):.2f}); "
            f"{'agency' if str(cap.get('rating_src', '')).startswith('agency') else 'synthetic'} rating "
            f"{cap.get('rating', '')}")
    else:
        add(14, "Global parameters identical to the date's config", NA, "Reference framework / human inputs")
    im, ig = get("DCF", "C53"), get("DCF", "D53")
    em = _n(spec.valuation["exit_mult"][0]) if not cap.get("exit_derived") else im
    gg = spec.valuation["g"][0]
    if econ and cap.get("exit_derived") and im:
        # the exit multiple is the perpetuity method's own implied multiple (fade years included): consistent by
        # construction; the implied-growth test only means something for a typed multiple
        add(15, "Terminal methods reconcile: implied multiple ±20% of the exit multiple, implied g ±1pt", PASS,
            f"Exit multiple derived from the perpetuity method ({im:.1f}x, incl. {spec.valuation['fade_years'][0]:.0f} "
            "fade years): consistent by construction")
    elif im and em and ig is not None:
        off_m, off_g = abs(em / im - 1), abs(ig - gg)
        st = PASS if off_m <= 0.20 and off_g <= 0.01 else FAIL
        add(15, "Terminal methods reconcile: implied multiple ±20% of the exit multiple, implied g ±1pt", st,
            f"perpetuity implies {im:.1f}x vs exit {em:.1f}x; exit implies {ig:.1%} growth vs {gg:.1%}" +
            ("" if st == PASS else ": fix the input that is wrong (RONIC, terminal growth, exit multiple or WACC) — "
                                   "averaging the two is not a fix"))
    pct = get("DCF", "H92")
    mh = [m for _, m in spec.margin_hist]
    tm = get("DCF", "C92")
    if econ and pct is not None and tm is not None and len(mh) >= 5:
        med = median(mh)
        cyc = spec.profile.get("cyclical")
        if cyc:  # the same mid-cycle as the defaults (a structural shift uses the recent cycle)
            from .spec import mid_cycle_margin
            med = mid_cycle_margin(spec.margin_hist)[0] or med
        st = PASS
        why = (f"terminal operating margin {tm:.1%} at the {pct:.0%} percentile of {len(mh)} years "
               f"({'mid-cycle' if cyc else 'median'} {med:.1%})")
        if cyc and tm < med - 0.02:
            st, why = FAIL, why + ": a cyclical's terminal year must be at mid-cycle"
        elif cyc and tm > med + 0.02:  # symmetric (PM round 4: peaks were never normalized)
            st, why = WARN, why + (": a cyclical's terminal year sits at mid-cycle — bring it to the median, or state "
                                   "the structural change (mix, pricing) with its margin effect in pt")
        elif pct > 0.9 or pct < 0.1:
            st, why = WARN, why + ": argue the structural change in the thesis or bring it inside the range"
        add(16, "Terminal margin within the 10–90th percentile of 10-year history; cyclicals at mid-cycle", st, why)
    else:
        add(16, "Terminal margin vs 10-year history", NA, "Fewer than 5 years of margin history" if econ else "")
    r6 = get("DCF", "C94")
    g6 = get("DCF", "C96")
    if econ and r6 is not None:
        st = FAIL if (r6 < -0.10 and _nz(g6) > 0.02) else (WARN if r6 < 0 and _nz(g6) > 0.02 else PASS)
        add(17, "Reinvestment consistent with growth (terminal = g ÷ RONIC by construction)", st,
            f"{spec.fc_labels[-1]} reinvests {r6:.0%} of NOPAT at {_nz(g6):.1%} growth" +
            (": negative reinvestment while growing — D&A likely includes amortization or capex is too low"
             if st == FAIL else ""))
    grow_mna = any(any(x > 0.005 for x in sg.mna) for sg in spec.segments)
    paid = any(x > 0 for x in spec.drivers["acq"][0]) or bool(spec.deals)
    mna1 = sum(_nz(sg.mna[0]) * _nz(sg.sales[2]) for sg in spec.segments if sg.mna)
    notes = " ".join(str(sg.mna_note or "") for sg in spec.segments).lower()
    acq_note = str((spec.drivers.get("acq") or ([], ""))[1] or "").lower()
    # a note that calls the deal pending or its price undisclosed is not "already paid" (PM round 3: CRM's note said
    # "Informatica annualization" and "pending Contentful and Fin" — the first word passed the gate for the second)
    unpriced = re.search(r"\b(pending|to be acquired|expected to close|not (been )?disclosed|undisclosed)\b",
                         notes + " " + acq_note)
    # ... unless the note says those deals sit on the Deals tab, outside the segment forecast (CRM v4)
    if unpriced and re.search(r"deals tab|in 'deals'|deals item|not (included )?in the (segment )?forecast|"
                              r"not a pending", notes + " " + acq_note):
        unpriced = None
    prior = any(w in notes for w in ("annualiz", "already paid", "paid in fy", "closed in fy", "prior year",
                                     "last year", "before the balance sheet")) and not unpriced
    s0_ = _nz(spec.hist.IS["sales"][2])
    ev_ = get("SUMMARY", "C8")
    evs_ = max(ev_ / s0_, 1.0) if ev_ and s0_ > 0 else 2.0
    st, why = PASS, ("Acquisitions are paid out of the DCF (UFCF counted) in the year paid" if econ else "")
    if grow_mna and not paid:
        st, why = FAIL, "M&A growth in the forecast with no acquisition cash (drivers.acq) or deal"
    elif econ and it.available and ytd_acq >= mna1 > 0.002 * s0_ and not unpriced:
        # the price paid this year is in the 10-Q cash flow: that is the price, not revenue × a multiple (PM round 6,
        # ADBE: acquisitions were raised from the $1,560mm paid to $1,663mm so this gate's arithmetic would tie)
        st, why = PASS, (f"{spec.fc_labels[0]} M&A revenue {mna1:,.0f}mm against {ytd_acq:,.0f}mm of acquisitions "
                         f"paid by {it.period_end:%b %-d, %Y} (10-Q cash flow: the disclosed price)")
    elif econ and it.available and mna1 > 0.002 * s0_ and acq1 <= ytd_acq * 1.02 + 1 and not prior:
        st, why = WARN, (f"{spec.fc_labels[0]} M&A revenue {mna1:,.0f}mm but only the {ytd_acq:,.0f}mm paid by "
                         f"{it.period_end:%b %-d, %Y} is deducted" +
                         (" and the note calls the deal pending / its price undisclosed" if unpriced else "") +
                         f": a price not disclosed is estimated (about {2 * mna1:,.0f}mm annual revenue × the "
                         f"company's own {evs_:.1f}x EV / sales ≈ {2 * mna1 * evs_:,.0f}mm) and paid in drivers.acq or "
                         "entered as a 'deals' item; or the M&A note says the revenue is from a deal already paid for")
    add(18, "Every acquisition whose revenue is forecast has its price deducted", st, why)
    lt_q = _nz(it.bs_cur.get("lt_inv")) if it.available else 0.0
    lt_inv = lt_q or (_nz(spec.hist.BS.get("lt_inv", [0, 0, 0])[2]) if hasattr(spec.hist, "BS") else 0.0)
    mcap = spec.price * (spec.basic_shares + spec.dilutive)
    stake = spec.valuation.get("other_stake", (0.0,))[0]
    hc = spec.valuation.get("stake_haircut", (0.25,))[0]
    st = (FAIL if lt_inv > 0.02 * mcap and not stake else WARN if lt_inv > 0.02 * mcap and stake < 0.5 * lt_inv
          else PASS)
    add(19, "Stakes and investments on the balance sheet are in the equity bridge", st,
        (f"{lt_inv:,.0f}mm of long-term investments on the latest balance sheet "
         f"(${(lt_inv * (1 - hc)) / max(1e-9, spec.basic_shares + spec.dilutive):,.2f}/share after the haircut), "
         f"{'none' if not stake else f'only {stake:,.0f}mm'} in the bridge") if st != PASS else
        (f"{stake:,.0f}mm in every bridge after a {hc:.0%} haircut (balance sheet: {lt_inv:,.0f}mm)"
         if stake and econ else "No material long-term investments on the balance sheet"))
    w_sotp = spec.valuation["w_sotp"][0]
    corp_cost = any(abs(x) > 0 for x in spec.corp_cost_pct)
    st = FAIL if (w_sotp > 0 and corp_cost and all(abs(m) < 0.01 for m in spec.corp_multiples) and not econ) else PASS
    add(20, "Sum of the parts: corporate costs capitalized (> 0x); segments on their comps' earnings basis", st,
        "Corporate capitalized at the blended segment multiple; EBITDA after stock comp" if econ else "")
    add(21, "3-year return over exactly 3.0 years, exit multiple from the same discount framework",
        PASS if econ else WARN, "Derived exit P/E (Summary C25)" if econ else "Typed exit P/E (reference)")
    # ---------------------------------------------------------------- decision layer
    sv = spec.scenario_values or {}
    add(22, "Bear, base and bull values with probabilities and skew", PASS if econ else NA,
        "Each scenario recalculated end to end when the deliverable is built (Summary rows 49–54)" if econ else "")
    add(23, "Reverse DCF shown", PASS if econ and get("DCF", "C105") is not None or
        isinstance(v.get(("DCF", "C105")), str) else (NA if not econ else FAIL), "DCF rows 103–108")
    k31, c80, nfade = get("DCF", "K31"), get("DCF", "C80"), get("DCF", "C85")
    if econ and k31 and k31 > 0 and c80 is not None and nfade and nfade >= 1:
        dr = c80 / k31 - 1
        g77 = _nz(get("DCF", "C77"))
        # two-sided (PM round 4: NFLX's free cash flow jumped 24% into 2032 because the fade's reinvestment missed
        # the content spend booked in other operating items): a jump above the fade year's growth + 5pt is as
        # discontinuous as a drop
        up = dr > g77 + 0.05
        add(27, "Free cash flow continuous into the fade (first fade year no more than 5% below the last forecast "
                "year, nor more than 5pt above its growth)", FAIL if dr < -0.05 or up else PASS,
            f"UFCF {spec.fc_labels[-1]} {k31:,.0f} → {c80:,.0f}mm in the first fade year ({dr:+.1%}; fade growth "
            f"{g77:.1%})" +
            (": the fade reinvests more than the forecast does — check terminal RONIC against the forecast's own "
             "incremental return (DCF C89)" if dr < -0.05 else
             ": the fade reinvests less than the forecast does — a reinvestment line (content, leases, deferred "
             "revenue) is missing from the reinvestment rate (DCF C94)" if up else ""))
    else:
        add(27, "Free cash flow continuous into the fade", NA, "No fade years" if econ else "")
    w_of = {"dcf_pg": "w_pg", "dcf_exit": "w_exit", "sotp": "w_sotp", "comps": "w_comps"}
    broken = [f"{k} {_nz(_n(out.get(k))):,.2f}" for k, w in w_of.items()
              if spec.valuation.get(w, (0,))[0] > 0 and (_n(out.get(k)) is None or _n(out.get(k)) <= 0)]
    fv_ = _n(out.get("fair_value"))
    if econ:
        add(29, "Every valuation method used is positive; fair value is positive",
            FAIL if broken or fv_ is None or fv_ <= 0 else PASS,
            (f"{', '.join(broken)}: weight set to 0 in the fair value. A method at or below zero means broken inputs "
             "(capex, interest, segment data or a debt-funded deficit): fix them before use" if broken else
             "Fair value is not positive" if fv_ is None or fv_ <= 0 else ""))
    m0, m1 = get("IS", f"E{IS['op_margin']}"), get("IS", f"F{IS['op_margin']}")
    if econ and m0 is not None and m1 is not None:
        # guided means the model is inside the guide (PM, ADBE: 36.6% -> 4.1% passed as "guided" against a 45% guide)
        guided = bool(spec.guidance.get("margin")) and any(
            str(r.get("metric") or "").startswith("Margin") and r.get("within") == "Yes" for r in guidance or [])
        st = WARN if abs(m1 - m0) > 0.05 and not guided else PASS
        add(30, "First-year operating margin within 5pt of last year's, unless guided", st,
            f"operating margin FY{spec.fy_hist[-1]} {m0:.1%} → {spec.fc_labels[0]} {m1:.1%}" +
            (": name the driver (one-offs in FY0, a charge the forecast drops, mix) or bring it back" if st == WARN
             else (" (guided)" if guided else "")))
    selfp = [p for p in spec.peers if p.get("ticker") == spec.ticker and "placeholder" in str(p.get("source") or "")]
    if econ:
        util = spec.profile.get("key") == "utility"
        add(31, "Trading comps from a selected peer set", (FAIL if util else WARN) if selfp or not spec.peers else PASS,
            ("No peers: a regulated utility is valued on P/E vs peers and its unlevered DCF is negative while the rate "
             "base grows, so the DCF-only fair value is not a valuation — add peers (Comps tab) before use" if util else
             "No peers: the comps weight moved to the two DCF methods — add peers or state why none trade comparably")
            if selfp or not spec.peers else f"{len(spec.peers)} peers")
        with_ev = [p for p in spec.peers if _n(p.get("ev")) and _n(p.get("ebitda"))]
        if spec.peers and not selfp and not with_ev and spec.valuation["comps_pe_weight"][0] < 1 and \
                spec.valuation["w_comps"][0] > 0:
            # a failed fetch is not a multiple (PM, ADBE: EV / EBITDA missing for all five peers, published as 0.0x)
            add(31, "Trading comps: peer EV / EBITDA present for the EV / EBITDA half", FAIL,
                f"None of the {len(spec.peers)} peers has EV / EBITDA (SEC fundamentals not read: a failed fetch?): "
                "rebuild, or set valuation.comps_pe_weight to 1 and say why EV / EBITDA does not apply")
        # comps at peer multiples assume the discount closes: the thesis, not evidence for it (PM round 6, ADBE at 9.6x
        # against a 16.4x median, comps at 20% added $37/share)
        from .workbook import _pe
        pes_ = sorted(x for x in (_pe(q) for q in spec.peers) if x)
        own_pe, w_c = get("SUMMARY", "C9"), spec.valuation["w_comps"][0]
        if pes_ and own_pe and own_pe > 0:
            med_ = pes_[len(pes_) // 2]
            if own_pe < 0.7 * med_ and w_c > 0.10 + 1e-9:
                add(48, "Comps do not carry the thesis (a company 30%+ below the peer P/E: comps weight ≤ 10%)", WARN,
                    f"P/E {own_pe:.1f}x against the peers' {med_:.1f}x median ({own_pe / med_ - 1:+.0%}) with comps at "
                    f"{w_c:.0%} of fair value: valuing at peer multiples assumes the discount closes — weight comps at "
                    "10% or less and state the discount ({peer_discount_pe}) as the variant view, or explain with its "
                    "number why it closes", r"\d")
        # an analyst input a guard refused is a failed gate, not a data note (PM, ADBE: Digital Media's 95.2% gross-
        # profit margin was refused and seven explanations claimed the fix)
        rj = spec.rejected or {}
        add(45, "Analyst inputs applied as written (none refused by an engine guard)", FAIL if rj else PASS,
            "; ".join(f"{f} = {r[1]} refused ({r[2]}); the model kept {r[3]}" for f, r in list(rj.items())[:4])
            + (": set a value the guard accepts, or fix the data that made the guard refuse it" if rj else ""))
    if econ:
        for gt in round3_gates(spec, v, out, IS):
            add(*gt)
        if _n(out.get("bear_fv")) and _n(out.get("bull_fv")) and _n(out.get("fair_value")) and _n(out.get("price")):
            g.append(scenario_gate(spec, out["bear_fv"], out["fair_value"], out["bull_fv"], out["price"]))
    stale = thesis_number_problems(spec, v) if econ else []
    add(28, "Thesis numbers are the model's (placeholders such as {fv} link them; typed ones must match)",
        FAIL if stale else PASS, "; ".join(stale[:4]) if stale else "")
    cons = spec.consensus or {}
    has_c = bool(cons.get("eps") or cons.get("g_fy1") is not None or cons.get("g_fy2") is not None)
    from .market import is_recent
    mcap = (spec.price or 0) * ((spec.basic_shares or 0) + (spec.dilutive or 0))
    big_gap = not has_c and is_recent(spec.as_of) and mcap > 10000 and econ
    add(24, "Consensus comparison shown, or its absence stated", WARN if big_gap else PASS,
        "Checks tab: model vs consensus" if has_c else
        (f"No consensus returned for a ${mcap / 1000:,.0f}bn company on a current date: a data failure (Yahoo did not "
         "answer), not an absence — rebuild, or quote the consensus you can source with its date" if big_gap else
         "No point-in-time consensus for this valuation date (stated on the Checks tab)"),
        r"\d" if big_gap else None)
    return g
