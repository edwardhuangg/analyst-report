"""Recalculate a built workbook and run the sanity checks an analyst would.

Needs the optional `formulas` package (pip install formulas). `check` returns the key outputs
and a list of problems; `review` adds the computed forecast and the guidance comparison, so an
agent (or the in-app AI analyst) can iterate: build -> review -> adjust overrides -> rebuild.
"""
from __future__ import annotations

import contextlib
import io
import re
import tempfile

KEYS = {
    "check": ("SUMMARY", "C45"), "price": ("SUMMARY", "C5"), "ev": ("SUMMARY", "C8"),
    "pe_fy1": ("SUMMARY", "C9"), "ev_ebitda_fy1": ("SUMMARY", "C10"),
    "dcf_pg": ("SUMMARY", "D15"), "dcf_exit": ("SUMMARY", "D16"), "sotp": ("SUMMARY", "D17"),
    "comps": ("SUMMARY", "D18"), "fair_value": ("SUMMARY", "G19"), "upside": ("SUMMARY", "G20"),
    "irr_3y": ("SUMMARY", "C29"), "wacc": ("DCF", "C17"), "tv_share_pg": ("DCF", "C52"),
    "implied_exit_multiple": ("DCF", "C53"), "implied_g_from_exit": ("DCF", "D53"),
    "implied_irr": ("DCF", "C105"), "req_terminal_g": ("DCF", "C106"), "req_terminal_margin": ("DCF", "C107"),
    "req_fade_years": ("DCF", "C108"), "exit_pe_3y": ("SUMMARY", "C25"), "roic_gw_t": ("DCF", "C93"),
}
FC = "FGHIJK"


def available() -> bool:
    try:
        import formulas  # noqa: F401
        return True
    except ImportError:
        return False


def recalc(xlsx: bytes) -> dict:
    import formulas  # optional dependency
    with tempfile.NamedTemporaryFile(suffix=".xlsx") as fh:
        fh.write(xlsx)
        fh.flush()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            sol = formulas.ExcelModel().loads(fh.name).finish().calculate()
    vals = {}
    for k, v in sol.items():
        m = re.match(r"'?\[(.+?)\](.+?)'?!([A-Z]+\d+)$", k)
        if not m:
            continue
        try:
            val = v.value[0][0]
        except Exception:
            val = v
        vals[(m.group(2).upper(), m.group(3))] = val
    return vals


def cached_values(xlsx: bytes) -> dict:
    """Every formula's computed result, as plain numbers / text, for writing into the file."""
    out = {}
    for k, v in recalc(xlsx).items():
        if hasattr(v, "dtype"):
            v = v.item()
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif type(v) is str:
            out[k] = v
    return out


def _n(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _problems(v: dict, out: dict, econ: bool = False) -> list[str]:
    probs = []
    if out["check"] != "OK":
        bs = [v.get(("BS", f"{c}43")) for c in "DEFGHIJK"]
        cf = [v.get(("CF", f"{c}42")) for c in "DEFGHIJK"]
        probs.append(f"Integrity check failed: BS {bs}, CF {cf}")
    errors = sorted({f"{s}!{c}" for (s, c), x in v.items() if isinstance(x, str) and x.startswith("#")})
    if errors:
        probs.append(f"{len(errors)} cells show Excel errors, e.g. {', '.join(errors[:8])}")
    px = _n(out["price"])
    for k in ("dcf_pg", "dcf_exit", "sotp", "comps"):
        x = _n(out[k])
        if x is None or x <= 0:
            probs.append(f"{k} value is {out[k]}: not usable (check drivers or set its weight to 0)")
        elif px and not (0.25 * px <= x <= 4 * px):
            probs.append(f"{k} value {x:,.2f} is far from the price {px:,.2f}: check the drivers behind it")
    tv = _n(out["tv_share_pg"])
    if tv is not None and tv > 0.9:
        probs.append(f"Terminal value is {tv:.0%} of DCF EV: the explicit forecast carries little weight")
    g, w = _n(out["implied_g_from_exit"]), _n(out.get("wacc"))
    if not econ and g is not None and (g < -0.01 or g > (min(0.07, w - 0.02) if w else 0.06)):
        probs.append(f"Exit multiple implies {g:.1%} perpetual growth at the model's {w or 0:.1%} WACC: inconsistent "
                     "with the perpetuity method. Reconcile both sides: the multiple against the peers' EV/EBITDA and "
                     "the company's through-cycle range, and the WACC against the market-implied cost of equity; do not "
                     "just set the multiple to the perpetuity-implied one (the two DCF methods are meant to triangulate)")
    for (s, c), x in v.items():  # Interim guidance check (rows flex with the segment count)
        if s == "INTERIM" and c.startswith("F") and x == "Check":
            label = v.get(("INTERIM", "A" + c[1:]))
            probs.append(f"Base case outside company guidance: {label}")
    return probs


def check(xlsx: bytes) -> tuple[dict, list[str]]:
    v = recalc(xlsx)
    out = {k: v.get(c) for k, c in KEYS.items()}
    probs = _problems(v, out)
    if _n(out["price"]) and _n(out["fair_value"]):
        out["upside"] = out["fair_value"] / out["price"] - 1
    return out, probs


def review(spec, xlsx: bytes, scenarios: bool = False) -> dict:
    """Key outputs, the computed base-case forecast, guidance vs model, and problems. scenarios=True also values
    the bear and bull (two more recalculations) so the scenario gate (33) is in the list the analyst sees."""
    from .workbook import build_workbook, is_layout, segments_layout
    v = recalc(xlsx)
    out = {k: v.get(c) for k, c in KEYS.items()}
    if scenarios and spec.framework != "reference":
        keep = spec.scenario
        try:
            for key, sc in (("bear_fv", 1), ("bull_fv", 3)):
                spec.scenario = sc
                out[key] = recalc(build_workbook(spec)).get(("SUMMARY", "G19"))
        except Exception:
            pass
        finally:
            spec.scenario = keep
    IS, SG = is_layout(spec.has_disc, spec.has_nonrec), segments_layout(spec)
    rows = {"Net sales": ("IS", IS["sales_b"]), "Sales growth": ("IS", IS["growth"]),
            "Organic growth": ("SEGMENTS", SG["t_org"]), "Segment margin": ("IS", IS["seg_margin"]),
            "EBIT": ("IS", IS["ebit"]), "Operating margin": ("IS", IS["op_margin"]),
            "EBITDA": ("IS", IS["ebitda"]), "Adjusted EPS": ("IS", IS["adj_eps"]),
            "Diluted EPS (GAAP)": ("IS", IS["eps"]), "Free cash flow": ("CF", 39), "Capex": ("CF", 19),
            "Unlevered FCF": ("DCF", 31), "Ending cash": ("BS", 6)}
    forecast = {label: [v.get((sh, f"{c}{r}")) for c in FC] for label, (sh, r) in rows.items()}
    g0 = 52 + len(spec.segments) + 2
    guidance = []
    for k in range(5):
        r = g0 + k
        lo, hi = v.get(("INTERIM", f"C{r}")), v.get(("INTERIM", f"D{r}"))
        if _n(lo) is not None and _n(hi) is not None:
            guidance.append({"metric": v.get(("INTERIM", f"A{r}")), "low": lo, "high": hi,
                             "model": v.get(("INTERIM", f"E{r}")), "within": v.get(("INTERIM", f"F{r}"))})
    roy = []
    r0 = 52 + len(spec.segments) + 10
    if spec.interim.available:
        for k in range(5):
            r = r0 + 2 + k
            roy.append({"metric": v.get(("INTERIM", f"A{r}")), "ytd": v.get(("INTERIM", f"B{r}")),
                        "implied_rest": v.get(("INTERIM", f"D{r}")), "prior_rest": v.get(("INTERIM", f"E{r}")),
                        "implied_growth": v.get(("INTERIM", f"F{r}")), "ytd_growth": v.get(("INTERIM", f"G{r}"))})
    probs = _problems(v, out, spec.framework != "reference")
    if _n(out["price"]) and _n(out["fair_value"]):
        out["upside"] = out["fair_value"] / out["price"] - 1
    probs += judgment_problems(spec, v, out, roy, guidance)
    probs += consistency_problems(spec, v, out, roy, IS)
    from . import gates as G
    plug_list = G.plugs(spec, v, IS)
    gate_list = G.gates(spec, v, out, IS, guidance, plug_list)
    if scenarios and spec.framework != "reference":
        gate_list += G.decision_gates(spec, out)
    for gt in gate_list:
        if gt["status"] in (G.FAIL, G.WARN):
            probs.append(f"Gate {'failed' if gt['status'] == G.FAIL else 'warning'} ({gt['id']}): {gt['gate']}"
                         + (f" — {gt['detail']}" if gt["detail"] else ""))
    return {"outputs": out, "forecast": forecast, "years": spec.fc_labels, "guidance": guidance,
            "rest_of_year": roy, "problems": probs, "gates": gate_list, "plugs": plug_list,
            "open_items": G.open_items(spec, gate_list, probs, plug_list)}


def consistency_problems(spec, v: dict, out: dict, roy: list, IS: dict) -> list[str]:
    """Internal consistency a PM checks first: the rest of the year must be plausible, the full year must
    contain what is already booked, cash must have a use, the thesis must quote the final numbers."""
    probs = []
    by = {str(r["metric"]).split(" (")[0].lower(): r for r in roy}
    sl, sp = by.get("net sales"), by.get("total segment profit")
    # on segment profit, which is before one-off charges: front-loaded charges (LLY's acquired IPR&D in Q1-Q3)
    # legitimately lift the rest-of-year GAAP margin, but the underlying margin must still be plausible
    if sl and sp and all(_n(x) for x in (sl["ytd"], sl["implied_rest"], sp["ytd"], sp["implied_rest"])):
        m_rest, m_ytd = sp["implied_rest"] / sl["implied_rest"], sp["ytd"] / sl["ytd"]
        m_prior = sp["prior_rest"] / sl["prior_rest"] if _n(sp.get("prior_rest")) and _n(sl.get("prior_rest")) else m_ytd
        lo, hi = min(m_ytd, m_prior), max(m_ytd, m_prior)
        if m_rest > hi + 0.08 or m_rest < lo - 0.08:
            probs.append(f"Not credible: {spec.fc_labels[0]} implies a rest-of-year segment margin of {m_rest:.0%} vs "
                         f"{m_ytd:.0%} year to date and {m_prior:.0%} in the same period last year: the FY1 segment "
                         "margin must be on the model's basis (before non-segment charges), not a company margin "
                         "defined differently")
    ytd_oi, fy_oi = _n(v.get(("INTERIM", "D12"))), _n(v.get(("IS", f"F{IS['other_inc']}")))
    sales = _n(v.get(("IS", f"F{IS['sales_b']}"))) or 0
    if spec.interim.available and ytd_oi is not None and fy_oi is not None and ytd_oi > 0.005 * sales \
            and fy_oi < 0.8 * ytd_oi:
        probs.append(f"{spec.fc_labels[0]} other income {fy_oi:,.0f}mm is below the {ytd_oi:,.0f}mm already booked "
                     f"in {spec.interim.label_cur}: put the booked one-off in drivers.nonrec (GAAP keeps it, adjusted "
                     "EPS excludes it) unless a loss is expected in the rest of the year")
    cash, ta = _n(v.get(("BS", "K6"))), _n(v.get(("BS", "K20")))
    if cash and ta and cash / ta > 0.35:
        probs.append(f"Cash builds to {cash:,.0f}mm ({cash / ta:.0%} of total assets) by {spec.fc_labels[-1]}: capital "
                     "return (buybacks, dividends, debt paydown) is under-assumed for the cash the model generates")
    # balance-sheet lines that cannot go negative (PM red flags on the first read of the BS)
    for r, label, fix in [(6, "Cash", "buybacks / debt repayment exceed the cash the model generates"),
                          (14, "PP&E", "capex is below depreciation for years: check capex_pct vs da_pct"),
                          (15, "Goodwill", "acquisitions (drivers.acq) must be cash paid, >= 0"),
                          (16, "Other intangibles", "amortization exceeds the intangibles left"),
                          (29, "Long-term debt", "repayment (drivers.debt_iss) exceeds the debt outstanding"),
                          (8, "Receivables", "check dso"), (9, "Inventories", "check dio")]:
        neg = [(lab, x) for lab, x in zip(spec.fc_labels, (_n(v.get(("BS", f"{c}{r}"))) for c in FC))
               if x is not None and x < -0.5]
        if neg:
            probs.append(f"Balance sheet: {label} turns negative in {neg[0][0]} ({neg[0][1]:,.0f}mm): {fix}")
    # pharma asset deals: the price is expensed as acquired IPR&D (a non-segment charge); entering it again as an
    # acquisition counts the cash twice and capitalizes goodwill / intangibles for an expensed asset (LLY run)
    acq = _n((spec.drivers.get("acq") or ([None],))[0][0])
    if (spec.profile or {}).get("key") == "pharma" and acq and acq > 0:
        from .spec import booked_charges
        booked = booked_charges(spec)
        if booked and booked > 0:
            probs.append(f"Double count risk: {spec.fc_labels[0]} acquisitions of {acq:,.0f}mm while {booked:,.0f}mm of "
                         f"acquired IPR&D / non-segment charges is already expensed in {spec.interim.label_cur}: "
                         "asset acquisitions expensed as IPR&D are a charge only (repo_pct), so drivers.acq must hold "
                         "only business combinations that create goodwill (set it to 0 otherwise)")
    return probs  # thesis numbers vs the model: gate 28 (every valuation figure, ±1.5%)


def judgment_problems(spec, v: dict, out: dict, roy: list, guidance: list = ()) -> list[str]:
    """Checks a PM would make on top of the arithmetic: does the thesis match the valuation, do the methods
    roughly agree, does the full year fit the year to date, are the peers comparable."""
    probs = []
    econ = spec.framework != "reference"
    up, irr, ke = _n(out.get("upside")), _n(out.get("irr_3y")), _n(v.get(("DCF", "C9")))
    stance = spec.stance_phrase
    if up is not None and irr is not None and ke is not None:
        if stance == "long" and up < -0.05 and irr < ke:
            probs.append(f"Thesis says long but fair value is {up:.0%} vs the price and the 3-year return "
                         f"{irr:.1%} is below the cost of equity {ke:.1%}: change the stance or explain the gap")
        if stance == "short" and up > 0.10 and irr > ke:
            probs.append(f"Thesis says short but fair value is {up:+.0%} vs the price: change the stance or explain")
        if abs(up) > 0.35:
            probs.append(f"Fair value is {up:+.0%} vs the price: before calling the market that wrong, check the "
                         "drivers that set the gap — for a cyclical, mid-cycle (not trough or peak) volumes and margins "
                         "in the last forecast years and the terminal value; for a compounder, the growth runway "
                         "(fade_years); the cost of capital. Keep the call only if the thesis shows why the market "
                         "is wrong")
    vals = {k: _n(out.get(k)) for k in ("dcf_pg", "dcf_exit", "sotp", "comps")}
    pos = {k: x for k, x in vals.items() if x and x > 0}
    if len(pos) >= 3 and max(pos.values()) > 1.8 * min(pos.values()):
        hi, lo = max(pos, key=pos.get), min(pos, key=pos.get)
        probs.append(f"Methods disagree: {hi} {pos[hi]:,.2f} vs {lo} {pos[lo]:,.2f}; fix the driver behind the "
                     "outlier or explain it and weight it accordingly")
    guided = {str(g["metric"]).split(" (")[0].lower() for g in guidance if g.get("within") == "Yes"}
    if "segment margin" in guided or "margin" in guided:  # a met margin guide sets the year's profit (NFLX)
        guided |= {"total segment profit", "operating income"}
    for row in roy:
        g, y, m = _n(row["implied_growth"]), _n(row["ytd_growth"]), str(row["metric"] or "")
        if g is None or y is None or m.split(" (")[0].lower() in guided:
            continue  # inside company guidance: the guidance explains the shape of the year
        tol = 0.10  # critique C6: reconcile any rest-of-year gap over 10 points with company commentary
        if abs(g - y) > tol and not m.startswith("Free cash"):
            probs.append(f"{spec.fc_labels[0]} implies rest-of-year {m.split(' (')[0].lower()} growth of {g:.0%} vs "
                         f"{y:.0%} year to date (Interim tab): explain (guidance, seasonality, deal, one-off) or fix")
    # cyclicals: the terminal value must capitalize a mid-cycle margin, not the trough (DE run: 2031E 15.2% vs a
    # 17.9% three-year average after a 21% peak and a 14% trough)
    if (spec.profile or {}).get("key") in ("industrial", "captive_finance", "semis", "energy") and not econ:
        from .workbook import is_layout
        r = is_layout(spec.has_disc, spec.has_nonrec)["seg_margin"]
        hist = [_n(v.get(("IS", f"{c}{r}"))) for c in "CDE"]
        hist = [x for x in hist if x is not None]
        last = _n(v.get(("IS", f"K{r}")))
        if len(hist) == 3 and last is not None and last < sum(hist) / 3 - 0.02 and max(hist) - min(hist) > 0.03:
            probs.append(f"Cyclical: {spec.fc_labels[-1]} segment margin {last:.1%} is below the last three years' "
                         f"average {sum(hist) / 3:.1%} ({', '.join(f'{x:.1%}' for x in hist)}): the terminal value "
                         "capitalizes a below-mid-cycle margin. Put the last forecast years at mid-cycle margins and "
                         "volumes, or state the structural reason they stay lower")
    em, cur = _n(spec.valuation["exit_mult"][0]), _n(out.get("ev_ebitda_fy1"))
    pm = sorted(p["ev"] / p["ebitda"] for p in spec.peers
                if _n(p.get("ev")) and _n(p.get("ebitda")) and p["ebitda"] > 0)
    if em and len(pm) >= 3 and not econ:  # economic framework: the reconciliation gate replaces this
        med = pm[len(pm) // 2]
        if em < 0.65 * med and (cur is None or em < 0.65 * cur):
            probs.append(f"Exit multiple {em:.1f}x is far below the peers' {med:.1f}x EV / LTM EBITDA median"
                         + (f" and the company's {cur:.1f}x forward" if cur else "") + ": that assumes the business "
                         "de-rates to an ex-growth multiple by the terminal year. Justify it (maturity, peak cycle) or "
                         "set it from the peers' and the company's through-cycle range; the perpetuity method already "
                         "carries the intrinsic view")
    pes = [(v.get(("COMPS", f"B{r}")), _n(v.get(("COMPS", f"F{r}")))) for r in range(5, 5 + len(spec.peers))]
    good = sorted(x for _, x in pes if x and x > 0)
    if len(good) >= 4:
        med = good[len(good) // 2]
        odd = [f"{t} {x:.1f}x" for t, x in pes if x and x > 0 and (x < 0.4 * med or x > 2.5 * med)]
        if odd:
            probs.append(f"Peer P/E outliers vs the {med:.1f}x median: {', '.join(odd)} — are they comparable "
                         "businesses (growth, margins, not being acquired)?")
    return probs


def _fmt(label: str, x) -> str:
    if _n(x) is None:
        return str(x) if x is not None else "–"
    if any(k in label.lower() for k in ("growth", "margin", "upside", "wacc", "share", "irr", "implied_g")):
        return f"{x:.1%}"
    if "eps" in label.lower() or label in ("price", "dcf_pg", "dcf_exit", "sotp", "comps", "fair_value"):
        return f"{x:,.2f}"
    if "multiple" in label or label.startswith(("pe_", "ev_ebitda")):
        return f"{x:.1f}x"
    return f"{x:,.0f}"


def review_text(r: dict) -> str:
    L = ["Valuation outputs (scenario in Assumptions!C4):"]
    L.append("  " + "; ".join(f"{k} {_fmt(k, x)}" for k, x in r["outputs"].items()))
    L.append("\nBase-case forecast | " + " | ".join(r["years"]))
    for label, vals in r["forecast"].items():
        L.append(f"  {label} | " + " | ".join(_fmt(label, x) for x in vals))
    if r["guidance"]:
        L.append(f"\nGuidance vs model ({r['years'][0]}):")
        for g in r["guidance"]:
            m = str(g["metric"])
            L.append(f"  {m}: guide {_fmt(m, g['low'])}–{_fmt(m, g['high'])}, model {_fmt(m, g['model'])} "
                     f"-> {g['within']}")
    if r.get("rest_of_year"):
        L.append(f"\nRest of {r['years'][0]} implied by the model vs year to date:")
        for x in r["rest_of_year"]:
            m = str(x["metric"])
            L.append(f"  {m}: YTD {_fmt(m, x['ytd'])}, implied rest of year {_fmt(m, x['implied_rest'])} -> "
                     f"implied growth {_fmt('growth', x['implied_growth'])} vs YTD growth {_fmt('growth', x['ytd_growth'])}")
    L.append("\nProblems found by the checker:" if r["problems"] else "\nNo problems found by the checker.")
    L.extend(f"  - {p}" for p in r["problems"])
    left = [(k, t) for k, t, w in (r.get("open_items") or []) if not w]
    if left:
        L.append("\nWarnings and open checks without an explanation (fix them, or explain each in "
                 "thesis.explanations under its key; any left unexplained holds the rating at Under review):")
        L.extend(f"  - key '{k}': {t[:160]}" for k, t in left)
    return "\n".join(L)
