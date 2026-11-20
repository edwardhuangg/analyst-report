"""Version-to-version bridge (PM round 3): when a model is republished, every change of input against the previous
version is valued in sequence, so a rating flip shows exactly what moved it. The audit trail measures overrides
against the pipeline default; this measures the new version against the last one the PM read.

Groups are applied cumulatively, from the previous overrides document to the new one, in a fixed order; each step is
the fair-value change it causes, so the steps add up exactly. A group is "judgment" (RONIC, growth, margins,
working capital, weights, capital return) or "data" (peers, stakes, deals, shares, tax and interest). A rating
change that comes mainly from judgment inputs needs an independent review before it ships (pipeline.rate)."""
from __future__ import annotations

import copy
import json

SEG_GROWTH = ("growth", "mna", "lines", "use_lines", "organic_last", "kpi")
GROUPS = [
    ("Terminal value: RONIC, terminal growth, fade years, exit multiple", True,
     [("valuation", k) for k in ("ronic", "g", "fade_years", "exit_mult", "exit_pe")]),
    ("Cost of capital: beta, cost of debt, tax rate", True, [("valuation", k) for k in ("beta", "kd", "tax_wacc")]),
    ("Method weights, peer discount, comps blend, SOTP multiples", True,
     [("valuation", k) for k in ("w_pg", "w_exit", "w_sotp", "w_comps", "peer_discount", "comps_pe_weight")] +
     [("corporate", "multiples"), ("segments", "*", "sotp")]),
    ("Peers (set and EPS)", False, [("peers",)]),
    ("Stakes, minorities, debt-like items", False,
     [("valuation", k) for k in ("other_stake", "stake_haircut", "nci_value", "captive_share")] +
     [("debt_like",), ("labels",)]),
    ("Deals (probability, synergies, multiples are judgment)", True, [("deals",)]),
    ("Revenue growth and M&A", True, [("segments", "*", k) for k in SEG_GROWTH] + [("notes", "growth")]),
    ("Margins and their basis", True, [("segments", "*", "margin"), ("seg_after_charges",), ("corporate", "cost_pct"),
                                       ("corporate", "sales"), ("notes", "margin")]),
    ("Working capital", True, [("drivers", k) for k in ("dso", "dio", "dpo", "accrued_pct", "oca_pct")]),
    ("Capex, D&A, stock comp, amortization, charges", True,
     [("drivers", k) for k in ("cogs_pct", "da_pct", "capex_pct", "sbc_pct", "amort", "repo_pct", "other_op")]),
    ("Capital return, acquisitions, financing", True,
     [("drivers", k) for k in ("buyback", "buyback_px", "debt_iss", "shares_issued", "stock_proc", "dps0",
                               "div_growth", "acq", "intang_share")]),
    ("Tax, interest, other income, one-offs", False,
     [("drivers", k) for k in ("etr", "int_rate", "int_yield", "other_nonop", "nci_ni", "nonrec")]),
    ("Share count, definitions, guidance", False,
     [("market",), ("adj_eps_ex_sbc",), ("adj_eps_ex_repo",), ("adj_eps_ex_nonrec",), ("adj_eps_amort_share",),
      ("adj_eps_tax_hist",), ("drivers", "adj_tax"), ("guidance",), ("adjusted_eps",), ("organic",)]),
    ("Scenario probabilities", True, [("scenario_probs",)]),
]
MISSING = object()


def _get(doc, path):
    cur = doc
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return MISSING
        cur = cur[k]
    return cur


def _set(doc, path, val):
    cur = doc
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    if val is MISSING:
        cur.pop(path[-1], None)
    else:
        cur[path[-1]] = copy.deepcopy(val)


def _expand(path, prev, new):
    if "*" not in path:
        return [path]
    i = path.index("*")
    names = set((_get(prev, path[:i]) if isinstance(_get(prev, path[:i]), dict) else {}) or {}) | \
        set((_get(new, path[:i]) if isinstance(_get(new, path[:i]), dict) else {}) or {})
    return [path[:i] + (n,) + path[i + 1:] for n in sorted(names)]


def _strip(doc):
    """Values only: notes and sources do not move value, and a reworded note is not a change."""
    if isinstance(doc, dict):
        return {k: _strip(v) for k, v in doc.items() if k not in ("note", "source", "sources", "sotp_note",
                                                                   "mna_note", "ev_source")}
    if isinstance(doc, list):
        return [_strip(x) for x in doc]
    return doc


def version_bridge(data, prev_doc: dict, new_doc: dict, progress=lambda m: None, label: str = "previous") -> dict:
    """Fair value under the previous document, each group of changes added in turn, and the new document."""
    from . import check, overrides, pipeline
    base = pipeline.make_spec(data, use_ai_thesis=False, guards=False)

    def model(doc, rating=False):
        s = copy.deepcopy(base)
        overrides.apply(s, copy.deepcopy(doc), source="version bridge")
        pipeline.fill_peer_ev(s)
        xlsx = pipeline.build(s, values=False)
        vals = check.cached_values(xlsx)
        fv = vals.get(("SUMMARY", "G19"))
        if not rating:
            return fv, None
        s.last_review = check.review(s, xlsx)
        s.scenario_values = pipeline.scenario_values(s, vals)
        if pipeline.bracket_price(s, vals):  # the same scenario rule as the deliverable
            s.scenario_values = pipeline.scenario_values(s, vals)
        r = pipeline.rate(s, vals)
        return fv, {"rating": r.get("rule") or r.get("rating"), "shown": r.get("rating"), "gap": r.get("gap"),
                    "irr_spread": r.get("irr_spread"), "pw": r.get("pw")}

    progress("Version bridge: the previous version on today's engine…")
    fv0, r0 = model(prev_doc, rating=True)
    cur, last, steps = copy.deepcopy(prev_doc), fv0, []
    for name, judgment, paths in GROUPS:
        moved = []
        for p in paths:
            for q in _expand(p, prev_doc, new_doc):
                a, b = _get(cur, q), _get(new_doc, q)
                if json.dumps(_strip(a), sort_keys=True, default=str) != json.dumps(_strip(b), sort_keys=True,
                                                                                     default=str):
                    _set(cur, q, b)
                    moved.append(".".join(str(x) for x in q))
        if not moved:
            continue
        progress(f"Version bridge: {name}…")
        fv, _ = model(cur)
        if isinstance(fv, (int, float)) and isinstance(last, (int, float)) and abs(fv - last) >= 0.005:
            steps.append({"group": name, "judgment": judgment, "effect": round(fv - last, 2), "keys": moved})
        last = fv
    fv1, r1 = model(new_doc, rating=True)
    if isinstance(fv1, (int, float)) and isinstance(last, (int, float)) and abs(fv1 - last) >= 0.01:
        steps.append({"group": "Other (thesis switches, interactions)", "judgment": False,
                      "effect": round(fv1 - last, 2), "keys": []})
    tot = sum(abs(x["effect"]) for x in steps) or 1.0
    jshare = sum(abs(x["effect"]) for x in steps if x["judgment"]) / tot
    return {"label": label, "prev_fv": fv0, "new_fv": fv1, "prev": r0, "new": r1, "steps": steps,
            "judgment_share": jshare,
            "rating_changed": bool(r0 and r1 and r0.get("rating") != r1.get("rating"))}


def needs_independent_review(b: dict | None) -> bool:
    return bool(b and b.get("rating_changed") and b.get("judgment_share", 0) >= 0.5)


def bridge_text(b: dict) -> str:
    L = [f"Previous version: fair value ${b['prev_fv']:,.2f}, rule rating {(b.get('prev') or {}).get('rating')}; "
         f"this version: ${b['new_fv']:,.2f}, rule rating {(b.get('new') or {}).get('rating')}. Judgment inputs "
         f"carry {b['judgment_share']:.0%} of the gross move."]
    for x in b["steps"]:
        L.append(f"  {'[judgment] ' if x['judgment'] else '[data] '}{x['group']}: {x['effect']:+,.2f}/share "
                 f"({', '.join(x['keys'][:8])})")
    return "\n".join(L)


def judgment_diff(prev_doc: dict, new_doc: dict, b: dict) -> dict:
    """The judgment inputs that moved, previous vs new value (with notes), for the independent reviewer."""
    out = {}
    for x in b["steps"]:
        if not x["judgment"]:
            continue
        for k in x["keys"]:
            q = tuple(k.split("."))
            out[k] = {"previous": _get(prev_doc, q), "new": _get(new_doc, q)}
            out[k] = {kk: (None if vv is MISSING else vv) for kk, vv in out[k].items()}
    return out


def independent_review(spec, data, client, prev_doc: dict, b: dict, progress=lambda m: None,
                       effort=None) -> list:
    """A reviewer that did not build the model challenges each judgment change behind a rating flip: keep it with
    evidence or revert it to the previous version. Its verdict is written into thesis.explanations."""
    from . import ai, overrides
    new_doc = json.loads(overrides.dumps(spec))
    doc = ai.challenge(client, bridge_text(b), judgment_diff(prev_doc, new_doc, b),
                       {k: (spec.thesis or {}).get(k) for k in ("variant_view", "breaker", "price_implies",
                                                                "consensus_notes")},
                       progress=progress, effort=effort)
    verdict = str(((doc.get("thesis") or {}).get("explanations") or {}).get("independent review") or "")
    warn = overrides.apply(spec, doc, source="independent review")
    if verdict:
        spec.thesis.setdefault("explanations", {})["independent review"] = verdict
    return warn
