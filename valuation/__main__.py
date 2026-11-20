"""Command line: build a model, write a research brief, or check a workbook.

    python -m valuation build HON --as-of 2025-08-29 --peers GE,RTX,ETN,EMR,ROK,JCI --overrides hon.json
    python -m valuation brief HON --as-of 2025-08-29 --out HON_brief.md
    python -m valuation check HON_Valuation_Model_2025-08-29.xlsx
    python -m valuation releases GE --as-of 2025-08-29          # earnings releases (EX-99.1) before a date
    python -m valuation doc <sec.gov URL> --grep "guidance|outlook"   # any SEC document as text

`brief` writes the research brief (Markdown) and the current assumptions (<ticker>_overrides.json)
for an analyst agent; `build --overrides` applies the agent's JSON and writes the workbook;
`check` recalculates it (needs `pip install formulas`) and lists problems.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date

from . import brief, check, overrides, pipeline, sec


def _args(argv):
    p = argparse.ArgumentParser(prog="python -m valuation")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "brief"):
        s = sub.add_parser(name)
        s.add_argument("ticker")
        s.add_argument("--as-of", default=str(date.today()), help="valuation date YYYY-MM-DD (default today)")
        s.add_argument("--peers", default="", help="comma-separated peer tickers")
        s.add_argument("--ai", action="store_true", help="use OpenAI to read releases, pick peers, write the thesis")
        s.add_argument("--analyst", action="store_true", help="also run the AI analyst pass (implies --ai)")
        s.add_argument("--effort", choices=["low", "medium", "high"], default=None,
                       help="AI analyst reasoning effort (default OPENAI_REASONING_EFFORT, else medium)")
        s.add_argument("--cik", default=None)
        s.add_argument("--allow-financials", action="store_true")
        s.add_argument("--out", default=None)
        if name == "build":
            s.add_argument("--overrides", default=None, help="overrides JSON from an analyst agent")
            s.add_argument("--check", action="store_true", help="recalculate and print checks (needs formulas)")
            s.add_argument("--previous", default=None,
                           help="overrides JSON of the version the PM last read: value bridge to it on the Checks tab "
                                "(and an independent AI review if a judgment-driven change flips the rating)")
            s.add_argument("--review", action="store_true",
                           help="with --overrides: run the AI review passes on them (no new analyst pass)")
            s.add_argument("--notes", default=None,
                           help="with --review: text file with the PM's critique of the previous version (the review "
                                "pass gets it and the reading pack)")
            s.add_argument("--save-overrides", action="store_true",
                           help="also write the final assumptions as <out>_overrides.json (edit and re-apply)")
    c = sub.add_parser("check")
    c.add_argument("xlsx")
    u = sub.add_parser("calibrate", help="universe check: implied IRRs vs costs of equity across built models")
    u.add_argument("xlsx", nargs="+")
    r = sub.add_parser("releases")
    r.add_argument("ticker")
    r.add_argument("--as-of", default=str(date.today()))
    r.add_argument("-n", type=int, default=4)
    d = sub.add_parser("doc")
    d.add_argument("url")
    d.add_argument("--grep", default=None, help="regex: print only matching lines (with one line of context)")
    d.add_argument("--limit", type=int, default=200000)
    return p.parse_args(argv)


def _report(xlsx: bytes, spec=None, rv=None):
    try:
        if rv is not None:
            out, probs = rv["outputs"], rv["problems"]
        elif spec is not None:  # with the spec: the judgment checks too (stance, rest of year, methods, peers)
            rv = check.review(spec, xlsx)
            out, probs = rv["outputs"], rv["problems"]
        else:
            out, probs = check.check(xlsx)
    except ImportError:
        print("check: `pip install formulas` to recalculate the workbook")
        return 0
    for k, v in out.items():
        print(f"  {k:24s} {v:,.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    print("Problems:" if probs else "No problems found.")
    for x in probs:
        print("  -", x)
    return 1 if any(x.startswith("Integrity") for x in probs) else 0


def calibrate(paths: list) -> int:
    """Critique C1: compare the IRR each price implies (reverse DCF) with the model's cost of equity across the
    coverage universe. A median implied IRR well below the median cost of equity means the terminal framework or
    the forecasts are too punitive (or the names are rich). The ERP stays the published market-wide figure: backing it
    out of the covered names' own prices would assume they are fairly priced. Fix globally, never one name at a time."""
    import statistics
    import openpyxl
    rows = []
    for pth in paths:
        try:
            wb = openpyxl.load_workbook(pth, data_only=True)
            d = wb["DCF"]
            irr, ke, rf, beta, erp = (d[c].value for c in ("C105", "C9", "C6", "C7", "C8"))
            req_g = d["C106"].value
            a = wb["Assumptions"]
            g = next((a.cell(r, 3).value for r in range(1, a.max_row + 1)
                      if str(a.cell(r, 1).value or "").startswith("Terminal growth rate")), None)
        except Exception as e:
            print(f"  {pth}: unreadable ({e})")
            continue
        if isinstance(irr, (int, float)) and isinstance(ke, (int, float)):
            rows.append((os.path.basename(pth), irr, ke, rf, beta, erp))
            pct = lambda x: f"{x:6.2%}" if isinstance(x, (int, float)) else f"{str(x or '-'):>6s}"  # noqa: E731
            print(f"  {os.path.basename(pth):45s} implied IRR {irr:6.2%}  cost of equity {ke:6.2%}  beta {beta:.2f}  "
                  f"terminal g {pct(g)}, price needs {pct(req_g)}")
        else:
            print(f"  {os.path.basename(pth):45s} implied IRR {irr}  (outside the solver range or reference model)")
    if len(rows) < 3:
        print("Need at least 3 models with an implied IRR")
        return 1
    m_irr, m_ke = statistics.median(r[1] for r in rows), statistics.median(r[2] for r in rows)
    rf = statistics.median(r[3] for r in rows)
    imp_erp = statistics.median((r[1] - r[3]) / r[4] for r in rows if r[4])
    print(f"Median implied IRR {m_irr:.2%} vs median cost of equity {m_ke:.2%} (gap {m_ke - m_irr:+.2%}); "
          f"ERP in use {statistics.median(r[5] for r in rows):.2%}; ERP the prices imply (median (IRR - rf) / beta) "
          f"{imp_erp:.2%} at a {rf:.2%} risk-free rate")
    if m_ke - m_irr > 0.01:
        print("The universe prices at a lower discount rate than the model: review the terminal framework (terminal "
              "growth vs the risk-free rate the ERP was measured against, RONIC, fade) and the forecasts for every "
              "name — do not adjust names one at a time, and do not back the ERP out of these prices")
    return 0


def main(argv=None) -> int:
    a = _args(argv if argv is not None else sys.argv[1:])
    if a.cmd == "check":
        with open(a.xlsx, "rb") as fh:
            return _report(fh.read())
    if a.cmd == "calibrate":
        return calibrate(a.xlsx)
    if a.cmd == "releases":
        co = sec.load_company(a.ticker)
        rels = [f for f in co.filings if f.form == "8-K" and "2.02" in f.items
                and f.filing_date <= date.fromisoformat(a.as_of)][:a.n]
        print(f"{co.name} (CIK {int(co.cik)}), fiscal year end {co.fiscal_year_end}")
        for f in rels:
            print(f"  {f.filing_date}  {sec.exhibit_99_url(f) or f.url}")
        return 0
    if a.cmd == "doc":
        text = sec.html_to_text(sec._get(a.url), limit=a.limit)
        if a.grep:
            lines = text.splitlines()
            rx = re.compile(a.grep, re.I)
            keep = sorted({j for i, ln in enumerate(lines) if rx.search(ln) for j in (i - 1, i, i + 1)
                           if 0 <= j < len(lines)})
            text = "\n".join(lines[j] for j in keep)
        print(text)
        return 0
    as_of = date.fromisoformat(a.as_of)
    peers = [x.strip().upper() for x in a.peers.split(",") if x.strip()] or None
    use_ai = a.ai or a.analyst or getattr(a, "review", False)
    log = lambda m: print("  ..", m, file=sys.stderr)  # noqa: E731
    data = pipeline.fetch(a.ticker, as_of, peers=peers, use_ai=use_ai, progress=log, cik=a.cik,
                          allow_financials=a.allow_financials)
    review_only = bool(getattr(a, "review", False) and a.overrides)
    spec = pipeline.make_spec(data, use_ai_thesis=use_ai and not review_only, analyst=a.analyst and not review_only,
                              progress=log, effort=a.effort, guards=not a.overrides)
    if a.cmd == "brief":
        out = a.out or f"{spec.ticker}_brief_{as_of}.md"
        with open(out, "w") as fh:
            fh.write(brief.build(spec, data))
        js = os.path.splitext(out)[0].replace("_brief", "") + "_overrides.json"
        with open(js, "w") as fh:
            fh.write(overrides.dumps(spec))
        print(f"wrote {out} and {js}")
        return 0
    if a.overrides:
        warn = overrides.apply(spec, overrides.load(a.overrides), source="overrides file")
        for w in warn:
            print("  warning:", w, file=sys.stderr)
        pipeline.fill_peer_ev(spec)
        if review_only and check.available():
            from . import ai
            client = ai.client()
            if client:
                notes = open(a.notes).read() if getattr(a, "notes", None) else ""
                w2, _ = pipeline.review_passes(spec, data, client, log, a.effort, "overrides file", notes=notes)
                for w in w2:
                    print("  warning:", w, file=sys.stderr)
                spec.flags.extend(f"AI analyst output: {w}" for w in w2)
                spec.ai_cost, spec.ai_usage = ai.usage_summary() if ai.USAGE else "", list(ai.USAGE)
    if getattr(a, "previous", None) and check.available():
        out_ = pipeline.bridge_previous(spec, data, overrides.load(a.previous), os.path.basename(a.previous), log,
                                        a.effort, use_ai)
        print(out_[0])
        for w in out_[1:]:
            print(w if w.startswith("After the independent review") else f"  warning: {w}",
                  file=sys.stdout if w.startswith("After") else sys.stderr)
    xlsx = pipeline.build(spec)
    rv = None
    if a.check and check.available():  # final checks go on the Cover, so the file carries its open items
        rv = check.review(spec, xlsx)
        spec.last_review = rv
        xlsx = pipeline.build(spec)
    out = a.out or f"{spec.ticker}_Valuation_Model_{as_of}.xlsx"
    with open(out, "wb") as fh:
        fh.write(xlsx)
    print(f"wrote {out} ({len(xlsx):,} bytes); provenance: " + " | ".join(spec.provenance))
    if a.save_overrides:
        js = os.path.splitext(out)[0] + "_overrides.json"
        with open(js, "w") as fh:
            fh.write(overrides.dumps(spec))
        print(f"wrote {js}")
        if a.check and check.available():  # the saved file must rebuild the model it was saved from
            s2 = pipeline.make_spec(data, use_ai_thesis=False, guards=False)
            overrides.apply(s2, overrides.load(js), source="round trip")
            pipeline.fill_peer_ev(s2)
            fv2 = check.cached_values(pipeline.build(s2, values=False)).get(("SUMMARY", "G19"))
            fv1 = ((rv or {}).get("outputs") or {}).get("fair_value")
            same = isinstance(fv1, (int, float)) and isinstance(fv2, (int, float)) and abs(fv1 - fv2) < 0.01
            print(f"round trip: {js} rebuilds to fair value {fv2:,.2f} vs {fv1:,.2f}" if same else
                  f"round trip MISMATCH: {js} rebuilds to {fv2} vs the model's {fv1}: an input is not exported")
    if spec.notes.get("analyst_summary"):
        print("Analyst summary:", spec.notes["analyst_summary"])
    if spec.ai_cost:
        print("AI cost:", spec.ai_cost)
        for r in spec.ai_usage:
            usd = f"${r['usd']:.3f}" if r["usd"] is not None else "price n/a"
            print(f"  {r['what']:28s} {r['model']:14s} in {r['input']:>7,} (cached {r['cached']:,}) "
                  f"out {r['output']:>6,} (reasoning {r['reasoning']:,}) {r['seconds']:>6.0f}s {usd}")
    if spec.data_notes:
        print(f"{len(spec.data_notes)} data notes:")
        for f in spec.data_notes:
            print("  -", f)
    return _report(xlsx, spec, rv) if a.check else 0


if __name__ == "__main__":
    sys.exit(main())
