"""Build and recalculate models across very different companies; fail on broken ones.

    PYTHONPATH=. python tests/regression.py                     # default basket, today
    PYTHONPATH=. python tests/regression.py HON@2025-08-29 NFLX@2025-08-29 MSFT

Needs network (SEC, Yahoo) and `pip install formulas`. Checks per model: BS balances and cash
ties every year, no Excel errors, every method positive, price within 25%-400% of each method;
financial companies must be refused cleanly. Also rebuilds HON with the reference overrides and
asserts it matches HON_Valuation_Model_2025-08-29.xlsx.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import date

sys.path.insert(0, ".")
from valuation import check, overrides, pipeline  # noqa: E402

BASKET = ["HON@2025-08-29", "NFLX@2025-08-29", "MSFT", "AAPL", "AMZN", "WMT", "CAT", "KO", "NVDA", "TSLA", "JPM"]
REFERENCE = {"fair_value": 231.764, "dcf_pg": 222.265, "dcf_exit": 225.645, "sotp": 236.150, "wacc": 0.0796}


def one(arg: str) -> bool:
    t, _, d = arg.partition("@")
    as_of = date.fromisoformat(d) if d else date.today()
    t0 = time.time()
    try:
        spec, xlsx, _ = pipeline.run(t, as_of)
    except ValueError as e:
        if "financial company" in str(e):
            print(f"ok   {arg:18s} refused as a financial company")
            return True
        print(f"FAIL {arg:18s} {e}")
        return False
    vals, probs = check.check(xlsx)
    hard = [p for p in probs if p.startswith(("Integrity", "dcf_pg value is", "dcf_exit value is", "sotp value is",
                                              "comps value is")) or "Excel errors" in p]
    status = "FAIL" if hard else "ok  "
    print(f"{status} {arg:18s} {time.time() - t0:4.0f}s px {vals['price']:,.2f} FV {vals['fair_value']:,.2f} "
          f"(DCF {vals['dcf_pg']:,.0f}/{vals['dcf_exit']:,.0f} SOTP {vals['sotp']:,.0f} comps {vals['comps']:,.0f}) "
          f"segments {len(spec.segments)} flags {len(spec.flags)}")
    for p in probs:
        print(f"       - {p}")
    return not hard


def reference() -> bool:
    doc = overrides.load("samples/HON_2025-08-29_overrides.json")
    spec, xlsx, warn = pipeline.run("HON", date(2025, 8, 29), overrides_doc=doc)
    vals, _ = check.check(xlsx)
    bad = {k: (vals[k], v) for k, v in REFERENCE.items() if abs(vals[k] - v) > 0.01 * abs(v)}
    print(("FAIL" if bad or warn else "ok  "), "HON reference overrides reproduce the reference model:",
          {k: round(vals[k], 3) for k in REFERENCE}, bad or "", warn or "")
    return not bad and not warn


if __name__ == "__main__":
    os.environ.setdefault("SEC_USER_AGENT", "fa-webapp regression test admin@example.com")
    args = sys.argv[1:] or BASKET
    results = []
    for a in args:
        try:
            results.append(one(a))
        except Exception:
            traceback.print_exc(limit=3)
            results.append(False)
    if not sys.argv[1:]:
        results.append(reference())
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
