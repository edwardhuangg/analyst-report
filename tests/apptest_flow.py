"""Headless run of the Streamlit flow: ticker -> defaults -> workbook bytes -> valuation shown.

    PYTHONPATH=. python tests/apptest_flow.py HON 2025-08-29 GE,RTX       # AI off
    APPTEST_AI=1 PYTHONPATH=. python tests/apptest_flow.py ETN 2026-11-12 ""   # AI analyst on (needs credits)
"""
import os
import sys
import tempfile
from datetime import date

from streamlit.testing.v1 import AppTest

ticker = sys.argv[1] if len(sys.argv) > 1 else "HON"
as_of = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2025, 8, 29)
peers = sys.argv[3] if len(sys.argv) > 3 else "GE,RTX,ETN,EMR,ROK,JCI"

use_ai = os.environ.get("APPTEST_AI") == "1"
at = AppTest.from_file("streamlit_app.py", default_timeout=900 if use_ai else 300)
at.run()
assert not at.exception, at.exception
if any(t.label == "Password" for t in at.text_input):  # the app's password gate (APP_PASSWORD in secrets)
    import tomllib
    pw = os.environ.get("APP_PASSWORD") or tomllib.load(open(".streamlit/secrets.toml", "rb")).get("APP_PASSWORD", "")
    [t for t in at.text_input if t.label == "Password"][0].set_value(pw)
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
at.text_input[0].set_value(ticker)
at.date_input[0].set_value(as_of)
at.text_input[1].set_value(peers)
for tg in at.toggle:  # AI off: deterministic run; on: AI + analyst agent
    tg.set_value(use_ai)
at.button[0].click()  # form submit
at.run()
assert not at.exception, at.exception
assert not at.error, [e.value for e in at.error]
spec = at.session_state["spec"]
print("loaded:", spec.name, spec.fc_labels[0], "segments", [s.name for s in spec.segments], "peers", len(spec.peers))
print("provenance:", spec.provenance)
print("analyst summary:", spec.notes.get("analyst_summary", "")[:300])
build = [b for b in at.button if b.label == "Build workbook"][0]
build.click()
at.run()
assert not at.exception, at.exception
x = at.session_state["xlsx"]
out = os.path.join(tempfile.gettempdir(), f"app_{ticker}.xlsx")
open(out, "wb").write(x)
print("workbook bytes:", len(x), "->", out)
rv = at.session_state["review"] if "review" in at.session_state else None
if rv:
    o = rv["outputs"]
    print("shown in step 3: check", o["check"], "| fair value", round(o["fair_value"], 2), "| upside",
          f"{o['upside']:+.1%}", "| problems", rv["problems"])
    print("metrics:", [m.label + " " + str(m.value) for m in at.metric])
    assert o["check"] == "OK"
