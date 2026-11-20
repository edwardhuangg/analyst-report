"""AI path with a mocked OpenAI client (no network call to OpenAI).

Covers release extraction, peer suggestion, point-in-time peer guidance, the analyst pass
(overrides format) and the workbook build. Run: PYTHONPATH=. python tests/test_ai_mock.py
"""
import json
import os
import sys
import tempfile
from datetime import date
from types import SimpleNamespace

sys.path.insert(0, ".")
from valuation import ai, pipeline  # noqa: E402

RESP = {
    "extract": {"release_date": "July 24, 2025",
                "guidance": {"sales": [40800, 41300], "organic": [0.04, 0.05], "margin": [0.23, 0.232],
                             "adj_eps": [10.45, 10.65], "fcf": [5400, 5800]},
                "adjusted_eps": {"fy": 9.89, "fy_prior": 9.52, "ytd_cur": 5.26, "ytd_prior": 4.74},
                "organic_fy_total": 0.03,
                "organic_fy_by_segment": {"Aerospace Technologies": 0.11, "Industrial Automation": -0.07,
                                          "Building Automation": 0.02, "Energy & Sustainability Solutions": 0.02},
                "organic_q_total": 0.05,
                "organic_q_by_segment": {"Aerospace Technologies": 0.06, "Industrial Automation": 0.0,
                                         "Building Automation": 0.08, "Energy & Sustainability Solutions": 0.06}},
    "peers": {"peers": [{"ticker": "GE", "overlap": "Aerospace"}, {"ticker": "RTX", "overlap": "Aerospace (Collins)"},
                        {"ticker": "EMR", "overlap": "Process & industrial automation"}],
              "segment_multiples": {"Aerospace Technologies": 20, "Industrial Automation": 13,
                                    "Building Automation": 17, "Energy & Sustainability Solutions": 12},
              "segment_notes": {"Aerospace Technologies": "GE Aerospace, RTX, Eaton aerospace"}},
    "peer_guidance": {"fiscal_year": 2025, "adj_eps_guidance": [5.6, 5.8], "basis": "adjusted"},
    "analyst": {"meta": {"summary": "Long into the separation."},
                "segments": {"Aerospace Technologies": {
                    "growth": {"base": [0.08, 0.07, 0.065, 0.06, 0.06, 0.055]},
                    "mna": {"values": [0.03, 0, 0, 0, 0, 0], "note": "CAES annualizes"}},
                    "Energy & Sustainability Solutions": {"sotp_split": [
                        {"name": "Solstice", "ebitda": 1000, "multiples": [8, 10, 13], "note": "Form 10"},
                        {"name": "UOP", "remainder": True, "multiples": [12, 14, 16], "note": "stays"}]}},
                "valuation": {"peer_discount": {"value": 0.2, "note": "Until separation completes"},
                              "other_stake": {"value": 2700, "note": "Quantinuum"}},
                "labels": {"peer_discount": "Conglomerate discount", "other_stake": "Quantinuum stake"},
                "thesis": {"stance": "long", "recommendation": "Long HON into the separations.",
                           "points": ["1. a", "2. b", "3. c", "4. d"], "model_says": ["• x", "• y", "• z"],
                           "risks": ["• r1", "• r2", "• r3", "• r4"], "monitor": ["• m1", "• m2"]}},
}


class Fake:
    calls = []

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        sys_msg, user = kw["messages"][0]["content"], kw["messages"][1]["content"]
        if "senior buy-side analyst" in sys_msg:
            key = "analyst"
        elif "Use only US-listed" in sys_msg:
            key = "peers"
        elif "adj_eps_guidance" in user:
            key = "peer_guidance"
        elif "extract numbers" in sys_msg:
            key = "extract"
        else:
            key = "thesis"
        Fake.calls.append(key)
        body = RESP.get(key) or RESP["analyst"]["thesis"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))])


ai.client = lambda: Fake()
d = pipeline.fetch("HON", date(2025, 8, 29), peers=None, use_ai=True)
s = pipeline.make_spec(d, use_ai_thesis=True, analyst=True)
print("calls:", Fake.calls)
assert "analyst" in Fake.calls and "thesis" not in Fake.calls, "analyst pass should supply the thesis"
assert "peer_guidance" in Fake.calls, "past valuation date: peer EPS should come from peer releases"
print("guidance:", s.guidance, "| adj EPS hist:", s.adj_eps_hist, "| reported:", s.adj_eps_reported)
print("peers:", [(p["ticker"], p["eps"], p["source"][:40]) for p in s.peers])
print("SOTP split:", s.sotp_splits)
print("labels:", s.labels, "| provenance:", s.provenance)
assert s.sotp_splits and s.labels["other_stake"] == "Quantinuum stake"
assert all(p["eps"] == 5.7 for p in s.peers)
out = os.path.join(tempfile.gettempdir(), "HON_ai_mock.xlsx")
open(out, "wb").write(pipeline.build(s))
print("wrote", out)
try:
    from valuation import check
    vals, probs = check.check(open(out, "rb").read())
    print("check:", vals["check"], "fair value", round(vals["fair_value"], 2), "problems:", probs)
    assert vals["check"] == "OK"
except ImportError:
    print("(pip install formulas to recalculate)")
