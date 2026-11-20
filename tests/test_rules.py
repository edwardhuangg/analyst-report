"""Unit checks of the round-3 to round-6 PM rules that need no network: comps fits, EV / EBITDA dispersion, IRR
parsing, the version-bridge paths, one-off peer EPS, the recommendation text and agency ratings.
Run: python tests/test_rules.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace  # noqa: E402

from valuation import bridge, capital, gates, pipeline  # noqa: E402
from valuation.workbook import ev_dispersed, ev_growth_fit, peer_eps_issue, peer_growth_fit  # noqa: E402

N = [0, 0]


def ok(name, cond, detail=""):
    N[0] += 1
    N[1] += bool(cond)
    print(("ok   " if cond else "FAIL ") + name, detail if not cond else "")


def peer(pe, g, ev=None, rg=None):
    return {"price": pe * 10.0, "eps": 10.0, "g_next": g, "ev": ev, "ebitda": 100.0 if ev else None, "rev_g_next": rg}


# a fit that pays less for more growth is rejected (CRM round 3: P/E = 17.6 - 10.6 x growth)
neg = [peer(20, 0.00), peer(18, 0.05), peer(16, 0.10), peer(15, 0.15), peer(13, 0.20), peer(12, 0.25)]
f = peer_growth_fit(neg)
ok("negative-slope P/E fit rejected", f["a"] is None and "pays less" in f["rejected"], f)
ok("rejected fit falls back to the median PEG", f["peg"] is not None, f)
pos = [peer(10, 0.02), peer(13, 0.06), peer(16, 0.10), peer(19, 0.14), peer(23, 0.18), peer(26, 0.22)]
f = peer_growth_fit(pos)
ok("positive, tight P/E fit used", f["a"] is not None and f["b"] > 0 and f["r2"] > 0.9, f)
ok("five peers: no regression", peer_growth_fit(pos[:5])["a"] is None)
# EV / EBITDA on revenue growth, and the dispersion rule
crm = [peer(10, .12, 885, .091), peer(21, .21, 3850, .187), peer(16, .36, 1393, .457), peer(16, .23, 2609, .111),
       peer(14, .19, 3165, .104), peer(22, .21, 1874, .205)]
ok("CRM's software peers: EV / EBITDA fit rejected", ev_growth_fit(crm)["a"] is None)
ok("CRM's software peers: EV / EBITDA too dispersed (P/E only)", "P/E only" in ev_dispersed(crm), ev_dispersed(crm))
tight = [peer(15, .05, m, .05) for m in (11.0, 11.5, 12.0, 12.5, 13.0, 13.5)]
ok("tight industrial EV / EBITDA set is usable", ev_dispersed(tight) == "")
# implied IRR text from the solver grid
ok("IRR text parsed", pipeline.irr_value("> 20%") == 0.20 and pipeline.irr_value("< 3%") == 0.03 and
   pipeline.irr_value(0.101) == 0.101 and pipeline.irr_value(None) is None)
# bridge paths
prev = {"segments": {"A": {"growth": 1, "note": "x"}, "B": {"growth": 2}}, "valuation": {"ronic": {"value": .5}}}
new = {"segments": {"A": {"growth": 1, "note": "y"}, "C": {"growth": 3}}, "valuation": {"ronic": {"value": .27}}}
ok("wildcard expands over both versions' segments",
   bridge._expand(("segments", "*", "growth"), prev, new) == [("segments", n, "growth") for n in "ABC"])
ok("a reworded note is not a change", bridge._strip(prev["segments"]["A"]) == bridge._strip(new["segments"]["A"]))
d = {"valuation": {"ronic": {"value": .5}}}
bridge._set(d, ("valuation", "ronic"), new["valuation"]["ronic"])
bridge._set(d, ("segments", "B", "growth"), bridge.MISSING)
ok("set / remove a path", d["valuation"]["ronic"]["value"] == .27)
ok("needs review only on a judgment-driven flip",
   bridge.needs_independent_review({"rating_changed": True, "judgment_share": 0.8}) and
   not bridge.needs_independent_review({"rating_changed": True, "judgment_share": 0.3}) and
   not bridge.needs_independent_review({"rating_changed": False, "judgment_share": 1.0}))
# round 4: one-off peer EPS (GOOGL: EPS falling 44% next year), basis errors (NVO at 1.8x), cyclical fits (DE)
googl = peer(15, -0.44)
ok("peer EPS falling 20%+ next year is excluded", peer_eps_issue(googl) != "" and peer_eps_issue(peer(25, .1)) == "")
set_ = [peer(20, .1), peer(22, .1), peer(18, .08), peer(1.8, -.02), peer(21, .12)]
ok("a P/E under a third of the others' median is a basis error", peer_eps_issue(set_[3], set_) != "" and
   peer_eps_issue(set_[0], set_) == "")
ok("a cyclical gets no growth fit", peer_growth_fit(pos, cyclical=True)["a"] is None and
   "cyclical" in peer_growth_fit(pos, cyclical=True)["how"])
reb = pos + [peer(30, 0.80)]
ok("a 40%+ rebound is left out of the fit", "rebounding" in peer_growth_fit(reb)["how"], peer_growth_fit(reb))
# the recommendation's call comes from the rule: the analyst's sentence keeps its substance, loses its call
ok("leading call stripped", gates.recommendation_text("Underweight / avoid adding: Caterpillar is executing well.")
   == "Caterpillar is executing well.")
ok("composed line stripped back to the sentence",
   gates.recommendation_text("Neutral, negative bias: Costco is an exceptional compounder.") ==
   "Costco is an exceptional compounder.")
th = {"recommendation_text": "The base case has enough upside to warrant ownership.",
      "points": ["1. Long-term growth is durable.", "2. We are Neutral with a positive bias."]}
calls = gates.stated_calls(SimpleNamespace(thesis=th))
ok("stated calls found ('warrant ownership', 'positive bias'), 'Long-term' ignored",
   {c for _, c, _ in calls} == {"long", "positive"}, calls)
ok("a negated call is not a call", gates.stated_calls(SimpleNamespace(thesis={"points": ["This is not a Short."]})) == [])
# agency ratings: middle of three, lower of two, Moody's mapped to the S&P scale
ok("agency rating: middle of three", capital.agency_rating({"S&P": "A", "Moody's": "A2", "Fitch": "A+"})[0] == "A")
ok("agency rating: lower of two", capital.agency_rating({"S&P": "AA-", "Moody's": "A1"})[0] == "A+")
ok("no agency rating -> None", capital.agency_rating({}) is None)
ok("synthetic rating tops out at AA", capital.synthetic_rating(99) == ("AA", 0.0070))
# saved market data fills a failed refetch, and never replaces a successful one
from valuation import overrides  # noqa: E402
sp = SimpleNamespace(capital={"own_raw": None, "spread": 0.01}, pe_hist=[], market_rf=None, short_interest={}, flags=[],
                     valuation={"rf": (0.0425, "")})
overrides._market_data(sp, {"market_data": {"own_beta_raw": 0.85, "pe_hist": [[2024, 40.0]], "rf": 0.0515}}, [])
ok("saved market data used when the refetch is empty", sp.capital["own_raw"] == 0.85 and sp.pe_hist == [(2024, 40.0)]
   and sp.market_rf == 0.0515 and abs(sp.capital["kd"] - 0.0615) < 1e-9)
sp2 = SimpleNamespace(capital={"own_raw": 0.9}, pe_hist=[(2024, 30.0)], market_rf=0.05, short_interest={}, flags=[],
                      valuation={})
overrides._market_data(sp2, {"market_data": {"own_beta_raw": 0.5, "pe_hist": [[2024, 10.0]], "rf": 0.04}}, [])
ok("a successful refetch is kept", sp2.capital["own_raw"] == 0.9 and sp2.pe_hist == [(2024, 30.0)] and
   sp2.market_rf == 0.05)
# round 6 (PM on the ADBE website run)
from datetime import date  # noqa: E402

from valuation import financials, spec as S  # noqa: E402
ok("an explanation claiming a fix is recognized", gates.FIXED_RX.search("Fixed by restoring Digital Media margin") and
   gates.FIXED_RX.search("Correcting the margin should restore bear/base/bull ordering") and
   not gates.FIXED_RX.search("Not fixed: the 10-Q has no notes table"))
sp = SimpleNamespace(thesis={"explanations": {"gate 16": "Terminal margin is brought back inside the range."}})
it = gates.open_items(sp, [{"id": 16, "gate": "Terminal margin within the range", "status": "WARN", "detail": ""}], [])
ok("a fix claimed on a check that still fires stays open", it[0][2] is None and "claims a fix" in it[0][1], it)
ok("peer EPS falling while revenue grows is a one-off (CRM)",
   peer_eps_issue({"price": 194.5, "eps": 16.75, "g_next": -0.045, "rev_g_next": 0.097}) != "" and
   peer_eps_issue({"price": 100.0, "eps": 5.0, "g_next": -0.02, "rev_g_next": 0.03}) == "")
msft = {"eps": 20.01, "eps_next": 24.20, "fy_end": "2027-06-30", "source": "Yahoo"}
pipeline.calendarize_peer(msft, date(2026, 11, 27))
ok("MSFT's June year calendarized to a November year", abs(msft["eps"] - 17.99) < 0.05 and msft["eps_fy"] == 20.01,
   msft)
e = msft["eps"]
pipeline.calendarize_peer(msft, date(2026, 11, 27))
ok("calendarizing twice changes nothing", msft["eps"] == e)
ok("five-column reconciliation line: the year is the last two",
   S._reported_eps_line("Non-GAAP diluted net income per share | $ | 5.50 | | $ | 4.81 | | $ | 5.31 | | $ | 20.94 | "
                        "| $ | 18.42 | | \n Shares used")[-2:] == (20.94, 18.42))
fake = SimpleNamespace(hist=SimpleNamespace(IS={"eps_dil": [12.36, 12.36, 16.70]}), interim=SimpleNamespace(
    available=False), fy_hist=[2023, 2024, 2025], nonrec_hist=[0, 0, 0], guidance={})
ok("this year's guide in last year's slot is caught (ADBE fy 24.50)",
   "guide" in S.adj_eps_problem(fake, {"fy": 24.50, "fy_prior": 20.94}, guide=[24.45, 24.50]))


class _Inst:  # Instance stand-in: {concept: value} on one date
    def __init__(self, f):
        self.f = f

    def value(self, concepts, end=None, start=None, dims=frozenset()):
        return next((self.f[c] for c in concepts if c in self.f), None)


ok("LongTermDebt on the face as the noncurrent line (ADBE)", financials._ltd_generic(_Inst(
    {"LongTermDebt": 6210e6, "DebtCurrent": 0.0, "DebtInstrumentCarryingAmount": 6150e6}), date(2025, 11, 28)) == 6210)
ok("LongTermDebt as total debt: current portion netted out", financials._ltd_generic(_Inst(
    {"LongTermDebt": 5650e6, "LongTermDebtCurrent": 1500e6, "DebtInstrumentCarryingAmount": 5650e6}),
    date(2024, 11, 29)) == 4150)
ok("a gross-profit segment may run above 95%",
   overrides._margin_cap(SimpleNamespace(profit=[0, 0, 93.0], sales=[0, 0, 100.0])) > 0.95 and
   overrides._margin_cap(SimpleNamespace(profit=[0, 0, 30.0], sales=[0, 0, 100.0])) == 0.95)
bs = SimpleNamespace(interim=SimpleNamespace(available=True, period_end=date(2026, 8, 28)), period_ends=[])
ok("a deal closed before the balance sheet is not a Deals-tab item (Semrush)",
   overrides._closed_before_bs(bs, {"status": "closed", "close": "2026-04-01"}) and
   not overrides._closed_before_bs(bs, {"status": "pending", "close": "2026-11-28"}))
# round 7 (PM on ADBE r6 and the second website run)
ok("scenario risk named as the reason for beta / terminal growth is recognized",
   overrides.RISK_RX.search("Beta raised to reflect AI disruption risk to the creative franchise") and
   overrides.RISK_RX.search("Terminal growth lowered for structural decline risk") and
   not overrides.RISK_RX.search("Beta relevered at the higher post-ASR leverage"))
ok("a measurement reason lets beta move", overrides.MEASURE_RX.search("relevered at D/E 23% after the debt raise"))
ok("output-targeting reasons are recognized",
   overrides.TARGET_RX.search("Bear made harsher to reduce the upside/downside skew") and
   overrides.TARGET_RX.search("If a large gap remains, the residual is the explicit variant view") and
   not overrides.TARGET_RX.search("Base margin follows the 45% guide"))
ok("next-quarter EPS target read from a targets table", S._next_quarter_eps_guide(
    "The following table summarizes Adobe's fourth quarter FY2026 targets \n Total revenue | $6.80 billion to $6.85 "
    "billion | \n Earnings per share 1 \n | GAAP $4.65 to $4.70 | Non-GAAP $6.30 to $6.35 | \n The following table "
    "summarizes Adobe's updated FY2026 targets \n Non-GAAP $24.45 to $24.50") == (6.30, 6.35))
spec7 = SimpleNamespace(thesis={"points": ["The fade runs to a 3.75% terminal growth rate.",
                                           "Risks: the P/E premium could compress.",
                                           "Buybacks are sized within expected free cash flow.",
                                           "Upside of {gap} to fair value."]},
                        valuation={"g": (0.0325, ""), "ronic": (0.6, ""), "beta": (1.16, "")},
                        fc_labels=[f"{y}E" for y in range(2026, 2032)],
                        peers=[{"price": 100.0, "eps": 6.0}, {"price": 100.0, "eps": 6.2}, {"price": 100.0, "eps": 5.8}])
v7 = {("DCF", "C74"): 0.0325, ("DCF", "C17"): 0.10, ("SUMMARY", "C9"): 9.6, ("SUMMARY", "C5"): 226.2}
for c, bb, f in zip("FGHIJK", (9000, 12000, 12500, 13000, 13000, 13000), (10000, 10800, 11000, 11100, 11200, 11300)):
    v7[("CF", f"{c}28")], v7[("CF", f"{c}39")] = -bb, f
pr = gates.thesis_number_problems(spec7, v7)
ok("typed terminal growth that differs from the model is caught", any("terminal growth" in p_ for p_ in pr), pr)
ok("a premium claim at a discount is caught", any("premium" in p_ for p_ in pr), pr)
ok("buybacks 'within free cash flow' above it are caught", any("buybacks" in p_ for p_ in pr), pr)
ok("{gap} renders (a known placeholder) and an unknown one is caught",
   not any("{gap}" in p_ for p_ in pr) and any("unrendered" in p_ for p_ in gates.thesis_number_problems(
       SimpleNamespace(thesis={"points": ["x {gapp} y"]}, valuation=spec7.valuation, peers=[], fc_labels=spec7.fc_labels), v7)))
# round 8 (PM on ADBE r7)
from valuation.workbook import peer_ni_growth  # noqa: E402
ok("a quarter's adjusted EPS is the first column of a 3+ column line; a 2-figure line is a guide",
   S._quarter_eps("Non-GAAP diluted net income per share | $ | 6.13 | | $ | 5.31 | | $ | 5.96 | | \n Shares | 395 "
                  "Non-GAAP diluted net income per share | $ | 6.30 | | | $ | 6.35 | \n") == 6.13)
ok("peer growth on a net-income basis (a buyback is not growth)",
   abs(peer_ni_growth({"g_next": 0.10, "share_chg": -0.05}) - 0.045) < 1e-9 and
   peer_ni_growth({"g_next": 0.10}) == 0.10)
v8 = dict(v7)
pr8 = gates.thesis_number_problems(SimpleNamespace(
    thesis={"explanations": {"gate 37": "Comps remain a 20% cross-check because the regression is rejected."}},
    valuation={**spec7.valuation, "w_comps": (0.10, "")}, peers=[], fc_labels=spec7.fc_labels), v8)
ok("a typed comps weight that differs from the model is caught", any("comps" in p_ for p_ in pr8), pr8)
# round 9 (PM on MRK and DE through the hosted app)
ok("a net-income guide in billions is not an EPS guide (DE)",
   S._in_billions("Net income guidance improved to $4.75 billion to $5.00 billion.", 4.75, 5.0) and
   not S._in_billions("Expected Non-GAAP EPS Range To Be Between $2.77 and $2.82", 2.77, 2.82))
step = [(2016 + k, m) for k, m in enumerate([0.11, 0.12, 0.115, 0.12, 0.11, 0.18, 0.21, 0.21, 0.19, 0.137])]
flat = [(2016 + k, m) for k, m in enumerate([0.14, 0.15, 0.13, 0.15, 0.14, 0.15, 0.14, 0.16, 0.15, 0.14])]
ok("a structural step-up sets mid-cycle on the recent cycle (DE); a steady history keeps the 10-year median",
   abs(S.mid_cycle_margin(step)[0] - 0.19) < 1e-9 and abs(S.mid_cycle_margin(flat)[0] - 0.145) < 0.006, 
   (S.mid_cycle_margin(step), S.mid_cycle_margin(flat)))
mrk = SimpleNamespace(hist=SimpleNamespace(IS={"eps_dil": [0, 0, 7.0], "sales": [0, 0, 65000.0], "ni_attr": [0, 0, 1]},
                                           CF={}), adj_eps_hist=[None, None, 8.98], eps_guide_charge=True)
ok("an EPS guide far below last year stands when the release says charges are in it (MRK)",
   S.guidance_problem(mrk, "adj_eps", 2.77, 2.82) == "" and
   S.guidance_problem(SimpleNamespace(**{**vars(mrk), "eps_guide_charge": False}), "adj_eps", 2.77, 2.82) != "")
ok("an input bent to fit another is output-targeting (DE capex)",
   overrides.TARGET_RX.search("Capex ramps to 8% of sales to make terminal reinvestment consistent with RONIC"))
print(f"{N[1]}/{N[0]} passed")
sys.exit(0 if N[0] == N[1] else 1)
