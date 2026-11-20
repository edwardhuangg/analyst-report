"""Parsing fixes that a PM would catch, asserted against the filings (needs network; SEC responses are cached).

    PYTHONPATH=. python tests/test_accounting.py
"""
import os
import sys
from datetime import date

sys.path.insert(0, ".")
from valuation import financials as F, pipeline, sec  # noqa: E402
from valuation.xbrl import Instance  # noqa: E402

AS_OF = date(2026, 11, 12)
results = []


def ok(name, cond, detail=""):
    results.append(bool(cond))
    print(("ok  " if cond else "FAIL"), name, detail)


def annual(t, as_of=AS_OF):
    co = sec.load_company(t)
    fs = sec.select_filings(co, as_of)
    return fs, F.parse_annual(Instance(sec.fetch_instance(fs.annual)), sec.income_statement_concepts(fs.annual),
                              sec.member_children(fs.annual), sec.balance_sheet_concepts(fs.annual))


if __name__ == "__main__":
    os.environ.setdefault("SEC_USER_AGENT", "fa-webapp accounting test admin@example.com")

    # DE: interest sits inside "costs and expenses" -> derived COGS must exclude it (no double count);
    # the finance arm's interest is operating; long-term debt is a company-extension tag
    fs, h = annual("DE")
    I = h.IS
    ni_model = I["pretax"][1] - I["tax"][1] + I["disc"][1]
    ok("DE FY2024 net income ties (no double-counted interest)", abs(ni_model - I["ni_total"][1]) < 2,
       f"{ni_model:,.0f} vs {I['ni_total'][1]:,.0f}")
    ok("DE captive finance detected", h.captive.get("segment") == "Financial Services", h.captive)
    ok("DE long-term debt found (de: extension tag)", (h.BS["ltd"][2] or 0) > 30000, f"{h.BS['ltd'][2]:,.0f}")
    # DE: own PP&E is an extension tag, equipment leased to customers is its own BS line and its depreciation is in
    # D&A -> PP&E includes the fleet and capex its net purchases (the PP&E roll went negative without them);
    # payables are reported with accrued liabilities (unclassified balance sheet)
    ok("DE PP&E includes the leased fleet", 15000 < h.BS["ppe"][2] < 16500, f"{h.BS['ppe'][2]:,.0f}")
    ok("DE capex on the same basis as D&A", -h.CF["capex"][2] > 0.9 * h.CF["da"][2],
       f"capex {-h.CF['capex'][2]:,.0f} vs D&A {h.CF['da'][2]:,.0f}")
    ok("DE accounts payable & accrued found", h.BS["ap"][2] > 10000, f"{h.BS['ap'][2]:,.0f}")

    # SHW: segment profit is reported before tax -> corporate restated to the operating basis
    _, h = annual("SHW")
    tot = sum(s.profit[2] for s in h.segments) + h.corp_profit[2]
    ok("SHW total segment profit on an operating basis", abs(tot - h.IS["opinc"][2]) < 2,
       f"{tot:,.0f} vs operating income {h.IS['opinc'][2]:,.0f}")

    # CRM: fiscal year labelled as the company does (year ended Jan 31, 2026 = FY2026)
    data = pipeline.fetch("CRM", AS_OF, use_ai=False)
    ok("CRM fiscal year label", data.hist.years[-1] == 2026, data.hist.years)

    # GOOGL: 10-Q SG&A includes sales & marketing (same definition as the 10-K); securities gains are one-offs
    data = pipeline.fetch("GOOGL", AS_OF, use_ai=False)
    ytd = data.interim.ytd_cur
    ok("GOOGL 10-Q SG&A includes S&M", ytd["sga"] > 35000, f"{ytd['sga']:,.0f}")
    ok("GOOGL FY2025 one-offs detected", (data.hist.IS["nonrec"][2] or 0) > 20000, data.hist.IS["nonrec"])
    ok("GOOGL Google Services split into lines", len(data.hist.segments[0].lines) >= 4,
       [n for n, _ in data.hist.segments[0].lines])

    # LLY / NFLX: revenue lines (drugs without therapeutic-area subtotals; regions)
    _, h = annual("LLY")
    names = [n for n, _ in h.segments[0].lines]
    ok("LLY lines by drug", "Mounjaro" in names and "Zepbound" in names and "Cardiometabolic Health" not in names, names)
    tot = sum(v[2] for _, v in h.segments[0].lines)
    ok("LLY lines add up to revenue", abs(tot - h.IS["sales"][2]) < 1, f"{tot:,.0f} vs {h.IS['sales'][2]:,.0f}")
    _, h = annual("NFLX")
    ok("NFLX lines by region", len(h.segments[0].lines) == 4, [n for n, _ in h.segments[0].lines])

    # NFLX: untagged one-off in 9M 2026 other income lands in FY1 one-offs (GAAP), not in adjusted EPS
    data = pipeline.fetch("NFLX", AS_OF, use_ai=False)
    spec = pipeline.make_spec(data, use_ai_thesis=False)
    ok("NFLX FY1 one-offs carry the YTD other income", spec.drivers["nonrec"][0][0] > 2000, spec.drivers["nonrec"][0])

    ok("NFLX interest income estimated when not reported", spec.drivers["int_yield"][0][0] > 0.01,
       spec.drivers["int_yield"][0][0])

    # LLY: the 10-Q cover counts ~50mm employee-benefit-trust shares that are not outstanding for EPS
    data = pipeline.fetch("LLY", AS_OF, use_ai=False)
    spec = pipeline.make_spec(data, use_ai_thesis=False)
    ok("LLY basic shares = EPS share count, not the cover", spec.basic_shares < 900, spec.basic_shares)

    # peers: a broken share count must not produce an EV (FOXA came out at 0.6x EV/EBITDA)
    snap = {"ticker": "FOXA", "price": 65.71}
    pipeline.peer_ev(snap, AS_OF)
    ok("FOXA EV/EBITDA is sane or absent", snap.get("ev") is None or 2 <= snap["ev"] / snap["ebitda"] <= 80, snap)

    # COST: warehouse counts from its filings, for a volume x price KPI build
    from valuation import facts
    data = pipeline.fetch("COST", AS_OF, use_ai=False)
    kh = facts.kpi_history(data.company.cik, data.hist.period_ends, AS_OF)
    ok("COST warehouse history", kh.get("Stores / warehouses (year-end)") == [890, 914, 939], kh)

    # CRM: the FY27 guide "$13.11 - $13.19" was read as FY26 / FY25 adjusted EPS and the quarter's $5.90 as the six
    # months; the release's reconciliation line (table cells) holds the results, and a guidance range is skipped
    from valuation.spec import _in_range, _reported_eps_line
    guide = "Non-GAAP diluted net income per share (2) \n | $3.11 - $3.13 | | $13.11 - $13.19 | \n Shares"
    line = ("Non-GAAP diluted net income per share | $ | 3.81 | | | $ | 2.78 | | | $ | 12.52 | | | $ | 10.20 | | \n"
            " Shares used")
    ok("CRM reconciliation line read, guidance range skipped",
       _reported_eps_line(guide + " " + line) == (3.81, 2.78, 12.52, 10.20) and _reported_eps_line(guide) == (),
       _reported_eps_line(guide + " " + line))
    ok("CRM guidance range endpoint is not a result", _in_range(guide, guide.index("13.11"), guide.index("13.11") + 5))
    data = pipeline.fetch("CRM", date(2026, 11, 12), use_ai=False)
    spec = pipeline.make_spec(data, use_ai_thesis=False)
    ok("CRM reported adjusted EPS from the reconciliation", spec.adj_eps_hist[1:] == [10.20, 12.52] and
       spec.interim.ytd_cur.get("adj_eps") == 9.73, (spec.adj_eps_hist, spec.interim.ytd_cur.get("adj_eps")))

    # data audit of 20 fresh names (2026-11-15): each a number a PM would have thrown the model out for
    from valuation.xbrl import humanize_member
    ok("Segment names read cleanly", [humanize_member(m) for m in ("mrk:PharmaceuticalsegmentMember",
                                                                   "duk:ElectricUtilitiesandInfrastructureMember",
                                                                   "ups:U.S.DomesticPackageMember")]
       == ["Pharmaceutical", "Electric Utilities & Infrastructure", "U.S. Domestic Package"])
    nov = date(2026, 11, 13)
    d = pipeline.fetch("MCD", nov, use_ai=False)  # weighted shares tagged in millions (709.9 'shares')
    ok("MCD share counts on the right scale", 650 < d.hist.IS["dil_shares"][2] < 800 and
       650 < (d.interim.q_basic or 0) < 800, (d.hist.IS["dil_shares"], d.interim.q_basic))
    d = pipeline.fetch("SLB", nov, use_ai=False)  # no cost-of-sales or operating-income tag: was a 97% margin
    ok("SLB operating income from pre-tax income", 0.05 < d.hist.IS["opinc"][2] / d.hist.IS["sales"][2] < 0.30,
       d.hist.IS["opinc"])
    d = pipeline.fetch("DUK", nov, use_ai=False)  # segment 'profit' is segment net income
    ok("DUK built on operating income, not segment net income", d.hist.segments[0].member == "total" and
       d.hist.segments[0].profit[2] > 5000, [(s_.name, s_.profit[2]) for s_ in d.hist.segments])
    d = pipeline.fetch("NEE", nov, use_ai=False)  # capex and interest only on company tags
    ok("NEE capex and interest read", -d.hist.CF["capex"][2] > 15000 and d.hist.IS["interest_exp"][2] > 2000,
       (d.hist.CF["capex"], d.hist.IS["interest_exp"]))
    ok("NEE capex: the total tag is not added to its own parts", all(23000 < -x < 26500 for x in d.hist.CF["capex"]),
       d.hist.CF["capex"])
    d = pipeline.fetch("VZ", nov, use_ai=False)  # capex on the other-productive-assets tag
    ok("VZ capex as reported, every year", [-x for x in d.hist.CF["capex"]] == [18767, 17090, 17011],
       d.hist.CF["capex"])
    d = pipeline.fetch("DIS", nov, use_ai=False)  # interest expense tagged negative
    ok("DIS interest expense positive", d.hist.IS["interest_exp"][2] > 1000, d.hist.IS["interest_exp"])
    d = pipeline.fetch("NKE", nov, use_ai=False)  # net interest tag plus separately tagged interest income
    ok("NKE gross interest, income not counted twice", 150 < d.hist.IS["interest_exp"][2] < 600,
       (d.hist.IS["interest_exp"], d.hist.IS["interest_inc"]))

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
