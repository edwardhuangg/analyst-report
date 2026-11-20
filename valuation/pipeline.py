"""Fetch everything for a ticker as of a date, then build the spec and workbook.

    data = fetch(ticker, as_of, peers, use_ai)       # SEC filings + market data (+ AI reading)
    spec = make_spec(data, use_ai_thesis, analyst)   # defaults (+ AI analyst judgment)
    overrides.apply(spec, doc)                       # optional: an external agent's judgment
    xlsx = build(spec)
"""
from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

from . import ai, check, facts, financials as F, market, overrides, sec
from .spec import SPLIT_RATIOS, ModelSpec, _add_years, build_spec, floor_booked_charges
from .workbook import build_workbook
from .xbrl import Instance

# banks, credit institutions, broker-dealers, insurers: interest and float are operating items,
# so an industrial 3-statement / EV-based model gives meaningless answers
FINANCIAL_SIC = ("60", "61", "62", "63")


@dataclass
class CompanyData:
    company: sec.Company
    as_of: date
    filings: sec.FilingSet
    hist: F.History
    interim: F.Interim
    market: dict
    urls: dict
    statements: list
    release: dict = field(default_factory=dict)
    peers: list = field(default_factory=list)
    thesis: Optional[dict] = None
    sotp_multiples: dict = field(default_factory=dict)
    sotp_notes: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    ai_used: bool = False
    release_texts: dict = field(default_factory=dict)
    margin_hist: list = field(default_factory=list)   # [(fiscal year, operating margin)], up to 10 years
    hist10: dict = field(default_factory=dict)        # {'sales': [(year, revenue)], 'acq': [(year, paid)]}
    own_actions: list = field(default_factory=list)   # [(date, Yahoo factor)] splits / spin-offs after the 10-K
    pe_hist: list = field(default_factory=list)       # [(fiscal year, P/E at the fiscal year end)], up to 10 years
    prior_q_texts: list = field(default_factory=list)  # release texts of the year's earlier quarters (gate 47)


def _split_adjust(hist: F.History, interim: F.Interim, ticker: str, flags: list):
    """Restate 10-K per-share history when a split happened between the 10-K and the 10-Q.

    A big change in the cover share count is only treated as a split when the 10-Q tags a
    split ratio or Yahoo shows one in the window; otherwise (mergers, big buybacks) it is flagged.
    """
    if not (interim.available and interim.cover_shares and hist.cover_shares):
        return 1.0
    ratio = interim.cover_shares / hist.cover_shares
    if 0.7 < ratio < 1.4:
        return 1.0
    f = None
    if interim.split_ratio and abs(interim.split_ratio - ratio) / ratio < 0.15:
        f, how = interim.split_ratio, "split ratio tagged in the 10-Q"
    else:
        acts = [r for _, r in market.corporate_actions(ticker, hist.period_ends[-1], interim.period_end)
                if r >= 1.5 or r <= 0.67]
        if acts:
            f, how = min(SPLIT_RATIOS, key=lambda r: abs(r - ratio) / r), "split shown by Yahoo"
    if f is None:
        flags.append(f"Cover share count changed {ratio:.2f}x between the 10-K and the 10-Q with no split on record "
                     "(merger, spin-off or large issuance?): per-share history NOT restated — check shares and EPS")
        return 1.0
    for k in ("dil_shares", "basic_shares"):
        hist.IS[k] = [None if v is None else v * f for v in hist.IS[k]]
    hist.IS["eps_dil"] = [None if v is None else v / f for v in hist.IS["eps_dil"]]
    hist.dps = [None if v is None else v / f for v in hist.dps]
    hist.cover_shares *= f
    label = f"{f:g}-for-1 split" if f >= 1 else f"1-for-{1 / f:g} reverse split"
    flags.append(f"Share count changed {ratio:.2f}x between the 10-K and the 10-Q ({how}): per-share history "
                 f"restated for a {label}")
    return f


def _consensus(ticker: str, fy_end: date, sales_last: Optional[float], flags: list) -> dict:
    """FY-matched consensus revenue growth for FY1/FY2 (recent valuation dates only)."""
    est = market.estimates(ticker, fy_end)
    if not est:
        return {}
    out = {"period": est["period"], "eps": est.get("eps")}
    if est["period"] == "0y":
        base = est.get("rev_0y_base")
        g1, g2 = est.get("rev_0y_growth"), est.get("rev_+1y_growth")
        if base and sales_last and abs(base / 1e6 / sales_last - 1) > 0.05:
            flags.append(f"Consensus revenue base ({base / 1e6:,.0f}mm) differs from the 10-K ({sales_last:,.0f}mm): "
                         "consensus not used to anchor growth")
            return out
    else:
        g1, g2 = est.get("rev_+1y_growth"), None
    if g1 is not None and g1 < -0.25:
        flags.append(f"Consensus revenue growth of {g1:.0%} for FY1 implies a perimeter change (spin-off / "
                     "divestiture) not in the 10-K: growth not anchored to it — model the new perimeter")
        return out
    out["g_fy1"], out["g_fy2"] = g1, g2
    return out


def fetch(ticker: str, as_of: date, peers: Optional[list] = None, use_ai: bool = True,
          progress: Callable[[str], None] = lambda m: None, cik: Optional[str] = None,
          allow_financials: bool = False) -> CompanyData:
    progress("Looking up the company on SEC EDGAR…")
    ai.reset_usage()  # the cost meter covers one build: fetch + spec
    co = sec.load_company(ticker, cik)
    if co.sic[:2] in FINANCIAL_SIC and not allow_financials:
        raise ValueError(f"{co.name} is a financial company (SIC {co.sic}, {co.sic_description}). This template is "
                         "an industrial 3-statement / EV model; banks and insurers need a P/B-ROE or dividend "
                         "discount model. Tick 'Build anyway' under Advanced to override.")
    fs = sec.select_filings(co, as_of)
    flags = []
    if co.sic[:2] in FINANCIAL_SIC:
        flags.append(f"SIC {co.sic} ({co.sic_description}): an industrial-style model fits banks/insurers poorly")

    progress(f"Reading the FY 10-K XBRL ({fs.annual.filing_date:%b %d, %Y})…")
    isc = sec.income_statement_concepts(fs.annual)
    hist = F.parse_annual(Instance(sec.fetch_instance(fs.annual)), isc, sec.member_children(fs.annual),
                          sec.balance_sheet_concepts(fs.annual))
    try:
        capex_history(hist, co.cik, as_of)
    except Exception:
        pass
    label = sec.fiscal_year_label(fs.annual, hist.years[-1], hist.period_ends[-1])
    if label != hist.years[-1]:
        flags.append(f"Fiscal years labelled as the company does: the year ended {hist.period_ends[-1]:%b %-d, %Y} "
                     f"is FY{label} (the XBRL tag says {hist.years[-1]})")
        hist.years = [y + label - hist.years[-1] for y in hist.years]
    interim = F.empty_interim(hist)
    if fs.quarterly:
        progress(f"Reading the latest 10-Q ({fs.quarterly.report_date:%b %d, %Y})…")
        try:
            q_inst, q_isc = Instance(sec.fetch_instance(fs.quarterly)), sec.income_statement_concepts(fs.quarterly)
            q = F.parse_quarterly(q_inst, hist, q_isc)
            n_k = [s for s in hist.segments if s.member not in ("other", "total")]
            if q.available and len(n_k) > 1 and len(q.missing_segments) == len(n_k) and len(q.new_segments) == 1:
                F.to_one_segment(hist)
                flags.append(f"Resegmented to one reportable segment ({q.new_segments[0]}) in the latest 10-Q: "
                             f"modelled as one segment on operating income + amortization + stock comp + other "
                             f"operating expense (the basis the company guides); the 10-K's "
                             f"{', '.join(q.missing_segments)} are kept as revenue lines with their history")
                q = F.parse_quarterly(q_inst, hist, q_isc)
            if q.available:
                interim = q
            else:
                flags.append("Latest 10-Q could not be matched to year-to-date periods; Interim uses FY data")
        except Exception as e:  # keep going with FY data only
            flags.append(f"10-Q parsing failed ({e}); Interim uses FY data")
    _split_adjust(hist, interim, ticker, flags)
    if interim.available and interim.missing_segments and interim.new_segments:
        flags.append(f"Resegmented: the latest 10-Q reports {', '.join(interim.new_segments)} instead of the 10-K's "
                     f"{', '.join(interim.missing_segments)}. The model keeps the 10-K segments (3 years of history); "
                     "the Interim segment table is blank until the 10-K is recast")
    elif interim.available and interim.missing_segments:
        flags.append(f"The latest 10-Q no longer reports {', '.join(interim.missing_segments)}: the business was "
                     "probably separated or sold after the 10-K. The forecast still starts from the 10-K perimeter "
                     "— remove it (Assumptions growth / M&A) or wait for recast filings")
    if interim.available and interim.disc_ops_ytd and abs(interim.disc_ops_ytd) > 0.02 * _nz_sales(hist):
        flags.append(f"The latest 10-Q shows {interim.disc_ops_ytd:,.0f}mm year-to-date income from discontinued "
                     "operations: a business is being separated, so the 10-K perimeter may overstate the go-forward "
                     "company")

    progress("Pulling market data (Yahoo Finance)…")
    last_period = interim.period_end if interim.available else hist.period_ends[-1]
    for d, ratio in market.corporate_actions(ticker, last_period, as_of):
        kind = "split" if ratio >= 1.5 or ratio <= 0.67 else "spin-off / special distribution"
        flags.append(f"Corporate action on {d:%b %d, %Y} (Yahoo factor {ratio:g}, likely a {kind}) after the latest "
                     f"filing period ({last_period:%b %d, %Y}): the filings may describe a different perimeter or "
                     "share count than the price — review price, shares and segments")
    info = market.info(ticker)
    px, px_date = market.close_on(ticker, as_of)
    rf = market.risk_free_on(as_of)
    beta, beta_note = market.beta_on(ticker, as_of)
    if beta is None and market.is_recent(as_of) and info.get("beta"):
        beta, beta_note = info["beta"], "Yahoo Finance 5-yr monthly beta (as of today)"
    fy_end_cur = _add_years(hist.period_ends[-1], 1)
    consensus, eps_est = {}, (None, "No point-in-time consensus for a past valuation date")
    if market.is_recent(as_of):
        try:
            consensus = _consensus(ticker, fy_end_cur, hist.IS["sales"][-1], flags)
            eps_est = market.fy_eps_estimate(ticker, fy_end_cur, as_of)
        except Exception:
            pass
    mk = {"name": info.get("longName") or info.get("shortName"), "price": px,
          "price_note": f"{ticker.upper()} close {px_date:%b %d, %Y} (Yahoo; later splits / spin-off adjustments "
                        "undone so it matches the share count of the time)" if px_date else
                        "Price not found: enter manually",
          "rf": rf, "rf_note": f"10-yr U.S. Treasury yield on {as_of:%b %d, %Y} (Yahoo ^TNX; check FRED DGS10)",
          "beta": beta, "beta_note": beta_note, "industry": info.get("industry"), "sector": info.get("sector"),
          "forward_pe": info.get("forwardPE") if market.is_recent(as_of) else None,
          "eps_est": eps_est, "consensus": consensus}
    if market.is_recent(as_of):  # crowding for a short (a current snapshot: no point-in-time history)
        mk["short"] = {k: info.get(k) for k in ("shortPercentOfFloat", "shortRatio", "dateShortInterest")
                       if info.get(k) is not None}
    for f in (fs.quarterly if interim.available else None, fs.annual):
        if f and not mk.get("ratings"):
            try:
                mk["ratings"] = sec.credit_ratings(f)
                mk["ratings_src"] = f"{f.form} filed {f.filing_date}"
            except Exception:
                pass

    urls = {"annual": fs.annual.url, "quarterly": fs.quarterly.url if fs.quarterly and interim.available else None}
    texts = {}
    for key, f in (("fy_release", fs.fy_release), ("q_release", fs.q_release if interim.available else None)):
        if f:
            try:
                u, t = sec.release_text(f)
                urls[key] = u
                if t:
                    texts[key] = (f, t)
            except Exception:
                pass

    prior_texts = []
    for f in (fs.prior_q_releases if interim.available else []):
        try:
            prior_texts.append(sec.release_text(f)[1] or "")
        except Exception:
            prior_texts.append("")
    data = CompanyData(company=co, as_of=as_of, filings=fs, hist=hist, interim=interim, market=mk, urls=urls,
                       statements=sec.statement_reports(fs.annual), flags=flags, release_texts=texts,
                       prior_q_texts=prior_texts)
    try:  # the 10-year ranges the terminal margin and the multiple are tested against (point in time)
        data.margin_hist = facts.margin_history(co.cik, as_of,
                                                interest_share=1 - (hist.captive.get("share") or 0.0))
        data.pe_hist = own_pe_history(ticker, co.cik, as_of)
        data.hist10 = facts.ten_year_history(co.cik, as_of)
    except Exception:
        pass
    try:  # a spin-off after the 10-K changes the company the price is for (gate 40)
        data.own_actions = market.corporate_actions(ticker, fs.annual.report_date or as_of, as_of)
    except Exception:
        pass

    client = ai.client() if use_ai else None
    overlap = {}
    seg_names = [s.name for s in hist.segments]
    if client and texts:
        progress("Reading the earnings releases (AI)…")
        try:
            data.release = ai.extract_releases(client, co.name, hist.years[-1], interim, seg_names, texts)
            data.ai_used = True
        except Exception as e:
            client = _ai_failed(e, "Earnings-release extraction", flags, client)
    if not peers and client:
        progress("Choosing peers (AI)…")
        try:
            sug = ai.suggest_peers(client, co.name, ticker, seg_names, mk.get("industry"), as_of)
            peers = [p["ticker"] for p in sug.get("peers", [])][:8]
            data.sotp_multiples = sug.get("segment_multiples") or {}
            data.sotp_notes = sug.get("segment_notes") or {}
            overlap = {p["ticker"].upper(): p.get("overlap", "") for p in sug.get("peers", [])}
            data.ai_used = True
        except Exception as e:
            client = _ai_failed(e, "Peer suggestion", flags, client)
    for t in (peers or [])[:8]:
        progress(f"Peer {t}…")
        try:
            snap = market.peer_snapshot(t, as_of)
            if overlap.get(t.upper()):
                snap["overlap"] = overlap[t.upper()]
            if snap["price"] and snap["eps"] is None:
                point_in_time_peer_eps(snap, as_of, client)
            if snap["price"]:
                peer_ev(snap, as_of)
                peer_beta(snap, as_of)
                data.peers.append(snap)
        except Exception:
            flags.append(f"Peer {t}: no market data")
    return data


def capex_history(hist: F.History, cik: str, as_of: date) -> None:
    """Capex on no standard tag (VZ): the 10-K's two balance sheets give the PP&E roll-forward for FY0 only, so the
    earlier years take year-end PP&E from earlier filings. History must not show zero capex beside positive D&A."""
    cx, cfo = hist.CF["capex"], hist.CF["cfo"]
    miss = [i for i in (0, 1) if not cx[i] and cfo[i] is not None]
    if not miss or not cx[2]:
        return
    cf = facts.company_facts(cik)

    def ppe(end):
        for c in F.C["ppe"]:
            rows = [r for r in facts._rows(cf, c, as_of) if not r.get("start_d") and abs((r["end_d"] - end).days) <= 8]
            if rows:
                return max(rows, key=lambda r: r["filed"])["val"] / F.MM
        return None
    done = []
    for i in miss:
        p1 = hist.BS["ppe"][i] if i == 1 and hist.BS["ppe"][1] else ppe(hist.period_ends[i])
        p0 = ppe(hist.period_starts[i] - timedelta(days=1))
        dep = hist.IS["dep"][i] or ((hist.IS["da"][i] or 0) - (hist.IS["amort"][i] or 0))
        if p1 and p0 and dep and p1 - p0 + dep > 0:
            cx[i] = -(p1 - p0 + dep)
            done.append(f"FY{hist.years[i]} {-cx[i]:,.0f}mm")
    if done:
        txt = "earlier years from the same roll-forward on year-end PP&E in earlier filings (" + ", ".join(done) + ")"
        hit = [k for k, f in enumerate(hist.flags) if "earlier years not tagged" in f]
        for k in hit:
            hist.flags[k] = hist.flags[k].replace("earlier years not tagged", txt)
        if not hit:
            hist.flags.append("Capital expenditures not on a standard tag: " + txt)


def _ai_failed(e: Exception, what: str, flags: list, client):
    """Record an AI failure; on quota / auth errors return None so later steps skip AI."""
    if ai.fatal(e):
        flags.append(f"AI switched off for this build: {what} failed with {str(e)[:160]} — check the OpenAI key / "
                     "credits; the model uses the filing-based defaults")
        return None
    flags.append(f"{what} failed: {str(e)[:200]}")
    return client


def _nz_sales(hist: F.History) -> float:
    return abs(hist.IS["sales"][-1] or 0.0)


def point_in_time_peer_eps(snap: dict, as_of: date, client) -> None:
    """Past valuation date: the peer's FY guidance from its last release before the date (AI), or
    its LTM GAAP EPS from SEC filings made by then."""
    t = snap["ticker"]
    if client:
        try:
            pco = sec.load_company(t)
            rel = next((f for f in pco.filings if f.form == "8-K" and "2.02" in f.items and f.filing_date <= as_of),
                       None)
            if rel:
                url, text = sec.release_text(rel)
                if text:
                    g = ai.peer_guidance(client, pco.name, text, rel.filing_date)
                    lo_hi = g.get("adj_eps_guidance")
                    if isinstance(lo_hi, list) and len(lo_hi) == 2 and all(isinstance(x, (int, float)) for x in lo_hi):
                        snap["eps"] = round(sum(lo_hi) / 2, 3)
                        snap["source"] = (f"{rel.filing_date:%b %-d, %Y} release: FY{g.get('fiscal_year') or ''} "
                                          f"adj. EPS guide ${lo_hi[0]:.2f}–{lo_hi[1]:.2f} (AI-read, verify); "
                                          + snap["source"].split("; ", 1)[-1])
                        return
        except Exception:
            pass
    try:
        pf = facts.peer_fundamentals(t, as_of)
        if pf.get("eps_ltm"):
            snap["eps"] = round(pf["eps_ltm"], 2)
            snap["eps_basis"] = "trailing"
            snap["source"] = (f"{pf['eps_basis']} GAAP diluted EPS from SEC filings made by {as_of:%b %-d, %Y} "
                              "(no point-in-time consensus; replace with FY guidance); "
                              + snap["source"].split("; ", 1)[-1])
    except Exception:
        pass


def peer_ev(snap: dict, as_of: date) -> None:
    """EV and LTM EBITDA from the peer's SEC filings made by the valuation date (point in time).
    Shares outstanding on the latest 10-K/10-Q cover x the price on the date, plus net debt."""
    pf = None
    for attempt in range(2):  # one retry: a transient SEC failure left every ADBE peer without EV / EBITDA
        try:
            pf = facts.peer_fundamentals(snap["ticker"], as_of)
            break
        except Exception:
            time.sleep(2)
    if pf is None:
        return
    if (pf.get("shares") or 0) > 1 and (pf.get("ebitda_ltm") or 0) > 0 and snap.get("price"):
        ev = snap["price"] * pf["shares"] + pf["net_debt"]
        if not 2 <= ev / pf["ebitda_ltm"] <= 80:  # a share count or EBITDA tag gone wrong, not a real multiple
            return
        snap["ev"] = round(ev, 0)
        snap["ebitda"] = round(pf["ebitda_ltm"], 0)
        snap["shares_mm"] = round(pf["shares"], 1)
        if pf.get("revenue_ltm"):
            snap["ebitda_margin"] = round(pf["ebitda_ltm"] / pf["revenue_ltm"], 4)
        if pf.get("share_chg") is not None:
            snap["share_chg"] = round(pf["share_chg"], 4)
        snap["shares_date"] = str(pf.get("shares_date") or "")
        snap["de"] = round((pf.get("debt") or 0.0) / (snap["price"] * pf["shares"]), 4)
        snap["ev_source"] = (f"SEC filings by {as_of:%b %-d, %Y}: {pf['shares']:,.0f}mm shares x price + net debt "
                             f"{pf['net_debt']:,.0f}mm; EBITDA {pf.get('ebitda_basis') or 'LTM'} (operating income + D&A)")


def calendarize_peer(snap: dict, fy1_end: date) -> None:
    """Peer consensus EPS for the twelve months ending at the company's FY1 year-end, when the peer's fiscal year
    ends three or more months away (PM, ADBE: MSFT's June-2027 year was set against Adobe's November-2026 year). A
    year ending later blends in the year before, estimated at next year's growth backwards; one ending earlier
    blends in the next year."""
    try:
        e0 = date.fromisoformat(str(snap.get("fy_end") or "")[:10])
    except ValueError:
        return
    eps0, eps1 = snap.get("eps"), snap.get("eps_next")
    m = round((e0 - fy1_end).days / 30.44)
    if snap.get("eps_fy") is not None or not isinstance(eps0, (int, float)) or eps0 <= 0 or not 3 <= abs(m) <= 11:
        return
    if not isinstance(eps1, (int, float)) or eps1 <= 0 or not -0.2 <= eps1 / eps0 - 1 <= 0.5:
        return  # no usable next year (or a one-off year): left on the peer's own year
    if m > 0:
        prev = eps0 / (eps1 / eps0)
        cal = (m * prev + (12 - m) * eps0) / 12
        how = (f"{m}/12 of the year to {e0.year - 1} (≈ {prev:.2f}: the {eps0:.2f} consensus ÷ next year's "
               f"{eps1 / eps0 - 1:+.1%} growth) + {12 - m}/12 of the year to {e0:%b %Y}")
    else:
        cal = ((12 + m) * eps0 + (-m) * eps1) / 12
        how = f"{12 + m}/12 of the year to {e0:%b %Y} ({eps0:.2f}) + {-m}/12 of the next ({eps1:.2f})"
    snap["eps_fy"], snap["eps"] = eps0, round(cal, 4)
    snap["source"] = f"{snap.get('source') or ''}; calendarized to the company's {fy1_end:%b %Y} year: {how}"


def peer_beta(snap: dict, as_of: date) -> None:
    """Raw 5-yr monthly regression beta for the bottom-up beta (valuation/capital.py)."""
    if snap.get("beta_raw") is None:
        try:
            snap["beta_raw"] = market.beta_on(snap["ticker"], as_of)[0]
        except Exception:
            snap["beta_raw"] = None


def own_pe_history(ticker: str, cik: str, as_of: date) -> list:
    """[(fiscal year, trailing P/E at the fiscal year end)] for up to 10 years: the company's own multiple band."""
    out = []
    for end, eps in facts.eps_history(cik, as_of):
        if not eps or eps <= 0:
            continue
        try:
            px, _ = market.close_on(ticker, end)
        except Exception:
            px = None
        if px:
            out.append((end.year, round(px / eps, 1)))
    return out


def apply_capital(spec: ModelSpec) -> None:
    """Re-run the beta method on the final peer set (an analyst may have changed the peers); keep an analyst's
    beta only inside the method band; refresh the rating-based cost of debt. Human files are left alone."""
    from . import capital
    if not spec.capital or spec.framework == "reference":
        return
    it = spec.interim
    eq = spec.price * (spec.basic_shares + spec.dilutive)
    debt = (_nz(it.bs_cur.get("st_debt")) + _nz(it.bs_cur.get("cur_ltd")) + _nz(it.bs_cur.get("ltd"))) \
        * (1 - spec.valuation.get("captive_share", (0.0,))[0])
    de = debt / eq if eq else 0.0
    peers = [p for p in spec.peers if p.get("ticker") and p["ticker"] != spec.ticker]
    b, how = capital.method_beta(spec.capital.get("own_raw"), peers, de, spec.valuation["tax_wacc"][0])
    old = spec.capital.get("beta")
    spec.capital.update({"beta": b, "beta_how": how, "de": de})
    src = spec.val_source.get("beta", "defaults")
    cur = spec.valuation["beta"][0]
    ai_b = spec.capital.get("beta_ai")
    if ai_b:  # an AI analyst's beta: kept inside the band around the method value for the final peers
        v, note = ai_b
        if str(note or "").startswith("Method beta") or not note:  # the brief's method text echoed: no judgment
            spec.valuation["beta"] = (b, how)
        else:
            nv = min(max(v, b - capital.BETA_BAND), b + capital.BETA_BAND)
            spec.valuation["beta"] = (round(nv, 4), note if nv == v else
                                      f"{note}; held within the method's {b:.2f} ± {capital.BETA_BAND:.2f} for the "
                                      f"final peer set (analyst {v:.2f})")
        return
    if "AI analyst" in src and abs(cur - b) <= capital.BETA_BAND + 1e-9:
        return  # an analyst's override inside the band, with its reason
    if src in ("defaults",) or "AI analyst" in src:
        if old is None or abs(cur - b) > 1e-9:
            spec.valuation["beta"] = (b, how)


def _with_price_date(source: str, snap: dict) -> str:
    """Drop any price wording the analyst wrote and state the market close actually used."""
    src = re.sub(r"(?i)[;,.]?\s*(price|close)[^;]*?(\d{4}|close)\.?(?=;|$)", "", source).strip(" ;,")
    return f"{src}; price = close {snap['price_date']:%b %-d, %Y} (Yahoo)" if snap.get("price_date") else src


def release_between(ticker: str, as_of: date, snap: str) -> Optional[str]:
    """Whether a consensus snapshot taken after the valuation date can stand for the consensus on it: '' when the
    peer filed no earnings release (8-K item 2.02) in between, else the release date; None when unknown."""
    try:
        end = date.fromisoformat(snap)
        co = sec.load_company(ticker)
        rel = [f.filing_date for f in co.filings if f.form == "8-K" and "2.02" in f.items and
               as_of < f.filing_date <= end]
    except Exception:
        return None
    return f"{min(rel):%b %-d, %Y}" if rel else ""


def fill_peer_ev(spec: ModelSpec) -> None:
    """After overrides: peers an AI named are re-priced at the market close on the valuation date (models have
    supplied invented prices), and every peer gets EV / EBITDA from its filings (point in time)."""
    moved = []
    for p in spec.peers:
        t = p.get("ticker")
        if not t or t == spec.ticker:
            continue
        if p.pop("_reprice", False) or not p.get("price"):
            try:
                px, d = market.close_on(t, spec.as_of)
            except Exception:
                px, d = None, None
            if px and (not p.get("price") or abs(p["price"] / px - 1) > 0.02):
                moved.append(f"{t} {p.get('price') or 'n/a'} -> {px:,.2f}")
                p["price"], p["ev"] = round(px, 2), None
                p["source"] = _with_price_date(str(p.get("source") or ""), {"price_date": d})
        if p.get("ev") is None and p.get("price"):
            peer_ev(p, spec.as_of)
        calendarize_peer(p, spec.fy_end_cur)
        if spec.framework != "reference":
            peer_beta(p, spec.as_of)
            # a spin-off, split or merger in the last 18 months: LTM EBITDA and the share count straddle the event
            # (HON as a peer showed 8.2x EV/EBITDA on half its share count) -> no EV/EBITDA unless pro forma
            try:
                acts = market.corporate_actions(t, spec.as_of - timedelta(days=548), spec.as_of)
            except Exception:
                acts = []
            if acts and p.get("ev") is not None and not p.get("pro_forma"):
                d, ratio = acts[-1]
                # a stock split leaves the perimeter alone (NOW's 5-for-1, Dec 2025): keep EV / EBITDA, with the share
                # count restated if it predates the split; a spin-off (a fractional Yahoo factor) is excluded
                split = (ratio >= 1.5 and abs(2 * ratio - round(2 * ratio)) < 0.02) or \
                        (0 < ratio <= 0.67 and abs(1 / ratio - round(1 / ratio)) < 0.02)
                if split and not p.get("shares_date"):  # a snapshot from saved overrides: re-read the filings
                    peer_ev(p, spec.as_of)
                sd, sh = str(p.get("shares_date") or ""), p.get("shares_mm")
                if split and sd and sd >= str(d):
                    p["ev_source"] = (str(p.get("ev_source") or "") + f"; {ratio:g}:1 split on {d:%b %-d, %Y}: "
                                      "share count dated after it")
                elif split and sd and sh:
                    p["ev"] = round(p["ev"] + p["price"] * sh * (ratio - 1), 0)
                    p["ev_source"] = (str(p.get("ev_source") or "") + f"; {ratio:g}:1 split on {d:%b %-d, %Y}: share "
                                      f"count (dated {sd}) restated x{ratio:g}")
                    if not 2 <= p["ev"] / max(p.get("ebitda") or 1.0, 1.0) <= 80:
                        p["ev"] = None
                else:
                    p["ev"] = None
                    p["ev_source"] = (f"Corporate action on {d:%b %-d, %Y} (Yahoo factor {ratio:g}): LTM EBITDA and "
                                      "shares are not pro forma, so EV / EBITDA is excluded")
            if p.get("eps_next") is None and p.get("eps") and market.is_recent(spec.as_of):
                p["eps_next"] = market.eps_next(t)
            if p.get("g_next") is None and p.get("eps") and market.is_recent(spec.as_of):
                p["g_next"] = market.eps_growth_next(t)
            if p.get("rev_g_next") is None and market.is_recent(spec.as_of):
                p["rev_g_next"] = market.revenue_growth_next(t)
            snap_d = str(p.get("snapshot") or "")[:10]
            if snap_d > str(spec.as_of) and p.get("pit") is None:
                p["pit"] = release_between(t, spec.as_of, snap_d)
    if moved:
        spec.flags.append("Peer prices replaced with the market close on the valuation date: " + "; ".join(moved))
    apply_capital(spec)


def make_spec(data: CompanyData, use_ai_thesis: bool = True, analyst: bool = False,
              progress: Callable[[str], None] = lambda m: None, effort: Optional[str] = None,
              guards: bool = True) -> ModelSpec:
    """Defaults from the filings; optionally the AI analyst loop, then the AI thesis."""
    spec = build_spec(data)
    ai_off = any(f.startswith("AI switched off") for f in data.flags)
    client = ai.client() if (use_ai_thesis or analyst) and not ai_off else None
    thesis_done = False
    if client and analyst:
        thesis_done, client = analyst_loop(spec, data, client, progress, effort)
    if guards and not spec.last_review and check.available():  # no analyst pass ran: the engine's own guards
        try:
            fill_peer_ev(spec)
            spec.last_review = mechanical_guards(spec, check.review(spec, build(spec, values=False)))
        except Exception as e:
            spec.flags.append(f"Engine guards skipped: {type(e).__name__}: {str(e)[:120]}")
    if client and use_ai_thesis and not thesis_done and ai.spent() < ai.budget():
        progress("Drafting the thesis (AI)…")
        try:
            spec.thesis = ai.write_thesis(client, spec)
            data.ai_used = True
        except Exception as e:
            _ai_failed(e, "Thesis drafting", spec.flags, client)
    if data.ai_used and not any(x.startswith("AI-assisted") for x in spec.sources):
        spec.sources.append("AI-assisted (verify): items marked AI in the notes and on the Cover")
    spec.ai_cost = ai.usage_summary() if ai.USAGE else ""
    spec.ai_usage = list(ai.USAGE)
    return spec


def analyst_loop(spec: ModelSpec, data: CompanyData, client, progress=lambda m: None,
                 effort: Optional[str] = None) -> tuple[bool, object]:
    """The in-app analyst agent: read -> set assumptions -> build -> check -> fix -> calibrate.

    1. the default model is built and recalculated so the analyst sees what the defaults imply;
    2. the analyst reads the brief + the reading pack (releases, MD&A, deal 8-Ks) and returns overrides;
    3. the model is rebuilt and checked; a review pass fixes guidance misses and inconsistencies;
    4. free cash flow is calibrated to the guidance midpoint exactly (other operating items).
    Returns (thesis written, client or None if AI had to be switched off).
    """
    from . import brief
    can_check = check.available()
    review0 = None
    if can_check:
        progress("Computing the default model for the analyst brief…")
        try:
            review0 = check.review(spec, build(spec, values=False))
            _audit_fv(spec, "defaults (filings and market data)", review0)
        except Exception:
            review0 = None
    progress("Collecting the analyst's reading: earnings releases, 10-K / 10-Q MD&A, 8-Ks since the 10-K…")
    try:
        docs = sec.analyst_documents(data.company, data.filings, data.as_of, data.release_texts)
    except Exception:
        docs = {k: t for k, (_, t) in (data.release_texts or {}).items()}
    progress(f"AI analyst ({ai.analyst_model()}, {effort or ai._effort()} effort) is setting scenarios, M&A, "
             "guidance, SOTP and the thesis…")
    try:
        doc = ai.analyst(client, brief.build(spec, data, review=review0), docs, progress=progress, effort=effort)
    except Exception as e:
        return False, _ai_failed(e, "AI analyst pass", spec.flags, client)
    data.ai_used = True
    _complete_peers(doc, data.as_of, client)
    warn = overrides.apply(spec, doc, source="AI analyst")
    fill_peer_ev(spec)  # the analyst's peers get betas, EV / EBITDA, splits and the point-in-time check (as the app)
    thesis_done = bool((doc.get("thesis") or {}).get("points"))
    if thesis_done and not (doc.get("thesis") or {}).get("about"):
        spec.thesis["about"] = (f"Drafted by the AI analyst ({ai.analyst_model()}) from the filings and releases "
                                f"available on {spec.as_of:%b %-d, %Y}; numbers above are live links. Verify.")
    if (doc.get("meta") or {}).get("summary"):
        spec.notes["analyst_summary"] = str(doc["meta"]["summary"])
    if can_check:
        w2, client = review_passes(spec, data, client, progress, effort, "AI analyst pass")
        warn += w2
    spec.flags.extend(f"AI analyst output: {w}" for w in warn)
    return thesis_done, client


def review_passes(spec: ModelSpec, data: CompanyData, client, progress=lambda m: None,
                  effort: Optional[str] = None, stage: str = "overrides", notes: str = "") -> tuple[list, object]:
    """Build -> check -> engine guards -> AI review of the issues -> one more pass on what a PM rejects outright or
    on open items that hold the rating -> engine guards. Returns (warnings, client or None if AI was switched off)."""
    warn = []
    try:
        rev = check.review(spec, build(spec, values=False))
        _audit_fv(spec, stage, rev)
        # mechanical guards first, so the review pass sees (and writes about) the numbers that ship
        rev = with_scenarios(spec, mechanical_guards(spec, rev))
        _audit_fv(spec, "engine guards", rev)
        if rev["problems"] and ai.spent() >= ai.budget():
            spec.flags.append(f"AI review pass skipped: this build already used ${ai.spent():.2f} of the "
                              f"${ai.budget():.2f} budget (OPENAI_MAX_COST_PER_RUN)")
        elif rev["problems"] or notes:
            progress(f"AI analyst is reviewing the built model ({len(rev['problems'])} issues from the "
                     "checker" + (", and the PM's critique" if notes else "") + ")…")
            current = overrides.export(spec)
            for sg in current["segments"].values():
                sg.pop("history", None)
                for ln in (sg.get("lines") or {}).values():
                    ln.pop("history", None)
            docs = None
            if notes:  # a critique names facts to check: the reading pack goes with it
                try:
                    docs = sec.analyst_documents(data.company, data.filings, data.as_of, data.release_texts)
                except Exception:
                    docs = None
            doc2 = ai.revise(client, overrides.SCHEMA_DOC, current, check.review_text(rev), progress=progress,
                             effort=effort, notes=notes, documents=docs)
            _complete_peers(doc2, data.as_of, client)
            if doc2.get("flags"):  # the review's flags are the final list: drop the first pass's
                spec.flags = [f for f in spec.flags if not f.startswith("Analyst (")]
            warn += overrides.apply(spec, doc2, source="AI analyst review")
            if doc2.get("peers"):
                fill_peer_ev(spec)
            rev = check.review(spec, build(spec, values=False), scenarios=True)
            _audit_fv(spec, "AI review pass", rev)
            hard = [p for p in rev["problems"] if HARD_RX.search(p)] + [  # and what holds the rating
                t for _, t, w in rev.get("open_items") or [] if not w]
            if hard and ai.spent() < ai.budget():  # one more pass on what a PM would reject outright
                progress(f"AI analyst is fixing {len(hard)} remaining issue(s)…")
                current = overrides.export(spec)
                for sg in current["segments"].values():
                    sg.pop("history", None)
                    for ln in (sg.get("lines") or {}).values():
                        ln.pop("history", None)
                doc3 = ai.revise(client, overrides.SCHEMA_DOC, current, check.review_text(rev), progress=progress,
                                 effort=effort)
                _complete_peers(doc3, data.as_of, client)
                if doc3.get("flags"):
                    spec.flags = [f for f in spec.flags if not f.startswith("Analyst (")]
                warn += overrides.apply(spec, doc3, source="AI analyst review")
                if doc3.get("peers"):
                    fill_peer_ev(spec)
                rev = check.review(spec, build(spec, values=False))
                _audit_fv(spec, "AI review (failed gates)", rev)
        # mechanical guards after the analyst: booked charges, guided growth / sales, guided FCF
        rev = mechanical_guards(spec, rev)
        _audit_fv(spec, "engine guards (final)", rev)
        n_audit = len(spec.audit)
        rev = revert_harmful(spec, rev)
        if len(spec.audit) > n_audit:
            rev = mechanical_guards(spec, rev)
            _audit_fv(spec, "engine revert of override groups that broke gates", rev)
        spec.last_review = rev
    except Exception as e:
        if ai.fatal(e):
            client = _ai_failed(e, "AI analyst review", spec.flags, client)
        else:
            spec.flags.append(f"AI analyst review step skipped: {str(e)[:160]}")
    return warn, client


def _audit_fv(spec: ModelSpec, stage: str, rev: Optional[dict]) -> None:
    fv = ((rev or {}).get("outputs") or {}).get("fair_value")
    if isinstance(fv, (int, float)):
        spec.audit_fv.append((stage, round(fv, 2)))


def _complete_peers(doc: dict, as_of: date, client) -> None:
    """The analyst names peers; prices and EPS come from market data / filings, not from the model."""
    peers = doc.get("peers")
    if not isinstance(peers, list):
        return
    out, conflicts = [], []
    for p in peers[:8]:
        if not isinstance(p, dict) or not p.get("ticker"):
            continue
        try:  # the price is always the market close on the date: a model must not supply it
            snap = market.peer_snapshot(str(p["ticker"]), as_of)
            if snap["price"] and snap["eps"] is None:
                point_in_time_peer_eps(snap, as_of, client)
        except Exception:
            continue
        if not snap.get("price"):
            continue
        peer_ev(snap, as_of)
        snap["overlap"] = p.get("overlap") or snap["overlap"]
        snap["name"] = p.get("name") or snap["name"]
        if isinstance(p.get("eps"), (int, float)) and p.get("source"):  # analyst read the guidance itself
            e, mkt = float(p["eps"]), snap.get("eps")
            if mkt and not 0.5 <= e / mkt <= 2:
                # far from consensus: one of the two is on another basis. The one whose P/E sits nearer the other
                # peers' is kept (NFLX after its 10:1 split: the analyst's pre-split EPS gave 1.8x; LLY round 5: NVO's
                # consensus is DKK per share against a USD ADR price, 1.8x, and the analyst's $3.55 per ADR was
                # discarded — the thesis then claimed a fix the data did not show)
                conflicts.append((snap, e, mkt, p))
            else:
                for _, r in market.corporate_actions(snap["ticker"], as_of - timedelta(days=550), as_of):
                    if r > 1.5 and e > 0 and snap["price"] / e < 5:  # P/E under 5x: an EPS from before the split
                        e /= r
                # a 'guidance' or 'framework' figure that is today's consensus to the cent is consensus (PM round 3:
                # MSFT gives no EPS guide; its "management framework" $24.20 was the +1y consensus, relabeled)
                cons = [x for x in (mkt, snap.get("eps_next")) if isinstance(x, (int, float)) and x > 0]
                relabeled = any(abs(e / x - 1) <= 0.005 for x in cons) and \
                    not re.search(r"(?i)consensus", str(p["source"]))
                snap["eps"] = e
                snap["source"] = _with_price_date(str(p["source"]), snap)
                if relabeled:
                    snap["source"] += (f"; equals the Yahoo consensus snapshot ({snap.get('snapshot')}) to within "
                                       "0.5%: consensus, not company guidance")
                else:  # the analyst's own reading replaces today's consensus: no look-ahead, its basis its own
                    snap["snapshot"] = None
                src = str(p["source"])
                snap["eps_basis"] = ("trailing" if re.search(r"(?i)\b(ltm|trailing|last twelve)\b", src) and
                                     not re.search(r"(?i)FY ?20\d\d guid|guidance midpoint|guide midpoint|midpoint|"
                                                   r"forward|consensus|estimate", src) else "forward")
        if snap.get("eps"):
            out.append(snap)
    pes = sorted(s_["price"] / s_["eps"] for s_ in out if s_.get("eps") and s_["eps"] > 0 and s_.get("price")
                 and all(s_ is not c[0] for c in conflicts))
    med = pes[len(pes) // 2] if pes else None
    for snap, e, mkt, p in conflicts:
        px_ = snap["price"]
        far = lambda x: abs(math.log((px_ / x) / med)) if med and x and x > 0 else 9.0  # noqa: E731
        if med and e > 0 and far(e) < far(mkt):
            snap["eps"], snap["snapshot"] = e, None
            snap["source"] = _with_price_date(str(p["source"]), snap) + (
                f"; Yahoo consensus {mkt:g} is {e / mkt:.1f}x off it (another basis: P/E {px_ / mkt:.1f}x vs the "
                f"peers' {med:.1f}x median), so the analyst's figure is used")
        else:
            snap["source"] += (f"; the analyst's EPS {e:g} ({p['source']}) is {e / mkt:.1f}x consensus "
                               "(pre-split or another basis): consensus used")
    if out:
        doc["peers"] = out
    else:
        doc.pop("peers", None)


HARD_RX = re.compile(r"Not credible|outside company guidance|Thesis quotes|already booked|Thesis says|"
                     r"Balance sheet:|Double count|Gate failed")


def calibrate_growth(spec: ModelSpec, rev: dict) -> bool:
    """FY1 organic growth and sales onto the guidance midpoint when the analyst left them outside the range.
    Both are linear in the FY1 inputs: organic shifts every growth unit's FY1 rate by the same amount; sales
    then moves the FY1 M&A / FX contribution (the usual gap between organic and reported guidance)."""
    f = rev.get("forecast") or {}
    org, sales = (f.get("Organic growth") or [None])[0], (f.get("Net sales") or [None])[0]
    g_org, g_sales = spec.guidance.get("organic"), spec.guidance.get("sales")
    units = [(sg, u) for sg in spec.segments for u in sg.units]
    base = sum(_nz(u.sales[2]) for _, u in units)
    free = [(sg, u) for sg, u in units if not u.kpi]
    free_base = sum(_nz(u.sales[2]) for _, u in free)
    moved = False
    if g_org and isinstance(org, (int, float)) and free_base and not g_org[0] - 0.0005 <= org <= g_org[1] + 0.0005:
        d = (sum(g_org) / 2 - org) * base / free_base
        for sg, u in free:
            for case in ("bear", "base", "bull"):
                u.growth[case][0] = round(u.growth[case][0] + d, 6)
            if u is not sg:
                sg.lines_explicit.update(("bear", "base", "bull"))
        spec.notes["growth"] = (spec.notes.get("growth") or spec.growth_note) + \
            f" {spec.fc_labels[0]} organic growth shifted {d:+.1%} onto the guidance midpoint {sum(g_org) / 2:.1%}."
        sales = sales + d * free_base if isinstance(sales, (int, float)) else sales
        spec.calibrated["organic"] = f"organic growth {d:+.1%}"
        moved = True
    if g_sales and isinstance(sales, (int, float)) and base and not g_sales[0] * 0.995 <= sales <= g_sales[1] * 1.005:
        d = (sum(g_sales) / 2 - sales) / base
        if g_org:  # organic is guided too: the residual is FX / M&A (bounded: a bigger gap is a perimeter question)
            d = max(-0.03, min(0.03, d))
            for sg in spec.segments:
                sg.mna[0] = round(sg.mna[0] + d, 6)
                sg.mna_note = f"{sg.mna_note}; {spec.fc_labels[0]} {d:+.1%} FX / M&A residual between the organic " \
                              f"and reported guidance (sales midpoint {sum(g_sales) / 2:,.0f}mm)"
            spec.calibrated["sales"] = f"FX / M&A residual {d:+.1%} (capped ±3pt)"
        else:  # calibrate through the operating driver, never a fake M&A line
            dd = d * base / free_base if free_base else d
            for sg, u in free:
                for case in ("bear", "base", "bull"):
                    u.growth[case][0] = round(u.growth[case][0] + dd, 6)
                if u is not sg:
                    sg.lines_explicit.update(("bear", "base", "bull"))
            spec.notes["growth"] = (spec.notes.get("growth") or spec.growth_note) + \
                f" {spec.fc_labels[0]} organic growth shifted {dd:+.1%} so sales = the guidance midpoint."
            spec.calibrated["sales"] = f"organic growth {dd:+.1%}"
        moved = True
    return moved


def _nz(v):
    return 0.0 if v is None else v


def mechanical_guards(spec: ModelSpec, rev: dict) -> dict:
    """Booked charges, guided growth / sales, guided FCF, then the cash floor; returns the review of the result."""
    if floor_booked_charges(spec) | calibrate_growth(spec, rev):
        rev = check.review(spec, build(spec, values=False))
    if calibrate_fcf(spec, rev):
        rev = check.review(spec, build(spec, values=False))
    if return_excess_cash(spec, rev):
        rev = check.review(spec, build(spec, values=False))
    if default_ronic(spec, rev) | cap_ronic(spec, rev):
        rev = check.review(spec, build(spec, values=False))
    for _ in range(3):  # new debt adds interest, which can push later years back below the floor (utilities)
        if not floor_cash(spec, rev):
            break
        rev = check.review(spec, build(spec, values=False))
    return rev


RONIC_DEFAULT = "Return on new invested capital"
RONIC_CAP = 0.60  # gates.RONIC_CAP and the overrides bound


def default_ronic(spec: ModelSpec, rev: dict) -> bool:
    """Terminal RONIC by default = the last forecast year's return on invested capital including goodwill (DCF C93),
    floor WACC, cap 60% (PM round 4: halfway between WACC and operating ROIC sat below the return including goodwill
    in all six models — COST 25% vs 32% with almost no goodwill, LLY 22% vs 46% — a framework-wide downward bias).
    The fade steps from the forecast's own incremental return to it. Only the engine's default is reset: an analyst
    who sets RONIC lower states a dated sector reason (gate 34)."""
    o = rev.get("outputs") or {}
    roic_t, wacc = o.get("roic_gw_t"), o.get("wacc")
    old, note = spec.valuation["ronic"]
    if not str(note or "").startswith(RONIC_DEFAULT) or spec.profile.get("ronic_rule") == "wacc" or \
            not isinstance(roic_t, (int, float)) or not isinstance(wacc, (int, float)) or roic_t <= 0:
        return False
    new = round(min(max(roic_t, wacc), RONIC_CAP) // 0.0025 * 0.0025, 4)
    if abs(new - old) < 0.0025:
        return False
    spec.valuation["ronic"] = (new, f"{RONIC_DEFAULT} = the {spec.fc_labels[-1]} return on invested capital "
                                    f"including goodwill ({roic_t:.1%}, DCF C93" +
                                    (f", capped at {RONIC_CAP:.0%}" if roic_t > RONIC_CAP else
                                     f", floored at WACC {wacc:.1%}" if roic_t < wacc else "") +
                                    "): new capital earns what the whole business earns on everything paid for it; "
                                    "lower it only for a dated sector reason (loss of exclusivity, competitive "
                                    "entry, recurring acquisitions not otherwise modeled)")
    return True


def cap_ronic(spec: ModelSpec, rev: dict) -> bool:
    """Gate 34 by construction (PM round 3: CRM's 55% RONIC was the explicit forecast's return, which excludes the
    acquisitions that bought the growth — Slack, Tableau, MuleSoft, Informatica): when goodwill and intangibles are
    a large share of capital and no recurring acquisitions are forecast, terminal RONIC is held at the return
    including goodwill of the last forecast year (DCF C93), never below WACC."""
    g34 = next((g for g in rev.get("gates") or [] if g.get("id") == 34), None)
    o = rev.get("outputs") or {}
    roic_t, wacc = o.get("roic_gw_t"), o.get("wacc")
    if not g34 or g34.get("status") != "FAIL" or not isinstance(roic_t, (int, float)) or \
            not isinstance(wacc, (int, float)):
        return False
    old, note = spec.valuation["ronic"]
    new = max(wacc, roic_t) // 0.0025 * 0.0025
    if new >= old:
        return False
    spec.valuation["ronic"] = (round(new, 4), f"{note.split('; held at the return including goodwill')[0]}; held at "
                               f"the return including goodwill ({spec.fc_labels[-1]} {roic_t:.1%}, floor WACC "
                               f"{wacc:.1%}): growth has been bought, and no recurring acquisitions are forecast "
                               f"(was {old:.1%})")
    spec.flags = [f for f in spec.flags if not f.startswith("Terminal RONIC held")]
    spec.flags.append(f"Terminal RONIC held at {new:.1%} (was {old:.1%}): the return including goodwill, because "
                      "goodwill and acquired intangibles are a large share of capital and the forecast has no "
                      "recurring acquisitions (gate 34)")
    return True


def with_scenarios(spec: ModelSpec, rev: dict) -> dict:
    """The review with bear / bull values (gate 33), recomputed only if the last one lacks them."""
    if "bear_fv" in (rev.get("outputs") or {}) or spec.framework == "reference":
        return rev
    return check.review(spec, build(spec, values=False), scenarios=True)


def return_excess_cash(spec: ModelSpec, rev: dict) -> bool:
    """Capital return by policy (PM critique: CRM's cash built to $58bn earning interest; the defaults took AVGO's
    to $792bn): from the second forecast year, cash above FY0's share of sales is returned as buybacks, so idle cash
    does not earn its way into EPS and the 3-year return. An analyst who means to hold the cash (a deal to fund, debt
    to repay) says so in thesis.explanations['cash build'] and the policy is not applied."""
    from . import gates as G
    held = [str(v) for k, v in ((spec.thesis or {}).get("explanations") or {}).items()
            if G.norm(k) in ("cash build", "capital return")]
    # holding the cash takes a named use (PM round 4: HD's "the engine may return surplus cash" switched the policy
    # off, and cash built to $44bn with zero buybacks)
    if any(re.search(r"(?i)\b(hold|held|keep|fund|repay|reserve|acquisition|deal|purchase price|maturit)", w) and
           not re.search(r"(?i)engine (may|will|can|to) return|return (surplus|excess) cash|not deliberately", w)
           for w in held):
        return False
    cash, sales = rev["forecast"].get("Ending cash"), rev["forecast"].get("Net sales")
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
    if not cash or not sales or not all(num(x) for x in list(cash) + list(sales)):
        return False
    s0, cash0 = spec.hist.IS["sales"][2] or 0.0, max(spec.hist.BS["cash"][2] or 0.0, 0.0)
    if s0 <= 0:
        return False
    share = max(cash0 / s0, 0.03)
    B, I = spec.hist.BS, spec.hist.IS
    ebitda0 = (I["opinc"][2] or 0.0) + (I["da"][2] or 0.0)
    debt0 = sum(B[k][2] or 0.0 for k in ("st_debt", "cur_ltd", "ltd"))
    lev0 = (debt0 - cash0) / ebitda0 if ebitda0 > 0 else 0.0
    repay = lev0 > 2.5  # a levered balance sheet (VZ ~3x) deleverages first
    key, cut = ("debt_iss", "; excess cash repays debt") if repay else ("buyback", "; raised so cash above")
    bb, bn = spec.drivers[key]
    bb, bn = list(bb), bn.split(cut)[0]
    sub, adds = 0.0, []
    for t in range(1, min(len(cash), len(bb))):
        cap = share * sales[t]
        x = (cash[t] - sub) - cap
        if x < max(50.0, 0.05 * cap):  # a small overshoot is not a policy question
            continue
        x = x // 50 * 50
        bb[t] += -x if repay else x
        sub += x
        adds.append(f"{spec.fc_labels[t]} {'-' if repay else '+'}{x:,.0f}mm")
    if not sub:
        return False
    what, fy0 = ", ".join(adds), spec.fy_hist[-1]
    use = (f"repays debt (net debt / EBITDA {lev0:.1f}x at FY{fy0}, above 2.5x)" if repay else "is returned as buybacks")
    spec.drivers[key] = (bb, f"{bn}{cut} " + (f"(net debt / EBITDA {lev0:.1f}x at FY{fy0}; " if repay else
                                              f"{share:.1%} of sales (FY{fy0}'s level) is returned (") + f"{what})")
    spec.flags = [f for f in spec.flags if not f.startswith("Excess cash")]
    spec.flags.append(f"Excess cash {use}: {what}. Cash is held at FY{fy0}'s {share:.1%} of sales from the second "
                      "forecast year; to hold it instead (a deal to fund), say why in thesis.explanations['cash build']")
    return True


def floor_cash(spec: ModelSpec, rev: dict) -> bool:
    """Capital return the model cannot fund: base-case ending cash must stay at or above a working minimum (the
    smaller of FY0 cash and 3% of sales). Buybacks are trimmed first, then debt repayment (LLY's analyst raised
    2030-31 buybacks to $55-65bn and the 2031 balance sheet showed -4.6bn of cash)."""
    cash, sales = rev["forecast"].get("Ending cash"), rev["forecast"].get("Net sales")
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
    if not cash or not sales or not all(num(x) for x in list(cash) + list(sales)):
        return False
    cash0 = max(spec.hist.BS["cash"][2] or 0.0, 0.0)
    bb, bn = spec.drivers["buyback"]
    di, dn = spec.drivers["debt_iss"]
    bb, di = list(bb), list(di)
    bn, dn = bn.split("; trimmed so ending cash")[0], dn.split("; repayment trimmed")[0]  # one note on a repeat pass
    add, cuts, repay_cut = 0.0, [], False
    for t in range(min(len(cash), len(bb), len(di))):
        short = min(cash0, 0.03 * sales[t]) - (cash[t] + add)
        if short <= 0:
            continue
        short = -(-short // 50) * 50  # whole $50mm, rounded up
        x = min(short, max(bb[t], 0.0))
        y = min(short - x, max(-di[t], 0.0))
        z = short - x - y  # dividends or capex above the cash generated (utilities, TXN, UPS): new borrowing
        bb[t] -= x
        di[t] += y + z
        add += x + y + z
        repay_cut |= y > 0 or z > 0
        if x or y or z:
            cuts.append(f"{spec.fc_labels[t]} " + " and ".join(
                ([f"buybacks -{x:,.0f}mm"] if x else []) + ([f"debt repayment -{y:,.0f}mm"] if y else []) +
                ([f"new debt +{z:,.0f}mm"] if z else [])))
    if not add:
        return False
    what = "; ".join(cuts)
    spec.drivers["buyback"] = (bb, f"{bn}; trimmed so ending cash stays above a working minimum ({what})")
    if repay_cut:
        spec.drivers["debt_iss"] = (di, f"{dn}; repayment trimmed / new debt raised so ending cash stays above a "
                                        "working minimum")
    spec.flags = [f for f in spec.flags if not f.startswith("Capital return trimmed")]
    spec.flags.append(f"Capital return trimmed to what the cash flow funds, the rest borrowed ({what}): ending cash "
                      f"stays at or above the smaller of FY{spec.fy_hist[-1]} cash and 3% of sales")
    return True


OUT_CELLS = {"fair_value": ("SUMMARY", "G19"), "dcf_pg": ("SUMMARY", "D15"), "dcf_exit": ("SUMMARY", "D16"),
             "sotp": ("SUMMARY", "D17"), "comps": ("SUMMARY", "D18"), "ke": ("DCF", "C9"),
             "div_yield": ("SUMMARY", "C11"), "price": ("SUMMARY", "C5"), "irr": ("DCF", "C105"),
             "wacc": ("DCF", "C17")}
# the rating rule (PM round 3: full convergence in 12 months made any value 5% above the price a Long, well inside
# the model's own error): the price closes a third of the gap to value a year; a Long needs the probability-weighted
# value 15% above the price (the error band) AND the price to offer a long-run IRR 2pt above WACC (reverse DCF)
CONVERGE, BAND, IRR_SPREAD = 1 / 3, 0.15, 0.02


def irr_value(x):
    """DCF!C105 as a number: the grid caps it at 3% / 20% and says so in text."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str) and x.strip().startswith(">"):
        return 0.20
    if isinstance(x, str) and x.strip().startswith("<"):
        return 0.03
    return None


def scenario_values(spec: ModelSpec, base_vals: dict) -> dict:
    """Bear / base / bull run end to end: the workbook rebuilt with the scenario selector at 1 and 3."""
    num = lambda x: x if isinstance(x, (int, float)) and not isinstance(x, bool) else None  # noqa: E731
    out = {"base": {k: num(base_vals.get(c)) for k, c in OUT_CELLS.items()}}
    keep = spec.scenario
    try:
        for key, sc in (("bear", 1), ("bull", 3)):
            spec.scenario = sc
            v = check.recalc(build_workbook(spec))
            out[key] = {k: num(v.get(c)) for k, c in OUT_CELLS.items()}
    except Exception as e:
        print(f"warning: scenario values not computed ({type(e).__name__}: {e})", file=sys.stderr)
    finally:
        spec.scenario = keep
    out["probs"] = spec.scenario_probs or [0.25, 0.5, 0.25]
    return out


DATE_RX = re.compile(r"(?i)\b(Q[1-4]\s*(FY)?\s*'?\d{2,4}|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
                     r"(\d{1,2},?\s+)?20\d{2}|20\d{2}-\d{2}(-\d{2})?|(first|second|third|fourth) quarter of 20\d{2}|"
                     r"(fy|fiscal)\s*'?20\d{2}|(h1|h2|1h|2h)\s*20\d{2})")


# low-volatility compounder (PM round 4: COST, a decade-long premium): beta, 10-year revenue σ, ROIC incl. goodwill
LOW_VOL = (0.85, 0.06, 0.15)
CROWDED = (0.10, 7.0)  # short interest % of float, days to cover
def low_vol_compounder(spec: ModelSpec, base: dict) -> str:
    beta = (spec.valuation.get("beta") or (None,))[0]
    from . import gates as G
    sig = G.growth_sigma(spec)
    roic = ((spec.last_review or {}).get("outputs") or {}).get("roic_gw_t") or (spec.capital or {}).get("roic_gw")
    if isinstance(beta, (int, float)) and beta < LOW_VOL[0] and sig is not None and sig < LOW_VOL[1] and \
            isinstance(roic, (int, float)) and roic > LOW_VOL[2]:
        return (f"a low-volatility compounder (beta {beta:.2f}, 10-year revenue σ {sig:.1%}, ROIC incl. goodwill "
                f"{roic:.0%})")
    return ""


def bracket_price(spec: ModelSpec, vals: dict) -> bool:
    """For a liquid large cap a bull below the price (or a bear above it) means the scenarios are miscalibrated, not
    that the market is wrong (PM round 4: COST, CAT and HD had the price above even the bull). The case is rebuilt
    from the reverse DCF: its margin path reaches the terminal operating margin the price requires (DCF C107) by the
    last forecast year, so the scenario says what the price needs."""
    sv = spec.scenario_values or {}
    px = (sv.get("base") or {}).get("price") or spec.price
    num = lambda x: x if isinstance(x, (int, float)) and not isinstance(x, bool) else None  # noqa: E731
    req, m6 = num(vals.get(("DCF", "C107"))), num(vals.get(("DCF", "D107")))
    if req is None or m6 is None or not px or not spec.segments:
        return False
    moved = []
    for case, sign in (("bull", 1), ("bear", -1)):
        v = num((sv.get(case) or {}).get("fair_value"))
        if v is None or (sign > 0 and v > px * 1.01) or (sign < 0 and v < px * 0.99):
            continue
        d = req - m6
        if sign * d <= 0.001:
            continue
        d = max(min(d, 0.15), -0.15)
        keep = {id(sg): list(sg.margin.get(case) or sg.margin["base"]) for sg in spec.segments}

        def bump(dd):
            for sg in spec.segments:
                b, c = sg.margin["base"], keep[id(sg)]
                n = len(b)
                sg.margin[case] = [round(max(x, y + dd * (t + 1) / n) if sign > 0 else min(x, y + dd * (t + 1) / n), 4)
                                   for t, (x, y) in enumerate(zip(c, b))]
        bump(d)
        # the requirement holds the explicit years at base, the scenario raises them too: scale back once so the
        # case lands just past the price (COST: a first pass put the bull 23% above it)
        sc0 = spec.scenario
        try:
            spec.scenario = 3 if sign > 0 else 1
            v2 = num(check.recalc(build_workbook(spec)).get(("SUMMARY", "G19")))
        finally:
            spec.scenario = sc0
        tgt = px * (1.05 if sign > 0 else 0.95)
        if v2 is not None and v2 != v and abs(v2 / px - 1) > 0.10:
            d2 = d * (tgt - v) / (v2 - v)
            if 0 < abs(d2) < abs(d):
                bump(d2)
                d = d2
        mh = [m_ for _, m_ in spec.margin_hist or []]
        hi_ = max(mh) if mh else None
        moved.append(f"{case} ${v:,.2f} was {'below' if sign > 0 else 'above'} the ${px:,.2f} price: its margins now "
                     f"move {d * 100:+.1f}pt from base by {spec.fc_labels[-1]}, toward the {req:.1%} terminal operating "
                     f"margin the price requires (base {m6:.1%})" +
                     (f" — {req - hi_:+.1%} above the 10-year high of {hi_:.1%}: the price needs more than history "
                      "has delivered, so check RONIC, fade years and WACC before calling the market wrong"
                      if sign > 0 and hi_ is not None and req > hi_ + 0.03 else ""))
    if not moved:
        return False
    spec.notes["scenario_rebuild"] = "Reverse-DCF scenario: " + "; ".join(moved)
    spec.flags = [f for f in spec.flags if not f.startswith("Scenario rebuilt from the reverse DCF")]
    spec.flags.append("Scenario rebuilt from the reverse DCF (price inside bear–bull): " + "; ".join(moved))
    return True


def rate(spec: ModelSpec, vals: dict) -> dict:
    """The rating is a function of the output (critique C3, PM rounds 3-4). The probability-weighted value must sit
    outside the model's error band (±15% of the price) AND the price must offer a long-run IRR (reverse DCF) at
    least 2pt better (Long) or worse (Short) than WACC, with the price inside bear–bull. A Long also needs 2:1 up /
    down skew. A Short needs a dated catalyst, a bull no more than +25%, written evidence that the gap closes, a
    crowding check, and — for a low-volatility compounder — a pair. Otherwise Neutral, with a positive / negative
    bias when the value is 5%+ away. The expected 12-month return assumes a third of the gap closes a year. No
    variant view -> Neutral. Any failed gate or unexplained warning -> Under review. The recommendation line is
    generated from the rule; an analyst override is labeled as one, with its reason."""
    sv = spec.scenario_values or {}
    base, bear, bull = (sv.get(k) or {} for k in ("base", "bear", "bull"))
    px, ke = base.get("price") or spec.price, base.get("ke")
    probs = sv.get("probs") or [0.25, 0.5, 0.25]
    vals_ = [bear.get("fair_value"), base.get("fair_value"), bull.get("fair_value")]
    th = spec.thesis or {}
    from . import gates as G
    gates = [g for g in (spec.last_review or {}).get("gates") or [] if g.get("id") not in (25, 26, 33, 41)]
    ok_vals = all(isinstance(x, (int, float)) for x in vals_) and px
    if ok_vals:
        gates.append(G.scenario_gate(spec, vals_[0], vals_[1], vals_[2], px))
        inside = vals_[0] < px < vals_[2]
        gates.append({"id": 41, "gate": "The price lies inside bear–bull (else the scenarios are miscalibrated, not the "
                                        "market)", "status": "PASS" if inside else "WARN",
                      "detail": (f"bear ${vals_[0]:,.2f} < price ${px:,.2f} < bull ${vals_[2]:,.2f}" if inside else
                                 f"price ${px:,.2f} outside bear ${vals_[0]:,.2f} – bull ${vals_[2]:,.2f}: rebuild the "
                                 "case from the reverse DCF (what the price requires) or say why the market is "
                                 "wrong on a liquid large cap") +
                                (f" ({spec.notes['scenario_rebuild']})" if spec.notes.get("scenario_rebuild") else "")})
        if spec.last_review is not None:
            spec.last_review["gates"] = gates
    failed = [g for g in gates if g.get("status") == "FAIL"]
    out = {"rating": "Neutral", "why": ""}
    if not ok_vals or ke is None:
        out.update(rating="Under review", why="Scenario values could not be computed")
    else:
        pw = sum(p * x for p, x in zip(probs, vals_)) / sum(probs)
        gap = pw / px - 1
        # a fairly priced stock earns its cost of equity; the price closes a third of the gap to value a year
        er = (1 + CONVERGE * gap) * (1 + ke) - 1
        alpha = er - ke
        irr, wacc = irr_value(base.get("irr")), base.get("wacc")
        spread = irr - wacc if irr is not None and isinstance(wacc, (int, float)) else None
        inside = vals_[0] < px < vals_[2]
        # skew with the probabilities attached (PM round 7: bull upside ÷ bear downside penalized an honest bear — the
        # ADBE bear went $115 -> $78 when its terminal growth was fixed and the Long flipped on a test the rule line
        # did not state): expected upside ÷ expected downside
        up_ = sum(p * max(x - px, 0.0) for p, x in zip(probs, vals_))
        dn_ = sum(p * max(px - x, 0.0) for p, x in zip(probs, vals_))
        skew = up_ / dn_ if inside and dn_ > 0 else None
        # sizing, not rating: a likely, deep bear is a binary setup (PM round 7: "a 35% chance of -67% argues for a
        # smaller position, not a different rating")
        p_bear, bear_dn = probs[0] / sum(probs), vals_[0] / px - 1
        binary = p_bear >= 0.25 and bear_dn <= -0.40
        texts = [str(x) for k in ("catalysts", "monitor", "points") for x in (th.get(k) or [])]
        dated = any(DATE_RX.search(t) for t in texts)
        variant = str(th.get("variant_view") or "").strip()
        out.update(expected_return=er, alpha=alpha, skew=skew, pw=pw, ke=ke, dated_catalyst=dated, gap=gap,
                   irr=irr, irr_spread=spread, variant=bool(variant), inside=inside,
                   sizing=(f"binary: a {p_bear:.0%} chance of {bear_dn:+.0%} in the bear — size below full until the "
                           "test that decides it" if binary else "full"))
        sp_ok = spread is not None
        # a short is more than a valuation gap (PM round 4): a dated catalyst, evidence the gap closes, a crowding
        # check, and a pair for a low-volatility compounder
        ex = {G.norm(k): str(v) for k, v in (th.get("explanations") or {}).items()}
        evidence = str(th.get("short_evidence") or ex.get("gap closes") or "").strip()
        pair = str(th.get("pair") or "").strip()
        lv = low_vol_compounder(spec, base)
        si = spec.short_interest or {}
        spf, dtc = si.get("shortPercentOfFloat"), si.get("shortRatio")
        crowded = (isinstance(spf, (int, float)) and spf >= CROWDED[0]) or \
            (isinstance(dtc, (int, float)) and dtc >= CROWDED[1])
        crowd_txt = (f"short interest {spf:.1%} of float, {dtc:.1f} days to cover" if isinstance(spf, (int, float))
                     and isinstance(dtc, (int, float)) else "short interest not available")
        short_val = gap <= -BAND and sp_ok and spread <= -IRR_SPREAD and inside and vals_[2] / px - 1 <= 0.25
        missing = [x for x, bad in (("no dated catalyst in the thesis", not dated),
                                    ("no evidence the gap closes (thesis.short_evidence: when the premium last "
                                     "compressed, and on what)", not re.search(r"\d", evidence)),
                                    (f"crowded ({crowd_txt}) with no borrow / squeeze plan in thesis.explanations"
                                     "['crowding']", crowded and not ex.get("crowding")),
                                    (f"{lv}: an outright short needs a pair (thesis.pair: the long leg)",
                                     bool(lv) and not pair)) if bad]
        if gap >= BAND and sp_ok and spread >= IRR_SPREAD and inside and skew is not None and skew >= 2:
            r = "Long"
        elif short_val and not missing:
            r = "Short"
        else:
            r = "Neutral"
        why = (f"Probability-weighted value ${pw:,.2f} is {gap:+.1%} vs the price (a call needs ±{BAND:.0%}, the "
               f"model's error band); the price implies a {irr:.1%} long-run IRR vs {wacc:.1%} WACC "
               f"({spread * 100:+.1f}pt; a call needs ±{IRR_SPREAD * 100:.0f}pt)" if sp_ok else
               f"Probability-weighted value ${pw:,.2f} is {gap:+.1%} vs the price; implied IRR not available")
        why += (f"; expected 12-month return {er:+.1%} with a third of the gap closing a year (alpha {alpha:+.1%} vs "
                f"cost of equity {ke:.1%}); " +
                (f"probability-weighted skew {skew:.1f}:1 (a Long needs 2:1)" if skew is not None else
                 f"price outside bear–bull (${vals_[0]:,.2f}–${vals_[2]:,.2f}): no call until the scenarios bracket it"))
        if r == "Long" and binary:
            why += f"; sized as a binary ({out['sizing'].split(': ', 1)[1].split(' —')[0]})"
        if r == "Short":
            why += f"; {crowd_txt}" + (f"; pair: {pair}" if pair else "") + \
                   "; borrow assumed general collateral" if not crowded else f"; {crowd_txt}"
        if r == "Neutral" and (gap >= 0.05 or gap <= -0.05):
            bias = "positive" if gap > 0 else "negative"
            out["bias"] = bias
            short_of = [x for x, ok in ((f"gap {gap:+.1%} inside ±{BAND:.0%}", abs(gap) < BAND),
                                        (f"IRR spread {spread * 100:+.1f}pt" if sp_ok else "no IRR",
                                         not sp_ok or abs(spread) < IRR_SPREAD),
                                        (f"probability-weighted skew {skew:.1f}:1 under 2:1" if skew is not None else "",
                                         gap > 0 and skew is not None and skew < 2)) if ok and x]
            if short_val and missing:
                short_of.append("the valuation gap is there but a short needs more: " + "; ".join(missing))
            why += f". Neutral with a {bias} bias" + (f" ({'; '.join(short_of)})" if short_of else "")
        if r != "Neutral" and not variant:
            why += f"; the rule says {r} but the thesis states no variant view vs consensus, so Neutral"
            r = "Neutral"
        out.update(rating=r, why=why, pair=pair if r == "Short" else "")
    opens = G.open_items(spec, [g for g in gates if g.get("id") not in (25, 26)],
                         (spec.last_review or {}).get("problems") or [], (spec.last_review or {}).get("plugs"))
    from . import bridge as BR
    vb = getattr(spec, "version_bridge", None)
    if BR.needs_independent_review(vb) and out.get("rating") != (vb.get("prev") or {}).get("rating"):
        w = next((str(v) for k, v in (th.get("explanations") or {}).items() if G.norm(k) == "independent review"),
                 None)
        opens.append(("independent review",
                      f"Rating {(vb.get('prev') or {}).get('rating')} → {out.get('rating')} against the {vb['label']} "
                      f"version, {vb['judgment_share']:.0%} of the value move from judgment inputs (Checks: version "
                      "bridge): needs an independent review before it ships", w))
    out["open_items"] = opens
    unexplained = [k for k, _, w in opens if not w]
    # the text says what the rule says (PM round 4: four of six recommendations contradicted the rule — CRM "Long"
    # under a Neutral rule, CAT "Underweight" under Short, NFLX "positive bias" under negative)
    ov = th.get("rating_override") or {}
    has_ov = isinstance(ov, dict) and ov.get("rating") and ov.get("reason")
    rule_call = {"Long": "long", "Short": "short"}.get(out.get("rating"), "")
    bias = out.get("bias") or ""
    contra = []
    for fld, call, phrase in G.stated_calls(spec):
        if call in ("long", "short") and call != rule_call:
            contra.append(f"{fld}: '…{phrase}…' states a {call.capitalize()} call, the rule says "
                          f"{out.get('rating')}{' (' + bias + ' bias)' if bias else ''}")
        elif call in ("positive", "negative") and call != bias:
            contra.append(f"{fld}: '…{phrase}…' states a {call} bias, the rule says "
                          f"{out.get('rating')}{' with a ' + bias + ' bias' if bias else ' with no bias'}")
    if has_ov:
        contra = []  # an override is the written divergence; it is labeled and logged instead
    if failed or unexplained:
        out["why"] = ((f"{len(failed)} gate(s) failed: " + "; ".join(f"({g['id']}) {g['gate']}" for g in failed[:4]) +
                       ". " if failed else "") +
                      (f"{len(unexplained)} warning(s) / open check(s) not explained in the thesis "
                       f"(thesis.explanations): {', '.join(unexplained[:4])}. " if unexplained else "") +
                      out.get("why", ""))
        out["rule"] = out["rating"]
        out["rating"] = "Under review"
    if contra and out["rating"] != "Under review":
        out["why"] = "The thesis text states a different call: " + "; ".join(contra[:3]) + ". " + out["why"]
        out["rule"] = out["rating"]
        out["rating"] = "Under review"
    if has_ov and out["rating"] != "Under review":
        if str(ov["rating"]).capitalize() != out["rating"]:
            out["override"] = f"Rule says {out['rating']}; analyst rates {ov['rating']}: {ov['reason']}"
            out["rule"] = out["rating"]
            out["rating"] = str(ov["rating"]).capitalize()
    # the recommendation line is the rule's (or the labeled override's), followed by the analyst's sentence
    rt = G.recommendation_text(th.get("recommendation_text") or th.get("recommendation"))
    th["recommendation_text"] = rt
    head = (f"Override: {out['rating']} (the rule says {out['rule']}; {ov['reason']})" if out.get("override") else
            f"Under review (rule: {out.get('rule')}{', ' + bias + ' bias' if bias else ''}, pending the open items)"
            if out["rating"] == "Under review" and out.get("rule") else
            "Under review" if out["rating"] == "Under review" else
            f"{out['rating']}, {bias} bias" if bias and out["rating"] == "Neutral" else
            f"{out['rating']}" + (f", as a pair against {out['pair']}" if out.get("pair") else "") +
            (", sized as a binary" if out["rating"] in ("Long", "Short") and
             str(out.get("sizing") or "").startswith("binary") else ""))
    th["recommendation"] = f"{head}: {rt}" if rt else head
    spec.thesis = th
    spec.rating = out
    stance = {"Long": "long", "Short": "short", "Neutral": "neutral on", "Under review": "neutral on"}
    spec.thesis["stance"] = stance.get(out["rating"], "neutral on")
    # gates 25-26 (decision layer) join the gate list shown on the Checks tab
    if spec.last_review is not None and isinstance(spec.last_review.get("gates"), list):
        gl = [g for g in spec.last_review["gates"] if g.get("id") not in (25, 26)]
        gl.append({"id": 25, "gate": "Rating and recommendation text follow the rule, or the override is labeled and "
                                     "written into the thesis",
                   "status": "FAIL" if contra else "PASS",
                   "detail": ("; ".join(contra[:3]) + ": rewrite the text to the rule's call, or write a "
                                                       "rating_override with its reason") if contra else
                   (out.get("override") or out["why"])})
        gl.append({"id": 26, "gate": "No unresolved failed gates; every warning and open check explained",
                   "status": "FAIL" if failed or unexplained else "PASS",
                   "detail": (f"{len(failed)} failed, {len(unexplained)} unexplained: rating held at Under review"
                              if failed or unexplained else "")})
        spec.last_review["gates"] = gl
    return out


def calibrate_fcf(spec: ModelSpec, rev: dict) -> bool:
    """FCF onto the guidance midpoint through working capital (receivable then payable days, at most ±15% each,
    held in every year so the change is a level, not a one-year swing) — never through a generic 'other' cash
    line (critique C6). What working capital cannot explain stays visible as a guidance miss."""
    g = spec.guidance.get("fcf")
    f = rev.get("forecast") or {}
    model = (f.get("Free cash flow") or [None])[0]
    sales = (f.get("Net sales") or [None])[0]
    if not g or not isinstance(model, (int, float)) or not isinstance(sales, (int, float)) or \
            g[0] * 0.995 <= model <= g[1] * 1.005:
        return False
    mid = sum(g) / 2
    gap = mid - model
    used = []
    cogs = sales * spec.drivers["cogs_pct"][0][0]
    for key, base, sign in (("dso", sales, -1), ("dpo", cogs, 1)):
        if abs(gap) < 1 or not base:
            break
        vals, note = spec.drivers[key]
        days = vals[0]
        if not days:
            continue
        step = max(-0.15 * days, min(0.15 * days, sign * gap * 365 / base))
        # the first year takes the step and it fades out by the fourth (PM round 2: a calibration held in every
        # year carried a one-year guide into the terminal value)
        fade = [1.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0]
        spec.drivers[key] = ([round(x + step * w, 1) for x, w in zip(vals, fade)],
                             f"{note}; {step:+.1f} days in {spec.fc_labels[0]} so {spec.fc_labels[0]} FCF meets the "
                             f"guide, fading to 0 by {spec.fc_labels[3]}")
        gap -= sign * step * base / 365
        used.append(f"{key.upper()} {step:+.1f} days")
    if not used:
        return False
    spec.calibrated["fcf"] = "working capital: " + ", ".join(used) + \
        (f"; {gap:,.0f}mm of the guide not explained by working capital" if abs(gap) > 0.01 * abs(mid) else "")
    return True


ATTRIBUTION = [("RONIC", ("valuation.ronic",)),
               ("Terminal growth and fade years", ("valuation.g", "valuation.fade_years")),
               ("Beta, cost of debt, tax rate", ("valuation.beta", "valuation.kd", "valuation.tax_wacc")),
               ("Exit multiple", ("valuation.exit_mult",)),
               ("Method weights, peer discount", ("valuation.w_pg", "valuation.w_exit", "valuation.w_sotp",
                                                  "valuation.w_comps", "valuation.peer_discount",
                                                  "valuation.comps_pe_weight")),
               ("Stakes and minorities", ("valuation.other_stake", "valuation.stake_haircut", "valuation.nci_value")),
               ("Working capital", ("drivers.dso", "drivers.dio", "drivers.dpo", "drivers.accrued_pct",
                                    "drivers.oca_pct")),
               ("Capex and depreciation", ("drivers.capex_pct", "drivers.da_pct")),
               ("Stock comp, amortization, restructuring", ("drivers.sbc_pct", "drivers.amort", "drivers.repo_pct",
                                                            "drivers.other_op")),
               ("Tax, interest, other non-operating", ("drivers.etr", "drivers.other_nonop", "drivers.int_rate",
                                                       "drivers.int_yield", "drivers.nonrec", "drivers.nci_ni")),
               ("Buybacks, dividends, debt", ("drivers.buyback", "drivers.buyback_px", "drivers.debt_iss",
                                              "drivers.shares_issued", "drivers.stock_proc", "drivers.dps0",
                                              "drivers.div_growth")),
               ("Acquisitions", ("drivers.acq", "drivers.intang_share")),
               ("Segment growth and M&A (base)", (".growth.base", ".mna")),
               ("Segment margins and their basis (base)", (".margin.base", "seg_after_charges"))]


def _first_values(spec: ModelSpec) -> dict:
    """Each overridden key's value before its first override (the defaults)."""
    first = {}
    for _, key, before, _after in spec.audit:
        first.setdefault(key, before)
    return first


def _override_groups(first: dict) -> list:
    """[(label, keys)] for the attribution groups an override touched."""
    out = []
    for label, pats in ATTRIBUTION:
        keys = [k for k in first if any(k == p or (p.startswith(".") and k.startswith("segments.") and
                                                   k.endswith(p)) for p in pats)]
        if keys and keys != ["seg_after_charges"]:
            out.append((label, keys))
    return out


def _revert_keys(s2: ModelSpec, keys: list, first: dict) -> None:
    """Put these keys back to their default values (in place)."""
    for k in keys:
        b = first[k]
        if k == "seg_after_charges":
            s2.set_single_segment_basis(int(b))
            s2.seg_after_charges = int(b)
        elif k.startswith("valuation."):
            n = k.split(".", 1)[1]
            s2.valuation[n] = (b, s2.valuation[n][1])
            if n == "exit_mult":
                s2.capital["exit_derived"] = True
        elif k.startswith("drivers."):
            n = k.split(".", 1)[1]
            s2.drivers[n] = (list(b), s2.drivers[n][1])
        else:
            _, name, rest = k.split(".", 2) if k.count(".") >= 2 else (None, None, None)
            sg = next((x for x in s2.segments if x.name == name), None)
            if sg is None:
                continue
            if rest == "mna":
                sg.mna = list(b)
            else:
                getattr(sg, rest.split(".")[0])["base"] = list(b)


def revert_harmful(spec: ModelSpec, rev: dict) -> dict:
    """The analyst's passes moved fair value 30% or more from the defaults and gates fail (PM, ADBE: $292 at defaults,
    $134 after the passes, through one refused input): each group of overrides is tried back at its defaults, and a
    group whose revert clears failed gates without adding any is reverted and flagged. Returns the review."""
    d0 = next((v for st, v in spec.audit_fv if st.startswith("defaults")), None)
    fv = (rev.get("outputs") or {}).get("fair_value")
    fails = lambda r: {g["id"] for g in r.get("gates") or [] if g["status"] == "FAIL"} - {25, 26}  # noqa: E731
    if not d0 or not isinstance(fv, (int, float)) or fv <= 0 or not fails(rev) or 0.7 <= fv / d0 <= 1 / 0.7:
        return rev
    import copy
    first = _first_values(spec)
    for label, keys in _override_groups(first):
        s2 = copy.deepcopy(spec)
        _revert_keys(s2, keys, first)
        try:
            r2 = check.review(s2, build(s2, values=False))
        except Exception:
            continue
        f0, f2 = fails(rev), fails(r2)
        fv2 = (r2.get("outputs") or {}).get("fair_value")
        if f2 < f0 and isinstance(fv2, (int, float)) and abs(fv2 - d0) < abs(fv - d0):
            for k in keys:
                spec.audit.append(("engine revert", k, _audit_value(spec, k), first[k]))
            _revert_keys(spec, keys, first)
            spec.flags.append(f"Reverted to defaults: {label.lower()} ({', '.join(k.split('.', 1)[-1] for k in keys)})"
                              f". The analyst's version moved fair value from ${d0:,.2f} (defaults) to ${fv:,.2f} "
                              f"while gates {', '.join(map(str, sorted(f0)))} failed; the revert clears gate(s) "
                              f"{', '.join(map(str, sorted(f0 - f2)))} (fair value ${fv2:,.2f})")
            rev, fv = r2, fv2
            if not fails(rev) or 0.7 <= fv / d0 <= 1 / 0.7:
                break
    return rev


def _audit_value(spec: ModelSpec, k: str):
    if k.startswith("valuation."):
        return spec.valuation[k.split(".", 1)[1]][0]
    if k.startswith("drivers."):
        return list(spec.drivers[k.split(".", 1)[1]][0])
    if k == "seg_after_charges":
        return spec.seg_after_charges
    _, name, rest = k.split(".", 2)
    sg = next((x for x in spec.segments if x.name == name), None)
    return None if sg is None else list(sg.mna) if rest == "mna" else list(getattr(sg, rest.split(".")[0])["base"])


def attribute_overrides(spec: ModelSpec, vals: dict) -> None:
    """Fair-value effect of each group of overrides: that group put back to its value before the first override,
    everything else as built (PM round 2: RONIC and working-capital overrides moved value $20-30 invisibly)."""
    import copy
    from .workbook import build_workbook
    fv = vals.get(("SUMMARY", "G19"))
    if not isinstance(fv, (int, float)):
        return
    first = _first_values(spec)
    out = []
    for label, keys in _override_groups(first):
        s2 = copy.deepcopy(spec)
        _revert_keys(s2, keys, first)
        try:
            v2 = check.cached_values(build_workbook(s2)).get(("SUMMARY", "G19"))
        except Exception:
            continue
        if isinstance(v2, (int, float)):
            out.append((label, [k.split(".", 1)[-1] for k in keys], round(fv - v2, 2)))
    spec.audit_impact = sorted(out, key=lambda x: -abs(x[2]))


def bridge_previous(spec: ModelSpec, data: CompanyData, prev_doc: dict, label: str = "previous",
                    progress: Callable[[str], None] = lambda m: None, effort: Optional[str] = None,
                    use_ai: bool = True) -> list:
    """Value bridge to the version the PM last read; when judgment inputs carry half the move or more and the rating
    flips, an independent AI review challenges each judgment change (PM round 6, ADBE: the price fell 4%, judgment
    moved fair value -$106 and the rating flipped from Long to Neutral with nothing stopping it). Returns
    [bridge text, warnings...]."""
    import json as _json
    from . import bridge
    out = []
    spec.version_bridge = bridge.version_bridge(data, prev_doc, _json.loads(overrides.dumps(spec)), progress, label)
    out.append(bridge.bridge_text(spec.version_bridge))
    if bridge.needs_independent_review(spec.version_bridge) and use_ai:
        client = ai.client()
        if client:
            out += bridge.independent_review(spec, data, client, prev_doc, spec.version_bridge, progress, effort)
            fill_peer_ev(spec)
            spec.version_bridge = bridge.version_bridge(data, prev_doc, _json.loads(overrides.dumps(spec)), progress,
                                                        label)
            out.append("After the independent review:\n" + bridge.bridge_text(spec.version_bridge))
            spec.ai_cost, spec.ai_usage = ai.usage_summary() if ai.USAGE else "", list(ai.USAGE)
    return out


def build(spec: ModelSpec, values: bool = True) -> bytes:
    """The workbook. values=True (deliverables) also stores every formula's computed result in the file, so it
    reads correctly in any viewer; internal check builds skip that (the checker recalculates anyway). In the
    economic framework the deliverable also carries the bear / base / bull values (each scenario recalculated
    end to end) and the rule-based rating."""
    xlsx = build_workbook(spec)
    if not values:
        return xlsx
    try:
        vals = check.cached_values(xlsx)
        if spec.framework != "reference":
            if spec.audit and not spec.audit_impact:
                try:
                    attribute_overrides(spec, vals)
                except Exception as e:  # attribution is a disclosure, never a reason to lose the deliverable
                    spec.flags.append(f"Override attribution skipped: {type(e).__name__}: {str(e)[:120]}")
            spec.scenario_values = scenario_values(spec, vals)
            if bracket_price(spec, vals):
                spec.scenario_values = scenario_values(spec, vals)
            rate(spec, vals)
            vals = check.cached_values(build_workbook(spec))
        return build_workbook(spec, values=vals)
    except Exception as e:  # e.g. the formulas library missing: Excel still calculates on open; say so on the Cover
        print(f"warning: formula results not stored ({type(e).__name__}: {e})", file=sys.stderr)
        return build_workbook(spec, unstored=True)


def run(ticker: str, as_of: date, peers: Optional[list] = None, use_ai: bool = False, analyst: bool = False,
        overrides_doc: Optional[dict] = None, progress: Callable[[str], None] = lambda m: None,
        cik: Optional[str] = None, allow_financials: bool = False) -> tuple[ModelSpec, bytes, list]:
    """One call: filings -> defaults (-> AI) (-> external overrides) -> workbook bytes."""
    data = fetch(ticker, as_of, peers=peers, use_ai=use_ai, progress=progress, cik=cik,
                 allow_financials=allow_financials)
    spec = make_spec(data, use_ai_thesis=use_ai, analyst=analyst, progress=progress, guards=not overrides_doc)
    warn = overrides.apply(spec, overrides_doc, source="external agent") if overrides_doc else []
    fill_peer_ev(spec)
    return spec, build(spec), warn
