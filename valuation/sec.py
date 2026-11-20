"""SEC EDGAR client: ticker lookup, filing lists, filing documents.

Everything here hits public EDGAR endpoints. SEC asks for a descriptive
User-Agent with a contact address; set SEC_USER_AGENT in the environment or
Streamlit secrets (e.g. "Your Name your.email@domain.com").
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import requests

DEFAULT_UA = "fa-webapp valuation-model-builder admin@example.com"
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fa_webapp_sec_cache")
_last_call = [0.0]


def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        try:  # Streamlit secrets, when running inside the app
            import streamlit as st
            ua = st.secrets.get("SEC_USER_AGENT")
        except Exception:
            ua = None
    return ua or DEFAULT_UA


def _get(url: str, *, binary: bool = False, max_age: float = 6 * 3600):
    """GET with a polite rate limit and a small disk cache (filings never change)."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()
    path = os.path.join(_CACHE_DIR, key)
    immutable = "/Archives/edgar/data/" in url  # filed documents are immutable
    if os.path.exists(path) and (immutable or time.time() - os.path.getmtime(path) < max_age):
        with open(path, "rb") as fh:
            raw = fh.read()
        return raw if binary else raw.decode("utf-8", errors="replace")

    wait = 0.12 - (time.time() - _last_call[0])  # SEC limit is 10 requests/second
    if wait > 0:
        time.sleep(wait)
    for attempt in range(3):
        resp = requests.get(url, headers={"User-Agent": _user_agent(),
                                          "Accept-Encoding": "gzip, deflate"}, timeout=60)
        _last_call[0] = time.time()
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
        return resp.content if binary else resp.text
    resp.raise_for_status()


def get_json(url: str):
    return json.loads(_get(url))


# ---------------------------------------------------------------- lookups
def lookup_ciks(ticker: str) -> list[tuple[str, str]]:
    """All (10-digit CIK, company title) pairs listed for a ticker (a new holding company
    can share the ticker with its predecessor registrant)."""
    data = get_json("https://www.sec.gov/files/company_tickers.json")
    t = ticker.upper().replace(".", "-")
    out = [(str(row["cik_str"]).zfill(10), row["title"]) for row in data.values() if row["ticker"].upper() == t]
    if not out:
        raise ValueError(f"Ticker {ticker!r} not found in SEC's company list")
    return out


def lookup_cik(ticker: str) -> tuple[str, str]:
    return lookup_ciks(ticker)[0]


@dataclass
class Filing:
    form: str
    accession: str
    filing_date: date
    report_date: Optional[date]
    primary_doc: str
    items: str
    cik: str

    @property
    def folder(self) -> str:
        return (f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/"
                f"{self.accession.replace('-', '')}/")

    @property
    def url(self) -> str:
        return self.folder + self.primary_doc

    @property
    def instance_url(self) -> str:
        """Extracted XBRL instance SEC publishes next to an inline-XBRL document."""
        return self.folder + re.sub(r"\.htm$", "", self.primary_doc) + "_htm.xml"


@dataclass
class Company:
    cik: str
    name: str
    ticker: str
    sic: str
    sic_description: str
    fiscal_year_end: str  # "MMDD"
    exchange: str
    filings: list[Filing] = field(default_factory=list)


def _d(s: str) -> Optional[date]:
    return date.fromisoformat(s) if s else None


def load_company(ticker: str, cik: Optional[str] = None) -> Company:
    """Company with its 10-K/10-Q/8-K list. When several registrants share the ticker,
    the first one that has filed a 10-K is used; `cik` forces a registrant (e.g. the
    predecessor of a new holding company that has not filed a 10-K yet)."""
    if cik:
        cik = re.sub(r"\D", "", cik).zfill(10)
        return _load_cik(ticker, cik, "")
    cands = lookup_ciks(ticker)
    for cik, title in cands:
        co = _load_cik(ticker, cik, title)
        if any(f.form == "10-K" for f in co.filings) or (cik, title) == cands[-1]:
            return co
    return co


def _load_cik(ticker: str, cik: str, title: str) -> Company:
    sub = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    batches = [sub["filings"]["recent"]]
    # heavy filers (debt programs, Form 4s) push 10-Ks off the first page: keep paging
    # until two years of 10-Ks are in hand
    for extra in sub["filings"].get("files", [])[:8]:
        if sum(f == "10-K" for b in batches for f in b["form"]) >= 3:
            break
        batches.append(get_json("https://data.sec.gov/submissions/" + extra["name"]))
    filings = []
    for r in batches:
        for i, form in enumerate(r["form"]):
            if form not in ("10-K", "10-Q", "8-K", "10-K/A", "20-F", "40-F"):
                continue
            filings.append(Filing(form=form, accession=r["accessionNumber"][i],
                                  filing_date=_d(r["filingDate"][i]),
                                  report_date=_d(r["reportDate"][i]),
                                  primary_doc=r["primaryDocument"][i],
                                  items=r.get("items", [""] * len(r["form"]))[i] or "",
                                  cik=cik))
    filings.sort(key=lambda f: f.filing_date, reverse=True)
    exch = (sub.get("exchanges") or [""])[0] or ""
    return Company(cik=cik, name=sub.get("name") or title, ticker=ticker.upper(),
                   sic=str(sub.get("sic") or ""), sic_description=sub.get("sicDescription") or "",
                   fiscal_year_end=sub.get("fiscalYearEnd") or "1231", exchange=exch,
                   filings=filings)


# ------------------------------------------------------- filing selection
@dataclass
class FilingSet:
    annual: Filing
    quarterly: Optional[Filing]          # latest 10-Q after the 10-K, if any
    fy_release: Optional[Filing]         # earnings 8-K for the 10-K's fiscal year
    q_release: Optional[Filing]          # earnings 8-K for the latest quarter
    prior_q_releases: list = field(default_factory=list)  # earnings 8-Ks for the year's earlier quarters


def select_filings(co: Company, as_of: date) -> FilingSet:
    avail = [f for f in co.filings if f.filing_date <= as_of]
    annual = next((f for f in avail if f.form == "10-K"), None)
    if annual is None:
        if any(f.form in ("20-F", "40-F") for f in avail):
            raise ValueError("This issuer files 20-F/40-F (foreign private issuer); "
                             "the model needs 10-K / 10-Q filings.")
        raise ValueError(f"{co.name} (CIK {int(co.cik)}) has no 10-K filed on or before {as_of:%b %d, %Y}. "
                         "If it is a newly formed holding company, enter the predecessor registrant's CIK "
                         "under Advanced options.")
    quarterly = next((f for f in avail if f.form == "10-Q" and f.report_date
                      and f.report_date > annual.report_date), None)
    releases = [f for f in avail if f.form == "8-K" and "2.02" in f.items]

    def near(target: Filing) -> Optional[Filing]:
        cands = [r for r in releases if -3 <= (target.filing_date - r.filing_date).days <= 45
                 and r.filing_date > (target.report_date or target.filing_date)]
        # prefer pure results releases over multi-item 8-Ks
        cands.sort(key=lambda r: (len(r.items.split(",")) > 3, (target.filing_date - r.filing_date).days))
        return cands[0] if cands else None

    fy_release = near(annual)
    q_release = near(quarterly) if quarterly else None
    # the year's earlier quarters: their reported adjusted EPS adds up to the year to date (gate 47)
    earlier = [f for f in avail if quarterly and f.form == "10-Q" and f.report_date and
               annual.report_date < f.report_date < quarterly.report_date]
    prior = [r for r in (near(f) for f in sorted(earlier, key=lambda f: f.report_date)) if r]
    return FilingSet(annual=annual, quarterly=quarterly, fy_release=fy_release, q_release=q_release,
                     prior_q_releases=prior)


# --------------------------------------------------------- documents
def filing_index(f: Filing) -> list[dict]:
    return get_json(f.folder + "index.json")["directory"]["item"]


def exhibit_99_url(f: Filing, strict: bool = False) -> Optional[str]:
    """URL of the press-release exhibit (EX-99.1) inside an 8-K. strict: EX-99 names only (no guessing,
    so a credit agreement filed as EX-10.1 is never mistaken for a release)."""
    items = filing_index(f)
    names = [it["name"] for it in items]
    pats = [r"ex-?99[-_.]?0?1\b", r"ex99[-_]?1", r"exhibit-?99[-_.]?1", r"ex-?99", r"exhibit-?99", r"99"]
    if strict:
        pats = pats[:5]
    else:
        pats.append(r"earn|release|press|results")  # e.g. GE: ge2q2025earningsrelease.htm
    for p in pats:
        for n in names:
            if n.lower().endswith((".htm", ".html")) and re.search(p, n.lower()) and n != f.primary_doc \
                    and "index" not in n.lower():
                return f.folder + n
    if strict:
        return None
    # last resort: the largest HTML document other than the 8-K cover itself
    docs = [it for it in items if it["name"].lower().endswith((".htm", ".html")) and it["name"] != f.primary_doc
            and "index" not in it["name"].lower() and not re.match(r"R\d+\.htm", it["name"])]
    if docs:
        return f.folder + max(docs, key=lambda it: int(it.get("size") or 0))["name"]
    return None


def statement_reports(f: Filing) -> list[dict]:
    """R-file pages (as rendered by EDGAR) for the primary statements and segment note."""
    try:
        xml = _get(f.folder + "FilingSummary.xml")
    except Exception:
        return []
    out = []
    for blk in re.findall(r"<Report instance=.*?</Report>", xml, re.S):
        def tag(t):
            m = re.search(rf"<{t}>(.*?)</{t}>", blk, re.S)
            return m.group(1).strip() if m else ""
        cat, short, html = tag("MenuCategory"), tag("ShortName"), tag("HtmlFileName")
        if not html:
            continue
        if cat == "Statements" or (cat == "Details" and "SEGMENT" in short.upper()):
            out.append({"file": html.replace(".htm", ""), "name": short.title(),
                        "category": cat, "url": f.folder + html})
    return out


def _linkbase(f: Filing, kind: str) -> Optional[str]:
    """A filing's XBRL linkbase (kind = pre / cal / def / lab). Usually named after the primary document,
    but filing agents often name the document differently from the taxonomy (DE: de-20251102x10k.htm
    with de-20251102_pre.xml), so fall back to the filing index."""
    try:
        return _get(f.folder + re.sub(r"\.htm$", "", f.primary_doc) + f"_{kind}.xml")
    except Exception:
        pass
    try:
        for it in filing_index(f):
            if it["name"].endswith(f"_{kind}.xml"):
                return _get(f.folder + it["name"])
    except Exception:
        pass
    return None


def member_children(f: Filing) -> dict[str, set[str]]:
    """Domain-member hierarchy from the definition linkbase: {'lly:CardiometabolicHealthMember':
    {'lly:MounjaroMember', ...}}, used to drop subtotal members from revenue breakdowns."""
    dfn = _linkbase(f, "def")
    if not dfn:
        return {}
    out: dict[str, set[str]] = {}
    for link in re.findall(r"<(?:\w+:)?definitionLink\b.*?</(?:\w+:)?definitionLink>", dfn, re.S):
        locs = {}
        for tag in re.findall(r"<(?:\w+:)?loc\b[^>]*>", link):
            lab = re.search(r'xlink:label="([^"]+)"', tag)
            href = re.search(r'xlink:href="[^"#]*#([^"]+)"', tag)
            if lab and href:
                pfx, _, name = href.group(1).partition("_")
                locs[lab.group(1)] = f"{pfx}:{name}"
        for tag in re.findall(r"<(?:\w+:)?definitionArc\b[^>]*>", link):
            if "domain-member" not in tag:
                continue
            fr = re.search(r'xlink:from="([^"]+)"', tag)
            to = re.search(r'xlink:to="([^"]+)"', tag)
            if fr and to and fr.group(1) in locs and to.group(1) in locs:
                out.setdefault(locs[fr.group(1)], set()).add(locs[to.group(1)])
    return out


def income_statement_concepts(f: Filing) -> set[str]:
    """Concepts presented on the income statement (R-page role), e.g. {'us-gaap:Revenues', ...}.

    Used to tell a GAAP operating-income line apart from an operating-income
    tag that only appears in the segment note.
    """
    return _statement_concepts(f, r"OPERATIONS|INCOME|EARNINGS", r"COMPREHENSIVE|PARENTHETICAL|BALANCE|CASH|EQUITY")


def balance_sheet_concepts(f: Filing) -> set[str]:
    """Concepts presented on the balance sheet, e.g. to tell a separate line (DE: equipment on operating leases)
    from a note disclosure of an amount already inside another line."""
    return _statement_concepts(f, r"BALANCE SHEET|FINANCIAL POSITION|FINANCIAL CONDITION", r"PARENTHETICAL")


def _statement_concepts(f: Filing, rx: str, not_rx: str) -> set[str]:
    """Concepts on the first primary statement whose R-page name matches rx (and not not_rx)."""
    try:
        xml = _get(f.folder + "FilingSummary.xml")
    except Exception:
        return set()
    role = None
    for blk in re.findall(r"<Report instance=.*?</Report>", xml, re.S):
        cat = re.search(r"<MenuCategory>(.*?)</MenuCategory>", blk)
        short = re.search(r"<ShortName>(.*?)</ShortName>", blk)
        r = re.search(r"<Role>(.*?)</Role>", blk)
        if not (cat and short and r) or cat.group(1) != "Statements":
            continue
        name = short.group(1).upper()
        if re.search(rx, name) and not re.search(not_rx, name):
            role = r.group(1)
            break
    if not role:
        return set()
    pre = _linkbase(f, "pre")
    if not pre:
        return set()
    m = re.search(r'<(?:\w+:)?presentationLink\b[^>]*xlink:role="%s".*?</(?:\w+:)?presentationLink>'
                  % re.escape(role), pre, re.S)  # with or without the link: prefix (filing agents differ)
    if not m:
        return set()
    out = set()
    for href in re.findall(r'xlink:href="[^"#]*#([^"]+)"', m.group(0)):
        pfx, _, name = href.partition("_")
        out.add(f"{pfx}:{name}")
    return out


def fiscal_year_label(f: Filing, dei_year: int, period_end: date) -> int:
    """The fiscal-year number the company itself uses for a 10-K whose year ends in Jan-Mar.

    Conventions differ (NVDA / CRM / WMT: the year it ends in; HD / TGT: the year it starts in) and
    the dei tag does not always follow the company's wording (CRM tags the year ended Jan 31, 2026
    as 2025 but calls it "fiscal 2026"), so count "fiscal 20xx" in the 10-K text."""
    if period_end.month > 3:
        return dei_year
    try:
        html = _get(f.url)
    except Exception:
        return dei_year
    counts = {y: len(re.findall(rf"(?i)\bfiscal(?:\s+year)?\s+{y}\b", html))
              for y in (period_end.year, period_end.year - 1)}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 5 and counts[best] > 1.5 * counts[min(counts, key=counts.get)] else dei_year


def html_to_text(html: str, limit: int = 60000) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", html)
    html = re.sub(r"(?i)</t[dh]>", " | ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;|&#xa0;", " ", text)
    text = (text.replace("&amp;", "&").replace("&#8217;", "'").replace("&#8220;", '"')
            .replace("&#8221;", '"').replace("&#8212;", "—").replace("&#8211;", "–"))
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text[:limit]


def section(f: Filing, start_rx: str, end_rx: str, limit: int = 30000) -> str:
    """Text of a 10-K / 10-Q section (e.g. MD&A). The table of contents names every item first, so
    the section is taken from the occurrence of the heading followed by the longest body."""
    text = html_to_text(_get(f.url), limit=10 ** 8)
    best = ""
    for m in re.finditer(start_rx, text):
        rest = text[m.start():]
        e = re.search(end_rx, rest[200:])
        body = rest[:200 + e.start()] if e else rest
        if len(body) > len(best):
            best = body
    return best[:limit]


DEAL_NOTE_RX = re.compile(r"(?i)(agreement to acquire|entered into an? (definitive |merger )?agreement|acquired all of the|"
                          r"acquisition.date fair value|consideration transferred|pending acquisitions?|business "
                          r"combinations?|merger agreement|expected to close|accelerated share repurchase|asr agreements?|"
                          r"initial delivery|final settlement|divestiture|held for sale)")


def notes_excerpt(f: Filing, limit: int = 7000) -> str:
    """The financial-statement notes on deals and share repurchases (prices, close dates, ASR terms): PM round 3 —
    CRM's Q2 10-Q disclosed Contentful (~$1.5bn) and Fin (~$3.6bn), but the reading pack had only the MD&A, so the
    analyst wrote "consideration not disclosed"."""
    lines = html_to_text(_get(f.url), limit=10 ** 8).splitlines()
    prose = lambda ln: "|" not in ln and len(ln) > 100  # noqa: E731 - sentences, not table rows that name a line
    hits = [i for i, ln in enumerate(lines) if prose(ln) and DEAL_NOTE_RX.search(ln) and
            not re.search(r"(?i)uses its best estimates|inherently uncertain|measurement period", ln)]
    # the short heading above a paragraph names the deal ("Contentful"); the sentence after carries the terms
    keep = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(lines) and
                   (j == i or (j < i and len(lines[j].strip()) < 60 and "|" not in lines[j]) or
                    (j > i and prose(lines[j])))})
    out, n = [], 0
    for j in keep:
        ln = lines[j].strip()
        if not ln or ln in out:
            continue
        out.append(ln[:900])
        n += len(out[-1])
        if n > limit:
            break
    return "\n".join(out)


AGENCY_RX = {"Moody's": re.compile(r"(?i)moody.?s"), "S&P": re.compile(r"(?i)\bS\s?&\s?P\b|standard\s*(&|and)\s*poor"),
             "Fitch": re.compile(r"(?i)\bfitch\b")}
MOODYS_TOK = re.compile(r"(?<![\w-])(Aaa|Aa[1-3]|A[1-3]|Baa[1-3]|Ba[1-3]|B[1-3])(?![\w-])")
SP_TOK = re.compile(r"(?<![\w-])(AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?)(?![\w+-]|\d)")


def credit_ratings(f: Filing) -> dict:
    """Long-term senior unsecured ratings the filer discloses (liquidity section or a ratings table), by agency
    (PM round 4: a synthetic AAA from interest coverage for all six names, actual AA to A)."""
    text = html_to_text(_get(f.url), limit=10 ** 8)
    out = {}
    for m in re.finditer(r"(?i)credit ratings?|debt ratings?|rated\b|ratings? (assigned|of our)", text):
        win = text[max(0, m.start() - 300): m.end() + 900]
        if re.search(r"(?i)short[- ]term|commercial paper", win[:120]) and not re.search(r"(?i)long[- ]term", win):
            continue
        for ag, rx in AGENCY_RX.items():
            if ag in out:
                continue
            for am in rx.finditer(win):
                seg = win[am.end(): am.end() + 140]
                tok = (MOODYS_TOK if ag == "Moody's" else SP_TOK).search(seg)
                if tok and not re.search(r"(?i)short[- ]term|commercial paper|outlook", seg[:max(tok.start(), 1)]):
                    out[ag] = tok.group(1)
                    out.setdefault("text", re.sub(r"\s+", " ", win[max(0, am.start() - 80): am.end() + 160]).strip())
                    break
        if len(out) >= 3:
            break
    return out


def analyst_documents(co: Company, fs: FilingSet, as_of: date, texts: dict) -> dict:
    """Reading pack for the AI analyst: earnings releases, MD&A, business overview, deal 8-Ks."""
    docs = {}
    for key, (f, t) in (texts or {}).items():
        # the latest release carries current guidance; the full-year one mostly repeats what extraction put in
        # the brief, so it is cut shorter
        label = "Full-year results release" if key == "fy_release" else "Latest quarter results release"
        docs[f"{label} (8-K EX-99.1 filed {f.filing_date})"] = t[:12000] if key == "fy_release" else t[:26000]
    mdna = r"(?i)(item\s*[27]\.?[\s|:.—–-]*)?management.s discussion and analysis"
    try:
        docs[f"10-K Item 1 Business (filed {fs.annual.filing_date})"] = section(
            fs.annual, r"(?i)item\s*1\.?[\s|:.—–-]*business\b", r"(?i)item\s*1a\.?[\s|:.—–-]*risk factors|risk factors",
            5000)
        docs[f"10-K Item 7 MD&A (filed {fs.annual.filing_date})"] = section(
            fs.annual, mdna, r"(?i)quantitative and qualitative disclosures|item\s*8\.?[\s|:.—–-]*financial statements",
            18000)
    except Exception:
        pass
    if fs.quarterly:
        try:
            docs[f"10-Q Item 2 MD&A ({fs.quarterly.report_date}, filed {fs.quarterly.filing_date})"] = section(
                fs.quarterly, mdna, r"(?i)quantitative and qualitative disclosures|controls and procedures", 16000)
        except Exception:
            pass
    for f, lim in ((fs.quarterly, 7000), (fs.annual, 5000)):
        if not f:
            continue
        try:
            t = notes_excerpt(f, lim)
            if t:
                docs[f"{f.form} notes: acquisitions, pending deals, share repurchases ({f.report_date}, filed "
                     f"{f.filing_date})"] = t
        except Exception:
            pass
    since = fs.annual.filing_date - timedelta(days=45)
    events = [f for f in co.filings if f.form == "8-K" and since <= f.filing_date <= as_of
              and re.search(r"\b(1\.01|2\.01|2\.05|2\.06|7\.01|8\.01)\b", f.items) and "2.02" not in f.items]
    if events:
        docs["8-K list since the 10-K (date: items)"] = "\n".join(f"{f.filing_date}: items {f.items}"
                                                                   for f in events[:25])
    for f in events[:5]:  # the press release if there is one, else the 8-K text describing the items
        try:
            url = exhibit_99_url(f, strict=True) or f.url
            t = html_to_text(_get(url), limit=2500)
            docs[f"8-K filed {f.filing_date} (items {f.items}) {url}"] = t
        except Exception:
            continue
    return {k: v for k, v in docs.items() if v and v.strip()}


def release_text(f: Filing) -> tuple[Optional[str], Optional[str]]:
    url = exhibit_99_url(f)
    if not url:
        return None, None
    return url, html_to_text(_get(url))


def fetch_instance(f: Filing) -> bytes:
    try:
        return _get(f.instance_url, binary=True)
    except requests.HTTPError:
        # older (non-inline) filings: find the instance in the index
        for it in filing_index(f):
            n = it["name"]
            if n.endswith(".xml") and not re.search(r"_(cal|def|lab|pre)\.xml$|FilingSummary", n):
                return _get(f.folder + n, binary=True)
        raise
