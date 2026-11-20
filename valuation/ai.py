"""Optional AI helpers (OpenAI): read earnings releases, suggest peers, read peer guidance,
run the analyst pass and its review, draft the thesis.

Everything here is optional. Without OPENAI_API_KEY the model still builds:
guidance cells stay blank for the analyst, peers come from the user, and the
thesis tab gets a template. The analyst pass returns an overrides document
(valuation/overrides.py), the same format an external agent produces.

Models (Streamlit secrets or environment):
    OPENAI_MODEL              extraction (numbers from releases)    default gpt-4.1-mini (fast, no reasoning)
    OPENAI_ANALYST_MODEL      judgment: assumptions, peers, thesis   default gpt-5.5
    OPENAI_REASONING_EFFORT   analyst & review reasoning effort      default medium (high: ~1.7x the cost)
    OPENAI_MAX_COST_PER_RUN   budget in $ for one build; optional AI steps are skipped once spent
                              (default by effort: low 1.00, medium 2.50, high 4.00, so the review passes still run)

Every call's token usage is recorded (`USAGE`) and priced with PRICES (per 1M tokens; reasoning tokens are
billed as output) so the app and CLI can show what a build cost.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from typing import Optional

FALLBACK_MODEL = "gpt-4.1"  # used when the configured model is not available on the key
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
# $ per 1M tokens (input, cached input, output) — OpenAI list prices; verify at openai.com/api/pricing
PRICES = {"gpt-5.5": (5.00, 0.50, 30.00), "gpt-5.4": (2.50, 0.25, 15.00), "gpt-5.4-mini": (0.75, 0.075, 4.50),
          "gpt-4.1": (2.00, 0.50, 8.00),
          "gpt-4.1-mini": (0.40, 0.10, 1.60), "gpt-4o-mini": (0.15, 0.075, 0.60), "gpt-4o": (2.50, 1.25, 10.00)}
USAGE: list = []  # one row per call: {what, model, input, cached, output, reasoning, seconds, usd}
# finished answers (and in-flight background response ids) by request: an interrupted or repeated build
# resumes or reuses them instead of paying again. OPENAI_CACHE=0 turns it off.
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fa_webapp_ai_cache")
# output-token caps (reasoning + answer) by effort: a stuck or rambling call cannot run up the bill
OUTPUT_CAP = {"low": 20000, "medium": 40000, "high": 64000}
QUEUE_LIMIT = 90  # seconds a background response may sit queued before the request is streamed directly


def reset_usage():
    USAGE.clear()


def spent() -> float:
    return sum(r["usd"] or 0 for r in USAGE)


BUDGET = {"low": 1.00, "medium": 2.50, "high": 4.00}  # default $ cap per build: a high-effort analyst pass alone
# can cost about $1, and a cap below that would silently skip the review passes that clear failed gates


def budget() -> float:
    default = BUDGET.get(_effort(), 1.00)
    try:
        return float(_secret("OPENAI_MAX_COST_PER_RUN") or default)
    except ValueError:
        return default


def usage_summary() -> str:
    if not USAGE:
        return "no AI calls"
    ins = sum(r["input"] for r in USAGE)
    outs = sum(r["output"] for r in USAGE)
    rea = sum(r["reasoning"] for r in USAGE)
    return (f"~${spent():.2f} for {len(USAGE)} AI calls ({ins:,} input / {outs:,} output tokens, "
            f"of which {rea:,} reasoning)")


def _record(what: str, model: str, usage, seconds: float):
    if usage is None:
        return
    g = lambda o, *names: next((getattr(o, n) for n in names if getattr(o, n, None) is not None), 0)  # noqa: E731
    inp = g(usage, "input_tokens", "prompt_tokens") or 0
    out = g(usage, "output_tokens", "completion_tokens") or 0
    det_in = g(usage, "input_tokens_details", "prompt_tokens_details")
    det_out = g(usage, "output_tokens_details", "completion_tokens_details")
    cached = (g(det_in, "cached_tokens") if det_in else 0) or 0
    reasoning = (g(det_out, "reasoning_tokens") if det_out else 0) or 0
    base = next((k for k in sorted(PRICES, key=len, reverse=True) if model.startswith(k)), None)
    usd = None
    if base:
        pi, pc, po = PRICES[base]
        usd = ((inp - cached) * pi + cached * pc + out * po) / 1e6
    USAGE.append({"what": what, "model": model, "input": int(inp), "cached": int(cached), "output": int(out),
                  "reasoning": int(reasoning), "seconds": round(seconds, 1), "usd": usd})


def _secret(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def client():
    key = _secret("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key, timeout=300, max_retries=1)
    except Exception:
        return None


def fatal(e: Exception) -> bool:
    """Errors that will repeat on every call (no credits, bad key): stop using AI for this run."""
    t = f"{type(e).__name__} {e}".lower()
    return any(k in t for k in ("insufficient_quota", "credit", "authentication", "invalid_api_key", "401",
                                "permissiondenied"))


def _model() -> str:
    return _secret("OPENAI_MODEL") or "gpt-4.1-mini"


def analyst_model() -> str:
    return _secret("OPENAI_ANALYST_MODEL") or "gpt-5.5"


def _effort() -> str:
    """Judgment passes (analyst, review) run at medium effort by default. Measured on CRM (2026-11-14, same engine,
    same peers): gpt-5.5 medium $1.97 with two review passes, the right adjusted-EPS definition and no plug;
    high $3.28 with one review pass (the budget ran out) and two failed gates; low (the v1 demos the PM
    critiqued) barely reasons. gpt-5.4-mini at high ran out of output tokens without an answer. Extraction steps
    stay at low effort."""
    return _secret("OPENAI_REASONING_EFFORT") or "medium"


def _parse(txt: str) -> dict:
    txt = (txt or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, re.S)
    if m:
        txt = m.group(1)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            return json.loads(txt[start:end + 1])
        raise


def _ask(cli, system: str, user: str, max_tokens: int = 1800, model: Optional[str] = None,
         effort: Optional[str] = None, progress=None, what: str = "", max_out: Optional[int] = None,
         limit: int = 600) -> dict:
    """One JSON call.

    Reasoning models (gpt-5*, o*) run through the Responses API in background mode and are polled, so a
    long analysis never hits an HTTP timeout (a timed-out request would be retried from scratch).
    Other models (and test doubles without `.responses`) use a plain chat completion.
    """
    model = model or _model()
    try:
        if model.startswith(REASONING_PREFIXES) and hasattr(cli, "responses"):
            txt, why = _background(cli, model, system, user, max_out or max(4 * max_tokens, 8000), effort or "low",
                                   progress, what, limit)
        else:
            txt, why = _chat(cli, model, system, user, max_tokens, effort, what)
        used = model if not model.startswith(REASONING_PREFIXES) else f"{model}, {effort or 'low'} effort"
    except Exception as e:
        if fatal(e) or model == FALLBACK_MODEL:
            raise
        if progress:
            progress(f"{model} failed ({str(e)[:80]}); retrying with {FALLBACK_MODEL}")
        txt, why = _chat(cli, FALLBACK_MODEL, system, user, max_tokens, effort, what)  # not enabled / timed out
        used = f"{FALLBACK_MODEL} after {model} failed"
    try:
        out = _parse(txt)
    except json.JSONDecodeError:
        raise ValueError(f"model returned invalid JSON{why}")
    if isinstance(out, dict):
        out["__model__"] = used
    return out


def _cache_path(cli, *parts) -> Optional[str]:
    if (_secret("OPENAI_CACHE") or "1") == "0" or not type(cli).__module__.startswith("openai"):
        return None  # test doubles and opted-out runs always call through
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest() + ".json")


def _cache_get(path) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return {}


def _cache_put(path, **kw):
    if path:
        with open(path, "w") as fh:
            json.dump(kw, fh)


def _from_cache(what: str, model: str, hit: dict):
    USAGE.append({"what": what + " (reused)", "model": model, "input": 0, "cached": 0, "output": 0, "reasoning": 0,
                  "seconds": 0.0, "usd": 0.0})
    return hit["text"], hit.get("why", "")


def _chat(cli, model, system, user, max_tokens, effort, what=""):
    path = _cache_path(cli, "chat", model, effort, max_tokens, system, user)
    hit = _cache_get(path) if path else {}
    if hit.get("text"):
        return _from_cache(what, model, hit)
    t0 = time.time()
    kw = dict(model=model, response_format={"type": "json_object"},
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if model.startswith(REASONING_PREFIXES):
        kw.update(max_completion_tokens=max(4 * max_tokens, 8000), reasoning_effort=effort or "low")
    else:
        kw.update(temperature=0.1, max_tokens=min(max_tokens, 16000))
    resp = cli.chat.completions.create(**kw)
    _record(what, model, getattr(resp, "usage", None), time.time() - t0)
    choice = resp.choices[0]
    cut = " (output cut off: raise the token budget)" if getattr(choice, "finish_reason", "") == "length" else ""
    if not cut:
        _cache_put(path, text=choice.message.content)
    return choice.message.content, cut


def _awake() -> float:
    """Seconds the machine has been awake: a timeout must not trip while a laptop sleeps (macOS monotonic stops in
    sleep). A response cancelled on wall time was paid for and lost (HD, round 5: lid closed mid-review)."""
    return time.monotonic()


def _retrieve(cli, rid, tries: int = 6):
    """Poll one background response; a dropped connection (sleep, network change) is retried, not fatal —
    the response keeps running on OpenAI's side."""
    for k in range(tries):
        try:
            return cli.responses.retrieve(rid, timeout=60)
        except Exception as e:
            if fatal(e) or k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def _background(cli, model, system, user, max_out, effort, progress, what="", limit=600):
    t0 = _awake()
    path = _cache_path(cli, "responses", model, effort, max_out, system, user)
    hit = _cache_get(path) if path else {}
    if hit.get("text"):
        return _from_cache(what, model, hit)
    r = None
    if hit.get("id"):  # an earlier run was interrupted while this request was in flight: pick it up
        try:
            r = _retrieve(cli, hit["id"])
            if progress:
                progress(f"… resuming {model} response started by an earlier run")
            if r.status in ("failed", "cancelled", "expired"):
                r = None
        except Exception:
            r = None
    if r is None:
        r = cli.responses.create(model=model, input=[{"role": "system", "content": system},
                                                     {"role": "user", "content": user}],
                                 reasoning={"effort": effort}, text={"format": {"type": "json_object"}},
                                 max_output_tokens=max_out, extra_body={"background": True}, timeout=120)
        _cache_put(path, id=r.id)
    last = t0
    while r.status in ("queued", "in_progress"):
        if r.status == "queued" and _awake() - t0 > QUEUE_LIMIT:
            # OpenAI's background queue can back up for many minutes while direct requests run at once:
            # a queued response has used no tokens, so cancel it and stream the same request instead
            try:
                cli.responses.cancel(r.id)
            except Exception:
                pass
            if progress:
                progress(f"… {model} background queue is slow; streaming the request directly")
            txt, why = _stream(cli, model, system, user, max_out, effort, progress, what, limit - (_awake() - t0))
            _cache_put(path, text=txt, why=why)
            return txt, why
        if _awake() - t0 > limit:
            try:
                cli.responses.cancel(r.id)
                _record(what + " (cancelled)", model, getattr(_retrieve(cli, r.id), "usage", None),
                        _awake() - t0)
            finally:
                raise TimeoutError(f"{model} did not finish within {limit // 60} minutes")
        time.sleep(3)
        r = _retrieve(cli, r.id)
        if progress and _awake() - last >= 30:
            last = _awake()
            progress(f"… {model} still reasoning ({_awake() - t0:.0f}s)")
    _record(what, model, getattr(r, "usage", None), _awake() - t0)
    if r.status == "completed" and r.output_text:
        _cache_put(path, id=r.id, text=r.output_text)
    if r.status != "completed":
        detail = getattr(r, "incomplete_details", None) or getattr(r, "error", None)
        if r.status == "incomplete" and r.output_text:
            return r.output_text, f" (incomplete: {detail})"
        raise RuntimeError(f"{model} response {r.status}: {detail}")
    return r.output_text, ""


def _stream(cli, model, system, user, max_out, effort, progress, what="", limit=600):
    """The same request as a streamed (synchronous) response: events keep the connection alive through a long
    reasoning phase, and the final event carries the usage."""
    t0, last, final = _awake(), _awake(), None
    events = cli.responses.create(model=model, input=[{"role": "system", "content": system},
                                                      {"role": "user", "content": user}],
                                  reasoning={"effort": effort}, text={"format": {"type": "json_object"}},
                                  max_output_tokens=max_out, stream=True, timeout=max(60, limit))
    for ev in events:
        typ = getattr(ev, "type", "")
        if typ in ("response.completed", "response.incomplete", "response.failed"):
            final = ev.response
        if progress and _awake() - last >= 30:
            last = _awake()
            progress(f"… {model} still reasoning ({_awake() - t0:.0f}s, streamed)")
    if final is None:
        raise RuntimeError(f"{model} stream ended without a final response")
    _record(what, model, getattr(final, "usage", None), _awake() - t0)
    if final.status != "completed":
        detail = getattr(final, "incomplete_details", None) or getattr(final, "error", None)
        if final.output_text:
            return final.output_text, f" (incomplete: {detail})"
        raise RuntimeError(f"{model} response {final.status}: {detail}")
    return final.output_text, ""


EXTRACT_SYS = (
    "You extract numbers from company earnings press releases for a financial model. "
    "Return strict JSON. Use null for anything not stated explicitly. Percentages as decimals "
    "(4% -> 0.04). Dollar amounts in $ millions (e.g. $5.4 billion -> 5400). Never estimate.")


def extract_releases(cli, company: str, fy: int, interim, seg_names: list, texts: dict) -> dict:
    """texts: {'fy_release': (Filing, text), 'q_release': (Filing, text)}"""
    parts = []
    for key, (f, text) in texts.items():
        label = f"FY{fy} results release" if key == "fy_release" else f"{interim.quarter_label} results release"
        parts.append(f"=== {label} (filed {f.filing_date}) ===\n{text[:28000]}")
    q = interim.quarter_label if interim.available else None
    schema = {
        "release_date": "date of the most recent release with full-year guidance, e.g. 'July 24, 2025'",
        "guidance": {
            "_comment": f"latest full-year guidance for FY{fy + 1} (low, high) from the MOST RECENT release. Companies "
                        "often guide both GAAP and non-GAAP figures: keep them apart",
            "sales": "[low, high] $mm or null",
            "organic": "[low, high] decimal ONLY when the release calls it organic growth (excluding acquisitions "
                       "and currency); reported or constant-currency growth that includes acquired revenue ('up "
                       "11%-12% Y/Y ... including 3pts Informatica contribution') is not organic: null",
            "margin": "[low, high] NON-GAAP / adjusted operating margin or segment margin, decimal; null if only a "
                      "GAAP margin is guided",
            "gaap_margin": "[low, high] GAAP operating margin decimal or null",
            "adj_eps": "[low, high] NON-GAAP / adjusted EPS $; null if only GAAP EPS is guided",
            "gaap_eps": "[low, high] GAAP EPS $ or null",
            "fcf": "[low, high] $mm or null (null when not guided; never 0 as a placeholder)",
            "net_income": "[low, high] $mm when the company guides net income instead of EPS ('net income guidance "
                          "$4.75 billion to $5.00 billion' = [4750, 5000]); never put it in adj_eps"},
        "adjusted_eps": {
            "_comment": "NON-GAAP / adjusted diluted EPS as the release reports it; every field null if the release "
                        "has no non-GAAP EPS (never put GAAP EPS here)",
            "fy": f"full fiscal year FY{fy} (twelve months, not year to date)",
            "fy_prior": f"full fiscal year FY{fy - 1}",
            "ytd_cur": f"year-to-date {interim.label_cur} (sum of quarters) or null",
            "ytd_prior": f"year-to-date {interim.label_prior} or null"},
        "organic_fy_total": f"FY{fy} total organic sales growth",
        "organic_fy_by_segment": {n: f"FY{fy} organic growth" for n in seg_names},
        "organic_q_total": f"{q} quarter organic growth" if q else None,
        "organic_q_by_segment": {n: f"{q} organic growth" for n in seg_names} if q else {},
    }
    user = (f"Company: {company}. Segments: {', '.join(seg_names)}.\n"
            f"Return JSON with exactly these keys:\n{json.dumps(schema, indent=1)}\n\n" + "\n\n".join(parts))
    out = _ask(cli, EXTRACT_SYS, user, 2000, what="read earnings releases")
    g = out.get("guidance") or {}
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
    out["guidance"] = {k: g.get(k) for k in ("sales", "organic", "margin", "adj_eps", "fcf", "net_income")
                       if isinstance(g.get(k), list) and len(g[k]) == 2 and all(num(x) for x in g[k])}
    adj = out.get("adjusted_eps") if isinstance(out.get("adjusted_eps"), dict) else {}
    out["adjusted_eps"] = {k: adj.get(k) if num(adj.get(k)) else None for k in ("fy", "fy_prior", "ytd_cur", "ytd_prior")}
    for k in ("organic_fy_total", "organic_q_total"):  # numbers only: models sometimes nest or comment
        if not num(out.get(k)):
            out[k] = None
    for k in ("organic_fy_by_segment", "organic_q_by_segment"):
        out[k] = {n: v for n, v in (out.get(k) or {}).items() if num(v)} if isinstance(out.get(k), dict) else {}
    return out


def suggest_peers(cli, company: str, ticker: str, seg_names: list, industry: Optional[str], as_of=None) -> dict:
    when = f" on {as_of}" if as_of else " today"
    sys = ("You are an equity research analyst. Return strict JSON. Use only US-listed tickers that "
           f"traded on NYSE/NASDAQ{when} and file 10-K/10-Q reports.")
    user = (f"Company: {company} ({ticker}); industry: {industry}; segments: {', '.join(seg_names)}.\n"
            "Pick 6 listed trading comparables (closest business overlap). For each segment give a reasonable "
            "base-case EV/EBITDA multiple from its peer group and a 6-12 word note naming those peers.\n"
            'JSON: {"peers": [{"ticker": "...", "overlap": "segment(s) it competes with"}], '
            '"segment_multiples": {"<segment>": 14.0}, "segment_notes": {"<segment>": "..."}}')
    return _ask(cli, sys, user, 1200, model=analyst_model(), effort="low", what="choose peers", max_out=6000)


def write_thesis(cli, spec) -> dict:
    h = spec.hist
    I = h.IS
    facts = {
        "company": spec.name, "ticker": spec.ticker, "as_of": str(spec.as_of), "price": spec.price,
        "fiscal_years": spec.fy_hist, "sales": I["sales"], "operating_income": I["opinc"],
        "net_income": I["ni_attr"], "diluted_eps": I["eps_dil"],
        "segments": {sg.name: {"sales": sg.sales, "profit": sg.profit} for sg in spec.segments},
        "cash_from_ops": h.CF["cfo"], "capex": h.CF["capex"],
        "net_debt_latest": (spec.interim.bs_cur.get("st_debt", 0) + spec.interim.bs_cur.get("cur_ltd", 0)
                            + spec.interim.bs_cur.get("ltd", 0) - spec.interim.bs_cur.get("cash", 0)),
        "guidance": spec.guidance, "base_case_growth": {sg.name: sg.growth["base"] for sg in spec.segments},
        "base_case_margin": {sg.name: sg.margin["base"] for sg in spec.segments},
        "peers": [p["ticker"] for p in spec.peers], "data_flags": spec.flags[:6],
    }
    sys = ("You are a buy-side analyst writing the thesis tab of a valuation model for a PM. Be specific, "
           "use the facts given, cite numbers, no hype. Do not state the model's fair value (it is linked "
           "live in the sheet). Return strict JSON.")
    user = ("Facts (USD mm):\n" + json.dumps(facts, default=str) + "\n\n"
            'JSON: {"stance": "long|short|neutral on", "recommendation": "1-2 sentences", '
            '"points": ["1. ...", "2. ...", "3. ...", "4. ..."] (each <= 230 chars), '
            '"model_says": ["• ...", "• ...", "• ..."], "risks": ["• ...", "• ...", "• ...", "• ..."], '
            '"monitor": ["• ...", "• ..."]}')
    out = _ask(cli, sys, user, 1600, model=analyst_model(), effort="low", what="draft thesis", max_out=8000)
    used = out.pop("__model__", analyst_model())
    out.setdefault("stance", "long")
    out["stance"] = out["stance"].replace(" on", "").strip() or "long"
    for k, n in (("points", 4), ("model_says", 3), ("risks", 4), ("monitor", 2)):
        out[k] = [str(x) for x in (out.get(k) or [])][:n] or ["[Analyst to complete]"]
    out["recommendation"] = str(out.get("recommendation") or "[Analyst to complete]")
    out["about"] = (f"Narrative drafted by AI ({used}) from the filings and the model's default drivers on "
                    f"{spec.as_of:%b %-d, %Y}; the numbers above are live links. Verify before use.")
    return out


def peer_guidance(cli, company: str, text: str, filed) -> dict:
    """A peer's full-year adjusted EPS guidance from its earnings release (point-in-time comps)."""
    user = (f"Company: {company}. Earnings release filed {filed}.\n"
            'Return JSON: {"fiscal_year": 2025, "adj_eps_guidance": [low, high] or null, '
            '"basis": "adjusted / GAAP, as stated"}. Use the latest full-year EPS guidance range in the release '
            "(a point estimate -> [x, x]); null if none is given.\n\n" + text[:24000])
    return _ask(cli, EXTRACT_SYS, user, 400, what=f"peer guidance {company}")


ANALYST_SYS = (
    "You are the senior buy-side analyst building a valuation model for a portfolio manager. A data pipeline "
    "has already parsed the SEC filings into a balanced 3-statement / DCF / comps / SOTP model with mechanical "
    "defaults; the brief shows the facts, the defaults and what the default model computes. Your job is the "
    "judgment layer: replace defaults with specific, defensible assumptions, each with a short note citing its "
    "basis (document + figure: '10-Q Q3 2026 MD&A: backlog $12.1bn +9%'). Rules: use ONLY information "
    "available on or before the valuation date (the documents provided and well-known facts dated before it); "
    "market data in the brief (share price, 10-yr Treasury yield, share count) are facts on the date: keep them; "
    "never change history (adj_eps_tax_hist stays within 3pt of the non-GAAP tax rate the releases state); a deal "
    "that closed before the balance-sheet date is in net debt and the forecast already, not a 'deals' item; when the "
    "price implies about zero growth, the bear is an explicit structural-decline case with its probability; "
    "count each risk once: a scenario risk (disruption, competition, a decline) lives in the bear case and its "
    "probability only — beta, terminal growth and fade years move for measurement reasons (leverage, business mix), "
    "never for that risk, and the base sits at consensus unless a variant view distinct from the bear's risk says "
    "otherwise; no input or explanation may give the skew, the gap or the rating as its reason (both are rejected); "
    "a CEO or CFO change within 12 months is a dated catalyst and the first risk (a new CEO often resets guidance); "
    "when the company trades 30%+ below the peer median, comps weigh 10% or less and the discount "
    "({peer_discount_pe}) is the variant view, with the return coming from the business, not a re-rating; frame the "
    "call as one number in thesis.price_implies: the bear probability the price implies ({p_bear_implied}) against "
    "yours, and where the call changes ({p_bear_long}); quote the probability-weighted 3-year return ({ret3y_pw}); "
    "drivers.shares_issued = stock comp ÷ the repurchase price; "
    "scenario probabilities are yours, with a basis in thesis.explanations['scenario probabilities'] (numbers: "
    "pipeline probabilities of success for a drug company, how often past cycles reached the bear for a cyclical) — "
    "never a template; a decline still running in FY6 (a loss of exclusivity) continues in fade_path to its trough, "
    "with terminal growth at the house rate; acquisitions paid this year whose cost was expensed as acquired IPR&D "
    "stay out of drivers.acq (say so in its note: the cost is already in EBIT); a company that guides net income "
    "goes in guidance.net_income, not adj_eps; frame the call on the side that can be reached ({p_bear_long} for a "
    "Long, {p_bear_short} for a Short); "
    "year 1 must land inside company guidance where guidance exists; bear < base < bull "
    "with stated reasons; growth, margins, multiples and the thesis must be consistent with each other. "
    "Never remove a real cost to hit a non-GAAP target: stock comp, amortization and recurring restructuring "
    "stay in EBIT and the DCF. Match the company's adjusted-EPS definition with the switches instead "
    "(adj_eps_ex_sbc = 1 when its non-GAAP EPS excludes stock comp, adj_eps_ex_repo = 1 when it also excludes "
    "restructuring / acquisition charges; adjusted_eps = null when the company reports no adjusted EPS — never "
    "copy GAAP or year-to-date figures into it) and treat one-off gains (e.g. investment "
    "gains) as non-recurring, not as run-rate other income: put one-offs already booked this year in drivers.nonrec "
    "(FY1) so GAAP EPS fits the year to date and adjusted EPS excludes them. "
    "Model the business the way its industry is analysed (the brief's industry section): set growth by revenue "
    "line (segments.<name>.lines) where lines are given, add a volume x price KPI build when the brief lists KPI "
    "history (store / warehouse counts) or the company discloses it, use fade_years for long growth runways, and "
    "treat captive-finance debt as operating. drivers.acq is cash paid for business combinations only (>= 0): "
    "an asset deal whose price is expensed as acquired IPR&D is already a charge (repo_pct) and must not be "
    "entered again; debt repayment cannot exceed the debt outstanding. Peer prices come from market data: give "
    "each peer's EPS and its "
    "source, never a price. "
    "The rating comes from the engine's rule, not from you: Long only when the probability-weighted value is 15%+ "
    "above the price (the model's error band) AND the price offers a long-run IRR (reverse DCF) 2pt+ above WACC "
    "with 2:1 skew and the price inside bear-bull; Short is the mirror image and also needs a dated catalyst, "
    "thesis.short_evidence (when the premium last compressed, by how much, on what — a premium that has persisted "
    "for a decade is a valuation view, not a short), a crowding check, and for a low-volatility compounder (beta "
    "under 0.85, low revenue sigma, high ROIC) a pair (thesis.pair: the long leg); everything else is Neutral with "
    "a bias. The recommendation's call (Long / Short / Neutral and its bias) is generated from the rule: write "
    "thesis.recommendation as one supporting sentence with no call in it, and never state a call or bias anywhere "
    "in the thesis that differs from the rule (gate 25: 'Long', 'Underweight', 'avoid adding', 'warrants "
    "ownership', a positive bias under a negative rule) — to disagree, write rating_override {rating, reason}, "
    "which is labeled an override and logged. For a liquid large cap the price sits inside bear-bull (gate 41): "
    "a bull below the price means the scenarios are miscalibrated, so build the bull from what the price requires "
    "(the reverse DCF: the terminal margin or years of growth it needs) or the Street bull; the engine rebuilds a "
    "bull below the price to the reverse-DCF margin. "
    "Base case = consensus (the brief gives FY1 / FY2 consensus) unless variant_view says why it differs and when "
    "that is tested; a management long-term target is the bull case, never the base anchor. Organic growth "
    "excludes acquired revenue until it laps a full year: deals closed during FY0 annualize in FY1 as segments mna "
    "(gate 36), never as organic or line growth. The bear case is the thesis breaker made numerical (its growth, "
    "margin and stock-comp levels), at least 15% below base, its growth about one historical standard deviation "
    "below base (the brief gives the 10-year sigma; gate 33). Terminal RONIC for a company whose growth was bought "
    "(goodwill and acquired intangibles a large share of capital, or acquisitions 2%+ of sales over 10 years) is "
    "at or below the return including goodwill (DCF C93), unless drivers.acq forecasts recurring acquisition "
    "spend (the 10-year average share of sales) with the growth it buys (gate 34). Terminal RONIC defaults to that "
    "return including goodwill (DCF C93): set it lower only for a dated sector reason — loss of exclusivity (the "
    "product and the year), competitive entry, recurring acquisitions not otherwise modeled — explained under "
    "'gate 34' with the year; never cut it for growth whose cost is already expensed in EBIT (business development, "
    "acquired IPR&D). Cyclicals end at mid-cycle from a peak as from a trough (gate 16: the 10-year median margin); "
    "a housing-linked retailer at a turnover trough carries the recovery in the base case or gives it an explicit "
    "probability. Pharma: products losing exclusivity inside or soon after the forecast erode in the lines (with "
    "the dates), and the terminal reflects the post-expiry mix and the pipeline that replaces it. Deals that closed "
    "before the valuation date are not pending: their cash is paid (drivers.acq) and their revenue is in mna. The "
    "rest of FY1 (forecast less year to date) looks like last year's (gate 43): a year-end working-capital ratio "
    "applied to a fast-growing year must not build working capital the 10-Q does not show. A KPI build ties to "
    "modeled revenue within 1% (gate 44). Buybacks above free cash flow "
    "are debt-funded: say so in the thesis (gate 39). Peers must "
    "be genuinely comparable (no acquisition targets, no outliers): a peer whose next-year consensus EPS falls 20%+ "
    "(one-offs in this year's EPS) or whose P/E is under a third or over 3x the others' is excluded automatically "
    "— fix its EPS (company-adjusted, in the currency of its price) to bring it back, and never claim a fix the "
    "data do not show (gate 42). Guidance is the company's stated figures: never "
    "restate it onto the model's basis to pass a check — explain a definitional gap in flags instead. "
    "Cost of capital is centralized: the risk-free rate, ERP and credit spreads are the same for every name on "
    "the date and you cannot change them; beta comes from the method (move it at most ±0.10, with a reason); the "
    "cost of debt follows the agency rating the filings disclose. "
    "Never set any input by reference to the share price: the model tests the price, it does not chase it. "
    "Calibrate to guidance only through operating drivers (margins, price / mix, organic growth, working "
    "capital, share count timing) — never tax, other income or a generic 'other' line — and make FY2 follow "
    "from FY1 without a plug unwinding. Reproduce the company's adjusted-EPS definition with the switches so "
    "history ties within $0.05; drivers.adj_tax is the non-GAAP tax rate the company states, never a rate solved to "
    "hit an EPS guide (it drives operating EPS and so every multiple: a plug there fails gate 11). Forecast from the latest 10-Q (debt, buybacks and acquisitions already done "
    "this year are in FY1). Every signed pending acquisition or separation goes in 'deals', whatever its size (a small "
    "deal still moves value by its premium), with a probability, its own revenue and margin and a peer multiple — "
    "never in the segment forecast; the target's EBITDA and margin are its reported figures (or labeled estimates), "
    "never backed out so the cross-check ties; synergies get 50% credit with the run-rate in year 3, at the lower of "
    "the deal multiple and the company's own EV / EBITDA, integration costs after tax (if every deal creates value, "
    "recheck each input: gate 38); acquisitions whose growth you forecast must be paid for in drivers.acq (a price "
    "not disclosed is estimated at the target's revenue x a peer EV / sales multiple, and said so); working-capital "
    "days and ratios are year-end balances: every forecast year ends on the fiscal year-end, so they start from the "
    "10-K's year-end level, never the 10-Q's seasonal one; the engine returns cash above FY0's share of sales as "
    "buybacks from FY2 (to hold it, explain under 'cash build'); separations "
    "forecast the remaining company plus a 'deals' item for what holders receive. Terminal value is NOPAT x "
    "(1 - g / RONIC) / (WACC - g); terminal growth defaults to the house 3.75% (long-run nominal GDP: go lower "
    "only for an industry reason, never toward the price): set RONIC so the terminal year is believable against "
    "10 years of margin history (cyclicals at mid-cycle); leave exit_mult derived or keep it within ±20%. "
    "Scenarios move the thesis-critical variables. The thesis answers the seven PM questions (price_implies, "
    "variant_view, scenarios, dated catalysts, breaker, sensitivities, rating) and the rating comes from the "
    "engine's rule. In thesis text write model numbers as placeholders ({fv}, {dcf_pg}, {comps}, {irr} ...), "
    "never typed. Never set a driver 'to tie' a guide: a non-operating line moved for calibration fails gate 11; "
    "a GAAP operating-margin guide already includes restructuring (do not deduct repo_pct again; set "
    "guidance.margin_basis). Read every 8-K after the balance sheet (closings, signings, share-repurchase "
    "settlements) and enter or dismiss it. Every warning and open check you leave needs an entry in "
    "thesis.explanations. Before returning, critique your own draft as a skeptical PM would: test each assumption "
    "against the gates above (definition tie-out, plugs, terminal reconciliation, 10-year margin range, value "
    "conservation) and fix what fails. One explanation per item, consistent with every other (never two accounts "
    "of the same share count); {irr} is the long-run IRR the price implies, not a 3-year return ({ret3y} is). "
    "A peer that gives no EPS guidance carries consensus, labeled consensus. When you revise a prior version "
    "after a critique, change only what evidence supports: a judgment change that raises value needs new "
    "evidence dated by the valuation date, not the critique's wording; every change is valued against the prior "
    "version and a judgment-driven rating change goes to an independent reviewer. "
    "Return strict JSON in the overrides format; omit fields you would not change; use segment and line names "
    "exactly as given. Numbers: decimals for percentages, $ millions for money, arrays of exactly six for forecast "
    "years.")


def analyst(cli, brief_text: str, documents: dict, progress=None, effort: Optional[str] = None) -> dict:
    """Judgment pass over the research brief + source documents -> overrides document.

    documents: {label: text} (earnings releases, MD&A excerpts, deal 8-Ks)."""
    parts = [brief_text, "# Source documents"]
    for label, text in (documents or {}).items():
        parts.append(f"=== {label} ===\n{text}")
    e = effort or _effort()
    out = _ask(cli, ANALYST_SYS, "\n\n".join(parts), 9000, model=analyst_model(), effort=e, progress=progress,
               what="analyst pass", max_out=OUTPUT_CAP.get(e, 20000), limit=900)  # queues can be slow at peak
    out.setdefault("meta", {})["author"] = f"AI analyst ({out.pop('__model__', analyst_model())})"
    return out


REVIEW_SYS = (
    "You are the senior buy-side analyst reviewing the model you just set up. You get your assumptions, what "
    "the built model computes from them, and the problems the model checker found. Fix the problems by "
    "changing assumptions (not history), keeping every change explainable: e.g. if year-1 sales miss "
    "guidance adjust organic growth or the M&A/FX contribution; if adjusted EPS misses, revisit amortization, "
    "non-segment charges, interest, buybacks or the adjusted-EPS definition (adj_eps_ex_sbc) — never by solving a tax "
    "rate (etr, adj_tax) onto the guide and never by "
    "zeroing stock comp or other real costs; if the exit multiple implies implausible perpetual growth "
    "reconcile both sides — the multiple against the peers' EV/EBITDA and the company's through-cycle range, "
    "and the WACC against the market-implied cost of equity — rather than setting the multiple to the "
    "perpetuity-implied one (the two DCF methods triangulate; a terminal multiple far below peers needs a "
    "reason such as maturity or a cycle peak); if a "
    "method is far from the price explain it or fix the driver behind it (growth runway via fade_years, margins, "
    "WACC inputs, the exit multiple vs the terminal year's capex intensity); if the rest of the year the model "
    "implies does not fit the year to date, fix FY1 (or record the one-off / seasonality in a note); if the stance "
    "contradicts the valuation, change the stance or the assumptions that drive it. A signed pending deal of "
    "any size goes in 'deals' (gate 32): the price leaves and its standalone value (its revenue and margin and a "
    "peer multiple) arrives. Working-capital days and ratios are year-end balances (gate 12): never carry the "
    "10-Q's seasonal balance into a full-year column. Cash above FY0's share of sales is returned by the engine "
    "unless you explain 'cash build'. The bear case is the breaker made numerical, 15%+ below base (gate 33); "
    "terminal RONIC stays at or below the return including goodwill when growth was bought, unless recurring "
    "acquisitions are forecast (gate 34); a base above consensus needs a variant view with its test date, and a "
    "management target belongs in the bull (gate 35); acquired revenue annualizes as mna, not organic (gate 36); "
    "a deal's standalone value from its price and from its EBITDA x multiple must agree (gate 38); debt-funded "
    "buybacks are stated (gate 39). History is locked: never move a history input (adj_eps_tax_hist more than 3pt "
    "from the stated non-GAAP rate, a past year's figure) to make a tie-out pass — if gate 1 fails, the reported "
    "figure or the definition switches are wrong, so check the release's reconciliation table. Interest far above "
    "the cost of debt on the debt read means debt is missing (gate 8): say so in a flag with the notes you can "
    "see, never solve it with a higher interest rate. An input the engine refused is gate 45: set a value the "
    "guard accepts or fix what made it refuse. An explanation that says fixed, corrected or restored while its "
    "check still fires is marked open: write what remains, not what you intended. A deal that closed before the "
    "balance-sheet date is already in net debt and the forecast: not a 'deals' item; the cash it cost is a fact, "
    "never raised to make a check tie; cost expensed as acquired IPR&D is not also an acquisition (cite it in the "
    "note). Never bend an input to fit another (capex raised so reinvestment fits RONIC is rejected): when RONIC "
    "sits under WACC (gate 49), the capital base is wrong, so say what is in it. A decline still running in FY6 goes "
    "in fade_path (gate 50). Scenario probabilities need their basis with numbers (gate 51). Count each risk once: a scenario risk sits in the bear and its probability, "
    "not also in beta, terminal growth, fade years or a base below consensus (the engine holds beta, terminal growth "
    "and fade years when their reason is scenario risk). Never give the skew, the gap or the rating as the reason "
    "for a change or an explanation (rejected). Text must match the model: no typed rate that differs from it, no "
    "'premium' at a discount, no buybacks 'within free cash flow' above it (gate 28). When the reverse DCF says the "
    "price implies about zero growth or less, the market is pricing decline: make the bear an explicit "
    "structural-decline case (revenue growth turning negative after year 3, terminal margin compression) with "
    "its probability in scenario_probs, and name the dated test (next guide, management change) in "
    "thesis.catalysts. The recommendation's call comes from the rule: thesis.recommendation is one "
    "supporting sentence with no call, and no thesis text states a call or bias the rule does not (gate 25; "
    "rating_override {rating, reason} to disagree). The price sits inside bear-bull (gate 41: build the bull from "
    "what the reverse DCF says the price requires). Terminal RONIC = the return including goodwill unless a dated "
    "sector reason with its year is under 'gate 34'; cyclicals end at mid-cycle from a peak too (gate 16); free "
    "cash flow continues into the fade within 5pt of its growth either way (gate 27: a reinvestment line missing "
    "from DCF C94); claims in explanations must be in the data (gate 42); the rest of FY1 looks like last year's "
    "(gate 43); a KPI build ties to revenue (gate 44). A Short also needs thesis.short_evidence (when the premium "
    "last compressed, on what), a crowding check and, for a low-volatility compounder, thesis.pair. "
    "A plug-table warning over 5% of revenue is explained under its own key with "
    "its cash or value effect in numbers. Never raise value on a judgment input without new evidence. "
    "{irr} is the long-run IRR at the price, not a 3-year return ({ret3y} is). Fix every failed gate "
    "first ('Gate failed (n)'), by the input the gate names. You may not change the risk-free rate or the ERP, "
    "and you may not move any input toward the share price (changes whose reason cites the price, the gap or "
    "the market-implied value are rejected); challenge drivers with evidence instead. Update the thesis "
    "'model_says' bullets to the final numbers using placeholders ({fv}, {dcf_pg}, {comps}, {bear}, {bull}, "
    "{irr}) instead of typed figures, and explain each warning or open check you leave in thesis.explanations "
    "under the key the checker gives (unexplained ones hold the rating at Under review). If you return "
    "flags, return the complete final list (it replaces the earlier flags; drop any a change has made stale). "
    "Unless source documents are attached below, this pass has no new documents: never cite a filing, disclosure or "
    "figure you were not given — an estimate "
    "is labeled an estimate with its arithmetic, and an item you cannot source is dismissed in "
    "thesis.explanations, not invented. Return strict JSON in the overrides "
    "format with ONLY the fields you change (notes included).")


def revise(cli, schema_doc: str, current: dict, review_text: str, progress=None,
           effort: Optional[str] = None, notes: str = "", documents: Optional[dict] = None) -> dict:
    """Second pass: the analyst fixes what the model checker found. notes: the PM's critique of the previous version
    (each point fixed with evidence or answered in thesis.explanations['pm review']); documents: the reading pack,
    so a fix can cite its source."""
    user = (f"## Overrides format\n{schema_doc}\n\n## Your current assumptions (overrides JSON)\n"
            f"{json.dumps(current, default=str)}\n\n## What the model computes and what the checker found\n"
            f"{review_text}\n\n" +
            (f"## The PM's critique of the previous version of this model\nAddress every point: fix it by the input it "
             f"names, citing a source document below or a stated estimate, or say why not. Put one paragraph per point "
             f"in thesis.explanations['pm review'] (what changed, its value effect). A critique is not evidence: move a "
             f"judgment input only on a figure from the documents.\n{notes}\n\n" if notes else "") +
            ("# Source documents (dated by the valuation date)\n" +
             "\n\n".join(f"=== {k} ===\n{v}" for k, v in documents.items()) + "\n\n" if documents else "") +
            "Return only the changed fields as an overrides JSON object.")
    e = effort or _effort()
    out = _ask(cli, REVIEW_SYS, user, 6000, model=analyst_model(), effort=e, progress=progress, what="review pass",
               max_out=OUTPUT_CAP.get(e, 20000) // 2, limit=900 if documents else 600)  # a reading pack takes longer
    out.setdefault("meta", {})["author"] = f"AI analyst review ({out.pop('__model__', analyst_model())})"
    return out


CHALLENGE_SYS = (
    "You are an independent reviewer on the investment committee. You did not build this model. The rating changed "
    "against the previous version the PM read, and most of the value move came from judgment inputs. You get the "
    "version-to-version bridge (each group's value effect), the judgment inputs that moved (previous vs new value, "
    "with the analyst's notes) and the thesis. For each moved input decide: KEEP only if its note cites evidence "
    "dated by the valuation date that the previous value missed (a filing, a disclosed figure, a guide); otherwise "
    "REVERT it to the previous value. Analysts revising after a critique tend to fix what was named and drift the "
    "unnamed judgment calls toward a higher value: treat any change that raises value without new evidence as "
    "unsupported. A management long-term target is not evidence for the base case. Never move an input toward the "
    "share price. Return strict JSON in the overrides format with ONLY the reverted fields set to their previous "
    "values (same structure as given), plus thesis.explanations['independent review']: one paragraph listing each "
    "input kept or reverted with its value effect in $/share from the bridge.")


def challenge(cli, bridge_txt: str, diff: dict, thesis: dict, progress=None, effort: Optional[str] = None) -> dict:
    """Independent review of a judgment-driven rating change (PM round 3 process lesson)."""
    user = (f"## Version bridge\n{bridge_txt}\n\n## Judgment inputs that moved (path: previous -> new)\n"
            f"{json.dumps(diff, default=str)}\n\n## Thesis\n{json.dumps(thesis, default=str)}\n\n"
            "Paths are dotted overrides keys (segments.<name>.growth, drivers.<key>, valuation.<key>). Return only "
            "the reverted fields as an overrides JSON object plus thesis.explanations['independent review'].")
    e = effort or _effort()
    out = _ask(cli, CHALLENGE_SYS, user, 4000, model=analyst_model(), effort=e, progress=progress,
               what="independent review", max_out=OUTPUT_CAP.get(e, 20000) // 2)
    out.setdefault("meta", {})["author"] = f"Independent review ({out.pop('__model__', analyst_model())})"
    return out
