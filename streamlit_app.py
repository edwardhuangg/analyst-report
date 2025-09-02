import streamlit as st
import pandas as pd
import yfinance as yf
import openai
import io
from datetime import datetime
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from http.client import RemoteDisconnected
import numpy as np
from typing import Optional, List
import re

# -------------------- Configuration -----------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
CREATED_BY = st.secrets.get("CREATED_BY", "Your Name / Org")
APP_VERSION = "1.0.0"
BUILD_TS = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

DISCLAIMER_MD = (
    """
**Disclaimer**  
This tool and the generated report are for **educational and informational purposes only** and do not constitute investment advice or a recommendation to buy or sell any security. The data is sourced from third parties (e.g., Yahoo Finance via `yfinance`) and may contain inaccuracies, delays, or omissions. All models (3-statement, DCF, Residual Income) are simplified and depend on user inputs and assumptions. **Do your own research** and consult a qualified financial professional before making investment decisions.

**Data/Model Notes**  
- Historical statements are taken as-reported and lightly normalized.  
- Cash flow convention: FCF ≈ OCF − CapEx; signs normalized to standard corporate finance conventions.  
- Two-stage DCF assumes either a fade from g₁ → g₂ or a constant g₁, with a standard Gordon Growth terminal value.  
- Residual Income is most appropriate for financials (banks/insurers) where book equity and ROE are primary drivers.  
- WACC inputs (rf, ERP, beta, D/V, Kd, tax) are user-editable and can materially change results.
    """
)

ABOUT_MD = (
    f"""
**Created by:** {CREATED_BY}  
**App version:** {APP_VERSION}  
**Build:** {BUILD_TS}

Questions or feedback? Add your contact info to Streamlit Secrets as `CREATED_BY` or edit this string in code.
    """
)

# -------------------- Info panel (always visible) ----------------
INFO_PANEL_MD = """
> **Heads-up: inputs matter.** The dashboard scaffolds a financial model for you, but it does **not** know the numbers in your head. The defaults are auto-filled from recent history and generic assumptions. If you don’t tweak them, it’s totally normal to see strange or extreme results at first.

**What to adjust right away**
- **3-Statement drivers:** Revenue growth, EBIT margin, CapEx/Rev, NWC/Rev, tax.
- **Discount rate:** Use the **WACC/CAPM** helper or set a manual **r**.
- **Terminal & stage growth:** Keep **g₂ < r**; pick a realistic **g₁** horizon.
- **Capital structure:** Shares outstanding & net debt (pulled from Yahoo—verify).
- **Comps:** Replace peer tickers with true like-for-like companies.

**Quick troubleshooting**
- Check **Analysis Window** and **Quarterly backfill** (short windows can be noisy).
- If CapEx sign looks odd, remember it’s often reported negative; the app normalizes, but verify.
- If valuation explodes, sanity-check **r**, **g₂**, and **Shares/Net Debt**.
"""

def render_info_panel():
    st.subheader("Information")
    st.caption("Quick pointers on how the app works and why early results may look wild until you tune inputs.")
    # Core how-to bullets (kept from your original copy)
    st.markdown(
        "- Use the sidebar to set the ticker and period, then **Generate Report**.\n"
        "- **3-Statement Model** lets you edit drivers; outputs feed the 3S-Driven DCF.\n"
        "- **WACC / CAPM** inputs drive discount rates (or keep a manual r).\n"
        "- **DCF** includes both a Two-Stage and a 3S-Driven approach.\n"
        "- **Residual Income (Financials)** is recommended for banks/insurers.\n"
        "- **Export** builds an Excel workbook with interactive tabs and charts.\n"
    )
    # The “why results may look odd” box
    st.info(INFO_PANEL_MD)
    st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)


# ---- OpenAI compatibility shim (new v1 client OR legacy 0.x) ---
def _make_create_chat():
    try:
        # Newer SDK (>=1.0)
        from openai import OpenAI
        _client = OpenAI(api_key=openai.api_key or None)
        def _create_chat(**kwargs):
            return _client.chat.completions.create(**kwargs)
        def _extract(resp):
            try:
                return resp.choices[0].message.content
            except Exception:
                return str(resp)
        return _create_chat, _extract
    except Exception:
        # Legacy SDK (<1.0)
        def _create_chat(**kwargs):
            model = kwargs.get("model")
            messages = kwargs.get("messages", [])
            temperature = kwargs.get("temperature")
            max_tokens = kwargs.get("max_tokens")
            return openai.ChatCompletion.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens
            )
        def _extract(resp):
            try:
                return resp["choices"][0]["message"]["content"]
            except Exception:
                return str(resp)
        return _create_chat, _extract

_create_chat, _extract_chat = _make_create_chat()

# -------------------- Session-state bootstrap -------------------
if "data_ready" not in st.session_state:
    st.session_state["data_ready"] = False
if "exec_summary" not in st.session_state:
    st.session_state["exec_summary"] = None
# Comps analysis + previous peer string for change detection
if "comps_analysis" not in st.session_state:
    st.session_state["comps_analysis"] = None
if "peer_input_prev" not in st.session_state:
    st.session_state["peer_input_prev"] = None
# 3S state holders
if "three_s" not in st.session_state:
    st.session_state["three_s"] = {}
if "dcf_choice" not in st.session_state:
    st.session_state["dcf_choice"] = "Two-Stage (Simple)"

# -------------------- Helpers: metrics & padding -----------------

def _cagr(series: pd.Series):
    try:
        s = pd.to_datetime(series.index, errors="coerce")
        ser = pd.Series(series.values, index=s).dropna()
        ser = ser.sort_index()  # oldest → newest
        if ser.empty or len(ser) < 2:
            return None
        start = float(ser.iloc[0])   # oldest
        end   = float(ser.iloc[-1])  # newest
        yrs = (ser.index[-1] - ser.index[0]).days / 365.25
        if yrs <= 0 or start <= 0:
            return None
        return ((end / start) ** (1 / yrs) - 1) * 100.0
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def compute_metrics(inc: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame, years: int):
    cutoff = datetime.today().replace(year=datetime.today().year - years)
    inc_p, bs_p, cf_p = inc[inc.index >= cutoff], bs[bs.index >= cutoff], cf[cf.index >= cutoff]

    eq_cols = [c for c in bs_p.columns if "equity" in c.lower()]
    eq_col = eq_cols[0] if eq_cols else None

    m, rev, cogs = {}, inc_p.get("Total Revenue"), inc_p.get("Cost Of Revenue")
    if rev is not None and cogs is not None:
        m[f"Gross Margin ({years}yr avg %)"] = ((rev - cogs) / rev * 100).mean()
    if rev is not None:
        if inc_p.get("Operating Income") is not None:
            m[f"Operating Margin ({years}yr avg %)"] = (inc_p.get("Operating Income") / rev * 100).mean()
        if inc_p.get("Net Income") is not None:
            m[f"Net Margin ({years}yr avg %)"] = (inc_p.get("Net Income") / rev * 100).mean()
        m[f"Revenue CAGR ({years}yr %)"] = _cagr(rev)

    if eq_col:
        if inc_p.get("Net Income") is not None:
            try:
                # Simple ROE approx: NI / avg equity
                eq = bs_p[eq_col]
                avg_eq = (eq.shift(-1).fillna(eq) + eq) / 2
                m[f"ROE ({years}yr avg %)"] = (inc_p.get("Net Income") / avg_eq * 100).mean()
            except Exception:
                pass
        invested = (bs_p.get("Total Liab", 0) + bs_p[eq_col]) if eq_col in bs_p.columns else None
        if invested is not None and inc_p.get("Net Income") is not None:
            m[f"ROIC ({years}yr avg %)"] = (inc_p.get("Net Income") * 0.8 / invested * 100).mean()

    if {"Total Cash From Operating Activities", "Capital Expenditures"}.issubset(cf_p.columns):
        fcf = cf_p["Total Cash From Operating Activities"] + cf_p["Capital Expenditures"]
        m[f"FCF CAGR ({years}yr %)"] = _cagr(fcf)

    assets = bs_p.get("Total Assets")
    if rev is not None and assets is not None:
        m[f"Asset Turnover ({years}yr avg x)"] = (rev / assets).mean()

    if {"Total Liab", eq_col}.issubset(bs_p.columns):
        try:
            m["Debt/Equity (latest)"] = (bs_p["Total Liab"] / bs_p[eq_col]).iloc[0]
        except Exception:
            pass

    return m

# ---------- Quarterly → Annual padding ----------

def _annualize_quarterly(qdf: pd.DataFrame, kind: str) -> pd.DataFrame:
    if qdf is None or qdf.empty:
        return pd.DataFrame()
    df = qdf.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    if kind == "flow":
        ann = df.resample("Y").sum()
    else:
        ann = df.resample("Y").last()
    ann = ann.dropna(how="all").sort_index(ascending=False)
    return ann


def _pad_annual_with_quarterly_if_needed(yt: yf.Ticker,
                                         inc: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame,
                                         target_n: int):
    def _nrows(df): return 0 if df is None or getattr(df, "empty", True) else len(df.index)

    need_inc = _nrows(inc) < target_n
    need_bs  = _nrows(bs)  < target_n
    need_cf  = _nrows(cf)  < target_n
    if not (need_inc or need_bs or need_cf):
        return inc, bs, cf

    try:
        q_inc = yt.quarterly_financials.T if need_inc else pd.DataFrame()
    except Exception:
        q_inc = pd.DataFrame()
    try:
        q_bs  = yt.quarterly_balance_sheet.T if need_bs else pd.DataFrame()
    except Exception:
        q_bs = pd.DataFrame()
    try:
        q_cf  = yt.quarterly_cashflow.T if need_cf else pd.DataFrame()
    except Exception:
        q_cf = pd.DataFrame()

    q_inc_ann = _annualize_quarterly(q_inc, "flow")   if need_inc else pd.DataFrame()
    q_bs_ann  = _annualize_quarterly(q_bs,  "stock")  if need_bs  else pd.DataFrame()
    q_cf_ann  = _annualize_quarterly(q_cf,  "flow")   if need_cf  else pd.DataFrame()

    def _pad(base_df, q_ann):
        def _n(df): return 0 if df is None or getattr(df, "empty", True) else len(df.index)
        if _n(base_df) >= target_n:
            return base_df
        base = base_df.copy() if _n(base_df) else pd.DataFrame()
        if q_ann is None or q_ann.empty:
            return base
        base_idx = base.index if _n(base) else pd.Index([])
        q_extra = q_ann[~q_ann.index.isin(base_idx)]
        combined = pd.concat([base, q_extra], axis=0) if _n(base) else q_extra
        combined = combined.dropna(how="all").sort_index(ascending=False)
        return combined.head(target_n)

    inc2 = _pad(inc, q_inc_ann) if need_inc else inc
    bs2  = _pad(bs,  q_bs_ann)  if need_bs  else bs
    cf2  = _pad(cf,  q_cf_ann)  if need_cf  else cf
    return inc2, bs2, cf2

# -------------------- Accounting / formatting ----------

def _make_accounting_formatter(decimals: int = 0, parens_for_neg: bool = True):
    fmt_spec = f"{{:,.{decimals}f}}"
    def _fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        try:
            val = float(x)
        except Exception:
            return x
        s = fmt_spec.format(abs(val))
        if decimals == 0 and "." in s:
            s = s.split(".")[0]
        return f"({s})" if (parens_for_neg and val < 0) else s
    return _fmt


def format_statement(df: pd.DataFrame, decimals: int = 0):
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = pd.to_numeric(out[c], errors="ignore")
    num_cols = out.select_dtypes(include=["number"]).columns
    fmt = _make_accounting_formatter(decimals=decimals, parens_for_neg=True)
    styler = out.style.format({c: fmt for c in num_cols}, na_rep="")
    return styler

# -------------------- UI helpers: flexible % inputs -----------------

def _parse_percent(s, default=0.0):
    """Return a *decimal* (e.g., '2.5%' -> 0.025). Accepts '2.5', '2.5%', '0.025', '25bp'."""
    try:
        if s is None: 
            return float(default)
        # handle numeric directly
        if isinstance(s, (int, float)):
            v = float(s)
            return v/100.0 if abs(v) >= 1.0 else v
        t = str(s).strip().lower().replace(",", "")
        if t == "":
            return float(default)
        # basis points
        if t.endswith("bp") or t.endswith("bps"):
            num = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
            return float(num[0]) / 10000.0 if num else float(default)
        had_pct = t.endswith("%")
        if had_pct:
            t = t[:-1]
        v = float(t)
        if had_pct or abs(v) >= 1.0:
            return v / 100.0
        return v
    except Exception:
        return float(default)


def percent_input(label, key, default=0.0, *, min_pct=None, max_pct=None, help=None, placeholder="e.g., 2.5 or 2.5% or 25bp"):
    """
    Text input that shows/accepts a percentage but returns & stores a decimal in st.session_state[key].
    Keeps the text box in-sync even if code changes the underlying decimal (e.g., after calibration).
    """
    txt_key = f"{key}__pctstr"
    cur_dec = float(st.session_state.get(key, default))

    # initialize / sync display string
    desired = f"{cur_dec*100:.3f}".rstrip("0").rstrip(".")
    if txt_key not in st.session_state:
        st.session_state[txt_key] = desired
    else:
        try:
            parsed_from_str = _parse_percent(st.session_state[txt_key], cur_dec)
            if abs(parsed_from_str - cur_dec) > 1e-9:
                st.session_state[txt_key] = desired
        except Exception:
            st.session_state[txt_key] = desired

    s = st.text_input(f"{label} (%)", key=txt_key, help=help, placeholder=placeholder)
    dec = _parse_percent(s, default)

    if min_pct is not None:
        dec = max(dec, float(min_pct) / 100.0)
    if max_pct is not None:
        dec = min(dec, float(max_pct) / 100.0)

    st.session_state[key] = dec
    return dec


# -------------------- Data fetch ------------

def _try_once_then_retry(fn):
    try:
        return fn()
    except Exception as e:
        transient = (
            isinstance(e, RequestsConnectionError) or
            isinstance(e, RemoteDisconnected) or
            "Remote end closed connection" in str(e)
        )
        if transient:
            return fn()
        raise


def _has_rows(df: pd.DataFrame) -> bool:
    return (df is not None) and hasattr(df, "empty") and (not df.empty)


@st.cache_data(ttl=3600, show_spinner="⏳ Fetching data …")
def fetch_financials(ticker: str, period: str = "5y", allow_quarterly_backfill: bool = False):
    try:
        yt = yf.Ticker(ticker)

        # Price
        try:
            price_df = _try_once_then_retry(lambda: yt.history(period=period, auto_adjust=False))
            if not _has_rows(price_df):
                raise RuntimeError("Empty price via history()")
        except Exception:
            price_df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
            if not _has_rows(price_df):
                raise RuntimeError("Empty price via download()")

        # Annual statements
        inc = _try_once_then_retry(lambda: yt.financials.T)
        bs  = _try_once_then_retry(lambda: yt.balance_sheet.T)
        cf  = _try_once_then_retry(lambda: yt.cashflow.T)
        if not (_has_rows(inc) and _has_rows(bs) and _has_rows(cf)):
            raise RuntimeError("Empty financial statements from Yahoo")

        # Target rows
        try:
            target_n = int(period[:-1]) if isinstance(period, str) and period.endswith("y") and period[:-1].isdigit() else 5
        except Exception:
            target_n = 5

            # 🔒 If backfill is OFF, cap N to the annual rows we truly have (no quarterly padding)
        if not allow_quarterly_backfill:
            existing_n = min([len(df.index) for df in (inc, bs, cf) if _has_rows(df)] or [0])
            if existing_n > 0:
                target_n = min(target_n, existing_n)

        # Only pad if explicitly allowed
        if allow_quarterly_backfill:
            inc, bs, cf = _pad_annual_with_quarterly_if_needed(yt, inc, bs, cf, target_n)
        else:
            # keep only the most recent target_n annual rows
            inc, bs, cf = inc.head(target_n), bs.head(target_n), cf.head(target_n)

        try:
            cf_q = yt.quarterly_cashflow.T
        except Exception:
            cf_q = pd.DataFrame()

        try:
            info = _try_once_then_retry(lambda: yt.info) or {}
        except Exception:
            info = {}

        def compute_net_debt(bs_df: pd.DataFrame, info_fallback: dict) -> float:
            def first(df, cols):
                for c in cols:
                    if c in df.columns:
                        return df[c].iloc[0]
                return None
            if bs_df is not None and not bs_df.empty:
                total_debt = first(bs_df, ["Total Debt","TotalDebt","Short Long Term Debt","Long Term Debt"])
                cash = first(bs_df, ["Cash And Cash Equivalents","Cash","CashAndCashEquivalents"])
                sti  = first(bs_df, ["Short Term Investments","ShortTermInvestments"])
                if total_debt is not None and (cash is not None or sti is not None):
                    return float((total_debt or 0.0) - ((cash or 0.0) + (sti or 0.0)))
            td = (info_fallback.get("totalDebt") or 0.0)
            c  = (info_fallback.get("cash") or 0.0)
            sti = (info_fallback.get("shortTermInvestments") or 0.0)
            return float(td - (c + sti))

        extras = {
            "Shares Outstanding (raw)": info.get("sharesOutstanding"),
            "Net Debt ($)": compute_net_debt(bs, info),
            "Market Cap ($)": info.get("marketCap"),
            "P/E Ratio": info.get("trailingPE"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "Sector": info.get("sector", ""),
            "Industry": info.get("industry", ""),
            "_info": info,
        }
        return price_df, inc, bs, cf, cf_q, extras

    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return None, None, None, None, None, {}

# -------------------- DCF (two-stage + inverse) -----------------

def _pick_first(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def _latest(series: pd.Series):
    return series.iloc[0] if (series is not None and not series.empty) else None

# --- Base FCF selection (TTM preferred) ---

def get_base_fcf(cf_annual: pd.DataFrame, cf_quarterly: Optional[pd.DataFrame] = None):
    """Return base FCF = OCF - CapEx (TTM from quarterlies preferred, else latest annual)."""
    def pick(df, names):
        for c in names:
            if df is not None and c in df.columns:
                return df[c]
        return None

    # TTM from quarterlies
    if cf_quarterly is not None and not cf_quarterly.empty:
        ocf_q = pick(cf_quarterly, [
            "Total Cash From Operating Activities", "Cash Flow From Operations",
            "Operating Cash Flow", "Net Cash Provided By Operating Activities"
        ])
        capex_q = pick(cf_quarterly, [
            "Capital Expenditures", "Capital Expenditure",
            "Purchase Of Property Plant And Equipment",
            "Investments In Property, Plant And Equipment"
        ])
        if ocf_q is not None and capex_q is not None:
            ocf_q = ocf_q.dropna().sort_index(ascending=False)
            capex_q = capex_q.dropna().sort_index(ascending=False)
            if not capex_q.empty and capex_q.iloc[:8].median() > 0:
                capex_q = -capex_q
            if len(ocf_q) >= 4 and len(capex_q) >= 4:
                ocf_4   = ocf_q.iloc[:4].sum()
                capex_4 = capex_q.iloc[:4].sum()
                return float(ocf_4 + capex_4)

    # Fallback: latest annual
    if cf_annual is not None and not cf_annual.empty:
        ocf_a = _pick_first(cf_annual, [
            "Total Cash From Operating Activities", "Cash Flow From Operations",
            "Operating Cash Flow", "Net Cash Provided By Operating Activities"
        ])
        capex_a = _pick_first(cf_annual, [
            "Capital Expenditures", "Capital Expenditure",
            "Purchase Of Property Plant And Equipment",
            "Investments In Property, Plant And Equipment"
        ])
        if ocf_a is not None and capex_a is not None:
            ocf_latest   = float(_latest(ocf_a) or 0.0)
            capex_latest = float(_latest(capex_a) or 0.0)
            if capex_latest > 0:
                capex_latest = -capex_latest
            return float(ocf_latest + capex_latest)

    return None


def pick_shares(yf_info: dict, inc_annual: pd.DataFrame, bs_annual: pd.DataFrame) -> Optional[int]:
    info_so = None
    try:
        if yf_info and "sharesOutstanding" in yf_info and yf_info["sharesOutstanding"]:
            info_so = int(yf_info["sharesOutstanding"])
    except Exception:
        info_so = None

    stmt_so = None
    for col in ["Diluted Average Shares", "Basic Average Shares",
                "Weighted Average Shs Out Dil", "Weighted Average Shs Out"]:
        if inc_annual is not None and col in inc_annual.columns:
            try:
                v = float(inc_annual[col].iloc[0])
                if info_so and v > 5 * (info_so or 1):
                    v = v / 1000.0
                if info_so and v < (info_so or 1) / 100.0:
                    v = v * 1_000_000.0
                stmt_so = int(v)
                break
            except Exception:
                pass

    if stmt_so is None:
        for col in ["Common Stock Shares Outstanding", "OrdinarySharesNumber"]:
            if bs_annual is not None and col in bs_annual.columns:
                try:
                    v = float(bs_annual[col].iloc[0])
                    if info_so and v > 5 * (info_so or 1):
                        v = v / 1000.0
                    if info_so and v < (info_so or 1) / 100.0:
                        v = v * 1_000_000.0
                    stmt_so = int(v)
                    break
                except Exception:
                    pass

    if info_so and stmt_so:
        return int(min([stmt_so, info_so], key=lambda x: abs(x - info_so)))
    return int(stmt_so or (info_so or 0)) or None


def dcf_two_stage_price(base_fcf, shares_out, net_debt, N, g1, g2, r, fade: bool = True):
    if base_fcf is None or shares_out in (None, 0):
        return None
    if r <= g2:
        return None
    years = list(range(1, N + 1))
    if fade:
        growth_path = [g1 + (g2 - g1) * (y - 1) / (N - 1 if N > 1 else 1) for y in years]
        proj_fcf = []
        f = base_fcf
        for gr in growth_path:
            f = f * (1 + gr)
            proj_fcf.append(f)
    else:
        proj_fcf = [base_fcf * (1 + g1) ** y for y in years]
    pv_fcf   = [fcf / (1 + r) ** y for y, fcf in zip(years, proj_fcf)]
    fcf_N1   = proj_fcf[-1] * (1 + g2)
    term_val = fcf_N1 / (r - g2)
    term_pv  = term_val / (1 + r) ** N
    ev       = sum(pv_fcf) + term_pv
    eq_v     = ev - (net_debt or 0.0)
    return eq_v / shares_out


def _bisect_solve(func, lo, hi, tol=1e-6, max_iter=60):
    f_lo = func(lo); f_hi = func(hi)
    if f_lo is None or f_hi is None:
        return None
    if f_lo == 0: return lo
    if f_hi == 0: return hi
    if f_lo * f_hi > 0: return None
    x_lo, x_hi = lo, hi
    for _ in range(max_iter):
        x_mid = (x_lo + x_hi) / 2
        f_mid = func(x_mid)
        if f_mid is None: return None
        if abs(f_mid) < tol: return x_mid
        if f_lo * f_mid <= 0:
            x_hi, f_hi = x_mid, f_mid
        else:
            x_lo, f_lo = x_mid, f_mid
    return (x_lo + x_hi) / 2


def solve_implied_r(target_px, base_fcf, shares_out, net_debt, N, g1, g2, fade: bool = True):
    lo = max(g2 + 0.001, 0.01); hi = 0.40
    def f(r):
        px = dcf_two_stage_price(base_fcf, shares_out, net_debt, N, g1, g2, r, fade=fade)
        return None if px is None else (px - target_px)
    return _bisect_solve(f, lo, hi)


def solve_implied_g1(target_px, base_fcf, shares_out, net_debt, N, r, g2, fade: bool = True):
    lo, hi = -0.50, 0.50
    def f(g1):
        px = dcf_two_stage_price(base_fcf, shares_out, net_debt, N, g1, g2, r, fade=fade)
        return None if px is None else (px - target_px)
    return _bisect_solve(f, lo, hi)

# --------------------  Financials (Residual Income) ---------

def detect_is_financials(extras: dict) -> bool:
    s = (extras.get("Sector") or "").lower()
    i = (extras.get("Industry") or "").lower()
    tokens = ["financial", "bank", "insurance", "insurer", "capital markets", "credit", "reinsurance", "asset management", "brokerage"]
    return any(t in s for t in tokens) or any(t in i for t in tokens)


def estimate_starting_bve_and_roe(inc: pd.DataFrame, bs: pd.DataFrame):
    eq_cols = [c for c in bs.columns if "equity" in c.lower()]
    bve0 = float(bs[eq_cols[0]].iloc[0]) if eq_cols else None
    try:
        ni = float(inc["Net Income"].iloc[0])
        eq_now = float(bs[eq_cols[0]].iloc[0])
        eq_prev = float(bs[eq_cols[0]].iloc[1]) if len(bs.index) > 1 else eq_now
        avg_eq = (eq_now + eq_prev) / 2.0
        roe0 = ni / avg_eq if avg_eq else None
    except Exception:
        roe0 = None
    return bve0, roe0


def residual_income_valuation(bv0, shares_out, ke, N, roe_start, payout, fade_to_ke=True, roe_terminal: Optional[float]=None, g_ri: float=0.0):
    """
    Value0 = BV0 + Σ PV[ (ROE_t - ke) * BV_{t-1} ] + PV(terminal RI, optional)
    """
    if bv0 is None or shares_out in (None, 0) or ke is None:
        return None, None, None
    N = int(max(1, N))
    payout = float(max(0.0, min(1.0, payout)))
    retention = 1.0 - payout

    years = list(range(1, N + 1))
    bv_beg = []
    roe_path = []
    earnings = []
    dividends = []
    bv_end = []
    ri = []
    pv_ri = []

    bv = float(bv0)
    for y in years:
        if fade_to_ke:
            roe_y = float(roe_start) + (float(ke) - float(roe_start)) * (y - 1) / (N - 1 if N > 1 else 1)
        else:
            tgt = float(roe_terminal if roe_terminal is not None else roe_start)
            roe_y = float(roe_start) + (tgt - float(roe_start)) * (y - 1) / (N - 1 if N > 1 else 1)

        bv_beg.append(bv)
        roe_path.append(roe_y)
        earn = bv * roe_y
        div  = earn * payout
        bv   = bv + (earn - div)
        earnings.append(earn)
        dividends.append(div)
        bv_end.append(bv)
        ri_y = (roe_y - ke) * bv_beg[-1]
        ri.append(ri_y)
        pv_ri.append(ri_y / ((1 + ke) ** y))

    term_pv = 0.0
    if not fade_to_ke:
        roe_T = roe_path[-1]
        bv_T = bv_end[-1]
        ri_N1 = (roe_T - ke) * bv_T
        if ke > g_ri:
            term_pv = (ri_N1 / (ke - g_ri)) / ((1 + ke) ** N)
        else:
            term_pv = 0.0

    value_equity = float(bv0) + sum(pv_ri) + term_pv
    implied_price = value_equity / shares_out

    df = pd.DataFrame({
        "Year": years,
        "BV_Beg": bv_beg,
        "ROE": roe_path,
        "Earnings": earnings,
        "Dividends": dividends,
        "BV_End": bv_end,
        "Residual Income": ri,
        "PV RI": pv_ri,
    })
    summary = {
        "BV0": bv0,
        "ke": ke,
        "payout": payout,
        "term_pv": term_pv,
        "Equity Value": value_equity,
        "Implied Price": implied_price
    }
    return implied_price, df, summary

# -------------------- NEW: 3-Statement Model --------------------

def _safe(val, default=0.0):
    try:
        return float(val) if pd.notna(val) else default
    except Exception:
        return default


def _col(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(dtype='float64')


def _latest_val(series: pd.Series, default=0.0):
    return _safe(series.iloc[0] if (series is not None and not series.empty) else default, default)


def derive_default_drivers(inc: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame) -> dict:
    rev = _col(inc, "Total Revenue")
    ebit= _col(inc, "Operating Income")
    dep = _col(cf,  "Depreciation", "Depreciation And Amortization")
    cap = _col(cf,  "Capital Expenditures", "Capital Expenditure")
    tca = _col(bs,  "Total Current Assets")
    tcl = _col(bs,  "Total Current Liabilities")
    cash= _col(bs,  "Cash And Cash Equivalents", "Cash")
    sdebt=_col(bs,  "Short Long Term Debt", "Short Term Debt")
    ldebt=_col(bs,  "Long Term Debt")

    om   = (ebit / rev).dropna()
    dep_pct   = (dep.abs() / rev).dropna()
    capex_pct = (cap.abs() / rev).dropna()

    # NWC = (CA - Cash) - (CL - ShortDebt)
    nwc_amt = (tca - cash) - (tcl - sdebt.fillna(0))
    nwc_pct = (nwc_amt / rev).dropna()

    # Effective tax rate (approx)
    tax = _col(inc, "Tax Provision"); ptx = _col(inc, "Pretax Income")
    eff_tax = (tax.abs() / ptx.abs()).replace([pd.NA, pd.NaT, float('inf')], pd.NA).dropna()

    # Interest rate from history
    iex = _col(inc, "Interest Expense"); debt_hist = (sdebt.fillna(0) + ldebt.fillna(0)).abs()
    int_rate = (iex.abs() / debt_hist.replace(0, pd.NA)).dropna()

    drivers = {
        "rev0": _latest_val(rev),
        "revg": float(round(rev.pct_change(-1).dropna().head(3).mean(), 4)) if len(rev) > 1 else 0.06,
        "ebit_margin": float(round(om.head(3).mean(), 4)) if not om.empty else 0.18,
        "tax_rate": float(min(max(eff_tax.head(3).mean() if not eff_tax.empty else 0.20, 0.0), 0.40)),
        "dep_pct_rev": float(round(dep_pct.head(3).mean(), 4)) if not dep_pct.empty else 0.04,
        "capex_pct_rev": float(round(capex_pct.head(3).mean(), 4)) if not capex_pct.empty else 0.05,
        "nwc_pct_rev": float(round(nwc_pct.head(3).mean(), 4)) if not nwc_pct.empty else 0.05,
        "interest_rate": float(round(int_rate.head(3).mean(), 4)) if not int_rate.empty else 0.03,
        "div_payout": 0.0,
        "cash0": _latest_val(cash),
        "ppe0": _latest_val(_col(bs, "Property Plant And Equipment Net", "Net Property Plant & Equipment")),
        "debt0": _latest_val(debt_hist),
        "equity0": _latest_val(_col(bs, "Total Stockholder Equity", "Total Equity", "Total Equity Gross Minority Interest")),
        "nwc0": _latest_val(nwc_amt),
    }
    for k in ("dep_pct_rev","capex_pct_rev","nwc_pct_rev"):
        drivers[k] = float(max(drivers[k], 0.0))
    return drivers


def project_three_statements(dr: dict, years: int):
    """Driver-based 3S build, returning projected Income, Balance, CashFlow and Unlevered FCF series."""
    N = years
    rev0 = dr["rev0"]; equity0 = dr["equity0"]; ppe0 = dr["ppe0"]; nwc0 = dr["nwc0"]; cash0 = dr["cash0"]; debt0 = dr["debt0"]
    g = dr["revg"]; m_ebit = dr["ebit_margin"]; t = dr["tax_rate"]; dep_pct = dr["dep_pct_rev"]; capex_pct = dr["capex_pct_rev"]; nwc_pct = dr["nwc_pct_rev"]
    r_int = dr["interest_rate"]; payout = dr["div_payout"]

    years_idx = [f"Y{n}" for n in range(1, N+1)]
    rev = []; ebit=[]; dep=[]; int_exp=[]; ebt=[]; tax=[]; ni=[]
    capex=[]; nwc_level=[]; dNWC=[]; ppe=[]; cash=[]; divs=[]
    cfo=[]; cfi=[]; cff=[]; fcf_u=[]; equity=[]

    rev_prev = rev0; ppe_prev = ppe0; nwc_prev = nwc0; cash_prev = cash0; equity_prev = equity0
    for _ in range(N):
        r = rev_prev * (1 + g); rev.append(r)
        d = dep_pct * r; dep.append(d)
        e = m_ebit * r; ebit.append(e)
        ii = r_int * debt0; int_exp.append(ii)
        ebt_i = e - ii
        tax_i = max(ebt_i, 0.0) * t
        ni_i = ebt_i - tax_i
        ebt.append(ebt_i); tax.append(tax_i); ni.append(ni_i)
        div_i = max(ni_i, 0.0) * payout; divs.append(div_i)

        nwc_i = nwc_pct * r; nwc_level.append(nwc_i)
        dNWC_i = nwc_i - nwc_prev; dNWC.append(dNWC_i)
        cap_i = capex_pct * r; capex.append(cap_i)

        ppe_i = ppe_prev + cap_i - d
        ppe.append(ppe_i)

        cfo_i = ni_i + d - dNWC_i
        cfi_i = -cap_i
        cff_i = -div_i
        cfo.append(cfo_i); cfi.append(cfi_i); cff.append(cff_i)

        cash_i = cash_prev + cfo_i + cfi_i + cff_i
        cash.append(cash_i)

        equity_i = equity_prev + ni_i - div_i
        equity.append(equity_i)

        fcf_i = e * (1 - t) + d - cap_i - dNWC_i
        fcf_u.append(fcf_i)

        rev_prev, ppe_prev, nwc_prev, cash_prev, equity_prev = r, ppe_i, nwc_i, cash_i, equity_i

    income_df = pd.DataFrame({
        "Year": years_idx, "Revenue": rev, "EBIT": ebit, "Depreciation": dep,
        "Interest": int_exp, "EBT": ebt, "Taxes": tax, "Net Income": ni
    })
    balance_df = pd.DataFrame({
        "Year": years_idx, "Cash": cash, "NWC": nwc_level, "PP&E": ppe,
        "Debt (assumed flat)": [debt0]*N, "Equity": equity
    })
    cashflow_df = pd.DataFrame({
        "Year": years_idx, "CFO": cfo, "CFI (CapEx)": cfi, "CFF": cff, "ΔCash": [cfo[i]+cfi[i]+cff[i] for i in range(N)]
    })
    fcf_series = pd.Series(fcf_u, index=years_idx, name="Unlevered FCF")
    equity_series = pd.Series(equity, index=years_idx)
    return income_df, balance_df, cashflow_df, fcf_series, equity_series, equity0


def build_dcf_from_series(fcf_series: pd.Series, r: float, g_term: float, net_debt: float, shares_out: float):
    if fcf_series is None or fcf_series.empty or r <= g_term:
        return None, None, None
    N = len(fcf_series)
    years = list(range(1, N+1))
    pv = [float(fcf_series.iloc[i]) / (1+r)**(i+1) for i in range(N)]
    term_fcf = float(fcf_series.iloc[-1]) * (1 + g_term)
    term_pv  = term_fcf / (r - g_term) / (1 + r)**N
    ev = sum(pv) + term_pv
    eq_v = ev - (net_debt or 0.0)
    px   = (eq_v / shares_out) if shares_out else None
    detail = pd.DataFrame({"Year": years + ["Terminal"], "FCF": list(fcf_series.values) + [term_fcf], "PV FCF": pv + [term_pv]})
    head = pd.Series({"Enterprise Value": ev, "Less: Net Debt": net_debt, "Equity Value": eq_v, "Shares Out": shares_out, "Implied Price": px}).to_frame("Value")
    return px, detail, head

# -------------------- UI & Styling ------------------------------

st.set_page_config(page_title="📈 Analyst Dashboard", layout="wide")

st.markdown(
"""
<style>
:root{
  --accent:#2e7efb; --accent-2:#5ad1ff;
  --text:#eaeaea; --text-dim:rgba(234,234,234,.85);
  --bg-main:#0f1217; --bg-card:#161a20; --border:rgba(255,255,255,.10);
  --st-header-height:56px;
}
html, body, .stApp, div[data-testid="stAppViewContainer"]{ background: var(--bg-main) !important; }
section[data-testid="stSidebar"]{ background: var(--bg-main) !important; border-right: 1px solid var(--border); }
header[data-testid="stHeader"]{ background: var(--bg-main) !important; box-shadow:none !important; border-bottom: 1px solid var(--border); height: var(--st-header-height) !important; }
.block-container{ max-width: 1200px; padding-top: calc(1rem + var(--st-header-height)) !important; }
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp code, h1,h2,h3,h4,h5,h6{ color: var(--text) !important; }
h1, h2, h3 { letter-spacing:.2px; }
h1 { font-size: 1.65rem; margin-bottom:.25rem; }
h2 { font-size: 1.25rem; margin:.35rem 0 .25rem; }
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea{
  background: var(--bg-card) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder, .stTextArea textarea::placeholder{ color: var(--text-dim) !important; }
label, .stMarkdown small, .stCaption { color: var(--text-dim) !important; }
.st-expander{
  border: 1px solid var(--border); border-radius: 12px !important; overflow: hidden; background: var(--bg-card);
}
.stTable table{ border: 1px solid var(--border); border-radius:10px; overflow:hidden; background: var(--bg-card); }
.stTable thead tr th{ background: rgba(255,255,255,0.04) !important; color: var(--text) !important; font-weight:700 !important; }
.stTable tbody td{ color: var(--text) !important; }
.stTable tbody tr:nth-child(even){ background: rgba(255,255,255,0.02); }
.stAlert{ border-radius:12px; background: var(--bg-card) !important; color: var(--text) !important; }
section[data-testid="stSidebar"] > div{ height:100%; display:flex; flex-direction:column; }
section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button{ width: 100% !important; position: sticky; top: calc(100vh - 72px); margin-top: 12px; padding: 12px 14px; border: 0 !important; border-radius: 12px !important; font-weight: 700; background-image: linear-gradient(135deg, var(--accent), var(--accent-2)) !important; background-color: var(--accent) !important; color: #fff !important; box-shadow: 0 8px 22px rgba(46,126,251,0.40) !important; }
.ui-card{ margin:12px 0; padding:16px 18px; border-radius:14px; border:1px solid var(--border); background: var(--bg-card); }
.ui-card .ui-card-title{ display:flex; align-items:center; gap:8px; margin:0 0 8px; font-weight:700; font-size:1.1rem; color:var(--text); }
.app-header{ margin:8px 0 10px; padding:14px 16px; border-radius:14px; border:1px solid var(--border); background: var(--bg-card); display:flex; gap:12px; align-items:center; }
.app-header .title{ flex:1; }
.app-header .title .eyebrow{ font-size:12px; text-transform:uppercase; color:var(--text-dim); }
.app-header .subtitle{ font-size:13px; color: var(--text-dim); margin-top:2px; }
.app-header .chips{ display:flex; gap:8px; flex-wrap:wrap; }
.app-header .chip{ padding:6px 10px; border-radius:999px; border:1px solid rgba(46,126,251,.35); background:rgba(46,126,251,.10); font-size:12px; color:var(--text); }
.section-break{ height:1px; background: var(--border); margin: 18px 0; }
.section-label{ font-size:12px; opacity:.85; text-transform:uppercase; letter-spacing:.08em; margin: 4px 0 6px; color: var(--text-dim); }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("""
<style id="fix-formsubmit-width">
/* Stretch ALL element-containers inside the sidebar, so nothing is clamped */
section[data-testid="stSidebar"] [data-testid="stElementContainer"]{
  width:100% !important; max-width:100% !important; min-width:0 !important;
  align-self:stretch !important; display:block !important;
}
/* If the container wraps content in an extra div, stretch that too */
section[data-testid="stSidebar"] [data-testid="stElementContainer"] > div{
  width:100% !important; max-width:100% !important; display:block !important;
}

/* Make submit **and** legacy buttons fill the container */
section[data-testid="stSidebar"] :is(.stFormSubmitButton, .stButton){ width:100% !important; display:block !important; }
section[data-testid="stSidebar"] :is(.stFormSubmitButton, .stButton) > *{ width:100% !important; display:block !important; }
section[data-testid="stSidebar"] :is(.stFormSubmitButton button, .stButton > button,
  button[data-testid="baseButton-primary"], button[data-testid="baseButton-secondary"]){
  width:100% !important; max-width:100% !important; min-width:0 !important; box-sizing:border-box !important;

  /* sticky visuals */
  position: sticky; top: calc(100vh - 72px);
  margin-top:12px; padding:12px 14px; border:0 !important; border-radius:12px !important; font-weight:700;
  background-image: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  background-color: var(--accent) !important; color:#fff !important;
  box-shadow: 0 8px 22px rgba(46,126,251,0.40) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ---------- Creator card (sidebar) ---------- */
section[data-testid="stSidebar"] .creator-card{
  margin-top:12px; padding:12px 14px; border-radius:12px; border:1px solid var(--border);
  background: var(--bg-card); display:flex; align-items:center; gap:12px;
}
section[data-testid="stSidebar"] .creator-card .avatar{
  width:36px; height:36px; border-radius:999px; display:grid; place-items:center; font-weight:700; letter-spacing:.4px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)); color:#fff;
}
section[data-testid="stSidebar"] .creator-card .meta{ display:flex; flex-direction:column; line-height:1.2; }
section[data-testid="stSidebar"] .creator-card .meta .label{ font-size:11px; text-transform:uppercase; color:var(--text-dim); }
section[data-testid="stSidebar"] .creator-card .meta .name{ font-weight:700; color:var(--text); }
section[data-testid="stSidebar"] .creator-card .actions{ margin-left:auto; }
section[data-testid="stSidebar"] .creator-card .link-btn{
  display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:8px;
  border:1px solid rgba(46,126,251,.35); background:rgba(46,126,251,.08);
  text-decoration:none; font-size:13px; color:var(--accent); transition: background .2s, transform .02s, box-shadow .2s;
}
section[data-testid="stSidebar"] .creator-card .link-btn:hover{ background:rgba(46,126,251,.12); box-shadow:0 6px 16px rgba(46,126,251,.25); }
section[data-testid="stSidebar"] .creator-card .link-btn svg{ width:16px; height:16px; display:block; }
</style>
""", unsafe_allow_html=True)


# ---------- Sidebar form ----------
with st.sidebar.form("report_form"):
    ticker = st.text_input("Ticker", st.session_state.get("ticker", "AAPL")).strip().upper()
    timeframe = st.selectbox(
        "Analysis Window",
        ["1y", "3y", "5y", "max"],
        index=["1y", "3y", "5y", "max"].index(st.session_state.get("timeframe", "5y")),
    )
    allow_q_backfill = st.checkbox(
        "Allow quarterly backfill when annuals < N",
        value=st.session_state.get("allow_q_backfill", False),
        help="If off, show only as-reported annual rows (no quarterly padding)."
    )
    submitted = st.form_submit_button("Generate Report", type="primary")


# ---------- Sidebar: About/Disclaimer ----------
with st.sidebar.expander("About & Disclaimer", expanded=False):
    st.markdown(
        """
**Status:** Current version includes a driver-based 3-Statement model, Two-Stage & 3S-Driven DCF, Residual Income for financials, AI write-ups, and Excel export.

**What’s inside (top → bottom):**
- **Generate Report** — Enter a ticker & window; fetches price and annual statements (pads with quarterly where needed).
- **Price Performance** — Closing-price line chart.
- **Financial Statements** — Income, Balance Sheet, Cash Flow (last N years).
- **Key Metrics & Ratios** — Margins, ROE/ROIC, asset turnover, FCF trends, and basic market multiples.
- **Comparable Companies** — Quick peer multiples table **+ AI Comps & Market Analysis**.
- **3-Statement Model (Driver-based)** — Edit growth/margins/CapEx/NWC/taxes; produces projected IS/BS/CF and **Unlevered FCF**.
- **Cost of Capital (WACC & CAPM)** — rf, ERP, beta, D/V, Kd, tax. You can use WACC as the DCF discount rate or keep a manual **r**.
- **Growth Consistency** — Cross-check: **g ≈ ROIC × reinvestment**.
- **DCF Valuation** — Two modes:
  - *Two-Stage (Simple):* optional fade **g₁ → g₂**; can auto-solve for **r** or **g₁** to match market.
  - *3S-Driven:* discounts the Unlevered FCFs produced by the 3-Statement model.
- **Residual Income (Financials)** — Bank/insurer-friendly equity valuation using BV₀, ROE path, payout, and **kₑ** (with optional terminal RI if you don’t fade ROE → kₑ).
- **Executive Summary (AI)** — Concise narrative reflecting your current settings and outputs.
- **Export** — One-click Excel workbook with **Overview, Price (chart), Income, Balance, CashFlow, Proj_Income, Proj_Balance, Proj_CashFlow, Proj_FCF, DCF Model, Residual Income, Comps, Key Metrics**. Interactive formulas & charts included.

**Data & model notes:** Data comes from public sources (e.g., Yahoo Finance via `yfinance`) and may have gaps or mapping quirks. FCF ≈ OCF − CapEx (TTM from quarterlies when available, else latest annual). Shares are inferred from Yahoo and statements. Net debt = total debt − (cash + STI). Models are simplified and sensitive to assumptions.

**How AI uses your inputs:**  
- **AI Comps & Market Analysis** summarizes relative valuation using your peer set and observed multiples.  
- **Executive Summary (AI)** synthesizes the full context (ticker, timeframe, 3S drivers, DCF/RI settings, WACC/CAPM inputs, fade options, etc.). Change inputs → regenerate.

**No investment advice:** For educational/informational use only — **not** financial advice, a recommendation, or a solicitation to buy/sell securities. Always do your own research and verify assumptions.
        """
    )

    # ---------- Sidebar: Creator card ----------
st.sidebar.markdown("""
<div class="creator-card">
  <div class="avatar">EH</div>
  <div class="meta">
    <div class="label">Created by</div>
    <div class="name">Edward Huang</div>
  </div>
  <div class="actions">
    <a class="link-btn" href="https://linkedin.com/in/edwardhuangg" target="_blank" rel="noopener noreferrer">
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M20.45 20.45h-3.57v-5.58c0-1.33-.02-3.04-1.86-3.04-1.86 0-2.14 1.45-2.14 2.95v5.67H9.31V9.75h3.43v1.46h.05c.48-.91 1.64-1.86 3.37-1.86 3.61 0 4.28 2.38 4.28 5.48v5.61zM5.34 8.29a2.07 2.07 0 1 1 0-4.14 2.07 2.07 0 0 1 0 4.14zM7.13 20.45H3.56V9.75h3.57v10.7zM22 2H2C.9 2 0 2.9 0 4v16c0 1.1.9 2 2 2h20c1.1 0 2-.9 2-2z"/>
      </svg>
      LinkedIn
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Header ----------
if st.session_state.get("data_ready"):
    _ticker = st.session_state["ticker"]
    _tf = st.session_state["timeframe"]
    _sector = (st.session_state.get("extras") or {}).get("Sector") or "—"
    _industry = (st.session_state.get("extras") or {}).get("Industry") or "—"
    st.markdown(f"""
    <div class="app-header">
      <div class="title">
        <div class="eyebrow">Analyst Dashboard</div>
        <h1>📊 Analyst Dashboard & Report Generator</h1>
        <div class="subtitle">AI-assisted valuation, tied 3-statement engine + DCF + RI. Excel export included.</div>
      </div>
      <div class="chips">
        <span class="chip">{_ticker}</span>
        <span class="chip">{_tf}</span>
        <span class="chip">{_sector}</span>
        <span class="chip">{_industry}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        """
    <div class="app-header">
      <div class="title">
        <div class="eyebrow">Analyst Dashboard</div>
        <h1>📊 Analyst Dashboard & Report Generator</h1>
        <div class="subtitle">AI-assisted valuation, tied 3-statement engine + DCF + RI. Excel export included.</div>
      </div>
      <div class="chips">
        <span class="chip">Ready</span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------- Always-on Information (show before any data is fetched) ----------
render_info_panel()

# ---------- Fetch on submit ----------
if submitted:
    price_df, inc, bs, cf, cf_q, extras = fetch_financials(ticker, timeframe, allow_quarterly_backfill=allow_q_backfill)
    st.session_state.update({
        "ticker": ticker, "timeframe": timeframe,
        "allow_quarterly_backfill": allow_q_backfill,
        "price_df": price_df, "inc": inc, "bs": bs, "cf": cf, "cf_q": cf_q, "extras": extras,
        "data_ready": price_df is not None,
        "years_back": int(timeframe.replace("y", "")) if timeframe != "max" else 5,
        # DCF defaults (editable)
        "N": 5, "g1": 0.10, "g2": 0.025, "r": 0.09, "fade": True,
        "exec_summary": None,
        "comps_analysis": None,
        "peer_input_prev": None,
        # reset 3S state
        "three_s": {},
    })

# ---------- Guard ----------
if not st.session_state["data_ready"]:
    st.info("Enter a ticker and click *Generate Report* to begin.")
    st.stop()

# ---------- Retrieve ----------
ticker      = st.session_state["ticker"]
timeframe   = st.session_state["timeframe"]
price_df    = st.session_state["price_df"]
inc         = st.session_state["inc"]
bs          = st.session_state["bs"]
cf          = st.session_state["cf"]
cf_q        = st.session_state.get("cf_q")
extras      = st.session_state["extras"]
years_back  = st.session_state["years_back"]

# ---------- Price chart ----------
st.subheader("Price Performance")
st.line_chart(price_df["Close"], height=300, use_container_width=True)

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)

# ---------- Financial statements (historical) ----------
desired_inc = ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income", "Net Income", "Ebitda",
               "Diluted Average Shares","Basic Average Shares","Weighted Average Shs Out Dil","Weighted Average Shs Out"]
inc_cols = [c for c in desired_inc if c in inc.columns] + [c for c in inc.columns if c not in desired_inc]

desired_bs = ["Total Assets", "Total Current Assets", "Total Liab", "Total Current Liabilities", "Total Debt", "Cash", "Cash And Cash Equivalents","Short Term Investments"]
equity_cols = [c for c in bs.columns if "equity" in c.lower()]
bs_cols = [c for c in desired_bs if c in bs.columns] + equity_cols + \
          [c for c in bs.columns if c not in desired_bs + equity_cols]

cf_pref = ["Total Cash From Operating Activities", "Capital Expenditures"]
cf_cols = [c for c in cf_pref if c in cf.columns] + [c for c in cf.columns if c not in cf_pref]

st.markdown('<div class="ui-card"><div class="ui-card-title">Income Statement</div>', unsafe_allow_html=True)
st.dataframe(format_statement(inc[inc_cols].head(years_back), decimals=0), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="ui-card"><div class="ui-card-title">Balance Sheet</div>', unsafe_allow_html=True)
st.dataframe(format_statement(bs[bs_cols].head(years_back), decimals=0), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="ui-card"><div class="ui-card-title">Cash Flow</div>', unsafe_allow_html=True)
st.dataframe(format_statement(cf[cf_cols].head(years_back), decimals=0), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Metrics ----------
metrics = compute_metrics(inc, bs, cf, years_back)
metrics.update({
    "Market Cap ($)": extras.get("Market Cap ($)"),
    "P/E Ratio": extras.get("P/E Ratio"),
    "EV/EBITDA": extras.get("EV/EBITDA"),
})

fmt_metrics = {}
for k, v in metrics.items():
    if isinstance(v, (int, float)):
        if "$" in k:
            fmt_metrics[k] = f"${v:,.0f}"
        elif "%" in k:
            fmt_metrics[k] = f"{v:.2f}%"
        elif "x" in k:
            fmt_metrics[k] = f"{v:.2f}x"
        else:
            fmt_metrics[k] = f"{v:.2f}"
    else:
        fmt_metrics[k] = v

ordered_keys = ["Market Cap ($)", "P/E Ratio", "EV/EBITDA"] + \
               [k for k in fmt_metrics if k not in ("Market Cap ($)", "P/E Ratio", "EV/EBITDA")]

st.subheader("Key Metrics & Ratios")
st.table(pd.DataFrame.from_dict({k: fmt_metrics[k] for k in ordered_keys}, orient="index", columns=["Value"]))

# ---------- Comps ----------
st.subheader("Comparable Companies")
st.caption("You control the comp set. Replace/add true peers by model/size/growth/region.")
peer_input = st.text_input("Peer tickers (comma-separated)", "MSFT,GOOG,AMZN")
peers = [p.strip().upper() for p in peer_input.split(",") if p.strip()]

rows = []
for peer in peers:
    try:
        info = yf.Ticker(peer).info or {}
        rows.append({
            "Ticker": peer,
            "P/E": info.get("trailingPE"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
            "P/B": info.get("priceToBook"),
            "P/S": info.get("priceToSalesTrailing12Months"),
            "Dividend Yield": info.get("dividendYield"),
        })
    except Exception:
        rows.append({"Ticker": peer})
comps_df = pd.DataFrame(rows).set_index("Ticker")
st.table(comps_df)

# ===== AI Comps & Market Analysis =====

def _prepare_comp_stats(df: pd.DataFrame):
    out = {}
    for col, as_mult in [("P/E", True), ("EV/EBITDA", True), ("P/B", True), ("P/S", True), ("Dividend Yield", False)]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                out[col] = {
                    "median": float(np.nanmedian(series)),
                    "min": float(np.nanmin(series)),
                    "max": float(np.nanmax(series)),
                    "count": int(series.notna().sum()),
                }
            else:
                out[col] = None
        else:
            out[col] = None
    return out


def _make_comps_prompt(ticker: str, extras: dict, comps_df: pd.DataFrame, stats: dict, timeframe: str, fmt_metrics: dict):
    sector = extras.get("Sector") or "—"
    industry = extras.get("Industry") or "—"
    target_pe = extras.get("P/E Ratio")
    target_ebitda = extras.get("EV/EBITDA")

    lines = [
        f"Ticker: {ticker}",
        f"Timeframe: {timeframe}",
        f"Sector: {sector} | Industry: {industry}",
        f"Target P/E: {target_pe} | Target EV/EBITDA: {target_ebitda}",
        f"Peers provided: {', '.join(comps_df.index.tolist()) or '—'}",
        "",
        "Peer multiples summary (median/min/max/count):"
    ]
    for col in ["P/E", "EV/EBITDA", "P/B", "P/S", "Dividend Yield"]:
        s = stats.get(col)
        if s:
            lines.append(f"- {col}: median={s['median']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}, n={s['count']}")
        else:
            lines.append(f"- {col}: n/a")
    lines.append("")
    lines.append("Other target metrics: " + "; ".join([f"{k}={v}" for k,v in fmt_metrics.items() if k in ("P/E Ratio","EV/EBITDA","Market Cap ($)")]))
    ctx = "\n".join(lines)

    prompt = (
        "You are an equity analyst. Write a compact analysis of the provided comp set and the market context.\n"
        "Deliver 3 short sections with bullets:\n"
        "1) Peer Set Sanity Check – Are these peers appropriate? Note any mismatches and suggest 1–3 adjustments. However, don't overcritique, if the comps chosen are somewhat relevant, only suggest a new company if significantly needed\n"
        "2) Relative Valuation – Where does the target sit vs comp medians on P/E and EV/EBITDA (cheap/fair/expensive)? Mention P/B or P/S if more relevant.\n"
        "3) Market Context & Takeaways – Brief comment on sector backdrop implied by peers, plus 2–3 concrete diligence items.\n"
        "Keep it crisp, numbers-driven, and readable\n\n"
        f"DATA:\n{ctx}"
    )
    return prompt


def generate_comps_analysis_if_needed():
    if not openai.api_key:
        st.info("Add an OpenAI API key in Secrets to enable AI comps analysis.")
        return
    if comps_df is None or comps_df.empty:
        st.session_state["comps_analysis"] = None
        return

    stats = _prepare_comp_stats(comps_df)
    prompt = _make_comps_prompt(ticker, extras, comps_df, stats, timeframe, fmt_metrics)
    with st.spinner("🔎 Analyzing comps & market …"):
        try:
            resp = _create_chat(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a rigorous, numbers-first equity analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=650,
            )
            st.session_state["comps_analysis"] = _extract_chat(resp)
        except Exception as e:
            st.session_state["comps_analysis"] = f"_Comps analysis unavailable_: {e}"

comps_changed = (st.session_state.get("peer_input_prev") != peer_input)
if submitted or comps_changed:
    generate_comps_analysis_if_needed()
st.session_state["peer_input_prev"] = peer_input

st.markdown("**Comps & Market Analysis (AI)**")
colA, colB = st.columns([1, 5])
with colA:
    if st.button("🔁 Refresh", use_container_width=True):
        generate_comps_analysis_if_needed()
with colB:
    if st.session_state.get("comps_analysis"):
        st.markdown(st.session_state["comps_analysis"])
    else:
        st.caption("Edit the peer list above (or click **Refresh**) to generate an AI summary of the comp set and market context.")

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)

# -------------------- NEW: 3-Statement Model (Driver-based) -----

st.markdown('<div class="section-label">Modeling</div>', unsafe_allow_html=True)
st.subheader("3-Statement Model (Driver-based)")

drivers = derive_default_drivers(inc, bs, cf)
with st.expander("Assumptions (edit to taste)"):
    c1, c2, c3 = st.columns(3)

    with c1:
        years_proj = st.slider("Projection Years (N)", 3, 10, 5)
        drivers["revg"]        = percent_input("Revenue Growth (g)", key="drv_revg",        default=float(drivers["revg"]))
        drivers["ebit_margin"] = percent_input("EBIT Margin",         key="drv_ebit_m",      default=float(drivers["ebit_margin"]))
        drivers["tax_rate"]    = percent_input("Tax Rate",            key="drv_tax",         default=float(drivers["tax_rate"]))

    with c2:
        drivers["dep_pct_rev"]   = percent_input("Depreciation / Revenue", key="drv_dep_pct",   default=float(drivers["dep_pct_rev"]))
        drivers["capex_pct_rev"] = percent_input("CapEx / Revenue",        key="drv_capex_pct", default=float(drivers["capex_pct_rev"]))
        drivers["nwc_pct_rev"]   = percent_input("NWC / Revenue",          key="drv_nwc_pct",   default=float(drivers["nwc_pct_rev"]))

    with c3:
        drivers["interest_rate"] = percent_input("Interest Rate on Debt",  key="drv_int_rate",  default=float(drivers["interest_rate"]))
        drivers["div_payout"]    = percent_input("Dividend Payout (of NI)",key="drv_payout",    default=float(drivers["div_payout"]), min_pct=0, max_pct=100)

inc_p, bs_p, cf_p, fcf_series_3s, equity_path_3s, equity0_3s = project_three_statements(drivers, years_proj)

st.markdown("**Income Statement (Projected)**")
st.dataframe( format_statement( inc_p.set_index("Year"), decimals=0 ), use_container_width=True )

st.markdown("**Balance Sheet (Projected)**")
st.dataframe( format_statement( bs_p.set_index("Year"), decimals=0 ), use_container_width=True )

st.markdown("**Cash Flow (Projected)**")
st.dataframe( format_statement( cf_p.set_index("Year"), decimals=0 ), use_container_width=True )

st.markdown("**Unlevered FCF from 3S**")
st.table( fcf_series_3s.reset_index().rename(columns={"index":"Year"}).style.format({"Unlevered FCF":"${:,.0f}"}) )

# cache to session for later sections/export
st.session_state["three_s"] = {
    "drivers": drivers,
    "inc": inc_p,
    "bs": bs_p,
    "cf": cf_p,
    "fcf": fcf_series_3s,
    "equity_path": equity_path_3s,
    "equity0": equity0_3s,
}

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)

# -------------------- WACC & Growth Consistency -----------------

st.markdown('<div class="section-label">Valuation Inputs</div>', unsafe_allow_html=True)
st.subheader("Cost of Capital (WACC)")
with st.expander("Fundamentals-based WACC (CAPM)", expanded=False):
    rf   = percent_input("Risk-free rate (rf)",          key="rf",   default=st.session_state.get("rf", 0.04))
    erp  = percent_input("Equity risk premium (ERP)",    key="erp",  default=st.session_state.get("erp", 0.05))
    beta = st.number_input("Equity beta (levered)",      key="beta", value=st.session_state.get("beta", 1.20), step=0.05, format="%.2f")
    kd   = percent_input("Cost of debt (pre-tax)",       key="kd",   default=st.session_state.get("kd", 0.05))
    tax  = percent_input("Tax rate (effective)",         key="tax",  default=st.session_state.get("tax", 0.21))
    dv   = percent_input("Target D/V (debt share of capital)", key="dv", default=st.session_state.get("dv", 0.00), min_pct=0, max_pct=100)

    D = max(0.0, min(1.0, float(dv)))
    E = 1.0 - D
    ke_capm = float(rf) + float(beta) * float(erp)
    kd_after = float(kd) * (1.0 - float(tax))
    wacc = ke_capm * E + kd_after * D

    st.metric("WACC (derived)", f"{wacc*100:.2f}%")
    use_wacc = st.checkbox("Use WACC as discount rate in DCF", key="use_wacc", value=st.session_state.get("use_wacc", False))

st.subheader("Growth Consistency (g ≈ ROIC × Reinvestment)")
with st.expander("Cross-check sustainable growth and reinvestment", expanded=False):
    roic_guess = 0.12
    try:
        for k in metrics:
            if k.startswith("ROIC") and isinstance(metrics[k], (int, float)):
                roic_guess = max(0.0, float(metrics[k]) / 100.0)
                break
    except Exception:
        pass

    roic = percent_input("ROIC (after-tax)",        key="roic",     default=st.session_state.get("roic", roic_guess))
    reinvest = percent_input("Reinvestment rate",   key="reinvest", default=st.session_state.get("reinvest", 0.00), min_pct=0, max_pct=100)
    implied_g = float(roic) * float(reinvest)
    st.write(f"Implied sustainable growth g ≈ **{implied_g*100:.2f}%**")

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)

# -------------------- DCF (two-stage + 3S-driven) --------------

st.markdown('<div class="section-label">Valuation Models</div>', unsafe_allow_html=True)
st.subheader("Discounted Cash-Flow (DCF) Valuation")

last_close = float(price_df["Close"].iloc[-1])
shares_out = pick_shares(extras.get("_info", {}), inc, bs)
if shares_out is None and extras.get("Market Cap ($)"):
    try:
        shares_out = int(extras["Market Cap ($)"] / last_close)
    except Exception:
        shares_out = extras.get("Shares Outstanding (raw)")
net_debt = extras.get("Net Debt ($)") or 0.0
base_fcf = get_base_fcf(cf, cf_quarterly=cf_q)

st.session_state.setdefault("base_fcf", base_fcf)
st.session_state.setdefault("shares_out", shares_out)
st.session_state.setdefault("net_debt", net_debt)
st.session_state.setdefault("last_close", last_close)

# Controls
colm1, colm2 = st.columns([1.2, 1])
with colm1:
    dcf_method = st.radio("DCF method", ["Two-Stage (Simple)", "3S-Driven (from model)"], horizontal=True, key="dcf_choice")
with colm2:
    st.caption("Two-Stage grows a single FCF; 3S-Driven discounts the unlevered FCFs produced by the 3-statement model.")

# Two-Stage inputs
percent_input("Stage-1 FCF Growth (g₁)", key="g1", default=st.session_state.get("g1", 0.10))
percent_input("Terminal Growth (g₂)",    key="g2", default=st.session_state.get("g2", 0.025))
percent_input("Discount Rate (r)",       key="r",  default=st.session_state.get("r", 0.09))
st.slider("Projection Years (N)", min_value=1, max_value=20, key="N", value=st.session_state.get("N", 5))
st.checkbox("Fade g₁ → g₂ across N (more realistic)", key="fade", value=st.session_state.get("fade", True))

st.radio("Calibrate to current price by solving for:", ["Discount rate (r)", "Stage-1 growth (g₁)"], key="calib_mode", horizontal=True)

# Calibrate button

def _calibrate_callback():
    s = st.session_state
    base = s.get("base_fcf"); sh = s.get("shares_out")
    nd = s.get("net_debt") or 0.0; N = int(s.get("N", 5))
    g2 = float(s.get("g2", 0.025)); last_px = s.get("last_close")
    fade = bool(s.get("fade", True))
    if base is None or not sh or not last_px:
        s["calib_error"] = "Need latest FCF, shares, and current price to calibrate."
        return
    mode = s.get("calib_mode", "Discount rate (r)")
    if mode == "Discount rate (r)":
        g1 = float(s.get("g1", 0.10))
        sol = solve_implied_r(last_px, base, sh, nd, N, g1, g2, fade=fade)
        if sol is None:
            s["calib_error"] = "No feasible r; adjust g₁/g₂ or N."
        else:
            s["r"] = float(round(sol, 4)); s["calib_error"] = ""
    else:
        r = float(s.get("r", 0.09))
        sol = solve_implied_g1(last_px, base, sh, nd, N, r, g2, fade=fade)
        if sol is None:
            s["calib_error"] = "No feasible g₁; adjust r/g₂ or N."
        else:
            s["g1"] = float(round(sol, 4)); s["calib_error"] = ""

st.button("📐 Auto-calibrate to market", on_click=_calibrate_callback)
if st.session_state.get("calib_error"):
    st.warning(st.session_state["calib_error"])

# Resolve inputs
_g1 = float(st.session_state["g1"])
_g2 = float(st.session_state["g2"])
_r_manual = float(st.session_state["r"])
r_eff = float(wacc) if st.session_state.get("use_wacc") else _r_manual
N  = int(st.session_state["N"]) ; fade = bool(st.session_state.get("fade", True))

# Compute both DCF flavors
implied_px_two = None; dcf_df_two = None; dcf_head_two = None
implied_px_3s  = None; dcf_df_3s  = None; dcf_head_3s  = None

try:
    implied_px_two = dcf_two_stage_price(base_fcf, shares_out, net_debt, N, _g1, _g2, r_eff, fade=fade)
    if implied_px_two is not None:
        # build detail for display
        years = list(range(1, N + 1))
        if fade:
            growth_path = [_g1 + (_g2 - _g1) * (y - 1) / (N - 1 if N > 1 else 1) for y in years]
            proj_fcf = []
            f = base_fcf
            for gr in growth_path:
                f = f * (1 + gr)
                proj_fcf.append(f)
        else:
            growth_path = [_g1 for _ in years]
            proj_fcf = [base_fcf * (1 + _g1) ** y for y in years]
        pv_fcf   = [fcf / (1 + r_eff) ** y for y, fcf in zip(years, proj_fcf)]
        fcf_N1   = proj_fcf[-1] * (1 + _g2)
        term_val = fcf_N1 / (r_eff - _g2)
        term_pv  = term_val / (1 + r_eff) ** N
        ev       = sum(pv_fcf) + term_pv
        eq_v     = ev - (net_debt or 0.0)
        dcf_df_two = pd.DataFrame({"Year": years + ["Terminal"], "FCF": proj_fcf + [fcf_N1], "PV FCF": pv_fcf + [term_pv]})
        term_pct = term_pv / ev if ev else None
        y1_pv = pv_fcf[0] if pv_fcf else None
        pv_mult = (sum(pv_fcf) / y1_pv) if (y1_pv and y1_pv != 0) else None
        dcf_head_two = pd.Series({
            "Enterprise Value": ev,
            "Less: Net Debt":   net_debt,
            "Equity Value":     eq_v,
            "Shares Out":       shares_out,
            "Implied Price":    implied_px_two,
            "Current Price":    last_close,
            "Base FCF (latest)":  base_fcf,
            "g₁ (stage-1)":       _g1,
            "g₂ (terminal)":      _g2,
            "r (discount)":       r_eff,
            "N (years)":          N,
            "Fade g₁→g₂":         fade,
            "Terminal PV / EV":   f"{term_pct:.1%}" if term_pct is not None else "—",
                        "PV(Years 1..N) / PV(Year1) (x)": f"{pv_mult:.1f}x" if pv_mult is not None else "—",
        }).to_frame("Value")
        dcf_head_two.loc["N (years)", "Value"] = f"{int(N)}"
except Exception as e:
    pass

try:
    implied_px_3s, dcf_df_3s, dcf_head_3s = build_dcf_from_series(
        st.session_state["three_s"].get("fcf"),
        r=r_eff, g_term=_g2, net_debt=net_debt, shares_out=shares_out
    )
except Exception:
    pass

# Show both in tabs
_tab_two, _tab_3s = st.tabs(["Two-Stage (Simple)", "3S-Driven (from model)"])
with _tab_two:
    if dcf_df_two is not None:
        st.table(dcf_df_two.style.format({"FCF": "${:,.0f}", "PV FCF": "${:,.0f}"}).hide(axis="index"))

        # custom per-field formatting
        def _fmt_val(k, v):
            if v is None: return "—"
            if k in {"Enterprise Value","Less: Net Debt","Equity Value","Implied Price","Current Price","Base FCF (latest)"}:
                return f"${v:,.0f}"
            if k in {"g₁ (stage-1)","g₂ (terminal)","r (discount)","Terminal PV / EV"}:
                try: return f"{float(v)*100:.2f}%"
                except: return str(v)
            if k in {"PV(Years 1..N) / PV(Year1) (x)"}:
                try: return f"{float(v):.1f}x"
                except: return str(v)
            if k in {"N (years)","Shares Out"}:
                try: return f"{int(v):,}"
                except: return str(v)
            if k == "Fade g₁→g₂":
                return "Yes" if bool(v) else "No"
            return f"{v}"

        _df = pd.DataFrame(
            {"Metric": list(dcf_head_two.index), "Value": [ _fmt_val(k, dcf_head_two.loc[k, "Value"]) for k in dcf_head_two.index ]}
        )
        st.table(_df)
    else:
        st.info("Two-stage DCF unavailable – check inputs (r > g₂, base FCF, shares).")

with _tab_3s:
    if dcf_df_3s is not None:
        st.table(dcf_df_3s.style.format({"FCF": "${:,.0f}", "PV FCF": "${:,.0f}"}).hide(axis="index"))
        st.table(dcf_head_3s.style.format({"Value": lambda x: f"${x:,.0f}" if isinstance(x,(int,float)) else x}))
    else:
        st.info("3S-driven DCF unavailable – ensure the 3-statement model produced FCFs and r > g₂.")

# Choose implied_px for downstream sections
if st.session_state["dcf_choice"] == "3S-Driven (from model)" and (implied_px_3s is not None):
    implied_px_chosen = implied_px_3s
else:
    implied_px_chosen = implied_px_two

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)

# -------------------- Residual Income (Financials) --------------

st.markdown('<div class="section-label">Financials Valuation</div>', unsafe_allow_html=True)
st.subheader("Residual Income (Banks/Insurers)")

auto_fin = detect_is_financials(extras)
use_financials_mode = st.checkbox(
    "Use Financials (Residual Income) mode",
    value=auto_fin,
    help=f"Auto-detected financials: {auto_fin} (Sector: {extras.get('Sector')}, Industry: {extras.get('Industry')})"
)

bve0_guess, roe0_guess = estimate_starting_bve_and_roe(inc, bs)
ke_default = float(
    (st.session_state.get("rf", 0.04) + st.session_state.get("beta", 1.20) * st.session_state.get("erp", 0.05))
)

col1, col2 = st.columns(2)
with col1:
    bve0 = st.number_input("Book Value of Equity (BV₀, $)", value=float(bve0_guess or 0.0), step=1_000_000.0, format="%.0f")
    ri_shares = st.number_input("Shares Outstanding", value=float(shares_out or 0), step=1_000.0, format="%.0f")
    ri_N = st.slider("Projection Years (N, RI)", min_value=1, max_value=20, value=st.session_state.get("ri_N", 5))
    payout = percent_input("Dividend Payout Ratio", key="ri_payout", default=0.00, min_pct=0, max_pct=100)
with col2:
    ke_in     = percent_input("Cost of Equity kₑ", key="ke_in",     default=float(ke_default))
    roe_start = percent_input("Starting ROE",      key="roe_start", default=float(roe0_guess or 0.10))
    fade_to_ke = st.checkbox("Fade ROE → kₑ by year N (Terminal RI ≈ 0)", value=True)
    if not fade_to_ke:
        roe_terminal = percent_input("Terminal ROE", key="ri_roe_terminal", default=float(roe0_guess or 0.10))
        g_ri = percent_input("Terminal g (for continuing RI)", key="ri_g", default=0.00)
    else:
        roe_terminal, g_ri = None, 0.0

ri_price, ri_df, ri_summary = residual_income_valuation(
    bv0=bve0, shares_out=int(ri_shares) if ri_shares else None, ke=float(ke_in),
    N=int(ri_N), roe_start=float(roe_start), payout=float(payout),
    fade_to_ke=fade_to_ke, roe_terminal=roe_terminal, g_ri=float(g_ri)
)

if ri_price:
    st.write(f"**Residual Income implied price:** ${ri_price:,.2f}")
    st.dataframe(ri_df.style.format({
        "BV_Beg": "${:,.0f}", "ROE": "{:.2%}",
        "Earnings": "${:,.0f}", "Dividends": "${:,.0f}",
        "BV_End": "${:,.0f}", "Residual Income": "${:,.0f}",
        "PV RI": "${:,.0f}"
    }), use_container_width=True)

# Persist RI context for export
st.session_state["ri_params"] = {
    "use_financials_mode": bool(use_financials_mode),
    "bv0": float(bve0 or 0.0),
    "shares": int(ri_shares or 0),
    "ke": float(ke_in or 0.0),
    "N": int(ri_N),
    "roe_start": float(roe_start or 0.0),
    "payout": float(payout or 0.0),
    "fade_to_ke": bool(fade_to_ke),
    "roe_terminal": float(roe_terminal) if (not fade_to_ke and roe_terminal is not None) else None,
    "g_ri": float(g_ri or 0.0),
    "ri_price": float(ri_price or 0.0),
}

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)

# -------------------- Executive Summary (AI) --------------------

st.subheader("Executive Summary (AI)")
st.caption("Click the button to (re)generate using the current valuation settings.")

if st.button("🧠 Generate / Refresh AI Summary"):
    try:
        # decide which valuation “counts”
        implied_pick = None
        method_label = None

        if use_financials_mode and ri_price:
            implied_pick = ri_price
            method_label = "Residual Income"
        else:
            # prefer whichever DCF tab is chosen and available
            if st.session_state["dcf_choice"] == "3S-Driven (from model)" and (implied_px_3s is not None):
                implied_pick = implied_px_3s
                method_label = "DCF (3S-Driven)"
            elif implied_px_two is not None:
                implied_pick = implied_px_two
                method_label = "DCF (Two-Stage)"

        upside_pct = None
        if implied_pick and st.session_state.get("last_close"):
            upside_pct = (implied_pick / st.session_state["last_close"] - 1.0) * 100.0

        method_label = (
            "Residual Income" if (use_financials_mode and ri_price)
            else ("DCF (3S-Driven)" if (st.session_state["dcf_choice"] == "3S-Driven (from model)" and implied_px_3s)
                else "DCF (Two-Stage)")
        )

        lines_used = []
        if method_label == "DCF (Two-Stage)" and implied_px_two:
            lines_used.append(f"DCF Two-Stage Implied Price: ${implied_px_two:,.2f}")
        elif method_label == "DCF (3S-Driven)" and implied_px_3s:
            lines_used.append(f"DCF 3S-Driven Implied Price: ${implied_px_3s:,.2f}")
        elif method_label == "Residual Income" and ri_price:
            lines_used.append(f"RI Implied Price: ${ri_price:,.2f}")


        context_lines = [
            f"Ticker: {ticker}",
            f"Current Price: ${st.session_state['last_close']:,.2f}",
            f"Market Cap: {fmt_metrics.get('Market Cap ($)','—')}",
            f"Net Debt: ${st.session_state['net_debt']:,.0f}  |  Shares Out: {st.session_state['shares_out'] or '—'}",
            f"Base FCF (latest): ${st.session_state['base_fcf']:,.0f}",
            f"Chosen Valuation Method: {method_label or '—'}",
        ] + lines_used + ([
            f"Upside/Downside (chosen): {upside_pct:.2f}% vs. market"
        ] if upside_pct is not None else [])

        what_true = []
        if st.session_state["dcf_choice"] == "3S-Driven (from model)" and implied_px_3s is not None:
            what_true.append("For 3S: margins, CapEx%Rev, and NWC%Rev drive FCF shape; stress test sensitivities.")
        if implied_px_two is not None:
            if fade:
                what_true.append(f"For two-stage: Stage-1 FCF must compound at ~{_g1:.2%} for {N} years (fading to {_g2:.2%}).")
            else:
                what_true.append(f"For two-stage: Stage-1 FCF must compound at ~{_g1:.2%} for {N} years.")
        if use_financials_mode and ri_price:
            what_true.append("Residual income adds value only when ROE exceeds kₑ; fading ROE → kₑ dampens terminal contribution.")

        summary_prompt = (
            "You are a buy-side partner reviewing an analyst's report.\n"
            "Write a crisp executive summary using ONLY the models provided in DATA.\n"
            "Do not mention models that are not listed.\n"
            "Structure: 1) Key Strengths  2) Key Risks  3) Valuation  4) What-Must-Be-True  5) Bottom Line.\n\n"
            "DATA:\n" + "\n".join("- " + line for line in context_lines) +
            ("\n\nWhat-must-be-true:\n" + "\n".join("- " + line for line in what_true) if what_true else "")
        )

        with st.spinner("✍️  Writing summary …"):
            resp = _create_chat(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a discerning buy-side partner."},
                    {"role": "user",   "content": summary_prompt},
                ],
                temperature=0.5,
                max_tokens=700,
            )
            st.session_state["exec_summary"] = _extract_chat(resp)
    except Exception as e:
        st.session_state["exec_summary"] = f"_Summary unavailable_: {e}"

if st.session_state["exec_summary"]:
    st.markdown(st.session_state["exec_summary"])
else:
    st.info("Click **Generate / Refresh AI Summary** to produce the write-up including the calibrated DCF/RI and 3S context.")

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)

# -------------------- Excel Report (extended with 3S) -----------

def build_excel_report(
    ticker: str,
    timeframe: str,
    price_df: pd.DataFrame,
    inc_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    comps_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    # DCF context
    base_fcf: float,
    shares_out: int,
    net_debt: float,
    g1: float,
    g2: float,
    r: float,
    N: int,
    current_price: float,
    fade_flag: bool,
    # WACC & Growth-consistency inputs
    use_wacc: bool = False,
    rf: float = 0.04,
    erp: float = 0.05,
    beta: float = 1.20,
    kd: float = 0.05,
    tax: float = 0.21,
    dv: float = 0.00,
    roic: float = 0.12,
    reinvest: float = 0.00,
    # Residual Income params
    fin_mode: bool = False,
    ri_bv0: float = 0.0,
    ri_shares: int = 0,
    ri_ke: float = 0.10,
    ri_N: int = 5,
    ri_roe_start: float = 0.10,
    ri_payout: float = 0.00,
    ri_fade_to_ke: bool = True,
    ri_roe_terminal: Optional[float] = None,
    ri_g: float = 0.00,
    # NEW: 3S projections
    proj_income_df: Optional[pd.DataFrame] = None,
    proj_balance_df: Optional[pd.DataFrame] = None,
    proj_cash_df: Optional[pd.DataFrame] = None,
    proj_fcf_series: Optional[pd.Series] = None,
):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        wb = writer.book
        if hasattr(wb, "set_calc_mode"):
            wb.set_calc_mode("auto")

        fmt_hdr   = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
        fmt_bold  = wb.add_format({"bold": True})
        fmt_money = wb.add_format({"num_format": "#,##0;(#,##0)", "align": "right"})
        fmt_usd   = wb.add_format({"num_format": "$#,##0;($#,##0)", "align": "right"})
        fmt_pct4  = wb.add_format({"num_format": "0.0000%", "align": "right"})
        fmt_pct2  = wb.add_format({"num_format": "0.00%", "align": "right"})
        fmt_num   = wb.add_format({"num_format": "#,##0", "align": "right"})
        fmt_txt   = wb.add_format({"align": "left"})
        fmt_note  = wb.add_format({"italic": True, "font_color": "#666"})
        fmt_title = wb.add_format({"bold": True, "font_size": 14})
        fmt_wrap  = wb.add_format({"text_wrap": True, "valign": "top"})

        # -------- Overview --------
        ov = wb.add_worksheet("Overview")
        ov.write("A1", f"{ticker} — Analyst Report", fmt_title)
        ov.write("A3", "Timeframe:", fmt_bold); ov.write("B3", timeframe)
        ov.write("A4", "Current Price:", fmt_bold); ov.write_number("B4", float(current_price or 0.0), fmt_usd)
        ov.write("A6", "Instructions:", fmt_bold)
        ov.write("A7", "Adjust assumptions on the 'DCF Model' and 'Residual Income' sheets. Implied prices update automatically.", fmt_note)
        ov.write("A9", "DCF Implied Price:", fmt_bold);   ov.write_formula("B9", "=IMPLIED_PRICE_DCF")
        ov.write("A10", "RI Implied Price:", fmt_bold);   ov.write_formula("B10", "=IMPLIED_PRICE_RI")
        ov.write("A12", "Chosen Price:", fmt_bold)
        ov.write_formula("B12", "=IF(CHOOSE_RI, IMPLIED_PRICE_RI, IMPLIED_PRICE_DCF)")
        ov.write("A13", "Upside vs Current:", fmt_bold)
        ov.write_formula("B13", "=(IF(CHOOSE_RI, IMPLIED_PRICE_RI, IMPLIED_PRICE_DCF)/B4)-1", fmt_pct2)
        ov.set_column("A:A", 22); ov.set_column("B:B", 18)

        # ---- FIXED & HARDENED ----
        def _index_looks_date_like(idx: pd.Index) -> bool:
            # True for Datetime/Period indexes or if most values parse to dates
            if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
                return True
            try:
                s = pd.Series(idx)
                parsed = pd.to_datetime(s, errors="coerce")
                return parsed.notna().mean() >= 0.6  # majority are valid dates
            except Exception:
                return False

        def write_df(sheetname, df, money_col_idxs=None, width=22):
            """Write any DataFrame to a sheet, with a robust 'Period' column."""
            ws = wb.add_worksheet(sheetname)
            if df is None or df.empty:
                ws.write("A1", f"No data for {sheetname}", fmt_note)
                return ws

            out = df.reset_index()

            # Ensure the first column is named 'Period' (regardless of prior name)
            first_col = out.columns[0]
            if first_col != "Period":
                out = out.rename(columns={first_col: "Period"})

            # Only coerce to dates if the *original* index looked date-like
            if _index_looks_date_like(df.index):
                out["Period"] = pd.to_datetime(out["Period"], errors="coerce").dt.date

            out = out.fillna("")
            out.to_excel(writer, sheet_name=sheetname, index=False)

            ws = writer.sheets[sheetname]
            for c in range(len(out.columns)):
                ws.write(0, c, out.columns[c], fmt_hdr)
            ws.freeze_panes(1, 1)
            ws.set_column(0, 0, 16)
            ws.set_column(1, len(out.columns)-1, width)

            if money_col_idxs:
                for c_idx in money_col_idxs:
                    ws.set_column(c_idx, c_idx, width, fmt_money)
            return ws

        # Historical statements
        income_priority = ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income", "Net Income", "Ebitda",
                           "Diluted Average Shares","Basic Average Shares","Weighted Average Shs Out Dil","Weighted Average Shs Out"]
        income_cols = [c for c in income_priority if c in inc_df.columns] + [c for c in inc_df.columns if c not in income_priority]
        write_df("Income",  inc_df[income_cols], money_col_idxs=[i for i,_ in enumerate(income_cols, start=1)])

        eq_cols = [c for c in bs_df.columns if "equity" in c.lower()]
        bs_priority = ["Total Assets","Total Current Assets","Total Liab","Total Current Liabilities","Total Debt","Cash","Cash And Cash Equivalents","Short Term Investments"] + eq_cols
        bs_cols_x = [c for c in bs_priority if c in bs_df.columns] + [c for c in bs_df.columns if c not in bs_priority]
        write_df("Balance", bs_df[bs_cols_x], money_col_idxs=[i for i,_ in enumerate(bs_cols_x, start=1)])

        cf_priority = ["Total Cash From Operating Activities","Capital Expenditures"]
        cf_cols_x = [c for c in cf_priority if c in cf_df.columns] + [c for c in cf_df.columns if c not in cf_priority]
        write_df("CashFlow", cf_df[cf_cols_x], money_col_idxs=[i for i,_ in enumerate(cf_cols_x, start=1)])

        # 3S projections
        if proj_income_df is not None and not proj_income_df.empty:
            write_df("Proj_Income", proj_income_df.set_index("Year"))
        else:
            wb.add_worksheet("Proj_Income").write("A1", "No projected income", fmt_note)
        if proj_balance_df is not None and not proj_balance_df.empty:
            write_df("Proj_Balance", proj_balance_df.set_index("Year"))
        else:
            wb.add_worksheet("Proj_Balance").write("A1", "No projected balance", fmt_note)
        if proj_cash_df is not None and not proj_cash_df.empty:
            write_df("Proj_CashFlow", proj_cash_df.set_index("Year"))
        else:
            wb.add_worksheet("Proj_CashFlow").write("A1", "No projected cash flow", fmt_note)
        if proj_fcf_series is not None and not proj_fcf_series.empty:
            fcf_df = proj_fcf_series.reset_index(); fcf_df.columns = ["Year","Unlevered FCF"]
            fcf_df.to_excel(writer, sheet_name="Proj_FCF", index=False)
            ws = writer.sheets["Proj_FCF"]; ws.write_row("A1", ["Year","Unlevered FCF"], fmt_hdr)
            ws.set_column(0, 0, 10); ws.set_column(1, 1, 20, fmt_usd)
        else:
            wb.add_worksheet("Proj_FCF").write("A1", "No 3S FCF", fmt_note)

        # Comps & metrics
        comps_out = comps_df.reset_index(drop=True) if (comps_df is not None and not comps_df.empty) else pd.DataFrame()
        if comps_out.empty:
            wb.add_worksheet("Comps").write("A1", "No comps data", fmt_note)
        else:
            comps_out.to_excel(writer, sheet_name="Comps", index=False)
            ws = writer.sheets["Comps"]
            for c in range(len(comps_out.columns)):
                ws.write(0, c, comps_out.columns[c], fmt_hdr)
            ws.freeze_panes(1, 1); ws.set_column(0, len(comps_out.columns)-1, 18)

        if metrics_df is None or metrics_df.empty:
            wb.add_worksheet("Key Metrics").write("A1", "No key metrics", fmt_note)
        else:
            metrics_df.to_excel(writer, sheet_name="Key Metrics", index=False)
            ws = writer.sheets["Key Metrics"]
            for c in range(len(metrics_df.columns)):
                ws.write(0, c, metrics_df.columns[c], fmt_hdr)
            ws.freeze_panes(1, 1); ws.set_column(0, len(metrics_df.columns)-1, 24)

        # Price with chart
        price_ws = wb.add_worksheet("Price")
        price_ws.write_row("A1", ["Date","Close"], fmt_hdr)
        px = price_df.reset_index()[["Date","Close"]].copy()
        px["Date"] = pd.to_datetime(px["Date"], errors="coerce")
        try: px["Date"] = px["Date"].dt.tz_localize(None)
        except Exception: pass
        px = px.dropna(subset=["Date"])
        date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})
        for i, row in enumerate(px.itertuples(index=False), start=1):
            dt = row.Date.to_pydatetime() if isinstance(row.Date, pd.Timestamp) else row.Date
            price_ws.write_datetime(i, 0, dt, date_fmt)
            price_ws.write_number(i, 1, float(row.Close), fmt_usd)
        price_ws.set_column(0, 0, 14); price_ws.set_column(1, 1, 14)
        npx = len(px)
        chart = wb.add_chart({"type": "line"})
        chart.add_series({"name": "Close", "categories": ["Price", 1, 0, npx, 0], "values": ["Price", 1, 1, npx, 1]})
        chart.set_title({"name": f"{ticker} Price"}); chart.set_y_axis({"num_format": "$#,##0"}); chart.set_legend({"position": "bottom"})
        price_ws.insert_chart("D2", chart, {"x_scale": 1.4, "y_scale": 1.2})

        # DCF Model (interactive, two-stage)
        dcf = wb.add_worksheet("DCF Model")
        dcf.set_column("A:A", 10); dcf.set_column("B:C", 20); dcf.set_column("D:D", 16); dcf.set_column("E:E", 24)
        dcf.write("A1", "Two-Stage DCF Model", fmt_title)
        dcf.write("A3", "Inputs", fmt_bold)
        dcf.write("A4", "Base FCF (latest)", fmt_txt);  dcf.write_number("B4", float(base_fcf or 0.0), fmt_usd)
        dcf.write("A5", "Stage-1 Growth (g₁)", fmt_txt); dcf.write_number("B5", float(g1 or 0.0), fmt_pct4)
        dcf.write("A6", "Terminal Growth (g₂)", fmt_txt); dcf.write_number("B6", float(g2 or 0.0), fmt_pct4)
        dcf.write("A7", "Discount Rate (r)", fmt_txt);   dcf.write_number("B7", float(r  or 0.0), fmt_pct4)
        dcf.write("A8", "Projection Years (N)", fmt_txt); dcf.write_number("B8", int(N or 0), fmt_num)
        dcf.write("A9", "Shares Outstanding", fmt_txt);   dcf.write_number("B9", float(shares_out or 0), fmt_num)
        dcf.write("A10", "Net Debt", fmt_txt);            dcf.write_number("B10", float(net_debt or 0.0), fmt_usd)
        dcf.write("A11", "Current Price", fmt_txt);       dcf.write_number("B11", float(current_price or 0.0), fmt_usd)
        dcf.write("A12", "Fade g₁→g₂ (0/1)", fmt_txt);   dcf.write_number("B12", 1 if fade_flag else 0, fmt_num)
        dcf.write("A13", "Use WACC (0/1)", fmt_txt);      dcf.write_number("B13", 1 if use_wacc else 0, fmt_num)
        dcf.write("A14", "Discount Rate (r_eff)", fmt_txt)

        dcf.write("D3",  "WACC (Fundamentals)", fmt_bold)
        dcf.write("D4",  "Risk-free (rf)", fmt_txt);                 dcf.write_number("E4",  float(rf or 0.0),  fmt_pct4)
        dcf.write("D5",  "Equity Risk Premium (ERP)", fmt_txt);      dcf.write_number("E5",  float(erp or 0.0), fmt_pct4)
        dcf.write("D6",  "Beta (levered)", fmt_txt);                 dcf.write_number("E6",  float(beta or 0.0), fmt_num)
        dcf.write("D7",  "Cost of Debt (pre-tax)", fmt_txt);         dcf.write_number("E7",  float(kd or 0.0),  fmt_pct4)
        dcf.write("D8",  "Tax Rate", fmt_txt);                       dcf.write_number("E8",  float(tax or 0.0), fmt_pct4)
        dcf.write("D9",  "Target D/V", fmt_txt);                     dcf.write_number("E9",  float(dv or 0.0),  fmt_pct4)
        dcf.write("D10", "Equity Weight (E/V)", fmt_txt);            dcf.write_formula("E10", "=1-E9")
        dcf.write("D11", "After-tax Kd", fmt_txt);                   dcf.write_formula("E11", "=E7*(1-E8)", fmt_pct4)
        dcf.write("D12", "Ke (CAPM)", fmt_txt);                      dcf.write_formula("E12", "=E4 + E6*E5", fmt_pct4)
        dcf.write("D13", "WACC", fmt_bold);                          dcf.write_formula("E13", "=E12*(1-E9)+E11*E9", fmt_pct4)
        dcf.write_formula("B14", "=IF($B$13=1, E13, $B$7)", fmt_pct4)

        # table body
        start_row = 16
        dcf.write(start_row-1, 0, "Year", fmt_hdr)
        dcf.write(start_row-1, 1, "FCF", fmt_hdr)
        dcf.write(start_row-1, 2, "PV FCF", fmt_hdr)
        dcf.write(start_row-1, 3, "g(y)", fmt_hdr)
        volatile_tail = "+0*($B$4+$B$5+$B$6+$B$7+$B$8+$B$9+$B$10+$B$11+$B$12+$B$13+$B$14+E13)"
        for i in range(1, 21):
            r0 = start_row + i - 1
            dcf.write_number(r0, 0, i, fmt_num)
            dcf.write_formula(r0, 3, f'=IF($B$8>={i}, IF($B$12=1, $B$5 + ($B$6-$B$5)*({i}-1)/MAX($B$8-1,1), $B$5), NA()){volatile_tail}')
            if i == 1:
                dcf.write_formula(r0, 1, f'=IF($B$8>={i}, $B$4*(1+D{r0+1}), NA()){volatile_tail}', fmt_usd)
            else:
                dcf.write_formula(r0, 1, f'=IF($B$8>={i}, B{r0}*(1+D{r0+1}), NA()){volatile_tail}', fmt_usd)
            dcf.write_formula(r0, 2, f'=IF($B$8>={i}, B{r0+1}/POWER(1+$B$14,{i}), NA()){volatile_tail}', fmt_usd)
        term_row = start_row + 21
        dcf.write(term_row-1, 0, "Terminal", fmt_hdr)
        dcf.write(term_row-1, 1, "FCF_(N+1)", fmt_hdr)
        dcf.write(term_row-1, 2, "Terminal PV", fmt_hdr)
        dcf.write_formula(term_row, 1, f'=IF($B$8>0, INDEX(B{start_row+1}:B{start_row+20}, $B$8)*(1+$B$6), NA()){volatile_tail}', fmt_usd)
        dcf.write_formula(term_row, 2, f'=IF($B$14>$B$6, B{term_row+1}/($B$14-$B$6)/POWER(1+$B$14,$B$8), NA()){volatile_tail}', fmt_usd)
        sum_row = term_row + 3
        dcf.write(sum_row,   0, "Enterprise Value",   fmt_bold)
        dcf.write(sum_row+1, 0, "Less: Net Debt",     fmt_txt)
        dcf.write(sum_row+2, 0, "Equity Value",       fmt_bold)
        dcf.write(sum_row+3, 0, "Shares Out",         fmt_txt)
        dcf.write(sum_row+4, 0, "Implied Price",      fmt_bold)
        dcf.write(sum_row+5, 0, "Upside vs Current",  fmt_txt)
        dcf.write(sum_row+6, 0, "Terminal PV / EV",   fmt_txt)
        dcf.write(sum_row+7, 0, "PV(Year1..N) / PV(Year1) (x)", fmt_txt)
        dcf.write_formula(sum_row, 1, f'=SUM(INDEX(C{start_row+1}:C{start_row+20},1):INDEX(C{start_row+1}:C{start_row+20},$B$8)) + C{term_row+1}{volatile_tail}', fmt_usd)
        dcf.write_formula(sum_row+1, 1, f'=$B$10', fmt_usd)
        dcf.write_formula(sum_row+2, 1, f'=B{sum_row+1}-B{sum_row+2}', fmt_usd)
        dcf.write_formula(sum_row+3, 1, f'=N($B$9)', fmt_num)
        dcf.write_formula(sum_row+4, 1, f'=IF(N(B{sum_row+3+1})>0, B{sum_row+2+1}/N(B{sum_row+3+1}), NA())', fmt_usd)
        dcf.write_formula(sum_row+5, 1, f'=IF(N($B$11)>0, B{sum_row+4+1}/N($B$11)-1, NA())', fmt_pct2)
        dcf.write_formula(sum_row+6, 1, f'=IF(B{sum_row+1}>0, C{term_row+1}/B{sum_row+1}, NA())', fmt_pct2)
        dcf.write_formula(sum_row+7, 1, f'=IF(INDEX(C{start_row+1}:C{start_row+20},1)>0, SUM(INDEX(C{start_row+1}:C{start_row+20},1):INDEX(C{start_row+1}:C{start_row+20},$B$8))/INDEX(C{start_row+1}:C{start_row+20},1), NA())', fmt_num)
        dcf.set_column("B:C", 20)
        wb.define_name("IMPLIED_PRICE_DCF", f"='DCF Model'!$B${(sum_row+4)+1}")

        # Residual Income sheet (financials)
        ri = wb.add_worksheet("Residual Income")
        ri.set_column("A:A", 10)
        ri.set_column("B:H", 18)
        ri.write("A1", "Residual Income (Financials)", fmt_title)

        # Inputs
        ri.write("A3", "Inputs", fmt_bold)
        ri.write("A4", "BV₀ (Book Value of Equity)", fmt_txt); ri.write_number("B4", float(ri_bv0 or 0.0), fmt_usd)
        ri.write("A5", "Shares Outstanding", fmt_txt);         ri.write_number("B5", float(ri_shares or 0), fmt_num)
        ri.write("A6", "Cost of Equity kₑ", fmt_txt)
        ri.write("A7", "Projection Years (N)", fmt_txt);       ri.write_number("B7", int(ri_N or 0), fmt_num)
        ri.write("A8", "Starting ROE", fmt_txt);               ri.write_number("B8", float(ri_roe_start or 0.0), fmt_pct4)
        ri.write("A9", "Dividend Payout Ratio", fmt_txt);      ri.write_number("B9", float(ri_payout or 0.0), fmt_pct4)
        ri.write("A10","Fade ROE→kₑ (0/1)", fmt_txt);          ri.write_number("B10", 1 if ri_fade_to_ke else 0, fmt_num)
        ri.write("A11","Terminal ROE (if fade=0)", fmt_txt);   ri.write_number("B11", float(ri_roe_terminal or 0.0), fmt_pct4)
        ri.write("A12","Terminal g (if fade=0)", fmt_txt);     ri.write_number("B12", float(ri_g or 0.0), fmt_pct4)

        # CAPM helper for k_e
        ri.write("D3",  "CAPM for kₑ (optional)", fmt_bold)
        ri.write("D4",  "Risk-free (rf)", fmt_txt);           ri.write_number("E4", float(rf or 0.0), fmt_pct4)
        ri.write("D5",  "Equity Risk Premium", fmt_txt);      ri.write_number("E5", float(erp or 0.0), fmt_pct4)
        ri.write("D6",  "Beta (levered)", fmt_txt);           ri.write_number("E6", float(beta or 0.0), fmt_num)
        ri.write("D8",  "kₑ (CAPM)", fmt_txt);                ri.write_formula("E8", "=E4+E6*E5", fmt_pct4)
        ri.write_formula("B6", "=E8", fmt_pct4)

        # Table header
        ri_start = 16
        for idx, h in enumerate(["Year","BV_Beg","ROE","Earnings","Dividends","BV_End","Residual Income","PV RI"]):
            ri.write(ri_start-1, idx, h, fmt_hdr)

        # Table body (20 rows; show N rows via IF)
        for i in range(1, 20+1):
            r0 = ri_start + i - 1
            ri.write_number(r0, 0, i, fmt_num)
            if i == 1:
                ri.write_formula(r0, 1, f'=$B$4')
            else:
                ri.write_formula(r0, 1, f'=F{r0}')
            # ROE path: fade to k_e if flag=1; else fade to Terminal ROE
            ri.write_formula(r0, 2, f'=IF($B$7>={i}, IF($B$10=1, $B$8 + ($B$6-$B$8)*({i}-1)/MAX($B$7-1,1), $B$8 + ($B$11-$B$8)*({i}-1)/MAX($B$7-1,1)), NA())', fmt_pct4)
            ri.write_formula(r0, 3, f'=B{r0+1}*C{r0+1}', fmt_usd)                 # Earnings
            ri.write_formula(r0, 4, f'=$B$9*D{r0+1}', fmt_usd)                    # Dividends
            ri.write_formula(r0, 5, f'=B{r0+1}+D{r0+1}-E{r0+1}', fmt_usd)         # BV_End
            ri.write_formula(r0, 6, f'=(C{r0+1}-$B$6)*B{r0+1}', fmt_usd)          # Residual Income
            ri.write_formula(r0, 7, f'=IF($B$7>={i}, G{r0+1}/POWER(1+$B$6,{i}), NA())', fmt_usd)  # PV RI

        # Terminal component (only if fade_to_ke = 0)
        ri_term_row = ri_start + 21
        ri.write(ri_term_row-1, 0, "Terminal", fmt_hdr)
        ri.write(ri_term_row-1, 6, "Terminal PV (RI)", fmt_hdr)
        ri.write_formula(
            ri_term_row, 6,
            f'=IF($B$10=0, ((INDEX(C{ri_start+1}:C{ri_start+20},$B$7)-$B$6)*INDEX(F{ri_start+1}:F{ri_start+20},$B$7))/($B$6-$B$12)/POWER(1+$B$6,$B$7), 0)',
            fmt_usd
        )

        # Sum & implied price
        ri_sum = ri_term_row + 3
        ri.write(ri_sum,   0, "Equity Value = BV₀ + ΣPV(RI) + Terminal", fmt_bold)
        ri.write(ri_sum+1, 0, "Implied Price (per share)", fmt_bold)
        ri.write_formula(
            ri_sum, 1,
            f'=$B$4 + SUM(INDEX(H{ri_start+1}:H{ri_start+20},1):INDEX(H{ri_start+1}:H{ri_start+20},$B$7)) + G{ri_term_row+1}',
            fmt_usd
        )
        ri.write_formula(ri_sum+1, 1, f'=IF($B$5>0, B{ri_sum+1}/$B$5, NA())', fmt_usd)

        # Named ranges for Overview
        wb.define_name("IMPLIED_PRICE_RI", f"='Residual Income'!$B${(ri_sum+1)+1}")
        wb.define_name("CHOOSE_RI", f"={ 'TRUE' if fin_mode else 'FALSE' }")

        # Charts
        pv_chart = wb.add_chart({"type": "column"})
        pv_chart.add_series({
            "name":       "PV of RI",
            "categories": ["Residual Income", ri_start, 0, ri_start+19, 0],
            "values":     ["Residual Income", ri_start, 7, ri_start+19, 7],
        })
        pv_chart.set_title({"name": "PV of Residual Income"})
        pv_chart.set_legend({"none": True})
        ri.insert_chart("J4", pv_chart, {"x_scale": 1.0, "y_scale": 1.0})

        pv_chart2 = wb.add_chart({"type": "column"})
        pv_chart2.add_series({
            "name":       "PV of FCFs",
            "categories": ["DCF Model", start_row, 0, start_row+19, 0],
            "values":     ["DCF Model", start_row, 2, start_row+19, 2],
        })
        pv_chart2.set_title({"name": "PV FCFs (Years 1..N)"})
        pv_chart2.set_legend({"none": True})
        dcf.insert_chart("E4", pv_chart2, {"x_scale": 1.0, "y_scale": 1.0})

        fcf_chart = wb.add_chart({"type": "line"})
        fcf_chart.add_series({
            "name": "FCF",
            "categories": ["DCF Model", start_row, 0, start_row+19, 0],
            "values":     ["DCF Model", start_row, 1, start_row+19, 1],
        })
        fcf_chart.add_series({
            "name": "PV FCF",
            "categories": ["DCF Model", start_row, 0, start_row+19, 0],
            "values":     ["DCF Model", start_row, 2, start_row+19, 2],
        })
        fcf_chart.set_title({"name": "FCF vs PV FCF"})
        fcf_chart.set_legend({"position": "bottom"})
        dcf.insert_chart("E20", fcf_chart, {"x_scale": 1.0, "y_scale": 1.0})

        # About & Disclaimer sheet
        about = wb.add_worksheet("About & Disclaimer")
        about.set_column("A:A", 120)
        about.write("A1", f"Created by: {CREATED_BY}", fmt_bold)
        about.write("A2", f"App version: {APP_VERSION}   |   Build: {BUILD_TS}")
        about.write("A4", "Disclaimer", fmt_bold)
        about.write("A5", DISCLAIMER_MD, fmt_wrap)
        about.write("A12", "Notes", fmt_bold)
        about.write("A13", "Historical data sourced via yfinance; values may differ from company filings.", fmt_wrap)

    return buffer.getvalue()

# -------------------- Export --------------------

st.subheader("Export")
st.caption("Generate an Excel workbook with all statements, models, charts, and the About/Disclaimer sheet.")

if st.button("📥 Build Excel Report"):
    metrics_df = pd.DataFrame(fmt_metrics.items(), columns=["Metric", "Value"])
    ri_params = st.session_state.get("ri_params", {})
    excel_bytes = build_excel_report(
        ticker=ticker,
        timeframe=timeframe,
        price_df=price_df,
        inc_df=inc,
        bs_df=bs,
        cf_df=cf,
        comps_df=comps_df.reset_index(),
        metrics_df=metrics_df,
        base_fcf=st.session_state.get("base_fcf") or 0.0,
        shares_out=st.session_state.get("shares_out") or 0,
        net_debt=st.session_state.get("net_debt") or 0.0,
        g1=_g1,
        g2=_g2,
        r=_r_manual,  # keep manual r as an input in the sheet
        N=N,
        current_price=st.session_state.get("last_close") or 0.0,
        fade_flag=fade,
        use_wacc=st.session_state.get("use_wacc", False),
        rf=st.session_state.get("rf", 0.04),
        erp=st.session_state.get("erp", 0.05),
        beta=st.session_state.get("beta", 1.20),
        kd=st.session_state.get("kd", 0.05),
        tax=st.session_state.get("tax", 0.21),
        dv=st.session_state.get("dv", 0.00),
        roic=st.session_state.get("roic", 0.12),
        reinvest=st.session_state.get("reinvest", 0.00),
        fin_mode=bool(ri_params.get("use_financials_mode", False)),
        ri_bv0=float(ri_params.get("bv0", 0.0)),
        ri_shares=int(ri_params.get("shares", 0)),
        ri_ke=float(ri_params.get("ke", 0.10)),
        ri_N=int(ri_params.get("N", 5)),
        ri_roe_start=float(ri_params.get("roe_start", 0.10)),
        ri_payout=float(ri_params.get("payout", 0.00)),
        ri_fade_to_ke=bool(ri_params.get("fade_to_ke", True)),
        ri_roe_terminal=(ri_params.get("roe_terminal", None)),
        ri_g=float(ri_params.get("g_ri", 0.00)),
        # 3S projections
        proj_income_df=st.session_state.get("three_s", {}).get("inc"),
        proj_balance_df=st.session_state.get("three_s", {}).get("bs"),
        proj_cash_df=st.session_state.get("three_s", {}).get("cf"),
        proj_fcf_series=st.session_state.get("three_s", {}).get("fcf"),
    )
    st.session_state["excel_bytes"] = excel_bytes

if st.session_state.get("excel_bytes"):
    st.download_button(
        "Click to download",
        data=st.session_state["excel_bytes"],
        file_name=f"{ticker}_analyst_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown('<div class="section-break"></div>', unsafe_allow_html=True)
