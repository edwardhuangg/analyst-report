# ================================================================
# Valuation Model Builder — ticker in, buy-side Excel model out
# ================================================================
# The web app only collects inputs. The deliverable is the .xlsx: a live,
# formula-driven 3-statement / DCF / comps / SOTP model in the layout of the
# HON_Valuation_Model template, built from the company's SEC filings.
import hmac
import os
from datetime import date

import pandas as pd
import streamlit as st

from valuation import ai, brief, check, market, overrides, pipeline
from valuation.spec import align_lines, no_peer_fallback

st.set_page_config(page_title="Valuation Model Builder", page_icon="📊", layout="wide")

for k in ("SEC_USER_AGENT", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_ANALYST_MODEL", "OPENAI_REASONING_EFFORT"):  # let library code see secrets
    try:
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])
    except Exception:
        pass

YEARS = 6

st.markdown("""
<style>
.block-container{padding-top:2rem; max-width:1200px;}
.step{font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; color:#1F4E79; font-weight:700; margin-bottom:-.4rem;}
section[data-testid="stSidebar"] .creator-card{margin-top:12px; padding:12px 14px; border-radius:12px;
  border:1px solid rgba(128,128,128,.25); display:flex; align-items:center; gap:12px;}
section[data-testid="stSidebar"] .creator-card .avatar{width:36px; height:36px; border-radius:999px; display:grid;
  place-items:center; font-weight:700; background:linear-gradient(135deg,#2E7EFB,#1F4E79); color:#fff;}
section[data-testid="stSidebar"] .creator-card .meta{display:flex; flex-direction:column; line-height:1.2;}
section[data-testid="stSidebar"] .creator-card .label{font-size:11px; text-transform:uppercase; opacity:.7;}
section[data-testid="stSidebar"] .creator-card .name{font-weight:700;}
section[data-testid="stSidebar"] .creator-card .actions{margin-left:auto;}
section[data-testid="stSidebar"] .creator-card a{font-size:13px; text-decoration:none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "1. **Pick a ticker and a valuation date.** The app pulls the 10-K, the latest 10-Q and the earnings "
        "releases from SEC EDGAR, plus price, rates and peers from Yahoo Finance.\n"
        "2. **Review the drivers.** Defaults come from the filings; change anything you disagree with.\n"
        "3. **Download the Excel model.** Every output is a live formula; change any blue input or the scenario "
        "selector in Excel and the whole model updates.\n\n"
        "**Analyst layer.** Filings give the history; the judgment (scenario paths, M&A, guidance, debt-like items, "
        "SOTP splits, peer guidance, thesis) comes from the AI analyst pass, or from an external agent: download the "
        "research brief in step 2, let the agent research and return the overrides JSON, upload it."
    )
    with st.expander("What's in the workbook"):
        st.markdown(
            "Cover · Summary (football field, weighted fair value, 3-year return) · Assumptions (bear/base/bull) · "
            "Interim (YTD, LTM, net debt, guidance check) · Segments · IS · BS · CF · Schedules · DCF (WACC, two "
            "terminal methods, sensitivity grids) · Comps · SOTP · Thesis. Balance sheet and cash checks must read "
            "0 in every year (Summary!C45)."
        )
    with st.expander("Disclaimer"):
        st.caption("Educational use only, not investment advice. Filings data is parsed automatically from XBRL "
                   "and market data comes from Yahoo Finance; AI-assisted items (guidance, peers, thesis) must be "
                   "verified. Defaults are a starting point, not a view.")
    st.markdown("""
<div class="creator-card">
  <div class="avatar">EH</div>
  <div class="meta"><div class="label">Created by</div><div class="name">Edward Huang</div></div>
  <div class="actions"><a href="https://linkedin.com/in/edwardhuangg" target="_blank" rel="noopener noreferrer">LinkedIn ↗</a></div>
</div>""", unsafe_allow_html=True)

# access: the OpenAI key spends money, so the app sits behind APP_PASSWORD (Streamlit secrets or environment). With a
# key configured and no password, the AI is switched off: the safe default for a public host
try:
    APP_PASSWORD = str(st.secrets.get("APP_PASSWORD") or os.environ.get("APP_PASSWORD") or "")
except Exception:
    APP_PASSWORD = os.environ.get("APP_PASSWORD") or ""
ai_locked = not APP_PASSWORD and bool(os.environ.get("OPENAI_API_KEY"))
if APP_PASSWORD and not st.session_state.get("authed"):
    st.title("Valuation Model Builder")
    with st.form("login"):
        pw_in = st.text_input("Password", type="password")
        entered = st.form_submit_button("Enter", type="primary")
    if entered:
        if hmac.compare_digest(pw_in.encode(), APP_PASSWORD.encode()):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong password")
    st.stop()

st.title("Valuation Model Builder")
st.caption("Ticker in → buy-side Excel model out (3-statement, DCF, comps, sum of the parts), built from SEC filings.")

ai_available = ai.client() is not None and not ai_locked
if ai_locked:
    st.warning("The AI analyst is off: an OpenAI key is configured but no APP_PASSWORD, so anyone could spend it. "
               "Set APP_PASSWORD in the app's secrets to switch it on.")

# ================================================================ step 1
st.markdown('<div class="step">Step 1</div>', unsafe_allow_html=True)
st.subheader("Company and valuation date")
with st.form("start"):
    c1, c2, c3 = st.columns([1, 1, 2])
    ticker = c1.text_input("Ticker", st.session_state.get("ticker", "HON")).strip().upper()
    as_of = c2.date_input("Valuation date", st.session_state.get("as_of", date.today()), max_value=date.today(),
                          help="The model uses only filings made on or before this date (e.g. Aug 29, 2025 "
                               "rebuilds the HON pitch from the FY2024 10-K and the Q2 2025 10-Q).")
    peer_txt = c3.text_input("Peer tickers (optional, comma-separated)", st.session_state.get("peer_txt", ""),
                             help="Leave blank to let AI pick peers" if ai_available else
                             "No OpenAI key configured: enter peers here (or add them in step 2)")
    c4, c45, c5 = st.columns([1, 1.3, 2.7])
    use_ai = c4.toggle("Use AI", value=ai_available, disabled=not ai_available,
                       help="Reads the earnings releases (guidance, adjusted EPS, organic growth), suggests peers, "
                            "reads peer guidance and drafts the thesis. Needs OPENAI_API_KEY in Streamlit secrets.")
    use_analyst = c45.toggle("AI analyst pass", value=ai_available, disabled=not ai_available,
                             help="The analyst agent (OPENAI_ANALYST_MODEL, default gpt-5.5) reads the releases, "
                                  "10-K / 10-Q MD&A and 8-Ks, sets scenario paths, M&A, guidance, debt-like items, "
                                  "SOTP multiples, peers and the thesis with source notes, then checks the built "
                                  "model and fixes what fails.")
    depth = c45.selectbox("Analyst depth", ["Fast (low)", "Standard (medium)", "Deep (high)"], index=1,
                          disabled=not ai_available, help="Reasoning effort of the analyst model. Fast takes ~2-5 "
                          "minutes; Standard ~5-12; Deep longer. Deeper = more careful reading and calibration.")
    with c5.expander("Advanced"):
        cik = st.text_input("SEC CIK override", "", help="Only needed when a ticker maps to a new holding company "
                                                       "that has not filed a 10-K yet (use the predecessor's CIK).")
        allow_fin = st.checkbox("Build anyway for banks / insurers", False,
                                help="The template is an industrial 3-statement / EV model; it gives misleading "
                                     "answers for banks and insurers.")
        prev_up = st.file_uploader("Previous version (assumptions .json)", type=["json"],
                                   help="The assumptions file of the version the PM last read (downloaded with an "
                                        "earlier build). The new file bridges its value to it, and if judgment "
                                        "changes flip the rating, an independent AI review challenges each one.")
    go = st.form_submit_button("Pull filings and build defaults", type="primary")

if go:
    st.session_state.update(ticker=ticker, as_of=as_of, peer_txt=peer_txt)
    for k in list(st.session_state):  # widgets keyed per company must not carry over
        if k.startswith(("v_", "seg_", "drivers", "peers_ed", "sotp_ed", "guide_ed")):
            del st.session_state[k]
    peers = [p.strip().upper() for p in peer_txt.split(",") if p.strip()]
    try:
        with st.status(f"Building {ticker} as of {as_of:%b %d, %Y}…", expanded=True) as status:
            data = pipeline.fetch(ticker, as_of, peers=peers or None, use_ai=use_ai,
                                  progress=status.write, cik=cik or None, allow_financials=allow_fin)
            spec = pipeline.make_spec(data, use_ai_thesis=use_ai, analyst=use_ai and use_analyst,
                                      progress=status.write, effort=depth.split("(")[1].rstrip(")"))
            status.update(label=f"{spec.name}: filings loaded", state="complete", expanded=False)
        if prev_up is not None and check.available():
            with st.status("Bridging to the previous version…", expanded=False) as status:
                for line in pipeline.bridge_previous(spec, data, overrides.parse(prev_up.getvalue().decode("utf-8")),
                                                     prev_up.name, status.write,
                                                     effort=depth.split("(")[1].rstrip(")"), use_ai=use_ai):
                    status.write(line)
        st.session_state["spec"] = spec
        st.session_state["data"] = data
        st.session_state.pop("xlsx", None)
    except Exception as e:
        st.error(f"Could not build {ticker}: {e}")
        st.stop()

spec = st.session_state.get("spec")
if spec is None:
    st.info("Enter a ticker and press **Pull filings and build defaults**.")
    st.stop()

# ================================================================ step 2
st.markdown('<div class="step">Step 2</div>', unsafe_allow_html=True)
st.subheader(f"Review the inputs — {spec.name} ({spec.ticker})")
u = spec.filing_urls
links = [f"[10-K FY{spec.fy_hist[-1]}]({u['annual']})"]
if u.get("quarterly"):
    links.append(f"[{spec.interim.quarter_label} 10-Q]({u['quarterly']})")
if u.get("fy_release"):
    links.append(f"[FY{spec.fy_hist[-1]} release]({u['fy_release']})")
if u.get("q_release"):
    links.append(f"[{spec.interim.quarter_label} release]({u['q_release']})")
st.markdown("Sources: " + " · ".join(links) +
            f" · {len(spec.segments)} segment{'s' if len(spec.segments) > 1 else ''} · "
            f"history {spec.hist_labels[0]}–{spec.hist_labels[-1]}, forecast {spec.fc_labels[0]}–{spec.fc_labels[-1]}")
if spec.notes.get("analyst_summary"):
    st.info(f"**AI analyst ({ai.analyst_model()}):** {spec.notes['analyst_summary']}")
st.caption("Assumptions set by: " + " · ".join(spec.provenance))
if spec.ai_cost:
    st.caption(f"OpenAI cost of this build: {spec.ai_cost} — budget ${ai.budget():.2f} per build "
               "(OPENAI_MAX_COST_PER_RUN)")
if spec.data_notes:
    with st.expander(f"Data notes ({len(spec.data_notes)}) — read before using the model", expanded=False):
        for f in spec.data_notes:
            st.markdown(f"- {f}")


def pct_frame(rows: dict) -> pd.DataFrame:
    return pd.DataFrame({k: [round(v * 100, 2) for v in vals] for k, vals in rows.items()},
                        index=spec.fc_labels).T


def from_pct(df: pd.DataFrame) -> dict:
    return {k: [float(x) / 100 for x in df.loc[k].tolist()] for k in df.index}


tabs = st.tabs(["Market & valuation", "Segments", "Other drivers", "Peers & SOTP", "Guidance & debt-like",
                "Thesis", "Agent hand-off"])

# ---------------------------------------------------- market & valuation
with tabs[0]:
    c = st.columns(4)
    spec.price = c[0].number_input("Share price ($)", value=float(spec.price), step=0.01, format="%.2f",
                                   help=spec.price_note)
    spec.basic_shares = c[1].number_input("Basic shares (mm)", value=float(spec.basic_shares), step=0.1,
                                          format="%.1f", help=spec.basic_note)
    spec.dilutive = c[2].number_input("Dilutive securities (mm)", value=float(spec.dilutive), step=0.1,
                                      format="%.1f", help=spec.dilutive_note)
    spec.scenario = ["Bear", "Base", "Bull"].index(
        c[3].selectbox("Scenario in the file", ["Bear", "Base", "Bull"], index=spec.scenario - 1)) + 1
    spec.seg_after_charges = int(st.checkbox(
        "Segment profit is already after amortization & stock comp (segments sum to GAAP operating income)",
        value=bool(spec.seg_after_charges), help=spec.seg_after_charges_note))
    V = spec.valuation
    st.markdown("**WACC and terminal value**")
    c = st.columns(7)
    for col, key, label, scale in [(c[0], "rf", "Risk-free %", 100), (c[1], "beta", "Beta", 1),
                                   (c[2], "erp", "ERP %", 100), (c[3], "kd", "Pre-tax cost of debt %", 100),
                                   (c[4], "tax_wacc", "Tax rate %", 100), (c[5], "g", "Terminal growth %", 100),
                                   (c[6], "exit_mult", "Exit EV/EBITDA x", 1)]:
        v = col.number_input(label, value=float(V[key][0]) * scale, step=0.05 if scale == 100 else 0.05,
                             format="%.2f", help=V[key][1] or None, key=f"v_{key}")
        V[key] = (v / scale, V[key][1])
    st.markdown("**Method weights (must sum to 100%)**")
    c = st.columns(4)
    for col, key, label in [(c[0], "w_pg", "DCF perpetuity"), (c[1], "w_exit", "DCF exit multiple"),
                            (c[2], "w_sotp", "Sum of the parts"), (c[3], "w_comps", "Trading comps")]:
        v = col.number_input(label + " %", value=float(V[key][0]) * 100, step=5.0, format="%.0f", key=f"v_{key}",
                             help=V[key][1] or None)
        V[key] = (v / 100, V[key][1])
    tot = sum(V[k][0] for k in ("w_pg", "w_exit", "w_sotp", "w_comps"))
    if abs(tot - 1) > 1e-6:
        st.warning(f"Weights sum to {tot:.0%}; the Summary divides by the total, so they will be rescaled.")
    c = st.columns(4)
    v = c[0].number_input("Discount to peer P/E %", value=float(V["peer_discount"][0]) * 100, step=5.0,
                          format="%.0f", help=V["peer_discount"][1])
    V["peer_discount"] = (v / 100, V["peer_discount"][1])
    v = c[1].number_input(f"Exit P/E for 3-year return (x, on {spec.fc_labels[4]} EPS)",
                          value=float(V["exit_pe"][0]), step=0.5, help=V["exit_pe"][1])
    V["exit_pe"] = (v, V["exit_pe"][1])
    c = st.columns(4)
    v = c[0].number_input("Fade years after the forecast (three-stage DCF)", min_value=0, max_value=10,
                          value=int(V["fade_years"][0]), step=1, help=V["fade_years"][1])
    V["fade_years"] = (v, V["fade_years"][1])
    v = c[1].number_input("Comps: weight on P/E % (rest EV/EBITDA)", value=float(V["comps_pe_weight"][0]) * 100,
                          step=10.0, format="%.0f", help=V["comps_pe_weight"][1])
    V["comps_pe_weight"] = (v / 100, V["comps_pe_weight"][1])
    if spec.captive:
        v = c[2].number_input(f"Captive-finance share of debt % ({spec.captive['segment']})",
                              value=float(V["captive_share"][0]) * 100, step=1.0, format="%.0f",
                              help=V["captive_share"][1])
        V["captive_share"] = (v / 100, V["captive_share"][1])
    if spec.profile:
        c[3].caption(f"Industry profile: **{spec.profile.get('label')}**")
    v = c[2].number_input("Other stakes not in EBITDA ($mm)", value=float(V["other_stake"][0]), step=50.0,
                          help=V["other_stake"][1])
    V["other_stake"] = (v, V["other_stake"][1])
    v = c[3].number_input("Noncontrolling interests ($mm)", value=float(V["nci_value"][0]), step=10.0,
                          help=V["nci_value"][1])
    V["nci_value"] = (v, V["nci_value"][1])

# ------------------------------------------------------------ segments
with tabs[1]:
    st.caption("Base case in %. Bear and bull are rebuilt from the base with the spreads below "
               "(year 1 uses half the growth spread). Everything stays editable in Excel.")
    names = pd.DataFrame({"Segment name": [sg.name for sg in spec.segments]})
    names = st.data_editor(names, hide_index=True, width="stretch", key="seg_names", disabled=False)
    st.markdown("**Organic growth, base case (% y/y)**")
    g = st.data_editor(pct_frame({sg.name: sg.growth["base"] for sg in spec.segments}),
                       width="stretch", key="seg_g")
    st.markdown("**Segment margin, base case (% of sales)**")
    m = st.data_editor(pct_frame({sg.name: sg.margin["base"] for sg in spec.segments}),
                       width="stretch", key="seg_m")
    st.markdown("**Acquisitions, divestitures & FX (% contribution to growth, all scenarios)**")
    mna = st.data_editor(pct_frame({sg.name: sg.mna for sg in spec.segments}), width="stretch",
                         key="seg_mna")
    c = st.columns(3)
    rebuild = c[2].checkbox("Rebuild bear / bull from the base ± spreads", value=False,
                            help="Off keeps the engine's or the AI analyst's bear / bull paths and moves them with "
                                 "any edit to the base case above.")
    g_spread = c[0].number_input("Growth spread, bear/bull (pt)", value=2.0, step=0.5, disabled=not rebuild) / 100
    m_spread = c[1].number_input("Margin spread at year 4+, bear/bull (pt)", value=2.0, step=0.5,
                                 disabled=not rebuild) / 100
    gd, md, mnad = from_pct(g), from_pct(m), from_pct(mna)
    for sg, new_name in zip(spec.segments, names["Segment name"].tolist()):
        base_g, base_m = gd[sg.name], md[sg.name]
        if rebuild:
            sg.growth = {"base": base_g,
                         "bear": [v - (g_spread / 2 if t == 0 else g_spread) for t, v in enumerate(base_g)],
                         "bull": [v + (g_spread / 2 if t == 0 else g_spread) for t, v in enumerate(base_g)]}
            sg.margin = {"base": base_m,
                         "bear": [v - min(m_spread, m_spread / 4 * (t + 1)) for t, v in enumerate(base_m)],
                         "bull": [v + min(m_spread, m_spread / 4 * (t + 1)) for t, v in enumerate(base_m)]}
        else:  # only an edit changes the model (the table rounds to 0.01%); bear / bull move with the base
            for kind, new in (("growth", base_g), ("margin", base_m)):
                paths = getattr(sg, kind)
                if [round(x * 100, 2) for x in paths["base"]] == [round(x * 100, 2) for x in new]:
                    continue
                if kind == "growth":
                    sg.lines_explicit.discard("base")  # a segment-level edit re-aligns its revenue lines
                d = [n - o for n, o in zip(new, paths["base"])]
                for case in ("bear", "bull"):
                    paths[case] = [v + x for v, x in zip(paths[case], d)]
                paths["base"] = new
        if rebuild:
            sg.lines_explicit.clear()
        if [round(x * 100, 2) for x in sg.mna] != [round(x * 100, 2) for x in mnad[sg.name]]:
            sg.mna = mnad[sg.name]
        align_lines(sg)
        if new_name and new_name.strip():
            sg.name = new_name.strip()

# -------------------------------------------------------- other drivers
LABELS = {
    "amort": ("Amortization of acquired intangibles", "$mm"), "sbc_pct": ("Stock compensation", "% sales"),
    "repo_pct": ("Repositioning & other non-segment charges", "% sales"),
    "other_nonop": ("Other non-operating income (pension & other)", "$mm"),
    "nonrec": ("Non-recurring gains (losses), excl. from adjusted EPS", "$mm"),
    "int_yield": ("Interest income yield on beginning cash", "%"), "int_rate": ("Interest rate on average debt", "%"),
    "etr": ("Effective tax rate", "%"), "nci_ni": ("Net income to NCI", "$mm"),
    "cogs_pct": ("Cost of sales (memo)", "% sales"), "da_pct": ("Depreciation & other amortization", "% sales"),
    "capex_pct": ("Capital expenditures", "% sales"), "dso": ("Days sales outstanding", "days"),
    "dio": ("Days inventory", "days"), "oca_pct": ("Other current assets", "% sales"),
    "dpo": ("Days payable", "days"), "accrued_pct": ("Accrued liabilities", "% sales"),
    "other_op": ("Other operating cash items, net", "$mm"), "acq": ("Acquisitions (cash)", "$mm"),
    "intang_share": ("Share of purchase price to intangibles", "%"), "debt_iss": ("Net debt issuance", "$mm"),
    "buyback": ("Share repurchases", "$mm"), "buyback_px": ("Average repurchase price", "$/share"),
    "shares_issued": ("Shares issued (options & RSUs)", "mm"), "stock_proc": ("Proceeds from stock issuance", "$mm"),
}
with tabs[2]:
    st.caption("Percent rows are shown in %. Hover a row's note in the workbook for the historical basis.")
    rows, notes = {}, {}
    for key, (label, unit) in LABELS.items():
        vals, note = spec.drivers[key]
        scale = 100 if "%" in unit else 1
        rows[f"{label} ({unit})"] = [v * scale for v in vals]
        notes[f"{label} ({unit})"] = note
    df = pd.DataFrame(rows, index=spec.fc_labels).T
    df["Basis"] = [notes[i] for i in df.index]
    df = st.data_editor(df, width="stretch", disabled=["Basis"], key="drivers", height=880)
    for key, (label, unit) in LABELS.items():
        scale = 100 if "%" in unit else 1
        r = f"{label} ({unit})"
        spec.drivers[key] = ([float(x) / scale for x in df.loc[r, spec.fc_labels].tolist()], spec.drivers[key][1])
    c = st.columns(2)
    dps = c[0].number_input(f"Dividend per share {spec.fc_labels[0]} ($)", value=float(spec.drivers["dps0"][0][0]),
                            step=0.01, format="%.2f", help=spec.drivers["dps0"][1])
    spec.drivers["dps0"] = ([dps], spec.drivers["dps0"][1])
    dg0 = float(spec.drivers["div_growth"][0][0]) * 100
    dg = c[1].number_input("Dividend growth after year 1 (%)", value=dg0, step=0.5, help=spec.drivers["div_growth"][1])
    if abs(dg - dg0) > 1e-9:  # an edit sets a flat path; otherwise the analyst's path stands
        spec.drivers["div_growth"] = ([dg / 100] * 5, spec.drivers["div_growth"][1])

# ------------------------------------------------------- peers & SOTP
with tabs[3]:
    c1, c2 = st.columns([3, 1])
    new_peers = c1.text_input("Replace peers (tickers, comma-separated)", "",
                              placeholder=", ".join(p["ticker"] for p in spec.peers))
    if c2.button("Fetch peer data", width="stretch") and new_peers.strip():
        snaps = []
        with st.spinner("Fetching peer prices and estimates…"):
            for t in [x.strip().upper() for x in new_peers.split(",") if x.strip()][:8]:
                try:
                    snap = market.peer_snapshot(t, spec.as_of)
                    if snap["price"] and snap["eps"] is None:
                        pipeline.point_in_time_peer_eps(snap, spec.as_of, ai.client() if use_ai else None)
                    if snap["price"]:
                        snaps.append(snap)
                except Exception:
                    st.warning(f"No market data for {t}")
        if snaps:
            spec.peers = snaps
            pipeline.fill_peer_ev(spec)  # EV / EBITDA, betas and the calendar year, as for the analyst's peers
            if spec.valuation["w_comps"][0] == 0:
                st.info("Peers added: give the comps method a weight on the first tab.")
    pdf = pd.DataFrame(spec.peers or [], columns=["name", "ticker", "overlap", "price", "eps", "source"])
    pdf = st.data_editor(pdf, num_rows="dynamic", width="stretch", key="peers_ed",
                         column_config={"price": st.column_config.NumberColumn("Price ($)", format="%.2f"),
                                        "eps": st.column_config.NumberColumn(f"{spec.fc_labels[0]} EPS ($)",
                                                                             format="%.2f")})
    # the table shows six columns of each peer; the rest (EV / EBITDA, consensus growth, beta inputs) must survive
    # the edit (ADBE: every peer lost its EV / EBITDA here, so the comps published a 0.0x multiple)
    full = {str(p.get("ticker") or "").upper(): p for p in spec.peers}
    edited = []
    for r in pdf.to_dict("records"):
        if not r.get("ticker"):
            continue
        p = dict(full.get(str(r["ticker"]).upper(), {}))
        for k in ("name", "ticker", "overlap", "price", "eps", "source"):
            x = r.get(k)
            p[k] = None if x is None or (isinstance(x, float) and pd.isna(x)) else \
                float(x) if k in ("price", "eps") else x
        edited.append(p)
    spec.peers = edited
    if not spec.peers:
        no_peer_fallback(spec, spec.eps_est)
    st.markdown("**SOTP EV/EBITDA multiples (bear / base / bull)**")
    sdf = pd.DataFrame([{"Segment": sg.name, "Bear": sg.sotp[0], "Base": sg.sotp[1], "Bull": sg.sotp[2],
                         "Note": sg.sotp_note} for sg in spec.segments] +
                       [{"Segment": "Corporate costs", "Bear": spec.corp_multiples[0], "Base": spec.corp_multiples[1],
                         "Bull": spec.corp_multiples[2], "Note": "Capitalized head-office cost"}])
    sdf = st.data_editor(sdf, hide_index=True, disabled=["Segment"], width="stretch", key="sotp_ed")
    for sg, (_, r) in zip(spec.segments, sdf.iloc[:-1].iterrows()):
        sg.sotp, sg.sotp_note = (float(r["Bear"]), float(r["Base"]), float(r["Bull"])), str(r["Note"] or "")
    last = sdf.iloc[-1]
    spec.corp_multiples = (float(last["Bear"]), float(last["Base"]), float(last["Bull"]))

# ------------------------------------------------- guidance & debt-like
with tabs[4]:
    st.markdown(f"**FY{spec.fy_fc[0]} company guidance** (for the check on the Interim tab; blank = not given)")
    gmeta = [("sales", "Net sales ($mm)", 1), ("organic", "Organic growth (%)", 100),
             ("margin", "Segment margin (%)", 100), ("adj_eps", "Adjusted EPS ($)", 1),
             ("fcf", "Free cash flow ($mm)", 1)]
    gdf = pd.DataFrame([{"Metric": label, "Low": (spec.guidance.get(k) or (None, None))[0] * s
                         if spec.guidance.get(k) else None,
                         "High": spec.guidance.get(k)[1] * s if spec.guidance.get(k) else None}
                        for k, label, s in gmeta])
    gdf = st.data_editor(gdf, hide_index=True, disabled=["Metric"], width="stretch", key="guide_ed")
    spec.guidance = {}
    for (k, _, s), (_, r) in zip(gmeta, gdf.iterrows()):
        if pd.notna(r["Low"]) and pd.notna(r["High"]):
            spec.guidance[k] = (float(r["Low"]) / s, float(r["High"]) / s)
    st.markdown("**Debt-like items outside net debt** (e.g. asbestos or environmental reserves, net of recoveries)")
    c = st.columns(4)
    dl = spec.debt_like
    dl["liab"] = c[0].number_input("Liability, latest ($mm)", value=float(dl["liab"]), step=10.0)
    dl["offset"] = c[1].number_input("Offsetting recoveries, latest ($mm)", value=float(dl["offset"]), step=10.0)
    dl["liab_fy"] = c[2].number_input("Liability, FY end ($mm)", value=float(dl["liab_fy"]), step=10.0)
    dl["offset_fy"] = c[3].number_input("Recoveries, FY end ($mm)", value=float(dl["offset_fy"]), step=10.0)
    if dl["liab"] or dl["offset"]:
        dl["note"] = "Entered by the analyst from the latest 10-Q / 10-K"

# ---------------------------------------------------------------- thesis
with tabs[5]:
    t = spec.thesis
    st.caption("Written to the Thesis tab; the valuation numbers there are live links to the model.")
    opts = ["long", "neutral on", "underweight", "short"]
    t["stance"] = st.selectbox("Stance", opts, index=opts.index(spec.stance_phrase))
    t["recommendation"] = st.text_area("Recommendation", t["recommendation"], height=80)
    for key, label in [("points", "Thesis points"), ("model_says", "What the model says"),
                       ("risks", "Key risks"), ("monitor", "What to monitor")]:
        txt = st.text_area(f"{label} (one per line)", "\n".join(t[key]), height=150 if key == "points" else 110)
        t[key] = [x for x in txt.splitlines() if x.strip()] or ["[Analyst to complete]"]

# ------------------------------------------------------- agent hand-off
with tabs[6]:
    st.markdown("**Add the analyst layer with an external agent** (Claude Code, ChatGPT, or a colleague). The brief "
                "holds the parsed filings, the source links, the checklist of judgments and the current assumptions "
                "as JSON; the agent researches, edits the JSON and you upload it here. Same format as the in-app AI "
                "analyst. From the terminal: `python -m valuation brief TICKER --as-of DATE`, then "
                "`python -m valuation build TICKER --as-of DATE --overrides file.json --check`.")
    c1, c2 = st.columns(2)
    data_obj = st.session_state.get("data")
    c1.download_button("⬇ Research brief (.md)", brief.build(spec, data_obj), width="stretch",
                       file_name=f"{spec.ticker}_brief_{spec.as_of:%Y-%m-%d}.md", mime="text/markdown")
    c2.download_button("⬇ Current assumptions (.json)", overrides.dumps(spec), width="stretch",
                       file_name=f"{spec.ticker}_overrides_{spec.as_of:%Y-%m-%d}.json", mime="application/json")
    up = st.file_uploader("Upload overrides JSON from the agent", type=["json", "md", "txt"], key="ovr_up")
    if up is not None and st.button("Apply overrides", type="primary"):
        try:
            doc = overrides.parse(up.getvalue().decode("utf-8", errors="replace"))
            warn = overrides.apply(spec, doc, source=f"uploaded file {up.name}")
            pipeline.fill_peer_ev(spec)
            for k in list(st.session_state):  # widgets must re-read the spec
                if k.startswith(("v_", "seg_", "drivers", "peers_ed", "sotp_ed", "guide_ed")):
                    del st.session_state[k]
            st.session_state["ovr_warn"] = warn
            st.session_state.pop("xlsx", None)
            st.rerun()
        except Exception as e:
            st.error(f"Could not apply the file: {e}")
    for w in st.session_state.get("ovr_warn", []):
        st.warning(w)

# ================================================================ step 3
st.markdown('<div class="step">Step 3</div>', unsafe_allow_html=True)
st.subheader("Build the Excel model")
c1, c2 = st.columns([1, 3])
if c1.button("Build workbook", type="primary", width="stretch"):
    with st.spinner("Writing the workbook…"):  # the deliverable (values, scenarios, rating) is built after the review
        st.session_state["xlsx"] = pipeline.build(spec, values=not check.available())
    st.session_state.pop("review", None)
    if check.available():
        with st.spinner("Checking the model, then running bear / base / bull and the rating…"):
            try:
                st.session_state["review"] = check.review(spec, st.session_state["xlsx"])
                spec.last_review = st.session_state["review"]  # the download carries the open checks
                st.session_state["xlsx"] = pipeline.build(spec)
            except Exception as e:
                st.session_state["review"] = {"error": str(e)}
                st.session_state["xlsx"] = pipeline.build(spec)
if st.session_state.get("xlsx"):
    fname = f"{spec.ticker}_Valuation_Model_{spec.as_of:%Y-%m-%d}.xlsx"
    c2.download_button(f"⬇ Download {fname}", st.session_state["xlsx"], file_name=fname, type="primary",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("⬇ Assumptions (.json): rebuilds this file; upload it as the previous version next time",
                       overrides.dumps(spec), file_name=fname.replace(".xlsx", "_overrides.json"),
                       mime="application/json")
    st.caption("Open in Excel (or Google Sheets / Numbers): formulas calculate on open. Check Summary!C45 = OK, "
               "then flip Assumptions!C4 between 1 / 2 / 3 for bear / base / bull.")
    rv = st.session_state.get("review")
    if rv and rv.get("error"):
        st.warning(f"Could not recalculate the workbook here ({rv['error']}); open it in Excel.")
    elif rv:
        o = rv["outputs"]
        num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
        st.markdown(f"**What the model says** (scenario: {['Bear', 'Base', 'Bull'][spec.scenario - 1]}; "
                    f"integrity check: {o.get('check')})")
        c = st.columns(4)
        rt0 = spec.rating or {}
        if num(rt0.get("pw")) and num(o["price"]):  # the decision number leads; the base case is in the scenarios
            c[0].metric("Probability-weighted value", f"${rt0['pw']:,.2f}",
                        f"{rt0['pw'] / o['price'] - 1:+.1%} vs ${o['price']:,.2f}")
        else:
            c[0].metric("Weighted fair value", f"${o['fair_value']:,.2f}" if num(o["fair_value"]) else "–",
                        f"{o['upside']:+.1%} vs ${o['price']:,.2f}" if num(o["upside"]) else None)
        c[1].metric("3-year annualized return", f"{o['irr_3y']:.1%}" if num(o["irr_3y"]) else "–")
        c[2].metric("WACC", f"{o['wacc']:.2%}" if num(o["wacc"]) else "–")
        c[3].metric(f"P/E · EV/EBITDA {spec.fc_labels[0]}",
                    f"{o['pe_fy1']:.1f}x · {o['ev_ebitda_fy1']:.1f}x" if num(o["pe_fy1"]) and num(o["ev_ebitda_fy1"])
                    else "–")
        V = spec.valuation
        methods = pd.DataFrame([
            {"Method": "DCF — perpetuity growth", "Value / share": o["dcf_pg"], "Weight": V["w_pg"][0]},
            {"Method": "DCF — exit multiple", "Value / share": o["dcf_exit"], "Weight": V["w_exit"][0]},
            {"Method": "Sum of the parts", "Value / share": o["sotp"], "Weight": V["w_sotp"][0]},
            {"Method": "Trading comps (P/E)", "Value / share": o["comps"], "Weight": V["w_comps"][0]}])
        methods["vs price"] = [x / o["price"] - 1 if num(x) and num(o["price"]) else None
                               for x in methods["Value / share"]]
        cc = st.columns([2, 3])
        cc[0].dataframe(methods, hide_index=True, width="stretch",
                        column_config={"Value / share": st.column_config.NumberColumn(format="$%.2f"),
                                       "Weight": st.column_config.NumberColumn(format="percent"),
                                       "vs price": st.column_config.NumberColumn(format="percent")})
        pct_rows = ("Sales growth", "Organic growth", "Segment margin", "Operating margin")
        fc = pd.DataFrame({k: v for k, v in rv["forecast"].items()}, index=rv["years"]).T
        fmt = fc.copy().astype(object)
        for r in fc.index:
            fmt.loc[r] = [("–" if not num(x) else f"{x:.1%}" if r in pct_rows else
                           f"{x:,.2f}" if "EPS" in r else f"{x:,.0f}") for x in fc.loc[r]]
        cc[1].dataframe(fmt, width="stretch")
        if rv["guidance"]:
            st.markdown(f"**{spec.fc_labels[0]} guidance vs the base case**")
            st.dataframe(pd.DataFrame(rv["guidance"]), hide_index=True, width="stretch")
        rt, sv = spec.rating or {}, spec.scenario_values or {}
        if rt:  # the rule-based rating and the decision layer (economic framework)
            st.markdown(f"**Rating (rule): {rt.get('rating')}**" +
                        (f" · sizing: {rt['sizing']}" if str(rt.get("sizing") or "").startswith("binary") else "") +
                        f" — {rt.get('why', '')}")
            if sv.get("bear") and sv.get("bull"):
                d = st.columns(4)
                d[0].metric("Bear", f"${sv['bear'].get('fair_value') or 0:,.2f}")
                d[1].metric("Base", f"${sv['base'].get('fair_value') or 0:,.2f}")
                d[2].metric("Bull", f"${sv['bull'].get('fair_value') or 0:,.2f}")
                irr = o.get("implied_irr")
                d[3].metric("IRR the price implies", f"{irr:.1%}" if num(irr) else str(irr or "–"),
                            f"vs WACC {o['wacc']:.1%}" if num(o.get("wacc")) else None, delta_color="off")
        gates = rv.get("gates") or []
        if gates:
            bad = [g for g in gates if g["status"] in ("FAIL", "WARN")]
            st.markdown(f"**Pre-publication gates:** {sum(g['status'] == 'PASS' for g in gates)} pass, "
                        f"{sum(g['status'] == 'WARN' for g in gates)} warn, "
                        f"{sum(g['status'] == 'FAIL' for g in gates)} fail (the Checks tab in the workbook has the "
                        "evidence, the plug table and the audit trail)")
            for g in bad:
                (st.error if g["status"] == "FAIL" else st.warning)(f"Gate {g['id']}: {g['gate']} — {g['detail']}")
        others = [p for p in rv["problems"] if not p.startswith("Gate ")]
        for p in others:
            st.warning(p)
        if not rv["problems"]:
            st.success("Checker: no problems (balances, no Excel errors, methods consistent, inside guidance).")
