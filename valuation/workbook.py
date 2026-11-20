"""Render a ModelSpec into the buy-side valuation workbook (HON template layout).

Tabs (in workbook order): Cover, Summary, Assumptions, Interim, Segments, IS, BS,
CF, Schedules, DCF, Comps, SOTP, Thesis. Every output is a live formula; hard-coded
numbers are history from filings (blue) or assumptions (blue on grey). Layouts flex
with the number of segments and peers but keep the template's row order.
"""
from __future__ import annotations

import io
import re
import textwrap
from datetime import date

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

from .spec import ModelSpec, _add_years, _nz, refresh_ytd_adj_eps

# ------------------------------------------------------------------ styles
NUM = {
    "mm": r"#,##0;\(#,##0\);\–",
    "pct": r"0.0%;\(0.0%\);\–",
    "usd": r'\$#,##0.00;"($"#,##0.00\);\–',
    "eps": r'\$0.00;"($"0.00\);\–',
    "x": r'0.0\x;\(0.0"x)";\–',
    "sh": r"#,##0.0;\(#,##0.0\);\–",
    "days": r"0.0;\(0.0\);\–",
    "dec": r"0.00;\(0.00\);\–",
    "chk": r"#,##0.0;\(#,##0.0\);\–",
    "date": r'mmm\ d", "yyyy',
    "int": "0",
}
COLORS = {"blue": "#0000FF", "black": "#000000", "green": "#008000", "white": "#FFFFFF",
          "grey": "#595959", "title": "#1F4E79"}
FILLS = {"key": "#BDD7EE", "hdr": "#D9E1F2", "band": "#1F4E79", "input": "#F2F2F2", "sel": "#FFFF00"}
LINK_RX = re.compile(r"^='?[A-Za-z][\w ]*'?!\$?[A-Z]{1,3}\$?\d+$")

H = ["C", "D", "E"]                     # history columns
P = ["F", "G", "H", "I", "J", "K"]      # forecast columns
ALL = H + P


def prev(c: str) -> str:
    return chr(ord(c) - 1)


class Sheet:
    def __init__(self, book: "Book", name: str, widths: dict, tab: str, last_col: str):
        self.book, self.name, self.last_col = book, name, last_col
        self.ws = book.wb.add_worksheet(name)
        self.ws.hide_gridlines(2)
        self.ws.set_tab_color(tab)
        for col, w in widths.items():
            a, _, b = col.partition(":")
            self.ws.set_column(f"{a}:{b or a}", w)
        self.rows = set()

    def put(self, ref: str, v, num=None, b=False, color=None, fill=None, top=False, size=None,
            wrap=False, valign=None, align=None):
        if v is None:
            return
        if color is None:
            if isinstance(v, str) and v.startswith("="):
                color = "green" if LINK_RX.match(v) else "black"
            elif isinstance(v, (int, float)):
                color = "blue"
            else:
                color = "black"
        fmt = self.book.fmt(num=num, b=b, color=color, fill=fill, top=top, size=size, wrap=wrap,
                            valign=valign, align=align)
        row = int(re.sub(r"[A-Z]", "", ref))
        self.rows.add(row)
        if isinstance(v, str) and v.startswith("="):
            cached = self.book.values.get((self.name.upper(), ref))
            if isinstance(cached, (int, float, str)) and not isinstance(cached, bool):
                # store the computed result next to the formula: viewers that do not recalculate (previews, Numbers,
                # Google Sheets import, Excel in manual mode) show the numbers instead of zeros
                self.ws.write_formula(ref, v, fmt, cached)
            else:
                self.ws.write_formula(ref, v, fmt)
        elif isinstance(v, date):
            self.ws.write_datetime(ref, v, fmt)
        elif isinstance(v, (int, float)):
            self.ws.write_number(ref, float(v), fmt)
        else:
            self.ws.write_string(ref, str(v), fmt)

    def fill_row(self, r: int, fill: str, start="A"):
        """Paint the empty cells of a band / header row across the sheet's width."""
        for c in range(ord(start), ord(self.last_col) + 1):
            self.ws.write_blank(f"{chr(c)}{r}", None, self.book.fmt(fill=fill))
        self.rows.add(r)

    def band(self, r: int, text: str):
        self.fill_row(r, "band")
        self.put(f"A{r}", text, b=True, color="white", fill="band")

    def header(self, r: int, first: str, labels: dict, note_col=None, note=None, right=True):
        self.fill_row(r, "hdr")
        self.put(f"A{r}", first, b=True, fill="hdr")
        for col, text in labels.items():
            self.put(f"{col}{r}", text, b=True, fill="hdr", align="right" if right else None,
                     color="green" if str(text).startswith("=") else "black")
        if note_col:
            self.put(f"{note_col}{r}", note, b=True, fill="hdr")

    def title(self, text: str, sub: str = None):
        self.put("A1", text, b=True, color="title", size=14)
        if sub:
            self.put("A2", sub, color="grey")

    def note(self, ref: str, text: str):
        self.put(ref, text, color="grey")

    def finish(self, tall: dict = None):
        tall = tall or {}
        for r in sorted(self.rows):
            self.ws.set_row(r - 1, tall.get(r, 17.35 if r == 1 else 15.0))


class Book:
    def __init__(self, values: dict = None):
        self.values = values or {}
        self.buf = io.BytesIO()
        self.wb = xlsxwriter.Workbook(self.buf, {"in_memory": True})
        self.wb.formats[0].set_font_name("Arial")
        self.wb.formats[0].set_font_size(10)
        self._cache = {}

    def fmt(self, num=None, b=False, color="black", fill=None, top=False, size=None, wrap=False,
            valign=None, align=None):
        key = (num, b, color, fill, top, size, wrap, valign, align)
        if key not in self._cache:
            d = {"font_name": "Arial", "font_size": size or 10, "bold": b,
                 "font_color": COLORS.get(color, color)}
            if num:
                d["num_format"] = NUM.get(num, num)
            if fill:
                d.update(bg_color=FILLS[fill], pattern=1)
            if top:
                d["top"] = 1
            if wrap:
                d["text_wrap"] = True
            if valign:
                d["valign"] = valign
            if align:
                d["align"] = align
            self._cache[key] = self.wb.add_format(d)
        return self._cache[key]

    def sheet(self, name, widths, tab, last_col):
        return Sheet(self, name, widths, tab, last_col)

    def close(self) -> bytes:
        self.wb.close()
        return self.buf.getvalue()


# ----------------------------------------------------------------- layouts
def assumptions_layout(s: ModelSpec) -> dict:
    """Assumptions rows. Growth blocks have one row per growth unit (a segment, or its revenue lines when
    they drive it); margin and M&A blocks one row per segment."""
    n, nu = len(s.segments), len(s.growth_units)
    r, row = {}, 20
    r["g_band"] = 19
    for grp in ("bear", "base", "bull", "sel"):
        r[f"g_{grp}"] = row
        for u in range(nu):
            r[f"g_{grp}_{u}"] = row + 1 + u
        row += nu + 1
    row += 1
    r["mna_band"] = row
    for i in range(n):
        r[f"mna_{i}"] = row + 1 + i
    row += n + 2
    r["m_band"] = row
    row += 1
    for grp in ("bear", "base", "bull", "sel"):
        r[f"m_{grp}"] = row
        for i in range(n):
            r[f"m_{grp}_{i}"] = row + 1 + i
        row += n + 1
    row += 1
    r["corp_band"], r["corp_sales"], r["corp_cost"] = row, row + 1, row + 2
    row += 4
    r["is_band"] = row
    is_keys = ["amort", "sbc_pct", "repo_pct", "other_nonop", "int_yield", "int_rate", "etr", "nci_ni"]
    if s.has_nonrec:
        is_keys.append("nonrec")
    if any(s.drivers.get("adj_tax", ([0.0],))[0]) or any(s.adj_eps_tax_hist):
        is_keys.append("adj_tax")  # the company taxes non-GAAP income at one rate (CRM)
    for i, k in enumerate(is_keys):
        r[k] = row + 1 + i
    row += len(is_keys) + 2
    r["cf_band"] = row
    for i, k in enumerate(["cogs_pct", "da_pct", "capex_pct", "dso", "dio", "oca_pct", "dpo", "accrued_pct",
                           "other_op", "acq", "intang_share", "debt_iss", "buyback", "buyback_px",
                           "shares_issued", "stock_proc", "dps0", "div_growth"]):
        r[k] = row + 1 + i
    row += 20
    r["val_band"] = row
    for i, k in enumerate(["rf", "beta", "erp", "kd", "tax_wacc", "g", "exit_mult"]):
        r[k] = row + 1 + i
    for i, k in enumerate(["w_pg", "w_exit", "w_sotp", "w_comps"]):
        r[k] = row + 9 + i
    extra = ["peer_discount", "exit_pe", "other_stake", "nci_value", "fade_years", "comps_pe_weight"]
    if s.framework != "reference":
        extra += ["ronic", "stake_haircut"]
    if s.captive:
        extra.append("captive_share")
    for i, k in enumerate(extra):
        r[k] = row + 14 + i
    return r


def segments_layout(s: ModelSpec) -> dict:
    """Segments rows: per segment sales, growth, organic, M&A, then its revenue lines (sales + organic
    growth) when they drive it; corporate and total; segment profit block; memo block."""
    n = len(s.segments)
    r, row = {}, 6
    for i, sg in enumerate(s.segments):
        r[f"s{i}"], r[f"s{i}_g"], r[f"s{i}_org"], r[f"s{i}_mna"] = row, row + 1, row + 2, row + 3
        row += 4
        if sg.use_lines and sg.lines:
            for k in range(len(sg.lines)):
                r[f"l{i}_{k}"], r[f"l{i}_{k}_org"] = row, row + 1
                row += 2
        row += 1
    r["corp"], r["total"], r["t_g"], r["t_org"] = row, row + 1, row + 2, row + 3
    r["p_band"] = row + 5
    row += 6
    for i in range(n):
        r[f"p{i}"], r[f"p{i}_m"] = row + 2 * i, row + 1 + 2 * i
    base = row + 2 * n
    r["corp_p"], r["corp_pct"], r["tp"], r["tp_m"] = base, base + 1, base + 2, base + 3
    r["memo_band"] = base + 5
    for i in range(n):
        r[f"memo{i}"] = base + 6 + i
    return r


def kpi_layout(s: ModelSpec) -> dict:
    """KPIs tab: one 15-row block per volume x price build."""
    r, row = {}, 6
    for b, (i, u) in enumerate(s.kpi_units):
        r[b] = {"band": row, "vol": row + 1, "vol_bear": row + 2, "vol_base": row + 3, "vol_bull": row + 4,
                "vol_sel": row + 5, "px": row + 6, "px_bear": row + 7, "px_base": row + 8, "px_bull": row + 9,
                "px_sel": row + 10, "implied": row + 11, "model": row + 12, "diff": row + 13}
        row += 15
    return r


def sotp_layout(s: ModelSpec) -> dict:
    """SOTP rows: one per segment, plus one per sub-split part under a split segment."""
    r, seg_rows, part_rows, valued = 5, {}, {}, []
    for i, sg in enumerate(s.segments):
        seg_rows[i] = r
        parts = s.sotp_splits.get(sg.name) or []
        r += 1
        if not parts:
            valued.append(seg_rows[i])
        for k in range(len(parts)):
            part_rows[(i, k)] = r
            valued.append(r)
            r += 1
    out = {"seg": seg_rows, "part": part_rows, "corp": r, "stake": r + 1, "ev": r + 2, "less": r + 3, "eq": r + 4,
           "ps": r + 5, "up": r + 6, "impl": r + 7, "mkt": r + 8, "note": r + 10}
    out["valued"] = valued + [r]
    return out


# thesis placeholders: the analyst writes {fv}, the file shows the live number (PM round 2: typed numbers went
# stale — the thesis quoted $224 fair value, $221 DCF and $269 comps against $217, $219 and $245)
LIVE = {"fv": ("Summary!$G$19", "$", "0.00"), "price": ("Summary!$C$5", "$", "0.00"),
        "upside": ("Summary!$G$20", "", "0.0%"), "dcf_pg": ("Summary!$D$15", "$", "0.00"),
        "dcf_exit": ("Summary!$D$16", "$", "0.00"), "sotp": ("Summary!$D$17", "$", "0.00"),
        "comps": ("Summary!$D$18", "$", "0.00"), "bear": ("Summary!$C$49", "$", "0.00"),
        "base": ("Summary!$C$50", "$", "0.00"), "bull": ("Summary!$C$51", "$", "0.00"),
        "pw": ("Summary!$C$52", "$", "0.00"), "etr": ("Summary!$C$53", "", "0.0%"),
        "alpha": ("Summary!$D$53", "", "0.0%"), "irr": ("DCF!$C$105", "", "0.0%"), "wacc": ("DCF!$C$17", "", "0.0%"),
        "ke": ("DCF!$C$9", "", "0.0%"), "pe": ("Summary!$C$9", "", "0.0x"), "exit_pe": ("Summary!$C$25", "", "0.0x"),
        "ret3y": ("Summary!$C$29", "", "0.0%"), "req_margin": ("DCF!$C$107", "", "0.0%"),
        "req_g": ("DCF!$C$106", "", "0.0%"), "gap": ("Summary!$C$59", "", "0.0%"),
        "irr_spread": ("Summary!$D$59", "", "0.0%"), "p_bear_implied": ("Summary!$C$62", "", "0%"),
        "p_bear_long": ("Summary!$C$63", "", "0%"), "peer_discount_pe": ("Summary!$C$64", "", "0%"),
        "ret3y_pw": ("Summary!$C$65", "", "0.0%"), "g": ("DCF!$C$74", "", "0.00%"), "beta": ("DCF!$C$7", "", "0.00")}


SYN_CREDIT = 0.5  # share of announced synergies credited in a deal's value


def peer_growth(p_: dict):
    """A peer's next-year EPS growth from one source: consensus +1y ÷ 0y (both Yahoo, same fiscal years). The
    analyst's guidance EPS against consensus +1y mixed bases and fiscal years (PM round 3: NOW, WDAY and MSFT showed
    ~0% growth because the guide was already the next fiscal year)."""
    g = p_.get("g_next")
    if isinstance(g, (int, float)) and not isinstance(g, bool):
        # an analyst EPS normalized for a one-off (snapshot cleared) moves the base the growth runs from (NFLX round 5:
        # GOOGL at $10.82 ex-gains against $14.93 next year is +38%, not the -24% off the gain-inflated $19.60); a
        # currency mismatch (NVO: DKK consensus vs a USD ADR, 6.6x) is never treated as a normalization
        e, en = p_.get("eps"), p_.get("eps_next")
        if p_.get("snapshot") is None and isinstance(e, (int, float)) and isinstance(en, (int, float)) and e > 0 \
                and g > -1:
            r = e / (en / (1 + g))  # the analyst's EPS against the consensus this year the growth was measured from
            if 0.3 <= r <= 3.5 and not 0.85 <= r <= 1.15:
                return en / e - 1
        return g
    if p_.get("eps") and p_.get("eps_next") and p_["eps"] > 0 and p_.get("eps_next_basis") == "same":
        return p_["eps_next"] / p_["eps"] - 1
    return None


def ev_growth_fit(peers: list) -> dict:
    """Growth-adjusted peer EV / EBITDA: OLS on next-year consensus REVENUE growth (an enterprise multiple, so not
    EPS growth that buybacks inflate), with the same sanity tests; else the plain median. PM round 3: a 22.4x median
    set by faster-growing NOW (38.5x) and WDAY (31.7x) was applied to CRM unadjusted."""
    pts = [(p_["ev"] / p_["ebitda"], p_["rev_g_next"]) for p_ in peers
           if p_.get("ev") and p_.get("ebitda") and p_["ebitda"] > 0 and
           isinstance(p_.get("rev_g_next"), (int, float)) and not isinstance(p_.get("rev_g_next"), bool)]
    return _fit([(m, g) for m, g in pts if 0 < m < 80 and -0.5 < g < 1.0], "EV / EBITDA", peg=False)


def ev_dispersed(peers: list) -> str:
    """Why the peer EV / EBITDA median cannot be applied as is: no sane growth fit and an interquartile range above
    half the median (CRM's software peers: 8.8x–38.5x, set by stock-comp intensity, not growth). '' when usable."""
    from statistics import median, quantiles
    m = sorted(p_["ev"] / p_["ebitda"] for p_ in peers if p_.get("ev") and p_.get("ebitda") and p_["ebitda"] > 0
               and 0 < p_["ev"] / p_["ebitda"] < 80)
    if len(m) < 4 or ev_growth_fit(peers)["a"] is not None:
        return ""
    q1, _, q3 = quantiles(m, n=4)
    med = median(m)
    if med > 0 and (q3 - q1) / med > 0.5:
        return (f"peer EV / EBITDA {m[0]:.1f}x–{m[-1]:.1f}x, interquartile range {(q3 - q1) / med:.0%} of the "
                f"{med:.1f}x median and no growth fit that holds: the comps value uses P/E only")
    return ""


ONE_OFF_G, CYCLE_G = -0.20, 0.40


def _pe(p_: dict):
    return p_["price"] / p_["eps"] if p_.get("price") and p_.get("eps") and p_["eps"] > 0 else None


def peer_eps_issue(p_: dict, peers: list = ()) -> str:
    """Why a peer's current-year EPS is not a P/E base: consensus EPS falling 20%+ next year means this year
    carries one-offs (PM round 4: GOOGL's $19.60 included ~$127bn of securities gains, 15x instead of ~27x), and
    rising 100%+ that a charge depressed it; a P/E under 0.4x or over 2.5x the other peers' median is a basis error (LLY v4: NVO at 1.8x on DKK EPS against a USD
    ADR price, MRK at 53x on EPS after an acquired-IPR&D charge). '' when the EPS is usable."""
    g = peer_growth(p_)
    if isinstance(g, (int, float)) and g < ONE_OFF_G:
        return (f"next-year consensus EPS {g:+.0%}: this year's EPS carries one-offs (gains, a reversal), so its "
                "P/E is excluded")
    if isinstance(g, (int, float)) and g > 1.0:
        return (f"next-year consensus EPS {g:+.0%}: this year's EPS is depressed by a one-off charge, so its P/E is "
                "excluded")
    rg = p_.get("rev_g_next")
    if isinstance(g, (int, float)) and isinstance(rg, (int, float)) and g < -0.03 and rg > 0.05:
        # EPS falling while revenue grows: gains in this year's EPS (PM, ADBE: CRM's $16.75 includes investment
        # gains, -4% next year on +9% revenue; its operating EPS is ~$13.75)
        return (f"next-year consensus EPS {g:+.0%} while revenue grows {rg:+.0%}: this year's EPS carries one-off "
                "gains, so its P/E is excluded")
    pe = _pe(p_)
    others = sorted(x for x in (_pe(q) for q in peers if q is not p_) if x)
    if pe and len(others) >= 3:
        med = others[len(others) // 2] if len(others) % 2 else (others[len(others) // 2 - 1] + others[len(others) // 2]) / 2
        if pe < 0.4 * med or pe > 2.5 * med:  # the checker's outlier band
            return (f"P/E {pe:.1f}x against the other peers' {med:.1f}x median: an EPS on another basis (currency, "
                    "GAAP after a one-off charge) — excluded until its EPS is fixed")
    return ""


def peer_ni_growth(p_: dict):
    """A peer's next-year growth on the company's basis: net income, so buybacks are not growth (PM round 7: the PEG
    set Adobe's net-income growth against the peers' per-share growth). Consensus EPS growth with the share count
    moving as it did over the last year; per-share growth when the share change is not known."""
    g = peer_growth(p_)
    sc = p_.get("share_chg")
    if not isinstance(g, (int, float)) or not isinstance(sc, (int, float)):
        return g
    return (1 + g) * (1 + sc) - 1


def peer_growth_fit(peers: list, cyclical: bool = False) -> dict:
    """Growth-adjusted peer P/E: OLS of P/E on next-year EPS growth when it is sane (6+ peers, slope > 0, R² ≥ 0.2),
    else the median PEG (3+ growing peers), else the plain median. PM round 3: the fit P/E = 17.6 − 10.6 × growth
    paid less for more growth and was applied anyway. Peers with one-off EPS are left out, and so are rebounds
    (40%+ growth) from the fit; for a cyclical no growth fit is used (PM round 4: DE at a trough P/E with rebound
    growth set the slope and pulled CAT's applied P/E to 14x against a 23.6x median)."""
    pts = [(p_["price"] / p_["eps"], peer_ni_growth(p_)) for p_ in peers
           if p_.get("price") and p_.get("eps") and p_["eps"] > 0 and peer_ni_growth(p_) is not None
           and not peer_eps_issue(p_, peers)]
    pts = [(pe, g) for pe, g in pts if 0 < pe < 150 and -0.5 < g < 1.0]
    rebound = [g for _, g in pts if g > CYCLE_G]
    if cyclical:
        out = _fit([], "P/E")
        out.update(n=len(pts), how="peer median: for a cyclical, next-year growth measures the cycle (trough "
                                   "rebounds, peak declines), not the franchise, so no growth fit is used")
        return out
    out = _fit([(pe, g) for pe, g in pts if g <= CYCLE_G], "P/E")
    if rebound and out["how"]:
        out["how"] += f" ({len(rebound)} peer(s) rebounding 40%+ left out of the fit)"
    return out


def _fit(pts: list, what: str, peg: bool = True) -> dict:
    out = {"a": None, "b": None, "r2": None, "n": len(pts), "peg": None, "how": "", "rejected": ""}
    if len(pts) >= 6:
        mx, my = sum(g for _, g in pts) / len(pts), sum(pe for pe, _ in pts) / len(pts)
        sxx = sum((g - mx) ** 2 for _, g in pts)
        syy = sum((pe - my) ** 2 for pe, _ in pts)
        if sxx > 1e-6 and syy > 1e-9:
            b_ = sum((g - mx) * (pe - my) for pe, g in pts) / sxx
            a_ = my - b_ * mx
            r2 = b_ ** 2 * sxx / syy
            out["r2"] = r2
            fit_txt = f"{what} = {a_:.1f} + {b_:.1f} × growth (R² {r2:.2f}, {len(pts)} peers)"
            if b_ > 0 and r2 >= 0.2:
                out.update(a=a_, b=b_, how=f"OLS across {len(pts)} peers: {fit_txt}")
            else:
                out["rejected"] = (f"regression rejected: {fit_txt}" +
                                   (" pays less for more growth" if b_ <= 0 else " explains too little"))
    elif pts:
        out["rejected"] = f"{len(pts)} peers with consensus growth (a regression needs 6)"
    if out["a"] is None and peg and len(pts) >= 3:
        pegs = sorted(pe / (g * 100) for pe, g in pts if g > 0.01)
        if len(pegs) >= 3:
            out["peg"] = pegs[len(pegs) // 2]
            out["how"] = (f"median PEG {out['peg']:.2f} across {len(pegs)} peers" +
                          (f" ({out['rejected']})" if out["rejected"] else ""))
    return out


def live_text(text):
    """'Fair value {fv} vs {price}' -> an Excel formula that shows the live numbers; plain text passes through."""
    if not isinstance(text, str) or not re.search(r"\{(" + "|".join(LIVE) + r")\}", text):
        return text
    parts = []
    for k, piece in enumerate(re.split(r"\{(\w+)\}", text)):
        if k % 2 and piece in LIVE:
            ref, pre, fmt = LIVE[piece]
            fmt_x = fmt.replace("x", "")
            num = f'TEXT({ref},"{fmt_x}")' + ('&"x"' if fmt.endswith("x") else "")
            parts.append(f'IF(ISNUMBER({ref}),"{pre}"&{num},{ref}&"")')
        else:
            piece = piece if k % 2 == 0 else "{" + piece + "}"
            for j in range(0, len(piece), 250):  # Excel: at most 255 characters per string literal
                parts.append('"' + piece[j:j + 250].replace('"', '""') + '"')
    return "=" + "&".join(p for p in parts if p != '""')


def is_layout(has_disc: bool, has_nonrec: bool = False) -> dict:
    keys = ["hdr", "band1", "sales", "cogs", "rnd", "sga", "oth_opex", "opinc_rep", None, "band2", "sales_b",
            "growth", "seg_profit", "seg_margin", "amort", "sbc", "repo", "ebit", "op_margin", "int_inc",
            "pension"] + (["nonrec"] if has_nonrec else []) + ["other_inc", "interest", "pretax", "tax", "etr"]
    if has_disc:
        keys.append("disc")
    keys += ["ni", "nci", "ni_attr", "shares", "eps", "adj_eps", "adj_tie", "op_eps", "adj_growth", None, "band3",
             "da", "ebitda",
             "ebitda_m", "seg_ebitda", "memo_cogs", "memo_cogs_pct"]
    return {k: 4 + i for i, k in enumerate(keys) if k}


# ------------------------------------------------------------------ build
def build_workbook(s: ModelSpec, values: dict = None, unstored: bool = False) -> bytes:
    """values: computed formula results to store in the file (pipeline.build); unstored marks a deliverable
    whose results could not be computed, so the Cover says to open it in Excel."""
    refresh_ytd_adj_eps(s)
    n = len(s.segments)
    A = assumptions_layout(s)
    SG = segments_layout(s)
    IS = is_layout(s.has_disc, s.has_nonrec)
    KP = kpi_layout(s)
    gunits = s.growth_units
    uidx = {}  # (segment index, line index or None) -> growth unit row index on Assumptions
    for u, (i, unit) in enumerate(gunits):
        sg_ = s.segments[i]
        uidx[(i, sg_.lines.index(unit) if unit is not sg_ else None)] = u
    INT_SEG0 = 48
    AP_LBL = ("Accounts payable & accrued liabilities (reported combined)"  # DE: unclassified balance sheet
              if "Accounts payable reported combined with accrued liabilities" in s.flags else "Accounts payable")
    INT_CORP, INT_TOT = 48 + n, 49 + n
    INT_G = 52 + n                       # guidance band row
    INT_ROY = INT_G + 10                 # rest-of-year check
    npeer = max(1, len(s.peers))
    CP = {k: v + npeer for k, v in {
        "med": 6, "mean": 7, "co": 8, "disc": 9, "applied": 10, "eps": 11, "implied": 12, "up": 13, "low": 15,
        "high": 16, "ev_band": 18, "ev_med": 19, "ev_mean": 20, "ev_co": 21, "ev_applied": 22, "ev_ebitda": 23,
        "ev_ev": 24, "ev_less": 25, "ev_ps": 26, "ev_up": 27, "ev_low": 28, "ev_high": 29, "bl_band": 31,
        "bl_w": 32, "bl_base": 33, "bl_low": 34, "bl_high": 35, "bl_up": 36, "note": 38,
        "ga_band": 40, "ga_g": 41, "ga_fit": 42, "ga_med": 43, "own_band": 45, "own_med": 46, "own_lo": 47,
        "own_hi": 48, "own_now": 49, "own_w": 50, "own_row": 51, "own_val": 52, "eps_band": 54, "eps_fy1": 55,
        "eps_mid": 56, "stake": 57}.items()}
    SO = sotp_layout(s)

    bk = Book(values)
    hist_lbl, fc_lbl = s.hist_labels, s.fc_labels
    yr_labels = dict(zip(ALL, hist_lbl + fc_lbl))
    units = f"($ in millions, FY ends {s.fye_text})"
    px = f"${s.price:,.2f}"
    a = lambda key, col="F": f"Assumptions!{col}{A[key]}"           # noqa: E731
    SW = "Assumptions!$C$14"                                         # segment profit after amort & SBC?
    EXSBC = "Assumptions!$C$15"                                      # adjusted EPS also excludes SBC?
    EXREPO = "Assumptions!$C$5"                                      # ... and non-segment charges (share)?
    EXNR = "Assumptions!$C$16"                                       # ... and one-off gains / losses?
    AMSH = "Assumptions!$C$17"                                       # share of amortization excluded
    aa = lambda key: f"Assumptions!$C${A[key]}"                      # noqa: E731
    stmt_w = {"A": 56, "B": 10, "C": 11, "D:K": 9.14, "L": 62}

    cover = bk.sheet("Cover", {"A": 34, "B": 90}, "#1F4E79", "C")
    summ = bk.sheet("Summary", {"A": 52, "B": 2, "C": 13, "D:E": 9.14, "F": 11, "G": 13, "H": 52}, "#1F4E79", "H")
    asm = bk.sheet("Assumptions", {"A": 52, "B": 12, "C": 10, "D:K": 9.14, "L": 70}, "#FFC000", "L")
    itm = bk.sheet("Interim", {"A": 58, "B": 13, "C:F": 11, "G": 44}, "#A6A6A6", "G")
    seg = bk.sheet("Segments", stmt_w, "#5B9BD5", "L")
    kpi = bk.sheet("KPIs", stmt_w, "#5B9BD5", "L") if s.kpi_units else None
    inc = bk.sheet("IS", stmt_w, "#5B9BD5", "L")
    bal = bk.sheet("BS", stmt_w, "#5B9BD5", "L")
    cfs = bk.sheet("CF", stmt_w, "#5B9BD5", "L")
    sch = bk.sheet("Schedules", stmt_w, "#5B9BD5", "L")
    dcf = bk.sheet("DCF", {"A": 56, "B": 11, "C:K": 10, "L": 62}, "#70AD47", "L")
    cmp_ = bk.sheet("Comps", {"A": 40, "B": 8, "C": 30, "D": 11, "E": 14, "F": 12, "G": 60, "H:I": 12, "J": 11,
                              "K": 46}, "#70AD47", "K")
    sotp = bk.sheet("SOTP", {"A": 44, "B": 14, "C": 13, "D": 11, "E:G": 8, "H:J": 11, "K": 60}, "#70AD47", "K")
    th = bk.sheet("Thesis", {"A": 60, "B": 30, "C:D": 12}, "#1F4E79", "D")
    econ_wb = s.framework != "reference"
    dl_ = bk.sheet("Deals", {"A": 40, "B": 12, "C": 12, "D": 11, "E": 13, "F": 11, "G": 90}, "#70AD47", "G") \
        if econ_wb else None
    ck = bk.sheet("Checks", {"A": 70, "B": 14, "C": 90, "D:H": 11}, "#C00000", "H") if econ_wb else None

    # ================================================================ Cover
    cover.put("A1", f"{s.name} ({s.exchange}: {s.ticker})", b=True, color="title", size=18)
    cover.put("A2", "Buy-side valuation model — 3-statement, DCF, comps, sum of the parts", b=True, size=12)
    cover.put("A3", f"Valuation date: {s.as_of:%B %-d, %Y} · Price {px} · {s.built_from}")
    if s.framework != "reference" and s.rating:
        cover.put("A4", "Rating (rule):", b=True)
        cover.put("B4", f"{s.rating.get('rating')} — {s.rating.get('why', '')}"[:240], b=True,
                  color="#C00000" if s.rating.get("rating") == "Under review" else "title")
    cover.put("A5", "Model check:", b=True)
    cover.put("B5", "=Summary!C45", b=True, fill="key")
    if s.framework != "reference":  # the decision number leads (PM round 7: the base value headlined at +63%)
        cover.put("A6", "Probability-weighted value ($):", b=True)
        cover.put("B6", "=Summary!C52", num="usd", b=True, fill="key")
        cover.put("C6", '=IFERROR(IF(Summary!C59>=0,"+","")&TEXT(Summary!C59,"0.0%")&" vs the price; base case $"&'
                        'TEXT(Summary!G19,"0.00")&"; bear probability the price implies "&TEXT(Summary!C62,"0%"),"")')
    else:
        cover.put("A6", "Weighted fair value ($):", b=True)
        cover.put("B6", "=Summary!G19", num="usd", b=True, fill="key")
    cover.put("A7", "Scenario:", b=True)
    cover.put("B7", "=Assumptions!D4", b=True)
    if unstored:
        cover.put("A8", "Formula results are not stored in this file: open it in Excel (it calculates on open) "
                        "before reading the numbers", b=True, color="#C00000")
    cover.band(9, "Colour legend")
    for r, (k, txt, kw) in enumerate([
            ("1,234", "Hard-coded input (history from filings, or an assumption you can change)", {"color": "blue"}),
            ("1,234", "Formula on the same sheet", {"color": "black"}),
            ("1,234", "Direct link to another sheet", {"color": "green"}),
            ("Yellow fill", "Scenario selector (Assumptions!C4)", {"fill": "sel"}),
            ("Blue fill", "Key output or check", {"fill": "key"})], start=10):
        cover.put(f"A{r}", k, b=True, **kw)
        cover.put(f"B{r}", txt)
    cover.band(16, "Tabs (in build order)")
    tabs = [("Assumptions", "Scenario selector, market data and every forecast driver"),
            ("Interim", s.interim_tab_text),
            ("Segments", "Sales and segment profit by segment" + (" and revenue line" if any(
                sg.use_lines and sg.lines for sg in s.segments) else "")),
            *([("KPIs", "Operating metrics: volume x price revenue build")] if s.kpi_units else []),
            ("IS", "Income statement: reported history and segment-based forecast"),
            ("Schedules", "PP&E, goodwill & intangibles, debt, working capital, shares, equity"),
            ("BS", "Balance sheet (balances every year: see check row)"),
            ("CF", "Cash flow statement and free cash flow"),
            ("DCF", "WACC, unlevered FCF, two terminal methods, sensitivity grids"),
            ("Comps", f"Peer P/E on {fc_lbl[0][:-1]} estimates and EV / LTM EBITDA"),
            ("SOTP", f"Sum of the parts on {fc_lbl[0]} EBITDA"),
            *([("Deals", "Pending acquisitions and separations, probability-weighted into fair value")]
              if s.framework != "reference" else []),
            ("Summary", "Football field, weighted fair value, 3-year return, key financials" +
             ("; bear / base / bull, what the price implies, rating" if s.framework != "reference" else "")),
            ("Thesis", f"{s.stance_phrase.split()[0].capitalize()} thesis, risks, catalysts"),
            *([("Checks", "Pre-publication gates, plug table, consensus, final check, audit trail")]
              if s.framework != "reference" else [])]
    for r, (t, d) in enumerate(tabs, start=17):
        cover.put(f"A{r}", t, b=True)
        cover.put(f"B{r}", d)
    src_row = 17 + len(tabs) + 1
    cover.band(src_row, "Sources")
    for r, line in enumerate(s.sources, start=src_row + 1):
        cover.put(f"A{r}", line)
    def wrapped(items):  # long notes on lines that fit columns A:B, so they read without scrolling right
        return [ln for it in items for ln in textwrap.wrap(str(it), 130, subsequent_indent="   ") or [""]]

    r = src_row + len(s.sources) + 2
    cover.band(r, "Who set the assumptions")
    prov = wrapped(s.provenance)
    for k, line in enumerate(prov, start=r + 1):
        cover.put(f"A{k}", line)
    notes = s.data_notes
    r += len(prov) + 2
    if notes:
        cover.band(r, f"Data notes ({len(notes)}): read before relying on the model")
        lines = wrapped(notes)
        for k, line in enumerate(lines, start=r + 1):
            cover.put(f"A{k}", line, color="grey")
        r += len(lines) + 2
    open_checks = (s.last_review or {}).get("problems") or []
    if open_checks:  # what the model checker still flags: the PM sees it without running anything
        cover.band(r, f"Open checks ({len(open_checks)}): flagged by the model checker — resolve or explain")
        for k, line in enumerate(wrapped(open_checks), start=r + 1):
            cover.put(f"A{k}", line, color="grey")
    cover.finish({1: 22.05})

    # ================================================================ Assumptions
    asm.title("Assumptions & scenario drivers",
              "Blue = input you can change · black = formula · green = link. Change the scenario in C4.")
    asm.put("A4", "Scenario selector (1 = Bear, 2 = Base, 3 = Bull)", b=True)
    asm.put("A5", "Adjusted EPS also excludes non-segment charges (restructuring, deal costs)? (1 = yes, 0 = no)")
    asm.put("C5", s.adj_eps_ex_repo, fill="input")
    asm.note("E5", "Match the company's non-GAAP EPS definition; the charges stay a cost in EBIT and the DCF")
    asm.put("C4", s.scenario, b=True, fill="sel")
    asm.put("D4", '=CHOOSE($C$4,"Bear","Base","Bull")', b=True)
    asm.band(6, "Market data at the valuation date")
    rows = [
        (7, "Valuation date", f"=DATE({s.as_of.year},{s.as_of.month},{s.as_of.day})", "date", s.as_of_note, "blue"),
        (8, "Share price ($)", s.price, "usd", s.price_note, None),
        (9, "Basic shares outstanding (mm)", s.basic_shares, "sh", s.basic_note, None),
        (10, "Dilutive securities (mm)", s.dilutive, "sh", s.dilutive_note, None),
    ]
    for r, label, v, num, note, color in rows:
        asm.put(f"A{r}", label)
        asm.put(f"C{r}", v, num=num, fill="input", color=color)
        asm.note(f"E{r}", note)
    asm.put("A11", "Diluted shares for valuation (mm)")
    asm.put("C11", "=C9+C10", num="sh")
    asm.note("E11", "Basic + dilutive")
    asm.put("A12", "Fiscal year end of current year")
    fy = s.fy_end_cur
    asm.put("C12", f"=DATE({fy.year},{fy.month},{fy.day})", num="date", fill="input", color="blue")
    if s.framework != "reference":  # cash flows count from the balance sheet that net debt comes from
        asm.put("A13", f"Share of {fc_lbl[0][:-1]} cash flow after the {s.interim.period_end:%b %-d, %Y} balance sheet")
        asm.put("C13", s.cf_share_bs, num="dec", fill="input")
        asm.note("E13", s.cf_share_note)
    else:
        asm.put("A13", f"Stub: fraction of {fc_lbl[0][:-1]} remaining after valuation date")
        asm.put("C13", "=MAX(0,(C12-C7)/365)", num="dec")
        asm.note("E13", f"Used to count only the {s.stub_text} cash flow in the DCF")
    asm.put("A14", "Segment profit already after amortization & stock comp? (1 = yes, 0 = no)")
    asm.put("C14", s.seg_after_charges, fill="input")
    asm.note("E14", s.seg_after_charges_note)
    asm.put("A15", "Adjusted EPS also excludes stock comp? (1 = yes, 0 = no)")
    asm.put("C15", s.adj_eps_ex_sbc, fill="input")
    asm.note("E15", "Match the company's non-GAAP EPS definition (most software companies exclude stock comp) so P/E "
                    "comps compare like with like; stock comp stays a cost in EBIT and the DCF")
    asm.put("A16", "Adjusted EPS also excludes one-off gains / losses (e.g. investment gains)? (1 = yes, 0 = no)")
    asm.put("C16", s.adj_eps_ex_nonrec, fill="input")
    asm.note("E16", "0 when the company's non-GAAP EPS includes them (CRM: strategic-investment gains): IS then shows "
                    "operating EPS without them, which every multiple uses")
    asm.put("A17", "Share of acquired-intangible amortization excluded from adjusted EPS (0–1)")
    asm.put("C17", s.adj_eps_amort_share, num="pct", fill="input")
    asm.note("E17", "1 = all of it; less when the company excludes only some deals' amortization (SHW: Valspar only)")
    asm.header(18, "Forecast drivers by year", yr_labels, "L", "Source / note")

    def scen_block(prefix, band_row, band_text, kind, unit, note):
        asm.band(band_row, band_text)
        for grp, label in (("bear", "Bear case"), ("base", "Base case"), ("bull", "Bull case")):
            r0 = A[f"{prefix}_{grp}"]
            asm.put(f"A{r0}", label, b=True)
            asm.note(f"L{r0}", note)
            for i, sg in enumerate(s.segments):
                r = A[f"{prefix}_{grp}_{i}"]
                asm.put(f"A{r}", f"   {sg.name}")
                asm.note(f"B{r}", unit)
                for j, c in enumerate(P):
                    asm.put(f"{c}{r}", getattr(sg, kind)[grp][j], num="pct", fill="input")
        r0 = A[f"{prefix}_sel"]
        asm.put(f"A{r0}", "Selected case (feeds the model)", b=True)
        for i, sg in enumerate(s.segments):
            r = A[f"{prefix}_sel_{i}"]
            asm.put(f"A{r}", f"   {sg.name}")
            asm.note(f"B{r}", unit)
            for c in P:
                asm.put(f"{c}{r}", f"=CHOOSE($C$4,{c}{A[f'{prefix}_bear_{i}']},{c}{A[f'{prefix}_base_{i}']},"
                                   f"{c}{A[f'{prefix}_bull_{i}']})", num="pct", fill="key")

    kmap = {id(u): b_ for b_, (_, u) in enumerate(s.kpi_units)}

    def unit_label(i, u):
        sg_ = s.segments[i]
        return sg_.name if u is sg_ else (f"{sg_.name}: {u.name}" if n > 1 else u.name)

    asm.band(A["g_band"], s.growth_band_text)
    for grp, label in (("bear", "Bear case"), ("base", "Base case"), ("bull", "Bull case")):
        r0 = A[f"g_{grp}"]
        asm.put(f"A{r0}", label, b=True)
        asm.note(f"L{r0}", s.growth_note)
        for u, (i, unit) in enumerate(gunits):
            r = A[f"g_{grp}_{u}"]
            asm.put(f"A{r}", f"   {unit_label(i, unit)}")
            asm.note(f"B{r}", "% y/y")
            for j, c in enumerate(P):
                if id(unit) in kmap:
                    k = KP[kmap[id(unit)]]
                    asm.put(f"{c}{r}", f"=(1+KPIs!{c}{k['vol_' + grp]})*(1+KPIs!{c}{k['px_' + grp]})-1", num="pct")
                else:
                    asm.put(f"{c}{r}", unit.growth[grp][j], num="pct", fill="input")
            if grp == "base" and unit is not s.segments[i] and getattr(unit, "note", ""):
                asm.note(f"L{r}", unit.note)
            elif grp == "base" and id(unit) in kmap:
                asm.note(f"L{r}", "Volume x price build on the KPIs tab")
    r0 = A["g_sel"]
    asm.put(f"A{r0}", "Selected case (feeds the model)", b=True)
    for u, (i, unit) in enumerate(gunits):
        r = A[f"g_sel_{u}"]
        asm.put(f"A{r}", f"   {unit_label(i, unit)}")
        asm.note(f"B{r}", "% y/y")
        for c in P:
            asm.put(f"{c}{r}", f"=CHOOSE($C$4,{c}{A[f'g_bear_{u}']},{c}{A[f'g_base_{u}']},{c}{A[f'g_bull_{u}']})",
                    num="pct", fill="key")
    asm.band(A["mna_band"], "Acquisitions, divestitures & FX contribution to growth (all scenarios)")
    for i, sg in enumerate(s.segments):
        r = A[f"mna_{i}"]
        asm.put(f"A{r}", f"   {sg.name}")
        asm.note(f"B{r}", "% y/y")
        for j, c in enumerate(P):
            asm.put(f"{c}{r}", sg.mna[j], num="pct", fill="input")
        asm.note(f"L{r}", sg.mna_note)
    scen_block("m", A["m_band"], s.margin_band_text, "margin", "% of sales", s.margin_note)

    asm.band(A["corp_band"], "Corporate & All Other")
    asm.put(f"A{A['corp_sales']}", "   Corporate & other sales")
    asm.note(f"B{A['corp_sales']}", "$mm")
    asm.put(f"A{A['corp_cost']}", "   Corporate costs (segment profit)")
    asm.note(f"B{A['corp_cost']}", "% of total sales")
    for j, c in enumerate(P):
        asm.put(f"{c}{A['corp_sales']}", s.corp_sales[j], num="mm", fill="input")
        asm.put(f"{c}{A['corp_cost']}", s.corp_cost_pct[j], num="pct", fill="input")
    asm.note(f"L{A['corp_sales']}", s.corp_sales_note)
    asm.note(f"L{A['corp_cost']}", s.corp_cost_note)

    DRIVER_META = {
        "amort": ("   Amortization of acquisition-related intangibles", "$mm", "mm"),
        "sbc_pct": ("   Stock compensation", "% of sales", "pct"),
        "repo_pct": ("   Repositioning & other non-segment charges", "% of sales", "pct"),
        "other_nonop": ("   Other non-operating income (pension & other)", "$mm", "mm"),
        "int_yield": ("   Interest income yield on beginning cash & ST investments", "%", "pct"),
        "int_rate": ("   Interest rate on average total debt", "%", "pct"),
        "etr": ("   Effective tax rate", "%", "pct"),
        "nci_ni": ("   Net income to noncontrolling interest", "$mm", "mm"),
        "nonrec": ("   Non-recurring gains (losses) in pre-tax income", "$mm", "mm"),
        "adj_tax": ("   Non-GAAP tax rate in adjusted EPS (company definition; 0 = adjustments at the effective "
                    "rate)", "%", "pct"),
        "cogs_pct": ("   Cost of sales (memo, for inventory & payables days)", "% of sales", "pct"),
        "da_pct": ("   Depreciation & other amortization", "% of sales", "pct"),
        "capex_pct": ("   Capital expenditures", "% of sales", "pct"),
        "dso": ("   Days sales outstanding", "days", "days"),
        "dio": ("   Days inventory (on cost of sales)", "days", "days"),
        "oca_pct": ("   Other current assets", "% of sales", "pct"),
        "dpo": ("   Days payable (on cost of sales)", "days", "days"),
        "accrued_pct": ("   Accrued liabilities", "% of sales", "pct"),
        "other_op": ("   Other operating cash items, net", "$mm", "mm"),
        "acq": ("   Acquisitions (cash)", "$mm", "mm"),
        "intang_share": ("   Share of purchase price to intangibles", "%", "pct"),
        "debt_iss": ("   Net debt issuance (repayment)", "$mm", "mm"),
        "buyback": ("   Share repurchases", "$mm", "mm"),
        "buyback_px": ("   Average repurchase price", "$/share", "usd"),
        "shares_issued": ("   Shares issued (options & RSUs)", "mm", "sh"),
        "stock_proc": ("   Proceeds from stock issuance", "$mm", "mm"),
        "dps0": (f"   Dividend per share ({fc_lbl[0]})", "$/share", "eps"),
        "div_growth": ("   Dividend growth", "% y/y", "pct"),
    }
    asm.band(A["is_band"], "Income statement drivers below segment profit")
    asm.band(A["cf_band"], "Cash flow & balance sheet drivers")
    for key, (label, unit, num) in DRIVER_META.items():
        if key not in A:
            continue
        r = A[key]
        asm.put(f"A{r}", label)
        asm.note(f"B{r}", unit)
        vals, note = s.drivers[key]
        cols = P if key not in ("dps0", "div_growth") else (["F"] if key == "dps0" else P[1:])
        for c, v in zip(cols, vals):
            asm.put(f"{c}{r}", v, num=num, fill="input")
        if key == "adj_tax":  # history years: the rate the company stated, for the tie-out
            for c, v in zip("CDE", s.adj_eps_tax_hist):
                if v:
                    asm.put(f"{c}{r}", v, num=num, fill="input")
        asm.note(f"L{r}", note)

    asm.band(A["val_band"], "Valuation inputs")
    VAL_META = {
        "rf": ("Risk-free rate (10-yr U.S. Treasury)", "pct"), "beta": ("Equity beta", "dec"),
        "erp": ("Equity risk premium", "pct"), "kd": ("Pre-tax cost of debt", "pct"),
        "tax_wacc": ("Tax rate for WACC", "pct"), "g": ("Terminal growth rate", "pct"),
        "exit_mult": ("Exit EV / EBITDA multiple", "x"),
        "w_pg": ("Weight: DCF (perpetuity growth)", "pct"), "w_exit": ("Weight: DCF (exit multiple)", "pct"),
        "w_sotp": ("Weight: Sum of the parts", "pct"), "w_comps": ("Weight: Trading comps (P/E)", "pct"),
        "peer_discount": (f"{s.label('peer_discount', 'Discount (premium)')} applied to peer P/E", "pct"),
        "exit_pe": (f"Exit P/E for 3-year return (on {fc_lbl[4]} EPS)", "x"),
        "other_stake": (f"{s.label('other_stake', 'Other investments / stakes not in EBITDA')} value (SOTP only)",
                        "mm"),
        "nci_value": ("Noncontrolling interests (incl. redeemable)", "mm"),
        "fade_years": (f"Fade years after {fc_lbl[-1]} (growth fades to terminal; 0 = two-stage DCF)", "int"),
        "comps_pe_weight": ("Comps: weight of P/E (rest = EV / LTM EBITDA)", "pct"),
        "captive_share": ("Captive-finance share of total debt (excluded from net debt)", "pct"),
        "ronic": ("Return on new invested capital (terminal reinvestment = g ÷ RONIC)", "pct"),
        "stake_haircut": ("Haircut on stakes outside EBITDA (tax, liquidity)", "pct"),
    }
    for key, (label, num) in VAL_META.items():
        if key not in A:
            continue
        r = A[key]
        v, note = s.valuation[key]
        asm.put(f"A{r}", label)
        if key == "exit_mult" and s.capital.get("exit_derived") and s.framework != "reference":
            # derived from the same framework as the perpetuity method: the forward EV/EBITDA that terminal
            # growth, RONIC and WACC imply (DCF!C53); type a number to override it (the checker gates the gap)
            asm.put(f"C{r}", "=ROUND(DCF!$C$53,1)", num=num, fill="input")
            note = ("Derived: the EV / EBITDA the perpetuity method implies at the terminal year (DCF!C53) — "
                    "terminal growth, RONIC and WACC set it. Overwrite only within ±20% and say why; "
                    f"today's market multiple: {note}")
        else:
            asm.put(f"C{r}", v, num=num, fill="input")
        asm.note(f"E{r}", note)
    asm.ws.freeze_panes(18, 2)
    asm.finish()

    # ================================================================ Interim
    it = s.interim
    itm.title(s.interim_title, s.interim_sub)
    itm.header(5, "($ in millions, except per share)",
               {"C": it.label_prior, "D": it.label_cur, "E": f"FY{s.fy_hist[-1]}", "F": it.ltm_label}, right=False)
    itm.put("G5", "Source", b=True, fill="hdr")
    src = s.interim_source
    flow = [(6, "Net sales", "sales"), (7, "Cost of products & services sold", "cogs"),
            (8, "Research & development", "rnd"), (9, "Selling, general & administrative", "sga"),
            (10, "Other operating expense (income), net", "oth_opex")]
    for r, label, k in flow:
        itm.put(f"A{r}", label)
        for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
            itm.put(f"{c}{r}", d.get(k), num="mm")
        itm.put(f"F{r}", f"=E{r}-C{r}+D{r}", num="mm")
        itm.note(f"G{r}", src)
    itm.put("A11", "Operating income (GAAP)", b=True)
    for c in "CDEF":
        itm.put(f"{c}11", f"={c}6-{c}7-{c}8-{c}9-{c}10", num="mm", b=True)
    for r, label, k in [(12, 'Other income, net (sign flipped from "other (income) expense")', "other_inc"),
                        (13, "Interest & other financial charges", "interest_exp")]:
        itm.put(f"A{r}", label)
        for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
            itm.put(f"{c}{r}", d.get(k), num="mm")
        itm.put(f"F{r}", f"=E{r}-C{r}+D{r}", num="mm")
        itm.note(f"G{r}", src)
    itm.put("A14", "Income before taxes")
    for c in "CDEF":
        itm.put(f"{c}14", f"={c}11+{c}12-{c}13", num="mm")
    itm.put("A15", "Tax expense")
    for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
        itm.put(f"{c}15", d.get("tax"), num="mm")
    itm.put("F15", "=E15-C15+D15", num="mm")
    itm.note("G15", src)
    itm.put("A16", "Net income (continuing operations)")
    for c in "CDEF":
        itm.put(f"{c}16", f"={c}14-{c}15", num="mm")
    itm.put("A17", "Net income to NCI")
    for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
        itm.put(f"{c}17", d.get("nci"), num="mm")
    itm.put("F17", "=E17-C17+D17", num="mm")
    itm.note("G17", src)
    itm.put("A18", f"Net income attributable to {s.short}", b=True)
    for c in "CDEF":
        itm.put(f"{c}18", f"={c}16-{c}17", num="mm", b=True)
    itm.put("A19", "Diluted EPS (GAAP)")
    for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
        itm.put(f"{c}19", d.get("eps"), num="eps")
    itm.put("F19", "=E19-C19+D19", num="eps")
    itm.note("G19", f"LTM EPS = FY{s.fy_hist[-1]} – {it.label_prior} + {it.label_cur} (sum of quarters)")
    itm.put("A20", "Adjusted EPS (company)" if s.adj_eps_reported else "Adjusted EPS (model definition)")
    for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
        itm.put(f"{c}20", d.get("adj_eps"), num="eps")
    itm.put("F20", "=E20-C20+D20", num="eps")
    itm.note("G20", s.adj_eps_note)
    itm.put("A21", "Total segment profit (non-GAAP)")
    for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
        itm.put(f"{c}21", d.get("seg_profit"), num="mm")
    itm.put("F21", "=E21-C21+D21", num="mm")
    itm.note("G21", src)
    itm.put("A22", "Segment margin")
    for c in "CDEF":
        itm.put(f"{c}22", f"=IF({c}6=0,0,{c}21/{c}6)" if c in "CD" else f"={c}21/{c}6", num="pct")
    for r, label, k in [(23, "Depreciation + amortization (cash flow statement)", "da")]:
        itm.put(f"A{r}", label)
        for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
            itm.put(f"{c}{r}", d.get(k), num="mm")
        itm.put(f"F{r}", f"=E{r}-C{r}+D{r}", num="mm")
        itm.note(f"G{r}", src)
    itm.put("A24", "EBITDA (operating income + D&A)", b=True)
    for c in "CDEF":
        itm.put(f"{c}24", f"={c}11+{c}23", num="mm", b=True)
    for r, label, k in [(25, "Cash from operations", "cfo"), (26, "Capital expenditures", "capex")]:
        itm.put(f"A{r}", label)
        for c, d in (("C", it.ytd_prior), ("D", it.ytd_cur), ("E", s.fy_interim)):
            itm.put(f"{c}{r}", d.get(k), num="mm")
        itm.put(f"F{r}", f"=E{r}-C{r}+D{r}", num="mm")
        itm.note(f"G{r}", src)
    itm.put("A27", "Free cash flow", b=True)
    for c in "CDEF":
        itm.put(f"{c}27", f"={c}25-{c}26", num="mm", b=True)

    itm.band(30, "Balance sheet and net debt at the latest report date")
    itm.header(31, "($ in millions)", {"C": f"{it.period_end:%b %-d, %Y}", "D": f"{s.period_ends[-1]:%b %-d, %Y}"},
               right=False)
    itm.put("G31", "Source", b=True, fill="hdr")
    bsrc = s.interim_bs_source
    for r, label, k in [(32, "Cash & cash equivalents", "cash"), (33, "Short-term investments", "sti"),
                        (34, "Commercial paper & short-term borrowings", "st_debt"),
                        (35, "Current maturities of long-term debt", "cur_ltd"), (36, "Long-term debt", "ltd")]:
        itm.put(f"A{r}", label)
        itm.put(f"C{r}", it.bs_cur.get(k), num="mm")
        itm.put(f"D{r}", it.bs_fy.get(k), num="mm")
        itm.note(f"G{r}", bsrc)
    itm.put("A37", "Total debt", b=True)
    itm.put("A38", "Net debt (debt – cash – ST investments)", b=True)
    for c in "CD":
        itm.put(f"{c}37", f"={c}34+{c}35+{c}36", num="mm", b=True, fill="key")
        itm.put(f"{c}38", f"={c}37-{c}32-{c}33", num="mm", b=True, fill="key")
    dl = s.debt_like
    itm.put("A39", s.label("debt_like", "Other debt-like liabilities (e.g. asbestos, environmental)"))
    itm.put("C39", dl["liab"], num="mm")
    itm.put("D39", dl["liab_fy"], num="mm")
    itm.note("G39", dl["note"])
    off_label = s.label("debt_like_offset", "Offsetting insurance recoveries / assets")
    if s.captive:
        itm.put("A40", "Captive-finance debt (funds financing receivables), excluded"
                + (f" + {off_label.lower()}" if dl["offset"] or dl["offset_fy"] else ""))
        itm.put("C40", f"={dl['offset']:g}+{aa('captive_share')}*C37", num="mm")
        itm.put("D40", f"={dl['offset_fy']:g}+{aa('captive_share')}*D37", num="mm")
        itm.note("G40", f"{s.captive['segment']}: captive share of debt (Assumptions) x total debt; its interest is "
                        "inside segment profit")
    else:
        itm.put("A40", off_label)
        itm.put("C40", dl["offset"], num="mm")
        itm.put("D40", dl["offset_fy"], num="mm")
        itm.note("G40", dl["note"])
    itm.put("A41", ("Debt-like items less captive-finance debt (negative = added back to equity value)"
                    if s.captive else s.label("debt_like_net", "Other debt-like items, net of offsets") + " (debt-like)"),
            b=True)
    for c in "CD":
        itm.put(f"{c}41", f"={c}39-{c}40", num="mm", b=True, fill="key")
    itm.put("A42", "Total shareowners' equity (incl. NCI)")
    itm.put("C42", it.bs_cur.get("equity_total"), num="mm")
    itm.put("D42", it.bs_fy.get("equity_total"), num="mm")
    itm.note("G42", bsrc)
    itm.put("A43", f"Noncontrolling interests ({s.period_ends[-1]:%b %-d, %Y})")
    itm.put("C43", f"={aa('nci_value')}", num="mm")

    itm.band(46, s.q_seg_title)
    itm.header(47, "Segment", dict(zip("BCDEFG", s.q_seg_headers)), right=False)
    qs = s.q_segments
    for i, row in enumerate(qs):
        r = INT_SEG0 + i
        itm.put(f"A{r}", row["name"])
        itm.put(f"B{r}", row.get("sales_cur"), num="mm")
        itm.put(f"C{r}", row.get("sales_prior"), num="mm")
        itm.put(f"D{r}", row.get("organic"), num="pct")
        itm.put(f"E{r}", row.get("profit_cur"), num="mm")
        itm.put(f"F{r}", f"=IF(B{r}=0,0,E{r}/B{r})", num="pct")
        itm.put(f"G{r}", row.get("margin_prior"), num="pct")
    for i in range(len(qs), n):  # segment present in the 10-K but not the 10-Q
        itm.put(f"A{INT_SEG0 + i}", s.segments[i].name)
    itm.put(f"A{INT_CORP}", "Corporate & All Other")
    itm.put(f"B{INT_CORP}", s.q_corp.get("sales_cur"), num="mm")
    itm.put(f"C{INT_CORP}", s.q_corp.get("sales_prior"), num="mm")
    itm.put(f"E{INT_CORP}", s.q_corp.get("profit_cur"), num="mm")
    itm.put(f"A{INT_TOT}", "Total", b=True)
    for c in "BCE":
        itm.put(f"{c}{INT_TOT}", f"=SUM({c}{INT_SEG0}:{c}{INT_CORP})", num="mm", b=True)
    itm.put(f"D{INT_TOT}", s.q_total.get("organic"), num="pct", b=True)
    itm.put(f"F{INT_TOT}", f"=IF(B{INT_TOT}=0,0,E{INT_TOT}/B{INT_TOT})", num="pct", b=True)
    itm.put(f"G{INT_TOT}", s.q_total.get("margin_prior"), num="pct", b=True)

    itm.band(INT_G, s.guidance_title)
    itm.header(INT_G + 1, "Metric", {"C": "Guide low", "D": "Guide high", "E": f"Model {fc_lbl[0]}",
                                      "F": "Within range?"}, right=False)
    itm.put(f"G{INT_G + 1}", "Base case only; tolerance ±0.5%", fill="hdr", color="grey")
    gd = s.guidance
    seg_m, gaap_m = f"Segments!F{SG['tp_m']}", f"IS!F{IS['op_margin']}"
    adj_m = (f"(IS!F{IS['ebit']}+IS!F{IS['amort']}*{AMSH}+IS!F{IS['sbc']}*{EXSBC}+IS!F{IS['repo']}*{EXREPO})"
             f"/IS!F{IS['sales_b']}")
    basis = str(s.margin_basis or "").lower()
    if basis.startswith("gaap"):
        margin_link = f"={gaap_m}"
    elif basis.startswith(("adj", "non")):
        margin_link = f"={adj_m}"
    elif basis.startswith("seg"):
        margin_link = f"={seg_m}"
    else:  # the guide's basis is the one closest to it: a GAAP guide vs a non-GAAP model was a false alarm
        mid_ = f"AVERAGE(C{INT_G + 4}:D{INT_G + 4})"
        margin_link = (f"=IF(COUNT(C{INT_G + 4}:D{INT_G + 4})<2,{seg_m},IF(AND(ABS({gaap_m}-{mid_})<=ABS({adj_m}-"
                       f"{mid_}),ABS({gaap_m}-{mid_})<=ABS({seg_m}-{mid_})),{gaap_m},IF(ABS({adj_m}-{mid_})<="
                       f"ABS({seg_m}-{mid_}),{adj_m},{seg_m})))")
    for k, (label, num, link) in enumerate([
            ("Net sales ($mm)", "mm", f"=IS!F{IS['sales_b']}"),
            ("Organic growth", "pct", f"=Segments!F{SG['t_org']}"),
            (f"Margin ({s.margin_basis or 'basis matched: segment, GAAP or adjusted operating'})", "pct",
             margin_link),
            ("Adjusted EPS ($)", "eps", f"=IS!F{IS['adj_eps']}"),
            ("Free cash flow ($mm)", "mm", "=CF!F39")]):
        r = INT_G + 2 + k
        lo, hi = gd.get(["sales", "organic", "margin", "adj_eps", "fcf"][k]) or (None, None)
        itm.put(f"A{r}", label)
        itm.put(f"C{r}", lo, num=num)
        itm.put(f"D{r}", hi, num=num)
        itm.put(f"E{r}", link, num=num, b=True)
        itm.put(f"F{r}", f'=IF(COUNT(C{r}:D{r})<2,"n/a",IF(AND(E{r}>=C{r}*0.995,E{r}<=D{r}*1.005),"Yes","Check"))')
    itm.note(f"A{INT_G + 8}", s.guidance_note)
    if it.available:  # what the full-year forecast leaves for the rest of the year, vs the year-to-date trend
        r0 = INT_ROY
        itm.band(r0, f"Rest of {fc_lbl[0][:-1]} implied by the model vs year to date")
        itm.header(r0 + 1, "Metric", {"B": f"{it.label_cur} actual", "C": f"Model {fc_lbl[0]}",
                                      "D": "Implied rest of year", "E": "Same period last year",
                                      "F": "Implied growth", "G": "YTD growth"}, right=False)
        for k, (label, row, link, num) in enumerate([
                ("Net sales ($mm)", 6, f"=IS!F{IS['sales_b']}", "mm"),
                ("Total segment profit ($mm)", 21, f"=IS!F{IS['seg_profit']}", "mm"),
                ("Operating income (EBIT, $mm)", 11, f"=IS!F{IS['ebit']}", "mm"),
                ("Adjusted EPS ($)", 20, f"=IS!F{IS['adj_eps']}", "eps"),
                ("Free cash flow ($mm)", 27, "=CF!F39", "mm")]):
            r = r0 + 2 + k
            itm.put(f"A{r}", label)
            itm.put(f"B{r}", f"=D{row}", num=num)
            itm.put(f"C{r}", link, num=num, b=True)
            itm.put(f"D{r}", f"=C{r}-B{r}", num=num)
            itm.put(f"E{r}", f"=E{row}-C{row}", num=num)
            itm.put(f"F{r}", f'=IF(N(E{r})<=0,"n/m",D{r}/E{r}-1)', num="pct", b=True)
            itm.put(f"G{r}", f'=IF(N(C{row})<=0,"n/m",D{row}/C{row}-1)', num="pct")
        itm.note(f"A{r0 + 8}", "A big gap between implied and year-to-date growth needs a reason (guidance, seasonality, "
                               "a deal closing, a one-off) — or the full-year assumptions are off.")
    itm.finish()

    # ================================================================ Segments
    seg.title("Segment build: sales and segment profit",
              "Forecast sales = prior year × (1 + organic growth + M&A/FX). Segment profit = sales × margin.")
    seg.header(4, units, yr_labels, "L", "Source / note")
    seg.band(5, "Net sales by segment")
    hs = s.hist_cols_available
    for i, sg in enumerate(s.segments):
        r, rg, ro, rm = SG[f"s{i}"], SG[f"s{i}_g"], SG[f"s{i}_org"], SG[f"s{i}_mna"]
        lines_on = sg.use_lines and sg.lines
        seg.put(f"A{r}", sg.name, b=True)
        for j, c in enumerate(H):
            seg.put(f"{c}{r}", sg.sales[j], num="mm", b=True)
        for c in P:
            if lines_on:
                seg.put(f"{c}{r}", "=" + "+".join(f"{c}{SG[f'l{i}_{k}']}" for k in range(len(sg.lines))),
                        num="mm", b=True)
            else:
                seg.put(f"{c}{r}", f"={prev(c)}{r}*(1+{c}{ro}+{c}{rm})", num="mm", b=True)
        seg.note(f"L{r}", s.seg_source + (f"; forecast = sum of the {len(sg.lines)} lines below" if lines_on else ""))
        seg.put(f"A{rg}", "   % growth (reported)")
        for j, c in enumerate(ALL[1:], start=1):
            if j >= 3 or (sg.sales[j] is not None and sg.sales[j - 1]):
                seg.put(f"{c}{rg}", f"={c}{r}/{prev(c)}{r}-1", num="pct")
        seg.put(f"A{ro}", "   Organic growth")
        if sg.organic_last is not None:
            seg.put(f"E{ro}", sg.organic_last, num="pct")
        elif sg.sales[1]:
            seg.put(f"E{ro}", f"=E{rg}", num="pct")
        for c in P:
            if lines_on:
                pc = prev(c)
                terms = "+".join(f"{pc}{SG[f'l{i}_{k}']}*{c}{SG[f'l{i}_{k}_org']}" for k in range(len(sg.lines)))
                seg.put(f"{c}{ro}", f"=({terms})/{pc}{r}", num="pct")
            else:
                seg.put(f"{c}{ro}", f"=Assumptions!{c}{A[f'g_sel_{uidx[(i, None)]}']}", num="pct")
        seg.note(f"L{ro}", s.organic_note if not lines_on else
                 "Forecast = sales-weighted organic growth of the lines (each set on Assumptions)")
        seg.put(f"A{rm}", "   Acquisitions, divestitures & FX")
        if sg.sales[1]:
            seg.put(f"E{rm}", f"=E{rg}-E{ro}", num="pct")
        for c in P:
            seg.put(f"{c}{rm}", f"=Assumptions!{c}{A[f'mna_{i}']}", num="pct")
        seg.note(f"L{rm}", f"{s.fy_hist[-1]} = reported growth – organic growth")
        if lines_on:
            for k, ln in enumerate(sg.lines):
                rl, rlo = SG[f"l{i}_{k}"], SG[f"l{i}_{k}_org"]
                seg.put(f"A{rl}", f"   {ln.name}")
                for j, c in enumerate(H):
                    seg.put(f"{c}{rl}", ln.sales[j], num="mm")
                for c in P:
                    seg.put(f"{c}{rl}", f"={prev(c)}{rl}*(1+{c}{rlo}+{c}{rm})", num="mm")
                if k == 0:
                    seg.note(f"L{rl}", f"10-K disaggregated revenue ({s.hist.lines_axis or 'line'}); forecast = "
                                       "prior year x (1 + organic + segment M&A/FX)")
                seg.put(f"A{rlo}", "      growth (organic)")
                for j, c in enumerate(H[1:], start=1):
                    if ln.sales[j] is not None and ln.sales[j - 1]:
                        seg.put(f"{c}{rlo}", f"={c}{rl}/{prev(c)}{rl}-1", num="pct")
                for c in P:
                    seg.put(f"{c}{rlo}", f"=Assumptions!{c}{A[f'g_sel_{uidx[(i, k)]}']}", num="pct")
    rc, rt = SG["corp"], SG["total"]
    seg.put(f"A{rc}", "Corporate & All Other")
    for j, c in enumerate(H):
        seg.put(f"{c}{rc}", s.corp_sales_hist[j], num="mm")
    for c in P:
        seg.put(f"{c}{rc}", f"=Assumptions!{c}{A['corp_sales']}", num="mm")
    seg.note(f"L{rc}", "Total net sales less reported segments (corporate, other, eliminations)")
    seg.put(f"A{rt}", "Total net sales", b=True)
    seg_rows = [SG[f"s{i}"] for i in range(n)] + [rc]
    for c in ALL:
        if c in H and s.hist.IS["sales"][H.index(c)] is None:
            continue
        seg.put(f"{c}{rt}", "=" + "+".join(f"{c}{x}" for x in seg_rows), num="mm", b=True, fill="key", top=True)
    seg.note(f"L{rt}", "Ties to 10-K net sales: " + " / ".join(f"{v:,.0f}" for v in s.hist.IS["sales"] if v))
    seg.put(f"A{SG['t_g']}", "   % growth (reported)")
    for j, c in enumerate(ALL[1:], start=1):
        if j >= 3 or (s.hist.IS["sales"][j] and s.hist.IS["sales"][j - 1]):
            seg.put(f"{c}{SG['t_g']}", f"={c}{rt}/{prev(c)}{rt}-1", num="pct")
    seg.put(f"A{SG['t_org']}", "   Organic growth (sales-weighted)")
    if s.organic_total_last is not None:
        seg.put(f"E{SG['t_org']}", s.organic_total_last, num="pct")
    else:
        seg.put(f"E{SG['t_org']}", f"=E{SG['t_g']}", num="pct")
    for c in P:
        pc = prev(c)
        terms = "+".join(f"{pc}{SG[f's{i}']}*{c}{SG[f's{i}_org']}" for i in range(n))
        seg.put(f"{c}{SG['t_org']}", f"=({terms})/{pc}{rt}", num="pct")
    seg.note(f"L{SG['t_org']}", s.organic_total_note)

    seg.band(SG["p_band"], "Segment profit (non-GAAP; excludes amortization of acquired intangibles, stock comp, "
                           "repositioning)" if s.segments_reported else
             "Segment profit (company total: operating income before other operating expense, + amortization of "
             "acquired intangibles + stock comp)")
    for i, sg in enumerate(s.segments):
        r, rmg = SG[f"p{i}"], SG[f"p{i}_m"]
        seg.put(f"A{r}", sg.name, b=True)
        for j, c in enumerate(H):
            seg.put(f"{c}{r}", sg.profit[j], num="mm", b=True)
        for c in P:
            seg.put(f"{c}{r}", f"={c}{SG[f's{i}']}*{c}{rmg}", num="mm", b=True)
        seg.note(f"L{r}", s.seg_source)
        seg.put(f"A{rmg}", "   Segment margin")
        for j, c in enumerate(H):
            if sg.profit[j] is not None and sg.sales[j]:
                seg.put(f"{c}{rmg}", f"={c}{r}/{c}{SG[f's{i}']}", num="pct")
        for c in P:
            seg.put(f"{c}{rmg}", f"=Assumptions!{c}{A[f'm_sel_{i}']}", num="pct")
    rcp, rpct, rtp, rtm = SG["corp_p"], SG["corp_pct"], SG["tp"], SG["tp_m"]
    seg.put(f"A{rcp}", "Corporate & All Other")
    for j, c in enumerate(H):
        seg.put(f"{c}{rcp}", s.corp_profit_hist[j], num="mm")
    for c in P:
        seg.put(f"{c}{rcp}", f"={c}{rt}*Assumptions!{c}{A['corp_cost']}", num="mm")
    seg.note(f"L{rcp}", s.corp_profit_note)
    seg.put(f"A{rpct}", "   % of total sales")
    for j, c in enumerate(ALL):
        if j >= 3 or s.corp_profit_hist[j] is not None:
            seg.put(f"{c}{rpct}", f"={c}{rcp}/{c}{rt}", num="pct")
    seg.put(f"A{rtp}", "Total segment profit", b=True)
    prow = [SG[f"p{i}"] for i in range(n)] + [rcp]
    hist_tp = [j for j in range(3) if all(sg.profit[j] is not None for sg in s.segments)]
    for j, c in enumerate(ALL):
        if j >= 3 or j in hist_tp:
            seg.put(f"{c}{rtp}", "=" + "+".join(f"{c}{x}" for x in prow), num="mm", b=True, fill="key", top=True)
            seg.put(f"{c}{rtm}", f"={c}{rtp}/{c}{rt}", num="pct")
    seg.note(f"L{rtp}", s.seg_total_note)
    seg.put(f"A{rtm}", "   Segment margin")
    seg.band(SG["memo_band"], "Memo: share of total sales (used to allocate D&A in the SOTP)")
    for i, sg in enumerate(s.segments):
        r = SG[f"memo{i}"]
        seg.put(f"A{r}", f"   {sg.name}")
        for j, c in enumerate(ALL):
            if j >= 3 or sg.sales[j] is not None:
                seg.put(f"{c}{r}", f"={c}{SG[f's{i}']}/{c}{rt}", num="pct")
    seg.ws.freeze_panes(4, 2)
    seg.finish()

    # ================================================================ KPIs
    if kpi is not None:
        kpi.title("Operating KPIs: volume × price revenue build",
                  "Revenue growth on Assumptions = (1 + volume growth) × (1 + price growth) – 1 for these lines. "
                  "History as disclosed by the company (releases / 10-K).")
        kpi.header(4, units, yr_labels, "L", "Source / note")
        for b_, (i, u) in enumerate(s.kpi_units):
            k, kd = KP[b_], u.kpi
            kpi.band(k["band"], f"{unit_label(i, u)}: {kd['volume']['name']} × {kd['price']['name']}")
            for part, lvl in (("volume", "vol"), ("price", "px")):
                d = kd[part]
                r = k[lvl]
                kpi.put(f"A{r}", d["name"] + (f" ({d['unit']})" if d["unit"] else ""), b=True)
                for j, c in enumerate(H):
                    kpi.put(f"{c}{r}", d["history"][j], num="sh", b=True)
                for c in P:
                    kpi.put(f"{c}{r}", f"={prev(c)}{r}*(1+{c}{k[lvl + '_sel']})", num="sh", b=True)
                for grp, glabel in (("bear", "Bear"), ("base", "Base"), ("bull", "Bull")):
                    rg = k[f"{lvl}_{grp}"]
                    kpi.put(f"A{rg}", f"   {glabel} case growth")
                    for j, c in enumerate(P):
                        kpi.put(f"{c}{rg}", d["growth"][grp][j], num="pct", fill="input")
                rs = k[lvl + "_sel"]
                kpi.put(f"A{rs}", "   Selected case growth")
                for c in P:
                    kpi.put(f"{c}{rs}", f"=CHOOSE(Assumptions!$C$4,{c}{k[lvl + '_bear']},{c}{k[lvl + '_base']},"
                                        f"{c}{k[lvl + '_bull']})", num="pct", fill="key")
            seg_row = SG[f"s{i}"] if u is s.segments[i] else SG[f"l{i}_{s.segments[i].lines.index(u)}"]
            kpi.put(f"A{k['implied']}", "Implied revenue = volume × price × scale ($mm)")
            kpi.put(f"A{k['model']}", "Revenue in the model (Segments)")
            kpi.put(f"A{k['diff']}", "   Difference (implied / model – 1)")
            for j, c in enumerate(ALL):
                if j < 3 and kd["volume"]["history"][j] is None:
                    continue
                kpi.put(f"{c}{k['implied']}", f"={c}{k['vol']}*{c}{k['px']}*{kd.get('scale', 1.0):.8g}", num="mm")
                kpi.put(f"{c}{k['model']}", f"=Segments!{c}{seg_row}", num="mm")
                kpi.put(f"{c}{k['diff']}", f"=IF({c}{k['model']}=0,0,{c}{k['implied']}/{c}{k['model']}-1)", num="pct")
            kpi.note(f"L{k['band']}", kd.get("note") or None)
            kpi.note(f"L{k['implied']}", "Scale calibrates units so the latest year ties to reported revenue")
            sg_ = s.segments[i] if isinstance(i, int) and 0 <= i < len(s.segments) else None
            if sg_ is not None and any(abs(m_) > 0.0005 for m_ in sg_.mna):
                kpi.note(f"L{k['diff']}", "The volume × price build is organic; the model's revenue also carries "
                                          "acquired / FX revenue (Assumptions M&A: " +
                                          ", ".join(f"{fc_lbl[t]} {m_:+.1%}" for t, m_ in enumerate(sg_.mna)
                                                    if abs(m_) > 0.0005) + "), which is the difference here")
        kpi.ws.freeze_panes(4, 2)
        kpi.finish()

    # ================================================================ IS
    I = s.hist.IS
    inc.title("Income statement", "History as reported in the 10-K; forecast built from segment profit down to EPS.")
    inc.header(4, units, yr_labels, "L", "Source / note")
    inc.band(IS["band1"], "Reported GAAP lines (history, 10-K Consolidated Statement of Operations)")
    stmt_src = s.is_source
    for k, label in [("sales", "Net sales"), ("cogs", "Cost of products & services sold"),
                     ("rnd", "Research & development"), ("sga", "Selling, general & administrative"),
                     ("oth_opex", "Other operating expense (income), net")]:
        r = IS[k]
        inc.put(f"A{r}", label)
        for j, c in enumerate(H):
            inc.put(f"{c}{r}", s.is_hist[k][j], num="mm")
        inc.note(f"L{r}", stmt_src if k != "oth_opex" else
                 "Impairments, gains & other operating items: reported operating lines less those shown")
    r = IS["opinc_rep"]
    inc.put(f"A{r}", "Operating income (reported)", b=True)
    for c in s.hcols:
        inc.put(f"{c}{r}", f"={c}{IS['sales']}-{c}{IS['cogs']}-{c}{IS['rnd']}-{c}{IS['sga']}-{c}{IS['oth_opex']}",
                num="mm", b=True, top=True)
    inc.note(f"L{r}", "Sales – COGS – R&D – SG&A – other operating items")
    inc.band(IS["band2"], "Segment-based build (history reconciles to GAAP; forecast drives the three statements)")
    R = IS
    inc.put(f"A{R['sales_b']}", "Net sales", b=True)
    for c in s.hcols:
        inc.put(f"{c}{R['sales_b']}", f"={c}{R['sales']}", num="mm", b=True)
    for c in P:
        inc.put(f"{c}{R['sales_b']}", f"=Segments!{c}{SG['total']}", num="mm", b=True)
    inc.put(f"A{R['growth']}", "   % growth")
    for c in ALL[1:]:
        if c in P or (c in s.hcols and prev(c) in s.hcols):
            inc.put(f"{c}{R['growth']}", f"={c}{R['sales_b']}/{prev(c)}{R['sales_b']}-1", num="pct")
    inc.put(f"A{R['seg_profit']}", "Total segment profit", b=True)
    for j, c in enumerate(ALL):
        if j >= 3 or j in hist_tp:
            inc.put(f"{c}{R['seg_profit']}", f"=Segments!{c}{SG['tp']}", num="mm", b=True)
    inc.put(f"A{R['seg_margin']}", "   Segment margin")
    for j, c in enumerate(ALL):
        if j >= 3 or j in hist_tp:
            inc.put(f"{c}{R['seg_margin']}", f"={c}{R['seg_profit']}/{c}{R['sales_b']}", num="pct")
    inc.put(f"A{R['amort']}", "   Less: amortization of acquired intangibles")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['amort']}", I["amort"][j], num="mm")
    for c in P:
        inc.put(f"{c}{R['amort']}", f"={a('amort', c)}", num="mm")
    inc.note(f"L{R['amort']}", s.amort_note)
    inc.put(f"A{R['sbc']}", "   Less: stock compensation")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['sbc']}", I["sbc"][j], num="mm")
    for c in P:
        inc.put(f"{c}{R['sbc']}", f"={c}{R['sales_b']}*{a('sbc_pct', c)}", num="mm")
    inc.note(f"L{R['sbc']}", "10-K cash flow statement")
    inc.put(f"A{R['repo']}", s.label("repo", s.repo_label))
    for j, c in enumerate(H):
        if j in hist_tp:
            inc.put(f"{c}{R['repo']}", f"={c}{R['seg_profit']}-{c}{R['opinc_rep']}-({c}{R['amort']}+{c}{R['sbc']})"
                                       f"*(1-{SW})", num="mm")
    for c in P:
        inc.put(f"{c}{R['repo']}", f"={c}{R['sales_b']}*{a('repo_pct', c)}", num="mm")
    inc.note(f"L{R['repo']}", "History = plug to reported operating income (restructuring, impairments, other)"
             + ("; amortization & stock comp are inside segment profit (Assumptions!C14 = 1)" if s.seg_after_charges
                else ""))
    inc.put(f"A{R['ebit']}", "Operating income (EBIT)", b=True)
    for c in s.hcols:
        inc.put(f"{c}{R['ebit']}", f"={c}{R['opinc_rep']}", num="mm", b=True, fill="key", top=True)
    for c in P:
        inc.put(f"{c}{R['ebit']}", f"={c}{R['seg_profit']}-({c}{R['amort']}+{c}{R['sbc']})*(1-{SW})-{c}{R['repo']}",
                num="mm", b=True, fill="key", top=True)
    inc.put(f"A{R['op_margin']}", "   Operating margin")
    for c in s.hcols + P:
        inc.put(f"{c}{R['op_margin']}", f"={c}{R['ebit']}/{c}{R['sales_b']}", num="pct")
    inc.put(f"A{R['int_inc']}", "   Interest income")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['int_inc']}", I["interest_inc"][j], num="mm")
    for c in P:
        pc = prev(c)
        if s.framework != "reference":  # cash net of half the year's distributions (COST's special dividend)
            inc.put(f"{c}{R['int_inc']}", f"=MAX(0,BS!{pc}6+BS!{pc}7+0.5*(CF!{c}28+CF!{c}29+CF!{c}20+CF!{c}26+CF!{c}27"
                                          f"+CF!{pc}39))"
                                          f"*{a('int_yield', c)}", num="mm")
        else:
            inc.put(f"{c}{R['int_inc']}", f"=(BS!{pc}6+BS!{pc}7)*{a('int_yield', c)}", num="mm")
    inc.note(f"L{R['int_inc']}", "10-K interest income; forecast on beginning cash" +
             (" plus half of last year's free cash flow (the year's own cash generation, without a circular "
              "reference), less half the year's buybacks, dividends and acquisitions net of debt and stock raised "
              "to fund them"
              if s.framework != "reference" else " (avoids a circular reference)"))
    NR = s.has_nonrec
    nr = (lambda c: f"-{c}{R['nonrec']}") if NR else (lambda c: "")
    inc.put(f"A{R['pension']}", "   Pension & other non-operating income" + (" (recurring)" if NR else ""))
    for c in s.hcols:
        inc.put(f"{c}{R['pension']}", f"={c}{R['other_inc']}-{c}{R['int_inc']}{nr(c)}", num="mm")
    for c in P:
        inc.put(f"{c}{R['pension']}", f"={a('other_nonop', c)}", num="mm")
    if NR:
        inc.put(f"A{R['nonrec']}", "   Non-recurring gains (losses), excluded from adjusted EPS")
        for j, c in enumerate(H):
            if c in s.hcols:
                inc.put(f"{c}{R['nonrec']}", s.nonrec_hist[j], num="mm")
        for c in P:
            inc.put(f"{c}{R['nonrec']}", f"={a('nonrec', c)}", num="mm")
        inc.note(f"L{R['nonrec']}", "10-K / 10-Q XBRL: gains (losses) on equity & debt securities, debt "
                                    "extinguishment; forecast on Assumptions (non-cash)")
    inc.put(f"A{R['other_inc']}", "Other income, net")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['other_inc']}", s.is_hist["other_inc"][j], num="mm")
    for c in P:
        inc.put(f"{c}{R['other_inc']}", f"={c}{R['int_inc']}+{c}{R['pension']}"
                                         + (f"+{c}{R['nonrec']}" if NR else ""), num="mm")
    inc.note(f"L{R['other_inc']}", "Pre-tax income – EBIT + interest expense (all non-operating items)")
    inc.put(f"A{R['interest']}", "Less: interest & other financial charges")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['interest']}", I["interest_exp"][j], num="mm")
    for c in P:
        inc.put(f"{c}{R['interest']}", f"=Schedules!{c}30", num="mm")
    inc.note(f"L{R['interest']}", s.interest_note)
    inc.put(f"A{R['pretax']}", "Income before taxes", b=True)
    for c in s.hcols + P:
        inc.put(f"{c}{R['pretax']}", f"={c}{R['ebit']}+{c}{R['other_inc']}-{c}{R['interest']}", num="mm", b=True,
                top=True)
    inc.note(f"L{R['pretax']}", "Ties to 10-K: " + " / ".join(f"{v:,.0f}" for v in I["pretax"] if v is not None))
    inc.put(f"A{R['tax']}", "Less: tax expense")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['tax']}", I["tax"][j], num="mm")
    for c in P:
        inc.put(f"{c}{R['tax']}", f"={c}{R['pretax']}*{a('etr', c)}", num="mm")
    inc.note(f"L{R['tax']}", stmt_src)
    inc.put(f"A{R['etr']}", "   Effective tax rate")
    for c in s.hcols + P:
        inc.put(f"{c}{R['etr']}", f"={c}{R['tax']}/{c}{R['pretax']}", num="pct")
    ni_formula = f"{{c}}{R['pretax']}-{{c}}{R['tax']}"
    if s.has_disc:
        inc.put(f"A{R['disc']}", "Discontinued operations & other after-tax items")
        for j, c in enumerate(H):
            inc.put(f"{c}{R['disc']}", I["disc"][j], num="mm")
        for c in P:
            inc.put(f"{c}{R['disc']}", "=0", num="mm")
        inc.note(f"L{R['disc']}", "Reported net income less (pre-tax income – tax); not forecast")
        ni_formula += f"+{{c}}{R['disc']}"
    inc.put(f"A{R['ni']}", "Net income", b=True)
    for c in s.hcols + P:
        inc.put(f"{c}{R['ni']}", "=" + ni_formula.format(c=c), num="mm", b=True)
    inc.put(f"A{R['nci']}", "Less: net income to noncontrolling interest")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['nci']}", I["nci"][j], num="mm")
    for c in P:
        inc.put(f"{c}{R['nci']}", f"={a('nci_ni', c)}", num="mm")
    inc.note(f"L{R['nci']}", stmt_src)
    inc.put(f"A{R['ni_attr']}", f"Net income attributable to {s.short}", b=True)
    for c in s.hcols + P:
        inc.put(f"{c}{R['ni_attr']}", f"={c}{R['ni']}-{c}{R['nci']}", num="mm", b=True, fill="key", top=True)
    inc.note(f"L{R['ni_attr']}", "Ties to 10-K: " + " / ".join(f"{v:,.0f}" for v in I["ni_attr"] if v is not None))
    inc.put(f"A{R['shares']}", "Diluted weighted average shares (mm)")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['shares']}", I["dil_shares"][j], num="sh")
    for c in P:
        inc.put(f"{c}{R['shares']}", f"=Schedules!{c}55", num="sh")
    inc.note(f"L{R['shares']}", s.shares_note)
    inc.put(f"A{R['eps']}", "Diluted EPS (GAAP)", b=True)
    for c in s.hcols + P:
        inc.put(f"{c}{R['eps']}", f"={c}{R['ni_attr']}/{c}{R['shares']}", num="eps", b=True)
    amort_txt = ("acquired-intangible amortization" if s.adj_eps_amort_share >= 0.999 else
                 f"{s.adj_eps_amort_share:.0%} of acquired-intangible amortization")
    repo_txt = ("non-segment charges" if s.adj_eps_ex_repo >= 0.999 else
                f"{s.adj_eps_ex_repo:.0%} of non-segment charges")
    excl = [amort_txt] + (["stock comp"] if s.adj_eps_ex_sbc else []) + \
        ([repo_txt] if s.adj_eps_ex_repo else []) + (["one-offs"] if s.has_nonrec and s.adj_eps_ex_nonrec else [])
    inc.put(f"A{R['adj_eps']}", "Adjusted EPS, company definition (excl. " + ", ".join(excl[:-1]) +
            (" & " if len(excl) > 1 else "") + excl[-1] + ")", b=True)

    def adj_f(c, etr, exnr=EXNR, ngt=None):  # the adjusted-EPS definition contract (Assumptions C5, C15-C17)
        one_off = f"-{c}{R['nonrec']}*{exnr}" if NR else ""
        items = f"({c}{R['amort']}*{AMSH}+{c}{R['sbc']}*{EXSBC}+{c}{R['repo']}*{EXREPO}{one_off})"
        base = f"({c}{R['ni_attr']}+{items}*(1-{etr}))/{c}{R['shares']}"
        if ngt:  # the company taxes non-GAAP pre-tax income at one rate (CRM): (pre-tax + adjustments) x (1 - rate)
            return (f"=IF({ngt}>0,(({c}{R['pretax']}+{items})*(1-{ngt})-{c}{R['nci']})/{c}{R['shares']},"
                    f"{base})")
        return "=" + base

    NGT = "adj_tax" in A
    ngt_of = lambda c: (a("adj_tax", c) if c in P else  # noqa: E731
                        (f"Assumptions!{c}{A['adj_tax']}" if s.adj_eps_tax_hist[H.index(c)] else None)) if NGT else None
    for j, c in enumerate(H):
        if c not in s.hcols:
            continue
        v = s.adj_eps_hist[j]
        inc.put(f"{c}{R['adj_eps']}", v if v is not None else adj_f(c, f"{c}{R['etr']}", ngt=ngt_of(c)), num="eps",
                b=True, fill="key")
        if v is not None:  # gate: the definition must reproduce what the company reported
            model_def = adj_f(c, f"{c}{R['etr']}", ngt=ngt_of(c))[1:]
            inc.put(f"{c}{R['adj_tie']}", f"=({model_def})-{c}{R['adj_eps']}", num="eps", color="grey")
    for c in P:
        inc.put(f"{c}{R['adj_eps']}", adj_f(c, a("etr", c), ngt=ngt_of(c)), num="eps", b=True, fill="key")
    inc.note(f"L{R['adj_eps']}", s.adj_eps_hist_note)
    inc.put(f"A{R['adj_tie']}", "   Tie-out: the model's definition less reported (gate: within $0.05)", color="grey")
    inc.put(f"A{R['op_eps']}", "Operating EPS (excl. one-offs): the EPS every multiple uses", b=True)
    for c in s.hcols + P:
        etr = f"{c}{R['etr']}" if c in H else a("etr", c)
        if c in H and s.adj_eps_hist[H.index(c)] is not None and not (NR and s.adj_eps_ex_nonrec < 1):
            inc.put(f"{c}{R['op_eps']}", f"={c}{R['adj_eps']}", num="eps", b=True)
        else:
            inc.put(f"{c}{R['op_eps']}", adj_f(c, etr, exnr="1", ngt=ngt_of(c)), num="eps", b=True)
    inc.note(f"L{R['op_eps']}", "Same as adjusted EPS when the company excludes one-offs; otherwise adjusted EPS "
                                "less one-off gains (CRM's investment gains are not run-rate earnings)")
    inc.put(f"A{R['adj_growth']}", "   % growth")
    for c in ALL[1:]:
        if c in P or (c in s.hcols and prev(c) in s.hcols):
            inc.put(f"{c}{R['adj_growth']}", f"={c}{R['adj_eps']}/{prev(c)}{R['adj_eps']}-1", num="pct")
    inc.band(R["band3"], "Supplementary")
    inc.put(f"A{R['da']}", "Depreciation & amortization (total)")
    for j, c in enumerate(H):
        inc.put(f"{c}{R['da']}", s.cf_hist["da"][j], num="mm")
    for c in P:
        inc.put(f"{c}{R['da']}", f"=Schedules!{c}8+Schedules!{c}18", num="mm")
    inc.note(f"L{R['da']}", "10-K cash flow statement (depreciation + amortization)")
    inc.put(f"A{R['ebitda']}", "EBITDA (EBIT + D&A)", b=True)
    for c in s.hcols + P:
        inc.put(f"{c}{R['ebitda']}", f"={c}{R['ebit']}+{c}{R['da']}", num="mm", b=True)
    inc.put(f"A{R['ebitda_m']}", "   EBITDA margin")
    for c in s.hcols + P:
        inc.put(f"{c}{R['ebitda_m']}", f"={c}{R['ebitda']}/{c}{R['sales_b']}", num="pct")
    inc.put(f"A{R['seg_ebitda']}", "Segment EBITDA (segment profit + depreciation & other amortization)")
    for j, c in enumerate(ALL):
        if j >= 3 or j in hist_tp:
            inc.put(f"{c}{R['seg_ebitda']}", f"={c}{R['seg_profit']}+{c}{R['da']}-{c}{R['amort']}*(1-{SW})",
                    num="mm")
    inc.note(f"L{R['seg_ebitda']}", 'Basis for the SOTP and the "adjusted EBITDA" peers quote')
    inc.put(f"A{R['memo_cogs']}", "   Memo: cost of sales (for inventory & payables days)")
    for j, c in enumerate(H):
        if c in s.hcols:
            inc.put(f"{c}{R['memo_cogs']}", s.memo_cogs_formula[j].format(c=c, **R), num="mm")
    for c in P:
        inc.put(f"{c}{R['memo_cogs']}", f"={c}{R['sales_b']}*{a('cogs_pct', c)}", num="mm")
    inc.put(f"A{R['memo_cogs_pct']}", "   Memo: cost of sales % of sales")
    for c in s.hcols + P:
        inc.put(f"{c}{R['memo_cogs_pct']}", f"={c}{R['memo_cogs']}/{c}{R['sales_b']}", num="pct")
    inc.ws.freeze_panes(4, 2)
    inc.finish()

    # ================================================================ BS
    B = s.hist.BS
    bal.title("Balance sheet", "History from the 10-K; forecast links to the schedules. "
                               "Cash comes from the cash flow statement.")
    bal.header(4, units, yr_labels, "L", "Source / note")
    bal.band(5, "Assets")
    bsrc_k = s.bs_source
    bs_lines = [
        (6, "Cash & cash equivalents", "cash", lambda c: f"=CF!{c}36", "Forecast = ending cash on the CF"),
        (7, "Short-term investments", "sti", lambda c: f"={prev(c)}7", "Held flat"),
        (8, "Accounts receivable, net", "ar", lambda c: f"=Schedules!{c}33", bsrc_k),
        (9, "Inventories", "inv", lambda c: f"=Schedules!{c}35", bsrc_k),
        (10, "Assets held for sale", "ahfs", lambda c: "=0", "Assumed disposed in the first forecast year"),
        (11, "Other current assets", "oca", lambda c: f"=Schedules!{c}37", bsrc_k + " (plug to total current assets)"),
    ]
    for r, label, k, f, note in bs_lines:
        bal.put(f"A{r}", label)
        for c, v in zip(("D", "E"), B[k][1:]):
            bal.put(f"{c}{r}", v, num="mm")
        for c in P:
            bal.put(f"{c}{r}", f(c), num="mm")
        bal.note(f"L{r}", note)
    bal.put("A12", "Total current assets", b=True)
    for c in ["D", "E"] + P:
        bal.put(f"{c}12", f"=SUM({c}6:{c}11)", num="mm", b=True, top=True)
    bs_lines = [
        (13, "Investments & long-term receivables", "lt_inv", lambda c: f"={prev(c)}13", "Held flat"),
        (14, "Property, plant & equipment, net", "ppe", lambda c: f"=Schedules!{c}9", bsrc_k),
        (15, "Goodwill", "goodwill", lambda c: f"=Schedules!{c}15", bsrc_k),
        (16, "Other intangible assets, net", "intang", lambda c: f"=Schedules!{c}19", bsrc_k),
        (17, "Operating lease right-of-use assets", "rou", lambda c: f"={prev(c)}17", "Held flat"),
        (18, "Deferred income taxes", "dta", lambda c: f"={prev(c)}18", "Held flat"),
        (19, "Other assets (incl. pension asset)", "other_assets",
         lambda c: f"={prev(c)}19+IS!{c}{IS['pension']}" + (f"+IS!{c}{IS['nonrec']}" if s.has_nonrec else "")
         + f"+MAX(0,-({prev(c)}33+{a('other_op', c)}))",
         "History = plug to total assets; grows by non-cash pension & other income (offset on the CF)"),
    ]
    for r, label, k, f, note in bs_lines:
        bal.put(f"A{r}", label)
        for c, v in zip(("D", "E"), B[k][1:]):
            bal.put(f"{c}{r}", v, num="mm")
        for c in P:
            bal.put(f"{c}{r}", f(c), num="mm")
        bal.note(f"L{r}", note)
    bal.put("A20", "Total assets", b=True)
    for c in ["D", "E"] + P:
        bal.put(f"{c}20", f"={c}12+SUM({c}13:{c}19)", num="mm", b=True, fill="key", top=True)
    bal.note("L20", "Ties to 10-K: " + " / ".join(f"{v:,.0f}" for v in B["assets"][1:]))
    bal.band(22, "Liabilities")
    bs_lines = [
        (23, AP_LBL, "ap", lambda c: f"=Schedules!{c}39", bsrc_k),
        (24, "Commercial paper & short-term borrowings", "st_debt", lambda c: f"=Schedules!{c}25", bsrc_k),
        (25, "Current maturities of long-term debt", "cur_ltd", lambda c: f"=Schedules!{c}26", bsrc_k),
        (26, "Accrued & other current liabilities", "accrued", lambda c: f"=Schedules!{c}41",
         bsrc_k + " (plug to total current liabilities)"),
        (27, "Liabilities held for sale", "lhfs", lambda c: "=0", "Transferred with the disposal"),
    ]
    for r, label, k, f, note in bs_lines:
        bal.put(f"A{r}", label)
        for c, v in zip(("D", "E"), B[k][1:]):
            bal.put(f"{c}{r}", v, num="mm")
        for c in P:
            bal.put(f"{c}{r}", f(c), num="mm")
        bal.note(f"L{r}", note)
    bal.put("A28", "Total current liabilities", b=True)
    for c in ["D", "E"] + P:
        bal.put(f"{c}28", f"=SUM({c}23:{c}27)", num="mm", b=True, top=True)
    bs_lines = [
        (29, "Long-term debt", "ltd", lambda c: f"=Schedules!{c}27", bsrc_k),
        (30, "Deferred income taxes", "dtl", lambda c: f"={prev(c)}30", "Held flat"),
        (31, "Postretirement benefit obligations", "postret", lambda c: f"={prev(c)}31", "Held flat"),
        (32, "Operating lease liabilities (noncurrent)", "lease_liab", lambda c: f"={prev(c)}32", "Held flat"),
        (33, "Other liabilities", "other_liab", lambda c: f"=MAX(0,{prev(c)}33+{a('other_op', c)})",
         "History = plug to total liabilities; moves with other operating cash items (an outflow beyond the "
         "balance builds other assets instead)"),
        (34, "Redeemable noncontrolling interest", "redeem_nci", lambda c: f"={prev(c)}34", "Held flat"),
    ]
    for r, label, k, f, note in bs_lines:
        bal.put(f"A{r}", label)
        for c, v in zip(("D", "E"), B[k][1:]):
            bal.put(f"{c}{r}", v, num="mm")
        for c in P:
            bal.put(f"{c}{r}", f(c), num="mm")
        bal.note(f"L{r}", note)
    bal.put("A35", "Total liabilities", b=True)
    for c in ["D", "E"] + P:
        bal.put(f"{c}35", f"={c}28+SUM({c}29:{c}34)", num="mm", b=True, top=True)
    bal.band(37, "Equity")
    bal.put("A38", f"{s.short} shareowners' equity")
    bal.put("A39", "Noncontrolling interest")
    for c, v1, v2 in zip(("D", "E"), B["equity"][1:], B["nci_bs"][1:]):
        bal.put(f"{c}38", v1, num="mm")
        bal.put(f"{c}39", v2, num="mm")
    for c in P:
        bal.put(f"{c}38", f"=Schedules!{c}66", num="mm")
        bal.put(f"{c}39", f"={prev(c)}39", num="mm")
    bal.note("L38", bsrc_k)
    bal.note("L39", "Held flat; NCI earnings distributed (CF)")
    bal.put("A40", "Total equity", b=True)
    bal.put("A41", "Total liabilities & equity", b=True)
    bal.put("A43", "CHECK: assets – liabilities & equity (must be 0)", b=True)
    for c in ["D", "E"] + P:
        bal.put(f"{c}40", f"={c}38+{c}39", num="mm", b=True)
        bal.put(f"{c}41", f"={c}35+{c}40", num="mm", b=True, fill="key", top=True)
        bal.put(f"{c}43", f"=ROUND({c}20-{c}41,3)", num="chk", b=True, fill="key")
    bal.ws.freeze_panes(4, 2)
    bal.finish()

    # ================================================================ CF
    CFH = s.cf_hist
    cfs.title("Cash flow statement", "History condensed from the 10-K; forecast derived from the IS and changes "
                                     "in the BS.")
    cfs.header(4, units, yr_labels, "L", "Source / note")
    cfs.band(5, "Operating activities")
    csrc = s.cf_source
    lines = [
        (6, "Net income (incl. NCI)", "ni", lambda c: f"=IS!{c}{IS['ni']}", csrc),
        (7, "Depreciation & amortization", "da", lambda c: f"=IS!{c}{IS['da']}", csrc),
        (8, "Stock compensation", "sbc", lambda c: f"=IS!{c}{IS['sbc']}", csrc),
        (9, "Non-cash pension & other income" + (" & one-offs" if s.has_nonrec else ""), "pension",
         lambda c: f"=-IS!{c}{IS['pension']}" + (f"-IS!{c}{IS['nonrec']}" if s.has_nonrec else ""),
         "History: non-cash pension line (if reported); forecast reverses the IS non-operating income"),
        (10, "Other operating items, net", "other_op", lambda c: f"={a('other_op', c)}",
         "History derived: reported CFO less the other lines"),
        (11, "(Increase) decrease in receivables", "ar", lambda c: f"=BS!{prev(c)}8-BS!{c}8", csrc),
        (12, "(Increase) decrease in inventories", "inv", lambda c: f"=BS!{prev(c)}9-BS!{c}9", csrc),
        (13, "(Increase) decrease in other current assets", "oca", lambda c: f"=BS!{prev(c)}11-BS!{c}11", csrc),
        (14, "Increase (decrease) in payables", "ap", lambda c: f"=BS!{c}23-BS!{prev(c)}23", csrc),
        (15, "Increase (decrease) in accrued liabilities", "accr", lambda c: f"=BS!{c}26-BS!{prev(c)}26", csrc),
    ]
    for r, label, k, f, note in lines:
        cfs.put(f"A{r}", label)
        for j, c in enumerate(H):
            cfs.put(f"{c}{r}", CFH[k][j], num="mm")
        for c in P:
            cfs.put(f"{c}{r}", f(c), num="mm")
        cfs.note(f"L{r}", note)
    cfs.put("A16", "Cash from operating activities", b=True)
    for c in s.hcols + P:
        cfs.put(f"{c}16", f"=SUM({c}6:{c}15)", num="mm", b=True, fill="key", top=True)
    cfs.note("L16", "Ties to 10-K: " + " / ".join(f"{v:,.0f}" for v in CFH["cfo"] if v is not None))
    cfs.band(18, "Investing activities")
    lines = [
        (19, "Capital expenditures", "capex", lambda c: f"=-Schedules!{c}7", csrc),
        (20, "Acquisitions, net of cash acquired", "acq", lambda c: f"=-Schedules!{c}12", csrc),
        (21, "Divestiture proceeds", "divest",
         lambda c: f"=(BS!{prev(c)}10-BS!{prev(c)}27)-(BS!{c}10-BS!{c}27)",
         "Forecast = net assets held for sale disposed, at book value"),
        (22, "Other investing activities", "other_inv", lambda c: "=0", "History derived as reported CFI less lines above"),
    ]
    for r, label, k, f, note in lines:
        cfs.put(f"A{r}", label)
        for j, c in enumerate(H):
            cfs.put(f"{c}{r}", CFH[k][j], num="mm")
        for c in P:
            cfs.put(f"{c}{r}", f(c), num="mm")
        cfs.note(f"L{r}", note)
    cfs.put("A23", "Cash from investing activities", b=True)
    for c in s.hcols + P:
        cfs.put(f"{c}23", f"=SUM({c}19:{c}22)", num="mm", b=True, fill="key", top=True)
    cfs.note("L23", "Ties to 10-K: " + " / ".join(f"{v:,.0f}" for v in CFH["cfi"] if v is not None))
    cfs.band(25, "Financing activities")
    lines = [
        (26, "Net debt issuance (repayment)", "debt", lambda c: f"=Schedules!{c}23", csrc),
        (27, "Proceeds from stock issuance", "stock", lambda c: f"={a('stock_proc', c)}", csrc),
        (28, "Share repurchases", "buyback", lambda c: f"=-Schedules!{c}48", csrc),
        (29, "Dividends paid", "divs", lambda c: f"=-Schedules!{c}57", csrc),
        (30, "NCI distributions & other financing", "nci_other", lambda c: f"=-IS!{c}{IS['nci']}",
         "History derived as reported CFF less lines above; forecast: NCI earnings paid out"),
    ]
    for r, label, k, f, note in lines:
        cfs.put(f"A{r}", label)
        for j, c in enumerate(H):
            cfs.put(f"{c}{r}", CFH[k][j], num="mm")
        for c in P:
            cfs.put(f"{c}{r}", f(c), num="mm")
        cfs.note(f"L{r}", note)
    cfs.put("A31", "Cash from financing activities", b=True)
    for c in s.hcols + P:
        cfs.put(f"{c}31", f"=SUM({c}26:{c}30)", num="mm", b=True, fill="key", top=True)
    cfs.note("L31", "Ties to 10-K: " + " / ".join(f"{v:,.0f}" for v in CFH["cff"] if v is not None))
    cfs.put("A33", "Effect of exchange rates & other")
    for j, c in enumerate(H):
        cfs.put(f"{c}33", CFH["fx"][j], num="mm")
    for c in P:
        cfs.put(f"{c}33", "=0", num="mm")
    cfs.note("L33", "History: change in cash less CFO, CFI and CFF (FX, discontinued operations)")
    cfs.put("A34", "Net change in cash", b=True)
    for c in s.hcols + P:
        cfs.put(f"{c}34", f"={c}16+{c}23+{c}31+{c}33", num="mm", b=True)
    cfs.put("A35", "Beginning cash")
    for j, c in enumerate(H):
        cfs.put(f"{c}35", CFH["begin"][j], num="mm")
    for c in P:
        cfs.put(f"{c}35", f"={prev(c)}36", num="mm")
    cfs.note("L35", csrc + " (cash incl. restricted cash)")
    cfs.put("A36", "Ending cash", b=True)
    for c in s.hcols + P:
        cfs.put(f"{c}36", f"={c}35+{c}34", num="mm", b=True, fill="key", top=True)
    cfs.note("L36", "Ties to 10-K: " + " / ".join(f"{v:,.0f}" for v in CFH["end"] if v is not None))
    cfs.band(38, "Free cash flow")
    cfs.put("A39", "Free cash flow (CFO – capex)", b=True)
    cfs.put("A40", "   FCF conversion (FCF / NI attributable)")
    cfs.put("A41", "   FCF per diluted share ($)")
    for c in s.hcols + P:
        cfs.put(f"{c}39", f"={c}16+{c}19", num="mm", b=True, fill="key", top=True)
        cfs.put(f"{c}40", f"={c}39/IS!{c}{IS['ni_attr']}", num="pct")
        cfs.put(f"{c}41", f"={c}39/IS!{c}{IS['shares']}", num="eps")
    cfs.note("L39", "CFO less capital expenditures")
    cfs.put("A42", "CHECK: CF ending cash – BS cash (must be 0)", b=True)
    for c in ["D", "E"] + P:
        cfs.put(f"{c}42", f"=ROUND({c}36-BS!{c}6,3)", num="chk", b=True, fill="key")
    cfs.ws.freeze_panes(4, 2)
    cfs.finish()

    # ================================================================ Schedules
    sch.title("Supporting schedules", "PP&E, goodwill & intangibles, debt & interest, working capital, shares & "
                                      "dividends, equity roll-forward.")
    sch.header(4, units, yr_labels, "L", "Source / note")
    sch.band(5, "PP&E")
    sch.put("A6", "Beginning PP&E, net")
    sch.put("A7", "Plus: capital expenditures")
    sch.put("A8", "Less: depreciation & other amortization")
    sch.put("A9", "Ending PP&E, net", b=True)
    for j, c in enumerate(H):
        sch.put(f"{c}7", -CFH["capex"][j] if CFH["capex"][j] is not None else None, num="mm")
        if c in s.hcols:
            sch.put(f"{c}8", f"=IS!{c}{IS['da']}-IS!{c}{IS['amort']}", num="mm")
    for c, v in zip(("D", "E"), B["ppe"][1:]):
        sch.put(f"{c}9", v, num="mm", b=True)
    for c in P:
        sch.put(f"{c}6", f"={prev(c)}9", num="mm")
        sch.put(f"{c}7", f"=IS!{c}{IS['sales_b']}*{a('capex_pct', c)}", num="mm")
        sch.put(f"{c}8", f"=IS!{c}{IS['sales_b']}*{a('da_pct', c)}", num="mm")
        sch.put(f"{c}9", f"={c}6+{c}7-{c}8", num="mm", b=True)
    sch.note("L7", "10-K cash flow statement")
    sch.note("L8", "History = total D&A – acquired-intangible amortization")
    sch.note("L9", "10-K balance sheet (history)")
    sch.band(11, "Goodwill & acquired intangibles")
    for r, label in [(12, "Acquisitions (cash paid)"), (13, "   of which: identified intangibles"),
                     (14, "   of which: goodwill"), (15, "Ending goodwill"), (16, "Beginning intangibles"),
                     (17, "Plus: acquired intangibles"), (18, "Less: amortization of acquired intangibles"),
                     (19, "Ending other intangible assets, net")]:
        sch.put(f"A{r}", label, b=r in (15, 19))
    for j, c in enumerate(H):
        sch.put(f"{c}12", -CFH["acq"][j] if CFH["acq"][j] is not None else None, num="mm")
        if c in s.hcols:
            sch.put(f"{c}18", f"=IS!{c}{IS['amort']}", num="mm")
    for c, v1, v2 in zip(("D", "E"), B["goodwill"][1:], B["intang"][1:]):
        sch.put(f"{c}15", v1, num="mm", b=True)
        sch.put(f"{c}19", v2, num="mm", b=True)
    for c in P:
        sch.put(f"{c}12", f"={a('acq', c)}", num="mm")
        sch.put(f"{c}13", f"={c}12*{a('intang_share', c)}", num="mm")
        sch.put(f"{c}14", f"={c}12-{c}13", num="mm")
        sch.put(f"{c}15", f"={prev(c)}15+{c}14", num="mm", b=True)
        sch.put(f"{c}16", f"={prev(c)}19", num="mm")
        sch.put(f"{c}17", f"={c}13", num="mm")
        sch.put(f"{c}18", f"=IS!{c}{IS['amort']}", num="mm")
        sch.put(f"{c}19", f"={c}16+{c}17-{c}18", num="mm", b=True)
    sch.note("L12", "10-K cash flow statement")
    sch.note("L15", "10-K balance sheet (history)")
    sch.note("L19", "10-K balance sheet (history)")
    sch.band(21, "Debt & interest expense")
    for r, label in [(22, "Beginning total debt"), (23, "Net issuance (repayment)"), (24, "Ending total debt"),
                     (25, "   Commercial paper & short-term borrowings"), (26, "   Current maturities of long-term debt"),
                     (27, "   Long-term debt"), (28, "Average total debt"), (29, "   Interest rate on average debt"),
                     (30, "Interest expense")]:
        sch.put(f"A{r}", label, b=r in (24, 30))
    for j, c in enumerate(H):
        sch.put(f"{c}23", CFH["debt"][j], num="mm")
    for c, k in zip(("D", "E"), (1, 2)):
        sch.put(f"{c}24", f"={c}25+{c}26+{c}27", num="mm", b=True)
        sch.put(f"{c}25", B["st_debt"][k], num="mm")
        sch.put(f"{c}26", B["cur_ltd"][k], num="mm")
        sch.put(f"{c}27", B["ltd"][k], num="mm")
    sch.put("E28", "=AVERAGE(D24,E24)", num="mm")
    sch.put("E29", f"=IF(E28=0,0,IS!E{IS['interest']}/E28" + (f"/(1-{aa('captive_share')})" if s.captive else "")
            + ")", num="pct")  # captive finance: the IS carries interest on the non-captive debt only
    for c in P:
        sch.put(f"{c}22", f"={prev(c)}24", num="mm")
        sch.put(f"{c}23", f"={a('debt_iss', c)}", num="mm")
        sch.put(f"{c}24", f"={c}22+{c}23", num="mm", b=True)
        sch.put(f"{c}25", f"={prev(c)}25", num="mm")
        sch.put(f"{c}26", f"={prev(c)}26", num="mm")
        sch.put(f"{c}27", f"={c}24-{c}25-{c}26", num="mm")
        sch.put(f"{c}28", f"=AVERAGE({prev(c)}24,{c}24)", num="mm")
        sch.put(f"{c}29", f"={a('int_rate', c)}", num="pct")
        sch.put(f"{c}30", f"={c}28*{c}29" + (f"*(1-{aa('captive_share')})" if s.captive else ""), num="mm", b=True)
    sch.note("L23", "10-K: debt issued less debt repaid (short- and long-term)")
    sch.note("L25", "10-K balance sheet; held flat in forecast")
    sch.note("L26", "Held flat (maturities assumed refinanced)")
    sch.note("L27", "Forecast = total debt less the two lines above")
    sch.note("L30", "Average debt avoids circularity because debt is an input, not a cash plug"
             + ("; excludes the captive-finance share (its interest is inside segment profit)" if s.captive else ""))
    sch.band(32, "Working capital")
    wc = [(33, "Accounts receivable", 8, f"IS!{{c}}{IS['sales_b']}*{{c}}34/365"),
          (35, "Inventories", 9, f"IS!{{c}}{IS['memo_cogs']}*{{c}}36/365"),
          (37, "Other current assets", 11, f"IS!{{c}}{IS['sales_b']}*{{c}}38"),
          (39, AP_LBL, 23, f"IS!{{c}}{IS['memo_cogs']}*{{c}}40/365"),
          (41, "Accrued liabilities", 26, f"IS!{{c}}{IS['sales_b']}*{{c}}42")]
    ratio = {34: ("   Days sales outstanding", f"={{c}}33/IS!{{c}}{IS['sales_b']}*365", "dso", "days"),
             36: ("   Days inventory (on cost of sales)", f"={{c}}35/IS!{{c}}{IS['memo_cogs']}*365", "dio", "days"),
             38: ("   % of sales", f"={{c}}37/IS!{{c}}{IS['sales_b']}", "oca_pct", "pct"),
             40: ("   Days payable (on cost of sales)", f"={{c}}39/IS!{{c}}{IS['memo_cogs']}*365", "dpo", "days"),
             42: ("   % of sales", f"={{c}}41/IS!{{c}}{IS['sales_b']}", "accrued_pct", "pct")}
    for r, label, bs_row, fc in wc:
        sch.put(f"A{r}", label)
        for c in ("D", "E"):
            sch.put(f"{c}{r}", f"=BS!{c}{bs_row}", num="mm")
        for c in P:
            sch.put(f"{c}{r}", "=" + fc.format(c=c), num="mm")
        rl, hf, key, num = ratio[r + 1]
        sch.put(f"A{r + 1}", rl)
        for c in ("D", "E"):
            sch.put(f"{c}{r + 1}", hf.format(c=c), num=num)
        for c in P:
            sch.put(f"{c}{r + 1}", f"={a(key, c)}", num=num)
    sch.put("A43", "Net working capital (AR + inventory + OCA – AP – accrued)", b=True)
    for c in ["D", "E"] + P:
        sch.put(f"{c}43", f"={c}33+{c}35+{c}37-{c}39-{c}41", num="mm", b=True)
    sch.put("A44", "(Increase) decrease in NWC — cash impact")
    for c in P:
        sch.put(f"{c}44", f"={prev(c)}43-{c}43", num="mm")
    sch.band(46, "Shares & dividends")
    frac_q = {"Q1": 0.25, "Q2": 0.5, "Q3": 0.75}.get(it.fiscal_period) if it.available else None
    yb = it.ytd_basic if frac_q else None
    wtd_ytd = (frac_q, yb) if (yb and s.framework != "reference" and s.ending_shares_hist and
                               0.5 * s.ending_shares_hist < yb < 1.5 * s.ending_shares_hist) else None
    for r, label in [(47, "Beginning basic shares (mm)"), (48, "Share repurchases ($mm)"),
                     (49, "Average repurchase price ($)"), (50, "   Shares repurchased (mm)"),
                     (51, "   Shares issued (mm)"), (52, "Ending basic shares (mm)"), (53, "Average basic shares (mm)"),
                     (54, "   Dilutive securities (mm)"), (55, "Diluted weighted average shares (mm)"),
                     (56, "Dividend per share ($)"), (57, "Dividends paid ($mm)")]:
        sch.put(f"A{r}", label, b=r in (52, 55, 57))
    sch.put("E52", s.ending_shares_hist, num="sh", b=True)
    sch.note("L52", s.ending_shares_note)
    for c in P:
        sch.put(f"{c}47", f"={prev(c)}52", num="sh")
        sch.put(f"{c}48", f"={a('buyback', c)}", num="mm")
        sch.put(f"{c}49", f"={a('buyback_px', c)}", num="usd")
        sch.put(f"{c}50", f"=IF({c}49=0,0,{c}48/{c}49)", num="sh")
        sch.put(f"{c}51", f"={a('shares_issued', c)}", num="sh")
        sch.put(f"{c}52", f"={c}47-{c}50+{c}51", num="sh", b=True)
        sch.put(f"{c}53", f"=AVERAGE({c}47,{c}52)", num="sh")
        if c == "F" and wtd_ytd:
            # the share count opens from the 10-Q (critique B4): today's count already holds this year's buybacks
            # and issuance, so the year end takes off only the rest of the year's; the 10-Q's weighted average
            # covers the year to date (CRM retired the ASR's shares early in the year, which the average of opening
            # and closing counts missed by 2%)
            fr, yb = wtd_ytd
            bb_ytd = (s.ytd_capital or {}).get("buyback") or 0.0
            sch.put("F52", f"=Assumptions!$C$9-IF(F49=0,0,MAX(0,F48-{bb_ytd:g})/F49)+F51*{1 - fr:g}", num="sh", b=True)
            sch.put("F50", "=F47+F51-F52", num="sh")
            sch.put("F53", f"={fr:g}*{yb:g}+{1 - fr:g}*AVERAGE(Assumptions!$C$9,F52)", num="sh")
            sch.note("L50", f"{fc_lbl[0]}: implied by the 10-Q count (the {bb_ytd:,.0f}mm bought back year to date at "
                            "its actual prices) plus the rest of the year at the assumed price")
            sch.note("L52", f"{s.ending_shares_note}. {fc_lbl[0]}: opens from today's count (Assumptions C9, 10-Q) "
                            f"less the rest of the year's buybacks, plus {1 - fr:.0%} of the year's issuance")
            sch.note("L53", f"{fc_lbl[0]}: {it.label_cur} weighted average {yb:,.1f}mm for {fr:.0%} of the year "
                            "(10-Q), then the average of today's count and the year-end count")
        sch.put(f"{c}54", "=Assumptions!$C$10", num="sh")
        sch.put(f"{c}55", f"={c}53+{c}54", num="sh", b=True)
        sch.put(f"{c}56", f"={a('dps0', 'F')}" if c == "F" else f"={prev(c)}56*(1+{a('div_growth', c)})", num="eps")
        sch.put(f"{c}57", f"={c}56*{c}53", num="mm", b=True)
    sch.band(59, f"{s.short} shareowners' equity roll-forward")
    for r, label in [(60, "Beginning equity"), (61, f"   Plus: net income attributable to {s.short}"),
                     (62, "   Plus: stock compensation (credited to paid-in capital)"),
                     (63, "   Plus: proceeds from stock issuance"), (64, "   Less: share repurchases"),
                     (65, "   Less: dividends"), (66, f"Ending {s.short} shareowners’ equity")]:
        sch.put(f"A{r}", label, b=r == 66)
    for c, v in zip(("D", "E"), B["equity"][1:]):
        sch.put(f"{c}66", v, num="mm", b=True)
    for c in P:
        sch.put(f"{c}60", f"={prev(c)}66", num="mm")
        sch.put(f"{c}61", f"=IS!{c}{IS['ni_attr']}", num="mm")
        sch.put(f"{c}62", f"=IS!{c}{IS['sbc']}", num="mm")
        sch.put(f"{c}63", f"={a('stock_proc', c)}", num="mm")
        sch.put(f"{c}64", f"=-{c}48", num="mm")
        sch.put(f"{c}65", f"=-{c}57", num="mm")
        sch.put(f"{c}66", f"=SUM({c}60:{c}65)", num="mm", b=True)
    sch.note("L66", "10-K balance sheet (history)")
    sch.ws.freeze_panes(4, 2)
    sch.finish()

    # ================================================================ DCF
    econ = s.framework != "reference"
    dcf.title("Discounted cash flow (unlevered)",
              (f"Mid-year convention. Cash flows count from the {it.period_end:%b %-d, %Y} balance sheet (the net "
               f"debt date): {s.cf_share_bs:.0%} of {fc_lbl[0]} by seasonal weight. Terminal value by value drivers: "
               "NOPAT × (1 – g ÷ RONIC) ÷ (WACC – g).") if econ else
              f"Mid-year convention; {fc_lbl[0]} counts only the {s.stub_text} stub after the "
              f"{s.as_of:%b %-d, %Y} valuation date.")
    dcf.band(5, "WACC")
    rows = [(6, "Risk-free rate", f"={aa('rf')}", "pct"), (7, "Equity beta", f"={aa('beta')}", "dec"),
            (8, "Equity risk premium", f"={aa('erp')}", "pct"),
            (9, "Cost of equity (CAPM: Rf + β × ERP)", "=C6+C7*C8", "pct"),
            (10, "Pre-tax cost of debt", f"={aa('kd')}", "pct"), (11, "Tax rate", f"={aa('tax_wacc')}", "pct"),
            (12, "After-tax cost of debt", "=C10*(1-C11)", "pct"),
            (13, "Equity market value ($mm)", "=Assumptions!$C$8*Assumptions!$C$11", "mm"),
            (14, f"Total debt{' excl. captive finance' if s.captive else ''}, {it.period_end:%b %-d, %Y} ($mm)",
             f"=Interim!$C$37*(1-{aa('captive_share')})" if s.captive else "=Interim!$C$37", "mm"),
            (15, "Equity weight", "=C13/(C13+C14)", "pct"), (16, "Debt weight", "=C14/(C13+C14)", "pct"),
            (17, "WACC", "=C15*C9+C16*C12", "pct")]
    for r, label, f, num in rows:
        dcf.put(f"A{r}", label, b=r in (9, 17))
        dcf.put(f"C{r}", f, num=num, b=r in (9, 17), fill="key" if r == 17 else None)
    dcf.note("E13", "Market values, not book values")
    if econ and s.capital:
        c_ = s.capital
        dcf.note("E7", f"Method beta: {c_.get('beta_how', '')}")
        dcf.note("E8", "One ERP for every name on the date (valuation/capital.py)")
        if c_.get("rating"):
            dcf.note("E10", (f"Agency rating {c_['rating']} (" + ("filings" if c_.get("rating_src") == "agency" else
                                                                  "analyst input: the filings name none") + ")"
                             if str(c_.get("rating_src", "")).startswith("agency") else
                             f"Synthetic rating {c_['rating']} (EBIT / interest)") +
                     f" → spread {c_.get('spread', 0):.2%}")
    dcf.band(20, "Unlevered free cash flow")
    dcf.header(21, "Unlevered free cash flow ($mm)", yr_labels)
    rows = [(22, "Net sales", lambda c: f"=IS!{c}{IS['sales_b']}", "mm", False),
            (23, "EBITDA", lambda c: f"=IS!{c}{IS['ebitda']}", "mm", False),
            (24, "EBIT (GAAP operating income)", lambda c: f"=IS!{c}{IS['ebit']}", "mm", True),
            (25, "   Tax rate", lambda c: f"={a('etr', c)}", "pct", False),
            (26, "NOPAT = EBIT × (1 – tax)", lambda c: f"={c}24*(1-{c}25)", "mm", False),
            (27, "   Plus: D&A", lambda c: f"=IS!{c}{IS['da']}", "mm", False),
            (28, "   Less: capex", lambda c: f"=CF!{c}19", "mm", False),
            (29, "   Less: increase in NWC", lambda c: f"=Schedules!{c}44", "mm", False),
            (30, "   Plus: other operating cash items", lambda c: f"=CF!{c}10", "mm", False)]
    for r, label, f, num, bold in rows:
        dcf.put(f"A{r}", label, b=bold)
        for c in P:
            dcf.put(f"{c}{r}", f(c), num=num, b=bold)
    dcf.note("L30", "Stock comp is NOT added back: treated as a real cost")
    dcf.put("A31", "Unlevered free cash flow", b=True)
    dcf.put("A32", f"Share of the year counted (after the {it.period_end:%b %-d, %Y} balance sheet)" if econ else
            "Fraction of year counted")
    dcf.put("A33", "UFCF counted, less acquisitions paid after the balance sheet (their growth is in the forecast)"
            if econ else "UFCF counted")
    dcf.put("A34", "Discount period (mid-year, years from valuation date)")
    dcf.put("A35", "Discount factor = 1 / (1 + WACC)^period")
    dcf.put("A36", "Present value of UFCF", b=True)
    ytd_acq = _nz(s.ytd_capital.get("acq"))
    for c in P:
        dcf.put(f"{c}31", f"={c}26+{c}27+{c}28+{c}29+{c}30", num="mm", b=True, fill="key", top=True)
        dcf.put(f"{c}32", "=Assumptions!$C$13" if c == "F" else 1, num="dec", fill=None if c == "F" else "input")
        if econ:  # value conservation: an acquisition whose revenue is forecast is paid for out of value
            acq = f"MAX(0,{a('acq', c)}-{ytd_acq:g})" if c == "F" else a("acq", c)
            dcf.put(f"{c}33", f"={c}31*{c}32-{acq}", num="mm")
            dcf.put(f"{c}34", "=MAX(0,(Assumptions!$C$12-Assumptions!$C$7)/365)/2" if c == "F" else
                    ("=MAX(0,(Assumptions!$C$12-Assumptions!$C$7)/365)+0.5" if c == "G" else f"={prev(c)}34+1"),
                    num="dec")
        else:
            dcf.put(f"{c}33", f"={c}31*{c}32", num="mm")
            dcf.put(f"{c}34", "=F32/2" if c == "F" else ("=$F32+0.5" if c == "G" else f"={prev(c)}34+1"), num="dec")
        dcf.put(f"{c}35", f"=1/(1+$C$17)^{c}34", num="dec")
        dcf.put(f"{c}36", f"={c}33*{c}35", num="mm", b=True)
    if econ:
        dcf.note("L32", s.cf_share_note or "Seasonal share of the year's cash flow after the balance-sheet date")
        if ytd_acq:
            dcf.note("L33", f"{fc_lbl[0]} acquisitions less the {ytd_acq:,.0f}mm already paid by the balance-sheet "
                            "date (in net debt)")
    # fade-stage layout: reference = UFCF grown at the fade rate; economic = NOPAT grown, reinvestment = g / RONIC
    FD = ({"g": 77, "nopat": 78, "reinv": 79, "fcf": 80, "per": 81, "pv": 82, "ron_k": 83, "g0": 84, "n": 85,
           "base": 86, "ronic": 87, "tr": 88, "ron0": 89} if econ else
          {"g": 77, "fcf": 78, "per": 79, "pv": 80, "g0": 82, "n": 83, "base": 84})
    if econ:
        dcf.put("A37", "NOPAT before acquired-intangible amortization (EBITA × (1 – tax)): where the fade and the "
                       "terminal value start")
        dcf.put("K37", f"=K26+IS!K{IS['amort']}*(1-K25)", num="mm", b=True)
        dcf.note("L37", "Amortization of acquired intangibles is non-cash and runs off; the explicit years add it back "
                        "in UFCF, so the fade starts from earnings before it (gate 27: no drop into the fade)")
    pv_r = f"$C${FD['pv']}:$L${FD['pv']}"
    g_ = aa("g")
    if econ:
        # a bear that declines does not re-accelerate to the house rate forever (PM round 6, ADBE: -10.5% a year,
        # then a fade back to +3.75%): its terminal growth is its own last-year growth, floored at 0%
        dcf.put("A74", "Terminal growth in the scenario shown (house rate; the bear never above its own last-year "
                       "growth, floor 0%)")
        dcf.put("C74", f"=IF(Assumptions!$C$4=1,MIN({g_},MAX(0,K22/J22-1)),{g_})", num="pct", b=True)
        g_ = "$C$74"
    if econ:
        tv = f"=$C${FD['base']}*(1+{g_})*(1-{g_}/$C${FD['ronic']})/($C$17-{g_})"
    else:
        tv = f"=$C${FD['base']}*(1+{g_})/($C$17-{g_})"
    dcf.band(39, "Valuation")
    dcf.header(40, "($mm, except per share)", {"C": "Perpetuity", "D": "Exit multiple"})
    nci_row = ((47, "Less: noncontrolling interests; plus stakes outside EBITDA (after haircut)",
                f"=-{aa('nci_value')}+{aa('other_stake')}*(1-{aa('stake_haircut')})", "=C47", "mm") if econ else
               (47, "Less: noncontrolling interests", f"=-{aa('nci_value')}", "=C47", "mm"))
    rows = [(41, f"Sum of PV of UFCF ({s.stub_start_text}–{fc_lbl[-1][:-1]}, perpetuity incl. fade years)",
             f"=SUM(F36:K36)+SUM(C{FD['pv']}:L{FD['pv']})", "=SUM(F36:K36)", "mm"),
            (42, f"Terminal value (perpetuity: after the fade years; exit: at {fc_lbl[-1][:-1]})",
             tv, f"=K23*{aa('exit_mult')}", "mm"),
            (43, "PV of terminal value", f"=C42/(1+$C$17)^($K$34+$C${FD['n']})", "=D42/(1+$C$17)^(K34+0.5)", "mm"),
            (44, "Enterprise value", "=C41+C43", "=D41+D43", "mm"),
            (45, f"Less: net debt ({it.period_end:%b %-d, %Y})", "=-Interim!$C$38", "=C45", "mm"),
            (46, ("Plus: captive-finance debt inside net debt (funded by financing receivables), net of "
                  "debt-like items" if s.captive else
                  "Less: " + s.label("debt_like_net", "other debt-like items, net").lower().replace(" (debt-like)", "")),
             "=-Interim!$C$41", "=C46", "mm"),
            nci_row,
            (48, "Equity value", "=SUM(C44:C47)", "=SUM(D44:D47)", "mm"),
            (49, "Diluted shares (mm)", "=Assumptions!$C$11", "=C49", "sh"),
            (50, "Implied value per share ($)", "=C48/C49", "=D48/D49", "usd"),
            (51, f"Upside (downside) vs {px}", "=C50/Assumptions!$C$8-1", "=D50/Assumptions!$C$8-1", "pct"),
            (52, "Terminal value as % of EV", "=C43/C44", "=D43/D44", "pct")]
    for r, label, f1, f2, num in rows:
        bold = r in (44, 48, 50)
        dcf.put(f"A{r}", label, b=bold)
        dcf.put(f"C{r}", f1, num=num, b=bold, fill="key" if r == 50 else None)
        dcf.put(f"D{r}", f2, num=num, b=bold, fill="key" if r == 50 else None)
    dcf.note("E43", "Perpetuity TV discounted at the last mid-year period; exit-multiple TV at year-end (period + 0.5)")
    dcf.put("A53", "Cross-check: implied exit EBITDA multiple / implied perpetual growth")
    dcf.put("C53", f"=(SUM({pv_r})+C43)*(1+$C$17)^$K$34/K23", num="x")
    if econ:  # g that makes the value-driver TV equal the exit TV: N(1+g)(1-g/R) = TV(w-g), smaller root
        N_, R_, T_ = "$K$37", f"$C${FD['ronic']}", "D42"
        bq = f"({N_}-{N_}/{R_}+{T_})"
        dcf.put("D53", f"=IFERROR(({bq}-SQRT({bq}^2-4*({N_}/{R_})*({T_}*$C$17-{N_})))/(2*{N_}/{R_}),"
                       f"($C$17*{T_}-{N_})/({T_}+{N_}))", num="pct")
        dcf.note("E53", "Gate: the implied multiple within ±20% of the exit multiple and the implied growth within "
                        "±1pt of terminal growth (Checks tab)")
    else:
        dcf.put("D53", "=($C$17*D42-K31)/(D42+K31)", num="pct")
    dcf.band(56, "Sensitivity: value per share — each cell is a full re-computation of the DCF")
    dcf.put("A57", "Perpetuity method: WACC (down) × terminal growth (across)", b=True)
    dcf.put("B58", "WACC \\ g", b=True, fill="hdr")
    steps = [-0.01, -0.005, 0, 0.005, 0.01]
    for c, d in zip("CDEFG", steps):
        dcf.put(f"{c}58", f"={g_}+({d:g})", num="pct", b=True, fill="hdr")

    def tv_pv(w, g, n=f"$C${FD['n']}"):
        if econ:
            return (f"$C${FD['base']}*(1+{g})*(1-{g}/$C${FD['ronic']})/({w}-{g})/(1+{w})^($K$34+{n})")
        return f"$C${FD['base']}*(1+{g})/({w}-{g})/(1+{w})^($K$34+{n})"

    fade_pv = (lambda w: f"SUMPRODUCT($C${FD['fcf']}:$L${FD['fcf']},1/(1+{w})^$C${FD['per']}:$L${FD['per']})")
    explicit = (lambda w: f"SUMPRODUCT($F$33:$K$33,1/(1+{w})^$F$34:$K$34)")
    bridge = "$C$45+$C$46+$C$47"
    for k, d in enumerate(steps):
        r = 59 + k
        dcf.put(f"B{r}", f"=$C$17+({d:g})", num="pct", b=True, fill="hdr")
        for c in "CDEFG":
            dcf.put(f"{c}{r}", f"=({explicit(f'$B{r}')}+{fade_pv(f'$B{r}')}+{tv_pv(f'$B{r}', f'{c}$58')}"
                               f"+({bridge}))/$C$49", num="usd",
                    b=(c == "E" and d == 0), fill="key" if (c == "E" and d == 0) else None)
    dcf.note("A59", "Centre cell (shaded) = base-case value above.")
    dcf.note("A60", "In Excel you can build the same grid with")
    dcf.note("A61", "Data > What-If > Data Table (Alt+A+W+T).")
    dcf.put("A66", "Exit-multiple method: WACC (down) × exit EV/EBITDA (across)", b=True)
    dcf.put("B67", "WACC \\ x", b=True, fill="hdr")
    for c, d in zip("CDEFG", [-2, -1, 0, 1, 2]):
        dcf.put(f"{c}67", f"={aa('exit_mult')}+({d})", num="x", b=True, fill="hdr")
    for k, d in enumerate(steps):
        r = 68 + k
        dcf.put(f"B{r}", f"=$C$17+({d:g})", num="pct", b=True, fill="hdr")
        for c in "CDEFG":
            dcf.put(f"{c}{r}", f"=(SUMPRODUCT($F$33:$K$33,1/(1+$B{r})^$F$34:$K$34)+$K$23*{c}$67/(1+$B{r})^($K$34+0.5)"
                               f"+({bridge}))/$C$49", num="usd",
                    b=(c == "E" and d == 0), fill="key" if (c == "E" and d == 0) else None)
    scale = {"type": "3_color_scale", "min_color": "#F8696B", "mid_type": "percentile", "mid_value": 50,
             "mid_color": "#FFFFFF", "max_color": "#63BE7B"}
    dcf.ws.conditional_format("C59:G63", scale)
    dcf.ws.conditional_format("C68:G72", scale)
    # ---- stage 2: growth fades linearly from the last explicit year's sales growth to terminal growth
    last_fy = s.fy_fc[-1]
    dcf.band(75, f"Stage 2: growth fades from {fc_lbl[-1]} to the terminal rate (three-stage DCF; perpetuity "
                 "method only)")
    dcf.header(76, "Fade year", {c: f"{last_fy + k}E" for k, c in enumerate("CDEFGHIJKL", start=1)})
    n_, g0_ = f"$C${FD['n']}", f"$C${FD['g0']}"
    dcf.put(f"A{FD['g']}", "Growth (linear fade to terminal growth)")
    if econ:
        dcf.put(f"A{FD['nopat']}", "NOPAT")
        dcf.put(f"A{FD['reinv']}", "Less: reinvestment = NOPAT × growth ÷ RONIC of the year")
        dcf.put(f"A{FD['ron_k']}", f"RONIC of the year: from {fc_lbl[-1]}'s implied return to terminal RONIC")
    dcf.put(f"A{FD['fcf']}", "Unlevered free cash flow", b=True)
    dcf.put(f"A{FD['per']}", "Discount period (years from valuation date)")
    dcf.put(f"A{FD['pv']}", "Present value", b=True)
    for k, c in enumerate("CDEFGHIJKL", start=1):
        dcf.put(f"{c}{FD['g']}", f"=IF({k}<={n_},{g0_}+({g_}-{g0_})*{k}/({n_}+1),0)", num="pct")
        if econ:
            base = "$K$37" if k == 1 else f"{prev(c)}{FD['nopat']}"
            dcf.put(f"{c}{FD['nopat']}", f"=IF({k}<={n_},{base}*(1+{c}{FD['g']}),0)", num="mm")
            # PM round 2: reinvesting g / terminal RONIC from the first fade year cut CRM's FCF 15% as growth
            # slowed; the return on new capital fades from the explicit forecast's own level instead
            # a negative reinvestment rate in the last year (customer prepayments fund growth) has no implied
            # return: the reinvestment rate steps from that level to g / RONIC instead (gate 27 continuity)
            rr_k = f"($C$94+($C${FD['tr']}-$C$94)*{k}/({n_}+1))"
            dcf.put(f"{c}{FD['ron_k']}", f"=IF({k}<={n_},IF(N($C$94)<=0,IF({rr_k}>0,MIN(1,{c}{FD['g']}/{rr_k}),1),"
                                         f"$C${FD['ron0']}+($C${FD['ronic']}-$C${FD['ron0']})*{k}/({n_}+1)),0)",
                    num="pct")
            dcf.put(f"{c}{FD['reinv']}", f"=IF({k}<={n_},MAX(0,IF(N($C$94)<=0,{c}{FD['nopat']}*{rr_k},"
                                         f"{c}{FD['nopat']}*{c}{FD['g']}/{c}{FD['ron_k']})),0)", num="mm")
            dcf.put(f"{c}{FD['fcf']}", f"={c}{FD['nopat']}-{c}{FD['reinv']}", num="mm", b=True)
        else:
            base = "$K$31" if k == 1 else f"{prev(c)}{FD['fcf']}"
            dcf.put(f"{c}{FD['fcf']}", f"=IF({k}<={n_},{base}*(1+{c}{FD['g']}),0)", num="mm", b=True)
        dcf.put(f"{c}{FD['per']}", f"=$K$34+{k}", num="dec")
        dcf.put(f"{c}{FD['pv']}", f"={c}{FD['fcf']}/(1+$C$17)^{c}{FD['per']}", num="mm", b=True)
    params = [(FD["g0"], f"Sales growth in {fc_lbl[-1]} (start of the fade)", "=K22/J22-1", "pct"),
              (FD["n"], "Fade years (Assumptions)", f"={aa('fade_years')}", "int")]
    if econ:
        params += [(FD["base"], "NOPAT in the year the perpetuity starts from",
                    f"=IF({n_}=0,$K$37,INDEX($C${FD['nopat']}:$L${FD['nopat']},{n_}))", "mm"),
                   (FD["ronic"], "Return on new invested capital (RONIC, Assumptions)", f"={aa('ronic')}", "pct"),
                   (FD["tr"], "Terminal reinvestment rate = g ÷ RONIC", f"={g_}/$C${FD['ronic']}", "pct"),
                   (FD["ron0"], f"Return on new capital implied by {fc_lbl[-1]} (growth ÷ reinvestment rate, "
                                "capped at 100%): where the fade starts; when the reinvestment rate is negative the "
                                "fade steps the reinvestment rate itself to g ÷ RONIC",
                    "=IF(N(C94)<=0,1,MIN(1,MAX(0.02,C84/C94)))", "pct")]
    else:
        params += [(FD["base"], "UFCF in the year the perpetuity starts from",
                    f"=IF({n_}=0,$K$31,INDEX($C${FD['fcf']}:$L${FD['fcf']},{n_}))", "mm")]
    for r, label, f, num in params:
        dcf.put(f"A{r}", label)
        dcf.put(f"C{r}", f, num=num)
    dcf.note(f"E{FD['n']}", "0 = two-stage DCF (perpetuity straight after the explicit years)." +
             (" The fade grows NOPAT and reinvests growth ÷ RONIC, with RONIC moving in equal steps from the "
              "return the last forecast year implies (C89) to terminal RONIC, so free cash flow does not jump at the "
              "start of the fade (gate 27); the perpetuity grid holds the fade path at base g." if econ else
              " The fade keeps the explicit year's margins and reinvestment; the perpetuity grid holds the fade "
              "path at base g."))
    if econ:
        # ---- terminal-year sanity table (gate: margin inside the 10-year range; reinvestment = g / RONIC)
        dcf.band(90, "Terminal-year sanity check")
        dcf.header(91, "Metric", {"C": fc_lbl[-1], "D": "Terminal", "E": "10-yr P10", "F": "10-yr median",
                                  "G": "10-yr P90", "H": "Percentile"}, right=False)
        mh = [m for _, m in s.margin_hist][-10:]
        hist_rng = f"$C$98:${chr(ord('C') + max(len(mh), 1) - 1)}$98"
        # financing side: equity + debt - cash (a captive arm's debt included: the receivables it funds are capital
        # that growth needs, and RONIC defaults to this return)
        ic = "(BS!K40+BS!K34+BS!K24+BS!K25+BS!K29-BS!K6-BS!K7-BS!K13)"
        ic_ = "BS!K40+BS!K34+BS!K24+BS!K25+BS!K29-BS!K6-BS!K7-BS!K13-BS!K15-BS!K16"
        sanity = [(92, "Operating margin", f"=IS!K{IS['op_margin']}", f"=IS!K{IS['op_margin']}", True),
                  (93, "Return on invested capital incl. goodwill (NOPAT ÷ equity + debt − cash − investments)",
                   f"=K26/{ic}", f"=$C${FD['ronic']}", False),
                  # every line between NOPAT and UFCF counts (PM round 4: NFLX's content spend above amortization
                  # sits in other operating items, the fade began at a 100% RONIC and FCF jumped 24% into 2032)
                  (94, "Reinvestment rate: capex + working capital + other operating items − depreciation (excl. "
                       "acquired amortization), share of NOPAT", f"=-(K27-IS!K{IS['amort']}+K28+K29+K30)/K37",
                   f"=$C${FD['tr']}", False),
                  (95, "Capex ÷ D&A", '=IFERROR(-K28/K27,"n/a")', None, False),
                  (99, "Operating ROIC (NOPAT ÷ invested capital excl. goodwill & acquired intangibles)",
                   f"=IFERROR(IF({ic_}<=0,\"n.m. (invested capital excl. goodwill is negative: customer prepayments "
                   f"fund operations)\",K26/({ic_})),\"n/a\")",
                   f"=$C${FD['ronic']}", False),
                  (96, "Growth", "=K22/J22-1", f"={g_}", False),
                  (97, "Terminal value as % of EV (perpetuity)", None, "=C52", False)]
        for r, label, f1, f2, pctl in sanity:
            dcf.put(f"A{r}", label)
            dcf.put(f"C{r}", f1, num="x" if r == 95 else "pct")
            dcf.put(f"D{r}", f2, num="pct")
            if pctl and mh:
                dcf.put(f"E{r}", f"=PERCENTILE({hist_rng},0.1)", num="pct")
                dcf.put(f"F{r}", f"=MEDIAN({hist_rng})", num="pct")
                dcf.put(f"G{r}", f"=PERCENTILE({hist_rng},0.9)", num="pct")
                dcf.put(f"H{r}", f'=COUNTIF({hist_rng},"<="&C{r})/COUNT({hist_rng})', num="pct")
        dcf.put("A98", f"Operating margin history ({s.margin_hist[0][0]}–{s.margin_hist[-1][0]}, 10-K, oldest first)"
                if mh else "Operating margin history: not available from the filings")
        for k, m in enumerate(mh):
            dcf.put(f"{chr(ord('C') + k)}98", m, num="pct")
        dcf.note("L92", "Gate: a terminal margin outside the 10th–90th percentile of history needs a structural "
                        "argument in the thesis; cyclicals end at mid-cycle")
        # ---- reverse DCF: what the price implies, one input at a time with the others at base
        dcf.band(101, "What the price implies (reverse DCF): each line solves for one input, the others at base")
        dcf.header(102, "", {"C": "Price implies", "D": "Model base"}, right=False)
        dcf.put("A103", "Equity value at the price ($mm)")
        dcf.put("C103", "=Assumptions!$C$8*Assumptions!$C$11", num="mm")
        dcf.put("A104", "Enterprise value at the price ($mm)")
        dcf.put("C104", f"=C103-({bridge})", num="mm")
        pr = "Assumptions!$C$8"
        # implied IRR: value per share at 35 discount rates (3%–20%), interpolated where it crosses the price
        rates = [round(0.03 + 0.005 * j, 4) for j in range(35)]
        cols = [xl_col_to_name(2 + j) for j in range(len(rates))]
        dcf.put("A112", "Helper: discount rate")
        dcf.put("A113", "Helper: DCF value per share at that rate")
        for c, rt in zip(cols, rates):
            dcf.put(f"{c}112", rt, num="pct")
            dcf.put(f"{c}113", f"=IF(ROUND({c}112-{g_},6)<=0.005,1E+9,({explicit(f'{c}112')}+{fade_pv(f'{c}112')}"
                               f"+{tv_pv(f'{c}112', g_)}+({bridge}))/$C$49)", num="usd")
        v1, vr, rr = f"$C$113:${cols[-1]}$113", "$C$113", f"$C$112:${cols[-1]}$112"
        k_ = f"MATCH({pr},{v1},-1)"
        dcf.put("A105", "Implied IRR: the discount rate at which the DCF equals the price")
        dcf.put("C105", f'=IFERROR(INDEX({rr},{k_})+(INDEX({v1},{k_})-{pr})/(INDEX({v1},{k_})-INDEX({v1},{k_}+1))'
                        f'*0.005,IF({pr}>{vr},"< 3%","> 20%"))', num="pct", b=True, fill="key")
        dcf.put("D105", "=$C$17", num="pct")
        dcf.note("E105", "vs WACC: a long-duration, low-risk equity can be fair at an IRR below the hurdle — the "
                         "question is whether that IRR is acceptable")
        # terminal growth the price requires: value at 37 growth rates (-2% to 7%)
        grs = [round(-0.02 + 0.0025 * j, 4) for j in range(37)]
        gcols = [xl_col_to_name(2 + j) for j in range(len(grs))]
        dcf.put("A114", "Helper: terminal growth")
        dcf.put("A115", "Helper: DCF value per share at that growth")
        for c, gr in zip(gcols, grs):
            dcf.put(f"{c}114", gr, num="pct")
            dcf.put(f"{c}115", f"=IF(ROUND($C$17-{c}114,6)<=0.005,1E+9,({explicit('$C$17')}+{fade_pv('$C$17')}"
                               f"+{tv_pv('$C$17', f'{c}114')}+({bridge}))/$C$49)", num="usd")
        g1, gr_ = f"$C$115:${gcols[-1]}$115", f"$C$114:${gcols[-1]}$114"
        kg = f"MATCH({pr},{g1},1)"
        dcf.put("A106", "Terminal growth the price requires (at the model's WACC)")
        dcf.put("C106", f'=IFERROR(INDEX({gr_},{kg})+({pr}-INDEX({g1},{kg}))/(INDEX({g1},{kg}+1)-INDEX({g1},{kg}))'
                        f'*0.0025,IF({pr}<$C$115,"< -2%","> 7%"))', num="pct")
        dcf.put("D106", f"={g_}", num="pct")
        dcf.put("A107", "Terminal operating margin the price requires (post-explicit NOPAT scaled)")
        dcf.put("C107", f'=IFERROR(IS!K{IS["op_margin"]}*(C104-SUM(F36:K36))/(C41-SUM(F36:K36)+C43),"n/a")',
                num="pct")
        dcf.put("D107", f"=IS!K{IS['op_margin']}", num="pct")
        # years of above-terminal growth after the explicit forecast: value for 0..20 fade years
        dcf.put("A108", f"Years of above-terminal growth after {fc_lbl[-1]} the price requires")
        dcf.put("A117", "Helper: fade years")
        steps_n = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
        fcols = [xl_col_to_name(2 + i) for i in range(len(steps_n))]
        K = 20
        for nf, c in zip(steps_n, fcols):
            dcf.put(f"{c}117", nf, num="int")
            for k in range(1, K + 1):
                gk = f"({g0_}+({g_}-{g0_})*{k}/({c}$117+1))"
                fr, pr_ = 117 + k, 117 + K + k
                prevf = "1" if k == 1 else f"{c}{fr - 1}"
                dcf.put(f"{c}{fr}", f"=IF({k}<={c}$117,{prevf}*(1+{gk}),{prevf})", num="dec")
                rk = f"($C${FD['ron0']}+($C${FD['ronic']}-$C${FD['ron0']})*{k}/({c}$117+1))"
                rrk = f"($C$94+($C${FD['tr']}-$C$94)*{k}/({c}$117+1))"
                dcf.put(f"{c}{pr_}", f"=IF({k}<={c}$117,$K$37*{c}{fr}*IF(N($C$94)<=0,1-{rrk},1-{gk}/{rk})"
                                     f"/(1+$C$17)^($K$34+{k}),0)", num="mm")
            tvr, valr = 118 + 2 * K, 119 + 2 * K
            dcf.put(f"{c}{tvr}", f"=$K$37*{c}{117 + K}*(1+{g_})*(1-{g_}/$C${FD['ronic']})/($C$17-{g_})"
                                 f"/(1+$C$17)^($K$34+{c}$117)", num="mm")
            dcf.put(f"{c}{valr}", f"=(SUM($F$36:$K$36)+SUM({c}{118 + K}:{c}{117 + 2 * K})+{c}{tvr}+({bridge}))/$C$49",
                    num="usd")
        dcf.put("A118", f"Helper: growth factor by fade year (rows 118–{117 + K}), PV of fade cash flows "
                        f"({118 + K}–{117 + 2 * K}), PV of terminal value ({118 + 2 * K})")
        dcf.put(f"A{119 + 2 * K}", "Helper: DCF value per share with that many fade years")
        vrow = 119 + 2 * K
        f1 = f"$C${vrow}:${fcols[-1]}${vrow}"
        nrow = f"$C$117:${fcols[-1]}$117"
        kf = f"MATCH({pr},{f1},1)"
        dcf.put("C108", f'=IF({g0_}<={g_},"n/a: growth already at terminal",IF({pr}<=$C${vrow},0,IF({pr}>${fcols[-1]}${vrow},'
                        f'"> 20",INDEX({nrow},{kf})+({pr}-INDEX({f1},{kf}))/(INDEX({f1},{kf}+1)-INDEX({f1},{kf}))'
                        f'*(INDEX({nrow},{kf}+1)-INDEX({nrow},{kf})))))', num="dec")
        dcf.put("D108", f"={n_}", num="dec")
        dcf.note("E106", "Holds the explicit forecast and fade path at base; changes only terminal growth")
        dcf.note("E107", "The terminal-stage margin at which the DCF equals the price, all else at base")
    dcf.finish()

    # ================================================================ Comps
    cmp_.title(f"Trading comparables (P/E), priced at {s.as_of:%b %-d, %Y}",
               f"Forward P/E on each company’s own current-fiscal-year EPS (consensus or guidance midpoint).")
    cmp_.header(4, "Company", {"B": "Ticker", "C": f"Overlap with {s.short}", "D": "Price ($)",
                               "E": f"{fc_lbl[0]} EPS ($)", "F": f"P/E {fc_lbl[0]}", "G": "Source",
                               "H": "EV ($mm)", "I": "LTM EBITDA", "J": "EV / EBITDA", "K": "EV & EBITDA source"},
                right=False)
    last = 4 + npeer
    peers = s.peers or [{"name": "Add a peer", "ticker": "", "overlap": "", "price": None, "eps": None,
                         "source": "No peers supplied"}]
    econ = s.framework != "reference"
    def trailing(p_):  # a trailing GAAP EPS in a forward column (CRM's MSFT row: "latest LTM GAAP diluted EPS")
        src = str(p_.get("source") or "")
        return (p_.get("eps_basis") == "trailing" or src.startswith("TRAILING") or "GAAP diluted EPS from SEC" in src
                or (bool(re.search(r"(?i)\b(ltm|trailing|last twelve)\b", src)) and
                    not re.search(r"(?i)FY ?20\d\d guid|guidance midpoint|guide midpoint|midpoint|forward|consensus|"
                                  r"estimate", src)))
    n_fwd = sum(1 for p_ in peers if p_.get("eps") and p_["eps"] > 0 and not trailing(p_))
    I0 = s.hist.IS
    co_margin = ((_nz(I0["opinc"][2]) + _nz(I0["da"][2])) / I0["sales"][2]
                 if I0.get("da") and I0["sales"][2] and I0["opinc"][2] is not None else None)
    if econ:
        cmp_.put("L4", "Net income growth (next FY: consensus EPS growth with last year's share-count change)",
                 b=True, fill="hdr")
        cmp_.put("M4", "Revenue growth (next FY, consensus)", b=True, fill="hdr")
    for k, p_ in enumerate(peers):
        r = 5 + k
        if econ and peer_ni_growth(p_) is not None:
            cmp_.put(f"L{r}", peer_ni_growth(p_), num="pct")
            if isinstance(p_.get("share_chg"), (int, float)):
                cmp_.note(f"L{r}", f"Consensus EPS growth {peer_growth(p_):+.1%}; diluted shares "
                                   f"{p_['share_chg']:+.1%} over the last year (SEC)")
        if econ and isinstance(p_.get("rev_g_next"), (int, float)):
            cmp_.put(f"M{r}", p_["rev_g_next"], num="pct")
        cmp_.put(f"A{r}", p_["name"])
        cmp_.put(f"B{r}", p_["ticker"])
        cmp_.put(f"C{r}", p_.get("overlap") or "")
        cmp_.put(f"D{r}", p_.get("price"), num="usd")
        cmp_.put(f"E{r}", p_.get("eps"), num="eps")
        if econ and n_fwd >= 3 and trailing(p_):  # a trailing GAAP EPS in a forward column (CRM: MSFT)
            cmp_.put(f"F{r}", "trailing")  # text: MEDIAN ignores it
        elif econ and n_fwd >= 4 and peer_eps_issue(p_, peers):
            cmp_.put(f"F{r}", "excluded")  # text: MEDIAN ignores it
            cmp_.note(f"F{r}", peer_eps_issue(p_, peers))
        else:
            cmp_.put(f"F{r}", f'=IF(N(E{r})<=0,"n/m",D{r}/E{r})', num="x")
        cmp_.note(f"G{r}", p_.get("source") or "")
        cmp_.put(f"H{r}", p_.get("ev"), num="mm")
        cmp_.put(f"I{r}", p_.get("ebitda"), num="mm")
        pm_ = p_.get("ebitda_margin")
        if econ and isinstance(pm_, (int, float)) and co_margin and pm_ < 0.5 * co_margin:
            # a multiple of depressed EBITDA does not transfer (PM round 6: HUBS and DOCU at 26x on GAAP margins
            # far below ADBE's 37%)
            cmp_.put(f"J{r}", "excluded")  # text: MEDIAN ignores it
            cmp_.note(f"J{r}", f"EBITDA margin {pm_:.0%} is under half of {s.short}'s {co_margin:.0%}: a multiple of "
                               "depressed EBITDA does not transfer, so it is left out of the EV / EBITDA median")
        else:
            cmp_.put(f"J{r}", f'=IF(OR(N(H{r})<=0,N(I{r})<=0),"n/m",H{r}/I{r})', num="x")
        cmp_.note(f"K{r}", p_.get("ev_source") or ("" if p_.get("ev") else "No EV / EBITDA from SEC filings"))
    rows = [(CP["med"], "Peer median", f"=MEDIAN(F5:F{last})", "x", False),
            (CP["mean"], "Peer mean", f"=AVERAGE(F5:F{last})", "x", False),
            (CP["co"], f"{s.short} at {px} on model {fc_lbl[0]} adj. EPS",
             f"=Assumptions!$C$8/IS!F{IS['op_eps']}", "x", False),
            (CP["disc"], s.label("peer_discount", "Discount (premium) to peers"), f"={aa('peer_discount')}", "pct",
             False),
            (CP["applied"], ("Applied peer P/E = growth-adjusted peer P/E × (1 – discount)"
                             if econ else "Applied multiple = median × (1 – discount)"),
             (f"=F{CP['ga_fit']}*(1-F{CP['disc']})" if econ else f"=F{CP['med']}*(1-F{CP['disc']})"), "x", True),
            (CP["eps"], (f"{s.short} {fc_lbl[0]} operating EPS" + (", mid-cycle basis" if s.profile.get("cyclical")
                                                                     else "") + " (model)") if econ else
             f"{s.short} {fc_lbl[0]} operating EPS (model)",
             f"=F{CP['eps_mid']}" if econ else f"=IS!F{IS['op_eps']}", "eps", False),
            (CP["implied"], ("Implied value per share ($): peer P/E on operating EPS blended with the own-history "
                             "P/E on GAAP EPS, plus stakes") if econ else "Implied value per share ($)",
             (f"=(1-F{CP['own_w']})*F{CP['applied']}*F{CP['eps']}+F{CP['own_w']}*N(F{CP['own_val']})+F{CP['stake']}"
              if econ else f"=F{CP['applied']}*F{CP['eps']}"), "usd", True),
            (CP["up"], f"Upside (downside) vs {px}", f"=F{CP['implied']}/Assumptions!$C$8-1", "pct", False),
            (CP["low"], "Range low: discount + 10 points",
             (f"=F{CP['implied']}*0.9" if econ else
              f"=F{CP['med']}*(1-F{CP['disc']}-0.1)*F{CP['eps']}"), "usd", False),
            (CP["high"], "Range high: discount – 10 points",
             (f"=F{CP['implied']}*1.1" if econ else
              f"=F{CP['med']}*(1-F{CP['disc']}+0.1)*F{CP['eps']}"), "usd", False)]
    for r, label, f, num, bold in rows:
        cmp_.put(f"A{r}", label, b=bold)
        cmp_.put(f"F{r}", f, num=num, b=bold, fill="key" if r == CP["implied"] else None)
    J_ = f"J5:J{last}"
    nci_ = aa("nci_value")
    shares_ = "Assumptions!$C$11"
    cmp_.band(CP["ev_band"], f"EV / LTM EBITDA (peers' EBITDA and net debt from their SEC filings, as of "
                             f"{s.as_of:%b %-d, %Y})")
    evf_ = ev_growth_fit(s.peers) if econ else {"a": None}
    rows = [(CP["ev_med"], "Peer median EV / LTM EBITDA", f"=IF(COUNT({J_})=0,0,MEDIAN({J_}))", "x", False),
            (CP["ev_mean"], "Peer mean", f"=IF(COUNT({J_})=0,0,AVERAGE({J_}))", "x", False),
            (CP["ev_co"], f"{s.short} today: EV / LTM EBITDA", "=Summary!C8/Interim!$F$24", "x", False),
            (CP["ev_applied"], (f"Applied multiple = peer EV / EBITDA at {s.short}'s {fc_lbl[1]} revenue growth "
                                "(regression on peers' consensus revenue growth; else the median) × (1 – discount)"
                                if econ else "Applied multiple = median × (1 – discount)"),
             (f"=MIN(MAX({evf_['a']:.4f}+{evf_['b']:.4f}*IS!G{IS['growth']},0.6*F{CP['ev_med']}),1.6*F{CP['ev_med']})"
              f"*(1-F{CP['disc']})" if econ and evf_["a"] is not None else f"=F{CP['ev_med']}*(1-F{CP['disc']})"),
             "x", True),
            (CP["ev_ebitda"], f"{s.short} LTM EBITDA ({it.ltm_label}, $mm)", "=Interim!$F$24", "mm", False),
            (CP["ev_ev"], "Implied enterprise value ($mm)", f"=F{CP['ev_applied']}*F{CP['ev_ebitda']}", "mm", False),
            (CP["ev_less"], "Less: net debt, debt-like items, NCI ($mm)" + ("; plus stakes" if econ else ""),
             f"=-(Interim!$C$38+Interim!$C$41+{nci_})" + (f"+{aa('other_stake')}*(1-{aa('stake_haircut')})"
                                                           if econ else ""), "mm", False),
            (CP["ev_ps"], "Implied value per share ($)",
             f"=IF(COUNT({J_})=0,0,(F{CP['ev_ev']}+F{CP['ev_less']})/{shares_})", "usd", True),
            (CP["ev_up"], f"Upside (downside) vs {px}", f"=IF(COUNT({J_})=0,0,F{CP['ev_ps']}/Assumptions!$C$8-1)", "pct",
             False),
            (CP["ev_low"], "Range low: applied multiple – 10%" if econ else "Range low: discount + 10 points",
             (f"=IF(COUNT({J_})=0,0,(F{CP['ev_applied']}*0.9*F{CP['ev_ebitda']}+F{CP['ev_less']})/{shares_})" if econ
              else f"=IF(COUNT({J_})=0,0,(F{CP['ev_med']}*(1-F{CP['disc']}-0.1)*F{CP['ev_ebitda']}+F{CP['ev_less']})"
              f"/{shares_})"), "usd", False),
            (CP["ev_high"], "Range high: applied multiple + 10%" if econ else "Range high: discount – 10 points",
             (f"=IF(COUNT({J_})=0,0,(F{CP['ev_applied']}*1.1*F{CP['ev_ebitda']}+F{CP['ev_less']})/{shares_})" if econ
              else f"=IF(COUNT({J_})=0,0,(F{CP['ev_med']}*(1-F{CP['disc']}+0.1)*F{CP['ev_ebitda']}+F{CP['ev_less']})"
              f"/{shares_})"), "usd", False)]
    for r, label, f, num, bold in rows:
        cmp_.put(f"A{r}", label, b=bold)
        cmp_.put(f"F{r}", f, num=num, b=bold, fill="key" if r == CP["ev_ps"] else None)
    if econ:
        cmp_.note(f"G{CP['ev_applied']}", (evf_["how"] or "plain median: " + (
            evf_["rejected"] or "fewer than 6 peers with consensus revenue growth")) + "; bounded to 0.6–1.6× the median")
    cmp_.band(CP["bl_band"], "Comps value used in the Summary")
    w_ = f"F{CP['bl_w']}"
    disp_ = ev_dispersed(s.peers) if econ else ""
    # one method should not disagree with itself by 2x (PM round 7, ADBE: $549 on the EV / EBITDA median against $242
    # on the growth-adjusted P/E): when the legs are 1.5x apart, the unadjusted EV / EBITDA leg is dropped
    legs = f"MAX(F{CP['implied']},F{CP['ev_ps']})/MAX(0.01,MIN(F{CP['implied']},F{CP['ev_ps']}))"
    rows = [(CP["bl_w"], ("Weight on P/E: 100% — " + disp_) if disp_ else
             "Weight on P/E (Assumptions; 100% when no peer EBITDA is available or the two legs are more than 1.5x "
             "apart)",
             "=1" if disp_ else f"=IF(COUNT({J_})=0,1,IF({legs}>1.5,1,{aa('comps_pe_weight')}))" if econ else
             f"=IF(COUNT({J_})=0,1,{aa('comps_pe_weight')})", "pct", False),
            (CP["bl_base"], "Comps value per share ($): P/E and EV/EBITDA blended",
             f"={w_}*F{CP['implied']}+(1-{w_})*F{CP['ev_ps']}", "usd", True),
            (CP["bl_low"], "   Low", f"={w_}*F{CP['low']}+(1-{w_})*F{CP['ev_low']}", "usd", False),
            (CP["bl_high"], "   High", f"={w_}*F{CP['high']}+(1-{w_})*F{CP['ev_high']}", "usd", False),
            (CP["bl_up"], f"Upside (downside) vs {px}", f"=F{CP['bl_base']}/Assumptions!$C$8-1", "pct", False)]
    for r, label, f, num, bold in rows:
        cmp_.put(f"A{r}", label, b=bold)
        cmp_.put(f"F{r}", f, num=num, b=bold, fill="key" if r == CP["bl_base"] else None)
    if econ:
        # growth-adjusted peer multiple: regression of P/E on next-year EPS growth (6+ peers) or median PEG; the
        # fit is computed from the peer inputs above and applied to the company's own growth (live)
        fit_ = peer_growth_fit(s.peers, bool(s.profile.get("cyclical")))
        a_, b_, peg, how = fit_["a"], fit_["b"], fit_["peg"], fit_["how"]
        cmp_.band(CP["ga_band"], "Growth-adjusted peer P/E (replaces a flat premium or discount)")
        # two years, not one: a single year's growth carries one-offs into the multiple (COST: FY28 interest income
        # falls after a special dividend, 4% growth, so the PEG fit priced it at 14x)
        # net income, not per share: buybacks are not growth the peers' multiples pay for (PM round 6: ~10 of
        # ADBE's 18% EPS growth was shares retired, and the PEG priced it at 23x)
        cmp_.put(f"A{CP['ga_g']}", f"{s.short} operating net income growth (model, {fc_lbl[0]} to {fc_lbl[2]}, a "
                                   "year; operating EPS × diluted shares, so buybacks do not count as growth)")
        ni_ = lambda c: f"IS!{c}{IS['op_eps']}*IS!{c}{IS['shares']}"  # noqa: E731
        cmp_.put(f"F{CP['ga_g']}", f"=IFERROR(({ni_('H')}/({ni_('F')}))^0.5-1,{ni_('G')}/({ni_('F')})-1)", num="pct")
        cmp_.put(f"A{CP['ga_fit']}", "Growth-adjusted peer P/E at that growth", b=True)
        med = f"F{CP['med']}"
        if a_ is not None:
            fit = f"{a_:.4f}+{b_:.4f}*F{CP['ga_g']}"
        elif peg is not None:
            fit = f"{peg:.4f}*MAX(F{CP['ga_g']},0.02)*100"
        elif s.profile.get("cyclical"):
            fit = med
        else:
            fit = med
            how = "peer median (no consensus growth for enough peers" + \
                  (f"; {fit_['rejected']}" if fit_["rejected"] else "") + ")"
        cmp_.put(f"F{CP['ga_fit']}", f"=MIN(MAX({fit},0.6*{med}),1.6*{med})", num="x", b=True)
        cmp_.note(f"G{CP['ga_fit']}", how + "; bounded to 0.6–1.6× the median")
        cmp_.put(f"A{CP['ga_med']}", "Memo: plain peer median P/E")
        cmp_.put(f"F{CP['ga_med']}", f"={med}", num="x")
        # the company's own multiple band (SHW, COST: a premium that has lasted decades is information)
        ph_pairs = [(yr, pe) for yr, pe in s.pe_hist if 0 < pe < 80]  # GAAP P/E above 80x: depressed earnings
        gaap0, adj0 = s.hist.IS["eps_dil"][2], s.adj_eps_hist[2]
        # a GAAP P/E band says nothing when adjusted EPS runs far above GAAP (CRM: 60x median, 250x 75th pct)
        comparable = not (gaap0 and adj0 and gaap0 > 0 and adj0 > 1.25 * gaap0)
        ph_pairs = ph_pairs if comparable else []
        ph = [pe for _, pe in ph_pairs]
        cmp_.band(CP["own_band"], f"{s.short}'s own P/E history (fiscal year-end price ÷ GAAP EPS, 10-K)")
        if ph:
            last_c = xl_col_to_name(1 + len(ph))
            rng = f"$B${CP['own_row']}:${last_c}${CP['own_row']}"
            for k_, (yr, pe) in enumerate(ph_pairs):
                cmp_.put(f"{xl_col_to_name(1 + k_)}{CP['own_row']}", pe, num="x")
            cmp_.put(f"A{CP['own_row']}", f"P/E by fiscal year ({ph_pairs[0][0]}–{ph_pairs[-1][0]})")
            for r, label, f in [(CP["own_med"], "Own median P/E", f"=MEDIAN({rng})"),
                                (CP["own_lo"], "Own 25th percentile", f"=PERCENTILE({rng},0.25)"),
                                (CP["own_hi"], "Own 75th percentile", f"=PERCENTILE({rng},0.75)")]:
                cmp_.put(f"A{r}", label)
                cmp_.put(f"F{r}", f, num="x")
        else:
            cmp_.put(f"A{CP['own_med']}", "Own median P/E: no 10-year EPS / price history" if comparable else
                     f"Own GAAP P/E history not meaningful: adjusted EPS "
                     f"${adj0:,.2f} vs GAAP ${gaap0:,.2f} in FY{s.fy_hist[-1]} (amortization, stock comp)")
            cmp_.put(f"F{CP['own_med']}", "n/a")
        cmp_.put(f"A{CP['own_now']}", f"Today: price ÷ {fc_lbl[0]} operating EPS")
        cmp_.put(f"F{CP['own_now']}", f"=Assumptions!$C$8/IS!F{IS['op_eps']}", num="x")
        cmp_.put(f"A{CP['own_w']}", "Weight of the own-history value in the P/E value (needs 5+ years of history)")
        cmp_.put(f"F{CP['own_w']}", 0.5 if len(ph) >= 5 else 0.0, num="pct", fill="input")
        cmp_.put(f"A{CP['own_val']}", f"Own-history value: median P/E × {fc_lbl[0]} GAAP EPS" +
                 (" (mid-cycle scaled)" if s.profile.get("cyclical") else "") + " (same basis as the history)")
        cmp_.put(f"F{CP['own_val']}", (f"=IFERROR(F{CP['own_med']}*IS!F{IS['eps']}*F{CP['eps']}/F{CP['eps_fy1']},0)"
                                        if ph else 0), num="usd")
        # EPS basis: cyclicals are never valued on trough EPS
        cmp_.band(CP["eps_band"], "EPS the multiple is applied to")
        cmp_.put(f"A{CP['eps_fy1']}", f"{fc_lbl[0]} operating EPS")
        cmp_.put(f"F{CP['eps_fy1']}", f"=IS!F{IS['op_eps']}", num="eps")
        mh = sorted(m for _, m in s.margin_hist)
        mid = mh[len(mh) // 2] if len(mh) >= 5 else None
        if s.profile.get("cyclical") and mid is not None:
            # symmetric (PM round 4): a peak is normalized down as a trough is normalized up
            cmp_.put(f"A{CP['eps_mid']}", f"Mid-cycle EPS: {fc_lbl[0]} EPS × 10-yr median operating margin "
                                          f"({mid:.1%}) ÷ {fc_lbl[0]} margin, when more than 2pt away (trough or "
                                          "peak)", b=True)
            cmp_.put(f"F{CP['eps_mid']}", f"=IF(ABS(IS!F{IS['op_margin']}-{mid:.4f})>0.02,F{CP['eps_fy1']}*{mid:.4f}"
                                          f"/IS!F{IS['op_margin']},F{CP['eps_fy1']})", num="eps", b=True)
        else:
            cmp_.put(f"A{CP['eps_mid']}", "EPS used (not a cyclical: the forecast year itself)", b=True)
            cmp_.put(f"F{CP['eps_mid']}", f"=F{CP['eps_fy1']}", num="eps", b=True)
        cmp_.put(f"A{CP['stake']}", "Stakes outside EBITDA per share (after haircut; not in EPS)")
        cmp_.put(f"F{CP['stake']}", f"={aa('other_stake')}*(1-{aa('stake_haircut')})/{shares_}", num="usd")
    cmp_.note(f"A{CP['note']}", "P/E uses each peer's current-year EPS (forward); EV / EBITDA uses LTM EBITDA from "
                                "filings for both sides (EV = price × shares outstanding + net debt). Peers with a "
                                "captive finance arm or negative EBITDA show n/m.")
    cmp_.finish()

    # ================================================================ SOTP
    econ = s.framework != "reference"
    cyc = econ and s.profile.get("cyclical")
    sotp.title(f"Sum of the parts ({fc_lbl[0]} EBITDA{', mid-cycle where the year is a trough' if cyc else ''})",
               "Each segment valued against its own peer group; corporate costs capitalized as a negative" +
               (" at the blended segment multiple; EBITDA after stock comp, like peers' GAAP EBITDA." if econ else "."))
    sotp.header(4, "Business", dict(zip("BCDEFGHIJ", ["Segment profit", "D&A less SBC" if econ else "Allocated D&A",
                                                     "EBITDA", "Bear x", "Base x", "Bull x", "Bear EV", "Base EV",
                                                     "Bull EV"])), "K", "Note", right=False)
    def valued_row(r, mults, note):
        for c, v in zip("EFG", mults):
            sotp.put(f"{c}{r}", v, num="x", fill="input")
        for c, m in zip("HIJ", "EFG"):
            sotp.put(f"{c}{r}", f"=$D{r}*{m}{r}", num="mm")
        sotp.note(f"K{r}", note)

    for i, sg in enumerate(s.segments):
        r = SO["seg"][i]
        parts = s.sotp_splits.get(sg.name) or []
        sotp.put(f"A{r}", sg.name + (" (total)" if parts else ""))
        if cyc:  # never value a cyclical segment on trough profit: FY1 sales x its 3-year average margin, if higher
            sotp.put(f"B{r}", f"=MAX(Segments!F{SG[f'p{i}']},Segments!F{SG[f's{i}']}"
                              f"*AVERAGE(Segments!C{SG[f'p{i}_m']}:E{SG[f'p{i}_m']}))", num="mm")
        else:
            sotp.put(f"B{r}", f"=Segments!F{SG[f'p{i}']}", num="mm")
        sotp.put(f"C{r}", (f"=(Schedules!F8+Schedules!F18*{SW}-IS!F{IS['sbc']}*(1-{SW}))*Segments!F{SG[f'memo{i}']}"
                           if econ else f"=(Schedules!F8+Schedules!F18*{SW})*Segments!F{SG[f'memo{i}']}"), num="mm")
        sotp.put(f"D{r}", f"=B{r}+C{r}", num="mm")
        if not parts:
            valued_row(r, sg.sotp, sg.sotp_note)
            continue
        sotp.note(f"K{r}", "Split below; not valued on this line")
        fixed = [SO["part"][(i, k)] for k, p_ in enumerate(parts) if not p_["remainder"]]
        for k, p_ in enumerate(parts):
            pr = SO["part"][(i, k)]
            sotp.put(f"A{pr}", f"   of which {p_['name']}")
            if p_["remainder"]:
                sotp.put(f"D{pr}", f"=D{r}" + "".join(f"-D{x}" for x in fixed), num="mm")
            else:
                sotp.put(f"D{pr}", p_["ebitda"], num="mm", fill="input")
            valued_row(pr, p_["multiples"], p_["note"])
    r = SO["corp"]
    if econ and s.repo_structural:  # MRK: segment profit is above R&D and unallocated costs (37% of sales)
        sotp.put(f"A{r}", "Corporate costs and operating expenses outside segment profit (R&D, unallocated), "
                          "capitalized")
        sotp.put(f"B{r}", f"=Segments!F{SG['corp_p']}-{a('repo_pct', 'G')}*IS!F{IS['sales_b']}", num="mm")
        sotp.note(f"L{r}", f"{fc_lbl[0]} sales at the {fc_lbl[1]} rate of non-segment costs (the first year also "
                           "carries one-off charges)")
    else:
        sotp.put(f"A{r}", "Corporate costs (capitalized)")
        sotp.put(f"B{r}", f"=Segments!F{SG['corp_p']}", num="mm")
    sotp.put(f"D{r}", f"=B{r}", num="mm")
    no_corp = not any(s.corp_cost_pct) and not any(v for v in s.hist.corp_profit if v) and \
        not (econ and s.repo_structural)
    seg_rows_valued = [x for x in SO["valued"] if x != SO["corp"]]
    if econ and not no_corp and all(abs(m) < 0.01 for m in s.corp_multiples):
        # corporate cost is a real, permanent cost: capitalized at the blended segment multiple, never at 0x
        den = "+".join(f"$D${x}" for x in seg_rows_valued) or "1"
        for c in "EFG":
            num_ = "+".join(f"$D${x}*{c}{x}" for x in seg_rows_valued) or "0"
            sotp.put(f"{c}{r}", f"=({num_})/({den})", num="x")
        for c, m in zip("HIJ", "EFG"):
            sotp.put(f"{c}{r}", f"=$D{r}*{m}{r}", num="mm")
        sotp.note(f"K{r}", "Capitalized at the EBITDA-weighted segment multiple (a corporate cost at 0x adds back "
                           "about a year's cost times the multiple)")
    else:
        valued_row(r, s.corp_multiples, s.note("corp_multiple", (
            "None: corporate costs sit inside segment profit (single-segment model)" if not s.segments_reported else
            "None: no corporate cost line is tagged — corporate costs are inside segment profit or in 'other "
            "non-segment items' on the IS") if no_corp else
            "Negative value: head-office costs not borne by the segments"))
    r = SO["stake"]
    sotp.put(f"A{r}", s.label("other_stake", "Other investments / stakes"))
    if econ:  # same haircut as the DCF and comps bridges
        sotp.put(f"H{r}", f"={aa('other_stake')}*(1-{aa('stake_haircut')})*0.75", num="mm")
        sotp.put(f"I{r}", f"={aa('other_stake')}*(1-{aa('stake_haircut')})", num="mm")
        sotp.put(f"J{r}", f"={aa('other_stake')}", num="mm")
    else:
        sotp.put(f"H{r}", f"={aa('other_stake')}*0.55", num="mm")
        sotp.put(f"I{r}", f"={aa('other_stake')}", num="mm")
        sotp.put(f"J{r}", f"={aa('other_stake')}*2", num="mm")
    sotp.note(f"K{r}", "Bear = 55%, Bull = 2× the base value (Assumptions)" if s.valuation["other_stake"][0] else
              "None identified; enter stakes or investments outside segment EBITDA on Assumptions")
    d_rows = [str(SO["seg"][i]) for i in range(n)] + [str(SO["corp"])]
    v_rows = [str(x) for x in SO["valued"]] + [str(SO["stake"])]
    r = SO["ev"]
    sotp.put(f"A{r}", "Enterprise value", b=True)
    sotp.put(f"D{r}", "=" + "+".join(f"D{x}" for x in d_rows), num="mm", b=True, top=True)
    for c in "HIJ":
        sotp.put(f"{c}{r}", "=" + "+".join(f"{c}{x}" for x in v_rows), num="mm", b=True, top=True)
    for c in "EFG":
        sotp.ws.write_blank(f"{c}{r}", None, bk.fmt(top=True))
    sotp.note(f"K{r}", "Column D = total segment EBITDA less corporate")
    less = f"=-(Interim!$C$38+Interim!$C$41+{aa('nci_value')})"
    for key, label, f, num, bold, fill in [
            ("less", "Less: net debt, " + s.label("debt_like_net", "other debt-like items").lower()
             .replace(" (debt-like)", "") + ", NCI", lambda c: less, "mm", False, None),
            ("eq", "Equity value", lambda c: f"={c}{SO['ev']}+{c}{SO['less']}", "mm", True, None),
            ("ps", "Value per diluted share ($)", lambda c: f"={c}{SO['eq']}/Assumptions!$C$11", "usd", True, "key"),
            ("up", f"Upside (downside) vs {px}", lambda c: f"={c}{SO['ps']}/Assumptions!$C$8-1", "pct", False, None),
            ("impl", f"Implied group EV / {fc_lbl[0]} EBITDA", lambda c: f"={c}{SO['ev']}/$D${SO['ev']}", "x",
             False, None)]:
        r = SO[key]
        sotp.put(f"A{r}", label, b=bold)
        for c in "HIJ":
            sotp.put(f"{c}{r}", f(c), num=num, b=bold, fill=fill)
    r = SO["mkt"]
    sotp.put(f"A{r}", f"Market today: EV / {fc_lbl[0]} segment EBITDA less corporate")
    sotp.put(f"I{r}", f"=(Assumptions!$C$8*Assumptions!$C$11+Interim!$C$38+Interim!$C$41+{aa('nci_value')})"
                      f"/D{SO['ev']}", num="x")
    sotp.note(f"A{SO['note']}", "D&A is allocated to segments by sales mix. Multiples are analyst judgment set "
                                "against each segment's peer group.")
    sotp.finish()

    # ================================================================ Deals (economic framework)
    if econ_wb:
        dl_.title("Pending deals and separations",
                  "Value conservation: an acquisition is worth what it adds less what it costs; a separation swaps the "
                  "business's value inside the company for what holders receive. Probability-weighted into fair value.")
        dl_.header(4, "Deal", {"B": "Kind", "C": "Close", "D": "Probability", "E": "Value ($mm)", "F": "Per share",
                               "G": "Note"}, right=False)
        cm = "Summary!$C$10"  # the company's EV / FY1 EBITDA: what the business is worth inside the group today
        rows_ = []
        r = 5
        for d in s.deals:
            kind = str(d.get("kind", "acquisition")).lower()
            p = d.get("probability")
            p = (1.0 if str(d.get("status", "")).lower() == "closed" else 0.5) if p is None else p
            mult = d.get("multiple")
            mult_f = f"{mult:g}" if isinstance(mult, (int, float)) else cm
            g_ = lambda k: float(d.get(k) or 0.0)  # noqa: E731
            if kind.startswith("sep") or kind.startswith("spin"):
                own = f"{g_('stake') or 1.0:g}*{g_('ebitda'):g}*{mult_f}+{g_('cash_received'):g}-{g_('separation_cost'):g}"
                inside = f"{g_('ebitda'):g}*{cm}" if d.get("in_forecast", True) else "0"
                val = f"=({own}-{g_('stranded_cost'):g}*{cm}-{inside})"
                how = (f"{g_('stake') or 1.0:.0%} of {g_('ebitda'):,.0f}mm EBITDA at "
                       f"{mult_f if isinstance(mult, (int, float)) else 'the company multiple'}x + "
                       f"{g_('cash_received'):,.0f}mm cash − {g_('separation_cost'):,.0f}mm costs − stranded "
                       f"{g_('stranded_cost'):,.0f}mm/yr capitalized − its value inside the company")
            else:
                ebitda_d = g_("ebitda") or g_("revenue") * g_("margin")
                prem = g_("premium") if d.get("premium") is not None else 0.30
                market_sa = d.get("premium") is not None or not ebitda_d
                if market_sa:  # a public target's undisturbed value: sub-scale EBITDA x a multiple misses it (Braze)
                    standalone = f"{g_('consideration'):g}/(1+{prem:g})"
                else:
                    standalone = f"{ebitda_d:g}*{mult_f}"
                # synergies are earned inside the acquirer: capitalized at the lower of the deal multiple and the
                # acquirer's own EV / EBITDA (CRM v4: Braze and Fin synergies at a 30x target multiple added $4/share)
                syn_m = f"MIN({mult_f},{cm})" if isinstance(mult, (int, float)) else cm
                # PM round 4: all three CRM deals came out accretive at 80% credit — acquirers on average create no
                # value at announcement. Half credit, the run-rate reached in year 3 (deferred two years at WACC),
                # integration costs after tax
                syn = (f"{g_('synergies'):g}*{syn_m}*{SYN_CREDIT:g}/(1+DCF!$C$17)^2"
                       f"-{g_('integration_cost'):g}*(1-DCF!$C$11)")
                created = f"{standalone}+{syn}-{g_('consideration'):g}"
                if d.get("in_forecast"):  # its revenue is already in the forecast: only synergies less price/costs
                    created = f"{syn}-{g_('consideration'):g}"
                val = f"=({created})"
                how = ((f"standalone value = price ÷ (1 + {prem:.0%} premium over the undisturbed price)" +
                        (f" (cross-check: {ebitda_d:,.0f}mm EBITDA x {mult_f}x = {ebitda_d * mult:,.0f}mm)"
                         if ebitda_d and isinstance(mult, (int, float)) else
                         "" if d.get("premium") is not None else " (no target EBITDA or premium entered)")
                        if market_sa else f"{ebitda_d:,.0f}mm EBITDA") +
                       f" + synergies {g_('synergies'):,.0f}mm a year ({SYN_CREDIT:.0%} credit, run-rate in year 3) "
                       f"at the lower of {mult_f if isinstance(mult, (int, float)) else 'the company multiple'}x and "
                       "the company's own EV / EBITDA − price "
                       f"{g_('consideration'):,.0f}mm − integration {g_('integration_cost'):,.0f}mm after tax")
            dl_.put(f"A{r}", str(d.get("name", "Deal")))
            dl_.put(f"B{r}", kind)
            dl_.put(f"C{r}", str(d.get("close", "")))
            dl_.put(f"D{r}", p, num="pct", fill="input")
            dl_.put(f"E{r}", val, num="mm")
            bf = g_("break_fee")
            dl_.put(f"F{r}", f"=(D{r}*E{r}+(1-D{r})*{bf:g})/Assumptions!$C$11", num="usd", b=True)
            dl_.note(f"G{r}", how + (f"; break fee {bf:+,.0f}mm if it fails" if bf else "") +
                     (f". {d['note']}" if d.get("note") else ""))
            rows_.append(r)
            r += 1
        if not rows_:
            dl_.put("A5", "No pending acquisitions or separations entered. A signed deal of any size goes here: "
                          "its standalone value (revenue, margin, a peer multiple) less the price, probability-weighted.")
            r = 6
        dl_.put(f"A{r + 1}", "Total per share, probability-weighted (added to the weighted fair value)", b=True)
        dl_.put(f"F{r + 1}", "=" + ("+".join(f"F{x}" for x in rows_) if rows_ else "0"), num="usd", b=True,
                fill="key", top=True)
        DEALS_TOTAL = f"Deals!$F${r + 1}"
        dl_.finish()
    else:
        DEALS_TOTAL = None

    # ================================================================ Checks (economic framework)
    if econ_wb:
        lr = s.last_review or {}
        ck.title("Checks: pre-publication gates, plugs, consensus, audit trail",
                 "A failed gate holds the rating at Under review. Computed when the file was built.")
        ck.header(4, "Gate", {"B": "Status", "C": "Evidence"}, right=False)
        colors = {"PASS": "#008000", "WARN": "#C55A11", "FAIL": "#C00000", "N/A": "#595959"}
        r = 5
        for gt in sorted(lr.get("gates") or [], key=lambda x: x["id"]):
            ck.put(f"A{r}", f"{gt['id']}. {gt['gate']}")
            ck.put(f"B{r}", gt["status"], b=True, color=colors.get(gt["status"]))
            ck.put(f"C{r}", gt.get("detail") or "")
            r += 1
        if not lr.get("gates"):
            ck.put(f"A{r}", "Gates are computed when the model is reviewed (build with the check step)")
            r += 1
        r += 1
        opens = (s.rating or {}).get("open_items") or []
        if opens:
            ck.band(r, "Warnings and open checks: each resolved or explained in the thesis before a rating")
            ck.header(r + 1, "Item (key for thesis.explanations)", {"B": "Status", "C": "Explanation"}, right=False)
            r += 2
            for key, text, why in opens:
                ck.put(f"A{r}", f"{key}: {text}"[:240])
                ck.put(f"B{r}", "explained" if why else "OPEN", b=True, color=None if why else "#C00000")
                ck.put(f"C{r}", live_text(why) or "Not explained: the rating is held at Under review")
                r += 1
            r += 1
        if s.post_bs_events or s.asr_open:
            ck.band(r, f"Filed after the {it.period_end:%b %-d, %Y} balance sheet (enter or dismiss each)")
            r += 1
            for e in s.post_bs_events:
                ck.put(f"A{r}", f"8-K {e['date']} (items {e['items']}): {e['what']}")
                ck.put(f"C{r}", e["url"])
                r += 1
            if s.asr_open:
                ck.put(f"A{r}", "Accelerated share repurchase still to settle at the 10-Q: the final share delivery "
                                "changes the count (market.basic_shares)")
                r += 1
            r += 1
        ck.band(r, f"Plug disclosure: {fc_lbl[0]} lines set away from their normal level")
        ck.header(r + 1, "Line", {"B": "Kind", "C": f"{fc_lbl[0]}", "D": "Normal", "E": "Δ $mm", "F": "% revenue",
                                  "G": "% EPS", "H": "Status"}, right=False)
        r += 2
        for pl in lr.get("plugs") or []:
            pct = abs(pl["fy1"]) < 1.5 and abs(pl["normal"]) < 1.5
            ck.put(f"A{r}", pl["line"])
            ck.put(f"B{r}", pl["kind"])
            ck.put(f"C{r}", pl["fy1"], num="pct" if pct else "mm")
            ck.put(f"D{r}", pl["normal"], num="pct" if pct else "mm")
            ck.put(f"E{r}", pl["delta_mm"], num="mm")
            ck.put(f"F{r}", pl["pct_rev"], num="pct")
            ck.put(f"G{r}", pl["pct_eps"], num="pct")
            ck.put(f"H{r}", pl["status"], b=True, color=colors.get(pl["status"]))
            r += 1
        if not lr.get("plugs"):
            ck.put(f"A{r}", "No first-year line is set away from its normal level")
            r += 1
        r += 1
        ck.band(r, "Model vs consensus")
        cons = s.consensus or {}
        ck.header(r + 1, "Metric", {"B": "Model", "C": "Consensus", "D": "Difference"}, right=False)
        r += 2
        cons_rows = []
        if cons.get("g_fy1") is not None:
            cons_rows.append((f"{fc_lbl[0]} revenue growth", f"=IS!F{IS['growth']}", cons["g_fy1"], "pct"))
        if cons.get("g_fy2") is not None:
            cons_rows.append((f"{fc_lbl[1]} revenue growth", f"=IS!G{IS['growth']}", cons["g_fy2"], "pct"))
        if cons.get("eps"):
            cons_rows.append((f"{fc_lbl[0]} adjusted EPS", f"=IS!F{IS['adj_eps']}", cons["eps"], "eps"))
        for label, mf, cv, num in cons_rows:
            ck.put(f"A{r}", label)
            ck.put(f"B{r}", mf, num=num)
            ck.put(f"C{r}", cv, num=num)
            ck.put(f"D{r}", f"=B{r}-C{r}" if num == "pct" else f"=B{r}/C{r}-1", num="pct")
            r += 1
        notes_ = (s.thesis or {}).get("consensus_notes")
        ck.put(f"A{r}", (str(notes_) if notes_ else "Why the model differs: state a reason for each gap over 3% "
                         "(thesis: consensus_notes)") if cons_rows else
               "No point-in-time consensus for this valuation date: the comparison is absent, not skipped")
        r += 2
        ck.band(r, "Final check: ten numbers re-derived from source")
        ck.header(r + 1, "Number", {"B": "Model", "C": "Source", "D": "Difference", "E": "Source used"}, right=False)
        r += 2
        q = s.interim
        frac = {"Q1": 0.25, "Q2": 0.5, "Q3": 0.75}.get(q.fiscal_period) if q.available else None
        ytd = q.ytd_cur if q.available else {}
        fin = [("Share price", "=Assumptions!$C$8", s.price, "Market close on the valuation date"),
               ("Diluted shares (mm)", "=Assumptions!$C$11", (s.basic_shares or 0) + (s.dilutive or 0),
                "10-Q / EPS share count + dilution"),
               ("Net debt ($mm)", "=Interim!$C$38", None, "Latest balance sheet (Interim)"),
               (f"{fc_lbl[0]} revenue ($mm)", f"=IS!F{IS['sales_b']}",
                sum(s.guidance["sales"]) / 2 if s.guidance.get("sales") else None, "Guidance midpoint"),
               (f"{fc_lbl[0]} adjusted EPS", f"=IS!F{IS['adj_eps']}",
                sum(s.guidance["adj_eps"]) / 2 if s.guidance.get("adj_eps") else (cons.get("eps")),
                "Guidance midpoint or consensus"),
               (f"{fc_lbl[0]} interest expense ($mm)", f"=ABS(IS!F{IS['interest']})",
                abs(ytd.get("interest_exp") / frac) if frac and ytd.get("interest_exp") else None,
                "Year to date, annualized"),
               (f"{fc_lbl[0]} D&A ($mm)", f"=IS!F{IS['da']}",
                (ytd.get("da") - _nz(ytd.get("contract_amort")) if ytd.get("contract_amort") else
                 ytd.get("da") - _nz((s.hist.IS.get("contract_amort") or [0, 0, 0])[2]) * frac) / frac
                if frac and ytd.get("da") else None,
                "Year to date, annualized" + (
                    f", less capitalized contract-cost amortization ({_nz(s.hist.IS['contract_amort'][2]):,.0f}mm in "
                    f"FY{s.fy_hist[-1]}: matched by new contract costs in operating cash flow, so the model carries "
                    "neither)" if _nz((s.hist.IS.get("contract_amort") or [0, 0, 0])[2]) or ytd.get("contract_amort")
                    else "")),
               (f"{fc_lbl[0]} capex ($mm)", "=-CF!F19",
                ytd.get("capex") / frac if frac and ytd.get("capex") else None, "Year to date, annualized"),
               ("Terminal value % of EV (perpetuity)", "=DCF!C52", 0.75, "Rule of thumb ceiling"),
               ("Weighted fair value ($)", "=Summary!G19", None, "Implied IRR at the price is on the Summary")]
        for label, mf, src, how in fin:
            fmt_ = ("pct" if "%" in label else "usd" if label in ("Share price", "Weighted fair value ($)") else
                    "eps" if "EPS" in label else "sh" if "shares" in label else "mm")
            ck.put(f"A{r}", label)
            ck.put(f"B{r}", mf, num=fmt_)
            if src is not None:
                ck.put(f"C{r}", src, num=fmt_)
                ck.put(f"D{r}", f"=IFERROR(B{r}/C{r}-1,0)", num="pct")
            ck.note(f"E{r}", how)
            r += 1
        r += 1
        if s.verification:
            ck.band(r, "Release figures checked against the source text (AI-read figures)")
            r += 1
            for item, val, found in s.verification:
                ck.put(f"A{r}", item)
                ck.put(f"B{r}", val)
                txt = {True: "found in the release text", "replaced": "read from the release's reconciliation "
                       "line (replaced the extracted figure)", "range": "found only as one end of a range "
                       "(guidance?), not as a result: verify"}.get(found, "NOT found in the release text: verify")
                ck.put(f"C{r}", txt, color=None if found in (True, "replaced") else "#C00000")
                r += 1
            r += 1
        vb = s.version_bridge
        if vb and vb.get("steps") is not None:
            pr_, nw_ = vb.get("prev") or {}, vb.get("new") or {}
            ck.band(r, f"Version bridge: this version vs the {vb.get('label', 'previous')} one (each group of changes "
                       "added in turn, valued on this engine)")
            r += 1
            ck.put(f"A{r}", f"Previous version: fair value (rule rating {pr_.get('rating')})", b=True)
            ck.put(f"B{r}", vb.get("prev_fv"), num="usd", b=True)
            r += 1
            for x in vb["steps"]:
                ck.put(f"A{r}", f"   {x['group']}")
                ck.put(f"B{r}", x["effect"], num="usd")
                ck.put(f"C{r}", "judgment" if x["judgment"] else "data")
                ck.put(f"D{r}", ", ".join(x["keys"])[:250])
                r += 1
            ck.put(f"A{r}", f"This version: fair value (rule rating {nw_.get('rating')})", b=True)
            ck.put(f"B{r}", vb.get("new_fv"), num="usd", b=True)
            r += 1
            ck.put(f"A{r}", "Share of the gross move from judgment inputs")
            ck.put(f"B{r}", vb.get("judgment_share"), num="pct")
            if vb.get("rating_changed"):
                ck.put(f"C{r}", "Rating changed: an independent review is required when judgment inputs carry half "
                                "the move or more" if vb.get("judgment_share", 0) >= 0.5 else
                       "Rating changed, mainly on data inputs")
            r += 2
        rt_ = s.rating or {}
        if rt_.get("override") or rt_.get("rule"):
            # PM round 4: a recommendation that differs from the rule is labeled and logged, never silent
            ck.band(r, "Rating log: the rule's call and any override")
            r += 1
            ck.put(f"A{r}", "Rule rating")
            ck.put(f"B{r}", rt_.get("rule") or rt_.get("rating"), b=True)
            r += 1
            ck.put(f"A{r}", "Published rating")
            ck.put(f"B{r}", rt_.get("rating"), b=True)
            ck.put(f"C{r}", rt_.get("override") or ("Held at Under review: " + str(rt_.get("why", ""))[:240]
                                                     if rt_.get("rating") == "Under review" else ""))
            r += 2
        ck.band(r, "Audit trail: who changed what, and the fair value after each pass")
        r += 1
        for stage, fv in s.audit_fv:
            ck.put(f"A{r}", f"Fair value after: {stage}")
            ck.put(f"B{r}", fv, num="usd")
            r += 1
        if s.audit_fv:  # reconciled to the published number (PM round 6)
            ck.put(f"A{r}", "Fair value in this file (Summary G19); difference from the last pass in C", b=True)
            ck.put(f"B{r}", "=Summary!G19", num="usd", b=True)
            ck.put(f"C{r}", f"=Summary!G19-B{r - 1}", num="usd")
            r += 1
        if s.audit_impact:
            ck.put(f"A{r}", "Fair-value effect of each group of overrides (that group reverted, all else as built)",
                   b=True)
            r += 1
            for grp, keys, dv in s.audit_impact:
                ck.put(f"A{r}", f"   {grp}")
                ck.put(f"B{r}", dv, num="usd")
                ck.put(f"C{r}", ", ".join(keys)[:250])
                r += 1
        for src_, key, before, after in s.audit[-120:]:
            ck.put(f"A{r}", f"{src_}: {key}")
            ck.put(f"C{r}", f"{before} → {after}")
            r += 1
        ck.finish()

    # ================================================================ Summary
    summ.title(f"{s.name} ({s.ticker}) — buy-side valuation summary",
               f"As of {s.as_of:%b %-d, %Y}. All values per share unless noted. Scenario shown below; change it "
               f"on Assumptions!C4.")
    rows = [(4, "Scenario", "=Assumptions!D4", None, True),
            (5, "Share price", "=Assumptions!$C$8", "usd", False),
            (6, "Equity market value ($mm)", "=Assumptions!$C$8*Assumptions!$C$11", "mm", False),
            (7, f"Net debt, {it.period_end:%b %-d, %Y} ($mm)", "=Interim!$C$38", "mm", False),
            (8, "Enterprise value ($mm)", f"=C6+C7+Interim!$C$41+{aa('nci_value')}", "mm", False),
            (9, f"P/E on {fc_lbl[0]} operating EPS (model)", f"=Assumptions!$C$8/IS!F{IS['op_eps']}", "x", False),
            (10, f"EV / {fc_lbl[0]} EBITDA", f"=C8/IS!F{IS['ebitda']}", "x", False),
            (11, "Dividend yield", "=Schedules!F56/Assumptions!$C$8", "pct", False)]
    for r, label, f, num, bold in rows:
        summ.put(f"A{r}", label)
        summ.put(f"C{r}", f, num=num, b=bold)
    summ.band(13, "Valuation summary")
    summ.header(14, "Method", {"C": "Low", "D": "Base", "E": "High", "F": "Weight", "G": "Weighted"},
                "H", "Basis", right=False)
    vrows = [(15, "DCF — perpetuity growth", "=DCF!D62", "=DCF!C50", "=DCF!F60", "w_pg",
              "Low/high: WACC ±0.5pt and g ∓0.5pt from the grid"),
             (16, "DCF — exit multiple", "=DCF!D71", "=DCF!D50", "=DCF!F69", "w_exit",
              "Low/high: WACC ±0.5pt and multiple ∓1x"),
             (17, "Sum of the parts", f"=SOTP!H{SO['ps']}", f"=SOTP!I{SO['ps']}", f"=SOTP!J{SO['ps']}", "w_sotp",
              "Bear / base / bull multiples"),
             (18, "Trading comps (P/E & EV/EBITDA)", f"=Comps!F{CP['bl_low']}", f"=Comps!F{CP['bl_base']}",
              f"=Comps!F{CP['bl_high']}", "w_comps", "Peer median multiples less the discount ±10 points (Comps tab)")]
    for r, label, lo, base, hi, w, basis in vrows:
        summ.put(f"A{r}", label)
        summ.put(f"C{r}", lo, num="usd")
        summ.put(f"D{r}", base, num="usd", b=True)
        summ.put(f"E{r}", hi, num="usd")
        if s.framework != "reference":  # a method at or below zero is broken, never averaged in (DUK: -$124 DCF)
            summ.put(f"F{r}", f"=IF(N(D{r})<=0,0,{aa(w)})", num="pct")
        else:
            summ.put(f"F{r}", f"={aa(w)}", num="pct")
        summ.put(f"G{r}", f"=D{r}*F{r}", num="usd")
        summ.note(f"H{r}", basis)
    summ.put("A19", "Weighted fair value", b=True)
    summ.put("F19", "=SUM(F15:F18)", num="pct", b=True, top=True)
    if DEALS_TOTAL:
        summ.put("A12", "Pending deals & separations, probability-weighted ($/share, Deals tab)")
        summ.put("C12", f"={DEALS_TOTAL}", num="usd")
        summ.put("G19", "=IF(F19=0,0,SUM(G15:G18)/F19)+C12", num="usd", b=True, fill="key", top=True)
    else:
        summ.put("G19", "=SUM(G15:G18)/F19", num="usd", b=True, fill="key", top=True)
    for c in "CDE":
        summ.ws.write_blank(f"{c}19", None, bk.fmt(top=True))
    summ.put("A20", "Upside (downside) vs current price", b=True)
    summ.put("G20", "=G19/C5-1", num="pct", b=True, fill="key")
    if s.framework != "reference":  # a fairly priced stock earns its cost of equity, not 0%
        summ.put("A21", "Expected 12-month total return (value grows at the cost of equity; price converges)",
                 b=True)
        summ.put("G21", "=G19*(1+DCF!C9)/C5-1", num="pct", b=True, fill="key")
    else:
        summ.put("A21", "Expected 12-month total return (incl. dividend yield)", b=True)
        summ.put("G21", "=G20+C11", num="pct", b=True, fill="key")
    if s.framework != "reference":
        # one discount framework: the exit multiple is the model's own value at the exit date over forward EPS,
        # and the horizon is exactly 3.0 years (forward EPS calendarized to the exit date)
        ex = _add_years(s.as_of, 3)
        ends = [_add_years(s.fy_end_cur, k) for k in range(6)]
        ia = next((k for k, e in enumerate(ends) if e >= ex), 5)
        ib = min(ia + 1, 5)
        w = max(0.0, min(1.0, (ends[ia] - ex).days / 365.0))
        eps_row = IS.get("op_eps", IS["adj_eps"])
        dw = []  # dividend weight of each fiscal year inside (valuation date, exit date]
        for k, e in enumerate(ends):
            start = _add_years(e, -1)
            ov = (min(e, ex) - max(start, s.as_of)).days
            if ov > 0:
                dw.append((P[k], ov / 365.0))
        divs = "+".join(f"Schedules!{c}56*{wt:.4f}" for c, wt in dw) or "0"
        summ.band(23, "Three-year return: the buy-side test, base case (probability-weighted in row 65)")
        rows = [(24, f"Forward operating EPS at {ex:%b %-d, %Y} (next 12 months, calendarized)",
                 f"={w:.4f}*IS!{P[ia]}{eps_row}+{1 - w:.4f}*IS!{P[ib]}{eps_row}", "eps", False),
                (25, "Exit P/E (derived: model value at exit ÷ forward EPS)", "=IFERROR(C26/C24,0)", "x", False),
                (26, f"Model value per share at {ex:%b %-d, %Y} (fair value compounded at the cost of equity, "
                     "less dividends paid)", "=G19*(1+DCF!C9)^3-C27*(1+DCF!C9)^1.5", "usd", False),
                (27, "Dividends received over the three years", f"={divs}", "eps", False),
                (28, "Holding period (years)", 3, "dec", False),
                (29, "Annualized return (dividends not reinvested)", "=((C26+C27)/C5)^(1/C28)-1", "pct", True),
                (30, "Hurdle rate (cost of equity)", "=DCF!C9", "pct", False),
                (31, "Clears the hurdle?", '=IF(C29>C30,"Yes","No")', None, True)]
        summ.put("H25", '="Derived, not typed: "&TEXT(C25,"0.0")&"x vs "&TEXT(C9,"0.0")&"x today, a "&'
                        'TEXT(1-C25/C9,"0%")&" de-rating the DCF assumes"')
        # the derived exit returns about the cost of equity plus today's mispricing by construction: show what
        # the stock returns if it keeps today's multiple, or moves to the peers' (PM round 2)
        summ.put("D23", "At today's P/E", b=True, fill="hdr")
        summ.put("E23", "At the peer P/E", b=True, fill="hdr")
        summ.put("D25", "=C9", num="x")
        summ.put("E25", f"=Comps!F{CP['applied']}", num="x")
        for c_ in "DE":
            summ.put(f"{c_}26", f"=$C$24*{c_}25", num="usd")
            summ.put(f"{c_}29", f"=(({c_}26+$C$27)/$C$5)^(1/$C$28)-1", num="pct", b=True)
            summ.put(f"{c_}31", f'=IF({c_}29>$C$30,"Yes","No")', b=True)
    else:
        summ.band(23, "Three-year return: the buy-side test")
        ex = s.exit_date
        rows = [(24, f"{fc_lbl[4]} adjusted EPS (forward EPS at {ex:%b %Y})", f"=IS!J{IS['adj_eps']}", "eps", False),
                (25, "Exit P/E", f"={aa('exit_pe')}", "x", False),
                (26, f"Implied share value at {ex:%b %-d, %Y}", "=C24*C25", "usd", False),
                (27, f"Dividends received (rest of {fc_lbl[0][:-1]} + {fc_lbl[1][:-1]}–{fc_lbl[3][:-1]})",
                 "=Schedules!F56*Assumptions!$C$13+SUM(Schedules!G56:I56)", "eps", False),
                (28, "Holding period (years)", f"=(DATE({ex.year},{ex.month},{ex.day})-Assumptions!$C$7)/365", "dec",
                 False),
                (29, "Annualized return (dividends not reinvested)", "=((C26+C27)/C5)^(1/C28)-1", "pct", True),
                (30, "Hurdle rate (cost of equity)", "=DCF!C9", "pct", False),
                (31, "Clears the hurdle?", '=IF(C29>C30,"Yes","No")', None, True)]
    for r, label, f, num, bold in rows:
        summ.put(f"A{r}", label, b=bold)
        summ.put(f"C{r}", f, num=num, b=bold, fill="key" if r in (29, 31) else None)
    summ.band(33, "Key financials")
    summ.header(34, "($mm, except per share)", {c: f"=IS!{ic}4" for c, ic in zip("CDEFG", "EFGHI")})
    kf = [(35, "Net sales", "sales_b", "mm"), (36, "  growth", "growth", "pct"),
          (37, "Segment margin", "seg_margin", "pct"), (38, "Operating income", "ebit", "mm"),
          (39, "Adjusted EPS", "adj_eps", "eps"), (40, "  growth", "adj_growth", "pct")]
    for r, label, key, num in kf:
        summ.put(f"A{r}", label)
        for c, ic in zip("CDEFG", "EFGHI"):
            summ.put(f"{c}{r}", f"=IS!{ic}{IS[key]}", num=num)
    for r, label, row, num in [(41, "Free cash flow", 39, "mm"), (42, "FCF per share", 41, "eps")]:
        summ.put(f"A{r}", label)
        for c, ic in zip("CDEFG", "EFGHI"):
            summ.put(f"{c}{r}", f"=CF!{ic}{row}", num=num)
    summ.put("A43", "Net debt / EBITDA" + (" (excl. captive-finance debt)" if s.captive else ""))
    for c, ic in zip("CDEFG", "EFGHI"):
        debt = f"(BS!{ic}24+BS!{ic}25+BS!{ic}29)" + (f"*(1-{aa('captive_share')})" if s.captive else "")
        summ.put(f"{c}43", f"=({debt}-BS!{ic}6-BS!{ic}7)/IS!{ic}{IS['ebitda']}", num="x")
    summ.put("A45", "Model integrity check (BS balances and cash ties, all years)", b=True)
    summ.put("C45", '=IF(AND(MAX(BS!D43:K43)=0,MIN(BS!D43:K43)=0,MAX(CF!D42:K42)=0,MIN(CF!D42:K42)=0),"OK","ERROR")',
             b=True, fill="key")
    if s.framework != "reference":
        sv, rt = s.scenario_values or {}, s.rating or {}
        summ.band(47, "Decision: bear / base / bull, what the price implies, rating")
        summ.header(48, "", {"C": "Value / share", "D": "vs price", "E": "Probability"}, "H", "Basis", right=False)
        probs = sv.get("probs") or [0.25, 0.5, 0.25]
        for r, key, label, pr_ in [(49, "bear", "Bear", probs[0]), (50, "base", "Base", probs[1]),
                                   (51, "bull", "Bull", probs[2])]:
            summ.put(f"A{r}", label)
            v = (sv.get(key) or {}).get("fair_value")
            summ.put(f"C{r}", v if v is not None else ("=G19" if key == "base" else None), num="usd")
            summ.put(f"D{r}", f"=IF(N(C{r})=0,\"n/a\",C{r}/$C$5-1)", num="pct")
            summ.put(f"E{r}", pr_, num="pct", fill="input")
        summ.note("H49", "Each scenario recalculated end to end (DCF, SOTP, comps) with Assumptions!C4 = 1 / 2 / 3 "
                         "when the file was built; rebuild after changing drivers")
        rows = [(52, "Probability-weighted value", "=SUMPRODUCT(C49:C51,E49:E51)/SUM(E49:E51)", "usd", True),
                (53, "Expected 12-month total return (value grows at the cost of equity; the price closes a "
                     "third of the gap to value a year); alpha vs cost of equity in D",
                 "=(1+(C52/C5-1)/3)*(1+DCF!C9)-1", "pct", True),
                (54, "Up / down skew, probability-weighted (expected upside ÷ expected downside; a Long needs 2:1)",
                 '=IF(AND(C51>C5,C49<C5),(E49*MAX(C49-C5,0)+E50*MAX(C50-C5,0)+E51*MAX(C51-C5,0))/'
                 '(E49*MAX(C5-C49,0)+E50*MAX(C5-C50,0)+E51*MAX(C5-C51,0)),IF(C49>=C5,"no downside","n/a"))', "dec",
                 False),
                (55, "Implied IRR at the price (reverse DCF) vs WACC", "=DCF!C105", "pct", True),
                (56, "Years of above-terminal growth the price requires (model in D)", "=DCF!C108", "dec", False),
                (57, "Terminal operating margin the price requires (model in D)", "=DCF!C107", "pct", False),
                (58, "Terminal growth the price requires (model in D)", "=DCF!C106", "pct", False),
                (59, "Gap to probability-weighted value; IRR spread over WACC in D (the rule: a Long needs ≥ +15%, "
                     "≥ +2pt, the price inside bear–bull and probability-weighted skew ≥ 2:1; a Short ≤ –15% and ≤ "
                     "–2pt with a dated catalyst and bull ≤ +25%)", "=C52/C5-1", "pct", True)]
        for r, label, f, num, bold in rows:
            summ.put(f"A{r}", label, b=bold)
            summ.put(f"C{r}", f, num=num, b=bold, fill="key" if r in (52, 53, 55) else None)
        summ.put("D53", "=C53-DCF!C9", num="pct", b=True)
        summ.put("D55", "=DCF!C17", num="pct")
        summ.put("D56", "=DCF!D108", num="dec")
        summ.put("D57", "=DCF!D107", num="pct")
        summ.put("D58", "=DCF!D106", num="pct")
        summ.put("D59", '=IFERROR(IF(ISNUMBER(DCF!C105),DCF!C105,IF(LEFT(DCF!C105,1)=">",0.2,0.03))-DCF!C17,"n/a")',
                 num="pct", b=True)
        summ.put("A60", "Rating (rule)", b=True)
        summ.put("C60", rt.get("rating") or "Under review", b=True, fill="key")
        if rt.get("sizing"):  # sizing is not the rating (PM round 7)
            summ.put("D60", f"Sizing: {rt['sizing']}", b=rt.get("sizing") != "full")
        summ.note("H60", rt.get("why") or "Rated when the file is built: Long if the probability-weighted value is "
                                          "15%+ above the price (the model's error band), the price offers an IRR "
                                          "2pt+ above WACC and probability-weighted skew is 2:1; Short at –15% and "
                                          "–2pt with a dated "
                                          "catalyst and bull ≤ +25%; else Neutral; Under review while a gate fails "
                                          "or a warning / open check is not explained in the thesis")
        if rt.get("override"):
            summ.put("A61", "Analyst override of the rule (in the thesis)")
            summ.note("C61", rt["override"])
        # the decision in one number (PM round 6): the bear probability the price implies, and where the call flips,
        # with the bull's probability held; the peer discount is the variant view comps would assume closes
        pb_ = "(E51/SUM(E49:E51))"
        top = f"((1-{pb_})*C50+{pb_}*C51)"
        dec = [(62, "Bear probability the price implies (bull probability held; base takes the rest)",
                f'=IFERROR(MAX(0,MIN(1-{pb_},({top}-C5)/(C50-C49))),"n/a")', "pct", True),
               (63, "Bear probability at which the value is 15% above the price (Long side, in C) and 15% below it "
                    "(Short side, in D)",
                f'=IFERROR(IF(OR(({top}-1.15*C5)/(C50-C49)<0,({top}-1.15*C5)/(C50-C49)>1-{pb_}),"not reachable",'
                f'({top}-1.15*C5)/(C50-C49)),"n/a")', "pct", False),
               (64, "Discount to the peer median: P/E (C) and EV / EBITDA (D); valuing at peer multiples assumes it "
                    "closes", f'=IFERROR(C9/Comps!F{CP["med"]}-1,"n/a")', "pct", False),
               (65, "Three-year return, probability-weighted (bear / base / bull values compounded at the cost of "
                    "equity); clears the hurdle in D",
                "=IFERROR(((C52*(1+DCF!C9)^3-C27*(1+DCF!C9)^1.5+C27)/C5)^(1/C28)-1,\"n/a\")", "pct", True)]
        for r, label, f, num, bold in dec:
            summ.put(f"A{r}", label, b=bold)
            summ.put(f"C{r}", f, num=num, b=bold, fill="key" if r in (62, 65) else None)
        summ.put("D63", f'=IFERROR(IF(OR(({top}-0.85*C5)/(C50-C49)<0,({top}-0.85*C5)/(C50-C49)>1-{pb_}),'
                        f'"not reachable",({top}-0.85*C5)/(C50-C49)),"n/a")', num="pct")
        summ.put("D64", f'=IFERROR(C10/Comps!F{CP["ev_med"]}-1,"n/a")', num="pct")
        summ.put("D65", '=IF(ISNUMBER(C65),IF(C65>C30,"Yes","No"),"n/a")', b=True)
    summ.finish()

    # ================================================================ Thesis
    t = s.thesis
    th.title(f"Investment thesis — {s.stance_phrase} {s.short} (written as of {s.as_of:%b %-d, %Y})",
             "The view a buy-side analyst would put in front of the PM, with the numbers linked from the model.")
    tall = {}

    def para(r, text):
        text = live_text(text) if s.framework != "reference" else text
        if isinstance(text, str) and text.startswith("="):
            th.ws.merge_range(f"A{r}:D{r}", "", bk.fmt(wrap=True, valign="top"))
            th.put(f"A{r}", text, wrap=True, valign="top")
        else:
            th.ws.merge_range(f"A{r}:D{r}", text, bk.fmt(wrap=True, valign="top"))
        th.rows.add(r)
        tall[r] = 30.0

    row = 4
    if s.framework != "reference":
        # Part F: the seven questions a PM answers in two minutes, live-linked where the model has the number
        num_txt = lambda ref, fmt, alt="n/a": f'IF(ISNUMBER({ref}),TEXT({ref},"{fmt}"),{ref}&"")'  # noqa: E731
        q = [("1. What does the price imply?",
              f'="At ${s.price:,.2f} the price implies an IRR of "&{num_txt("DCF!C105", "0.0%")}&" vs a "&'
              f'{num_txt("DCF!C17", "0.0%")}&" WACC; equivalently "&IF(AND(ISNUMBER(DCF!C108),DCF!C108=0),'
              f'"no growth above terminal after {fc_lbl[-1]}",{num_txt("DCF!C108", "0.0")}&" years of '
              f'above-terminal growth after {fc_lbl[-1]}")&" (model: "&{num_txt("DCF!D108", "0")}&") or a "&'
              f'{num_txt("DCF!C107", "0.0%")}&" terminal operating margin (model: "&{num_txt("DCF!D107", "0.0%")}&")."',
              t.get("price_implies")),
             ("2. Where do we differ from consensus, and why?", None,
              t.get("variant_view") or "No variant view stated: the rating defaults to Neutral until there is one."),
             ("3. Bear / base / bull, probabilities, expected return and skew",
              f'="Bear $"&{num_txt("Summary!C49", "0.00")}&" / base $"&{num_txt("Summary!C50", "0.00")}&" / bull $"&'
              f'{num_txt("Summary!C51", "0.00")}&" at "&TEXT(Summary!E49,"0%")&" / "&TEXT(Summary!E50,"0%")&" / "&'
              f'TEXT(Summary!E51,"0%")&"; probability-weighted $"&{num_txt("Summary!C52", "0.00")}&" ("&'
              f'{num_txt("Summary!C59", "0.0%")}&" vs the price; implied IRR spread over WACC "&'
              f'{num_txt("Summary!D59", "0.0%")}&"), expected 12-month total return "&'
              f'{num_txt("Summary!C53", "0.0%")}&" with a third of the gap closing a year, "&IF(ISNUMBER(Summary!C54),'
              f'"up / down skew "&TEXT(Summary!C54,"0.0")&":1","bear and bull on the same side of the price (no skew)")'
              f'&"."',
              None),
             ("4. What will prove us right or wrong, and when?", None,
              "; ".join(t.get("catalysts") or t.get("monitor") or []) or "No dated catalyst stated."),
             ("5. The one risk that breaks the thesis", None,
              t.get("breaker") or ((t.get("risks") or ["—"])[0])),
             ("6. How confident: the two inputs value is most sensitive to",
              f'="Perpetuity DCF: WACC ±0.5pt moves it $"&{num_txt("DCF!E62", "0")}&"–$"&{num_txt("DCF!E60", "0")}&'
              f'"; terminal growth ±0.5pt moves it $"&{num_txt("DCF!D61", "0")}&"–$"&{num_txt("DCF!F61", "0")}&'
              f'"; exit multiple ±1x moves the exit DCF $"&{num_txt("DCF!D70", "0")}&"–$"&{num_txt("DCF!F70", "0")}&"."',
              t.get("sensitivities")),
             ("7. Rating, and how it follows from 1–6", None,
              f"{(s.rating or {}).get('rating', 'Under review')}: {(s.rating or {}).get('why', '')}" +
              (f" Override: {s.rating['override']}" if (s.rating or {}).get("override") else ""))]
        th.band(row, "The seven questions (critique Part F)")
        row += 1
        for title, formula, text in q:
            th.put(f"A{row}", title, b=True)
            row += 1
            if formula:
                th.ws.merge_range(f"A{row}:D{row}", "", bk.fmt(wrap=True, valign="top"))
                th.put(f"A{row}", formula, wrap=True, valign="top")
                th.rows.add(row)
                tall[row] = 30.0
                row += 1
            if text:
                para(row, text)
                row += 1
        row += 1
    th.band(row, "Recommendation")
    para(row + 1, t["recommendation"])
    head_ = ([(row + 2, "Probability-weighted value ($)", "=Summary!C52", "usd"),
              (row + 3, "vs the price", "=Summary!C59", "pct"),
              (row + 4, "Bear probability the price implies (ours in Summary E49)", "=Summary!C62", "pct"),
              (row + 5, "3-year annualized return, probability-weighted", "=Summary!C65", "pct"),
              (row + 6, "Base case: weighted fair value ($)", "=Summary!G19", "usd")]
             if s.framework != "reference" else
             [(row + 2, "Weighted fair value ($)", "=Summary!G19", "usd"),
              (row + 3, "12-month upside", "=Summary!G20", "pct"),
              (row + 4, "3-year annualized return", "=Summary!C29", "pct"),
              (row + 5, "DCF (perpetuity) value", "=DCF!C50", "usd"),
              (row + 6, "SOTP base value", f"=SOTP!I{SO['ps']}", "usd")])
    for r, label, f, num in head_:
        th.put(f"A{r}", label, b=True)
        th.put(f"B{r}", f, num=num, b=True, fill="key")
    row += 8
    for title, items in [(f"Thesis in {len(t['points'])} points", t["points"]),
                         ("What the model says (be honest with the PM)", t["model_says"]),
                         ("Key risks / what would make us wrong", t["risks"]),
                         ("What to monitor", t["monitor"])]:
        th.band(row, title)
        for k, text in enumerate(items):
            para(row + 1 + k, text)
        row += len(items) + 2
    ex = t.get("explanations") or {}
    if ex:
        th.band(row, "Warnings and open checks: how each is resolved")
        for k, (key, why) in enumerate(ex.items()):
            para(row + 1 + k, f"{key}: {why}")
        row += len(ex) + 2
    th.band(row, "About this draft")
    para(row + 1, t["about"])
    th.finish(tall)

    # summary sheet first-visible
    cover.ws.activate()
    return bk.close()
