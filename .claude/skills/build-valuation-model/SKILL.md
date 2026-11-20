---
name: build-valuation-model
description: Build a buy-side Excel valuation model (HON template) for any US-listed ticker and valuation date, then act as the analyst — research the filings, earnings releases and peers, write the overrides JSON (scenario paths, M&A, guidance, debt-like items, SOTP splits, peer guidance, thesis) and rebuild until the checks pass. Use when asked to value a company, build/upgrade a valuation model, or make the HON-quality version of a generated model.
---

# Build a valuation model as the analyst agent

The app's pipeline turns SEC XBRL + market data into a balanced 13-tab model with mechanical
defaults. Your job is the judgment layer that makes it HON-quality, expressed as an overrides
JSON. `samples/HON_2025-08-29_overrides.json` is the gold standard: applied to HON as of
2025-08-29 it reproduces `HON_Valuation_Model_2025-08-29.xlsx` to the cent. Match that depth.

Run everything from the repo root with the project's Python (the one that has xlsxwriter,
yfinance and openpyxl; `pip install formulas` enables `--check`). Set
`SEC_USER_AGENT="Name email"` in the environment.

## 1. Brief

```bash
python -m valuation brief TICKER --as-of YYYY-MM-DD [--peers A,B,C]
```

Writes `TICKER_brief_DATE.md` (facts parsed from the filings, source links, data notes,
checklist) and `TICKER_DATE_overrides.json` (current assumptions in override format). Read
the brief fully, especially **Data notes**: they name problems you must resolve (perimeter
changes, resegmentation, splits, consensus mismatches).

## 2. Research (point-in-time: nothing published after the valuation date)

- Company documents: the URLs in the brief. Read them as text with
  `python -m valuation doc URL --grep "guidance|outlook|organic|acquisition"` (SEC needs the
  User-Agent this tool sends; WebFetch is often blocked by sec.gov).
- Other filings before the date: `python -m valuation releases TICKER --as-of DATE -n 6`
  lists earnings releases; the EDGAR link in the brief lists 8-Ks (deals: items 1.01 / 2.01 /
  8.01), Form 10s for spin-offs, investor-day decks.
- Peers: for each, `python -m valuation releases PEER --as-of DATE -n 1`, then read its
  full-year adjusted EPS guidance with `doc --grep "EPS"`. Price on the date: the brief's
  peers already carry it (Yahoo close, unadjusted for later splits). Use WebSearch only for
  facts the filings lack (bond yields, rating, historical multiple ranges), and say so in the
  note.

## 3. Write the overrides JSON

Edit the exported JSON (keep keys you don't change, or drop them). Work through the brief's
checklist. Rules that separate a PM-grade model from a mechanical one:

- Every changed number gets a `note` citing document + figure ("Q2 2025 release: $5.60–5.80").
- Year 1 base case sits inside guidance through operating drivers only: organic growth, segment
  margin, price / mix, working-capital days, share-count timing. Never tax, other income or a
  generic "other" line (the Checks tab plug table fails a non-operating plug, and FY2 must not
  jump because a plug unwinds). The Interim tab's "Within range?" column must read Yes.
- Adjusted EPS definition contract: set `adj_eps_ex_sbc`, `adj_eps_ex_repo` (share of non-segment
  charges), `adj_eps_ex_nonrec` (0 when the company includes one-off gains, e.g. CRM's investment
  gains) and `adj_eps_amort_share` so history reproduces reported adjusted EPS within $0.05.
- The forecast opens from the latest 10-Q: debt raised, buybacks (ASR) and acquisitions already done
  this year belong in FY1; first-year interest within ±15% of the annualized run-rate.
- Bear/base/bull differ for a reason you can state in `notes.growth` / `notes.margin`.
- Announced deals: growth contribution in `mna` (annualize in year 2), cash in `drivers.acq`,
  funding in `drivers.debt_iss`. Pending spin-offs: `sotp_split` with the Form 10 EBITDA.
  `drivers.acq` is cash paid for business combinations (>= 0): an asset deal expensed as acquired IPR&D is
  already a charge (`repo_pct`), never an acquisition too. Repayment cannot exceed the debt outstanding, and
  buybacks must be fundable: the engine caps both and writes a data note when it does.
- Cost of capital is centralized: you cannot change the risk-free rate or the ERP; beta moves at most ±0.10
  from the method value (with a reason); never set an input by reference to the share price.
- Terminal value is NOPAT × (1 − g ÷ RONIC) ÷ (WACC − g): `valuation.ronic` defaults to the last forecast
  year's ROIC including goodwill (DCF C93); set it lower only for a dated sector reason (loss of
  exclusivity with product and year, competitive entry, recurring acquisitions not otherwise modeled)
  under `thesis.explanations["gate 34"]` with the year, about WACC when commoditizing. The exit multiple
  is derived from the same framework; if you set one, keep it within ±20% of the perpetuity-implied
  multiple. Keep the terminal margin inside the 10–90th percentile of 10-year history (cyclicals at
  mid-cycle) unless the thesis argues a structural change.
- Signed pending deals of any size (a small one still moves value by its premium; gate 32): a `deals`
  item (probability, consideration, the target's own revenue and margin or EBITDA and a peer multiple,
  synergies, integration cost, break fee), not segment growth. Acquired revenue is never free: a price
  that is not disclosed is estimated (revenue x a peer EV / sales multiple) and said so. Separations:
  forecast the remaining company and add a `deals` item of kind "separation".
- Working-capital days and ratios are year-end balances: every forecast year ends on the fiscal year-end,
  so they start from the 10-K's year-end level, never the 10-Q's seasonal one (gate 12 names the line).
- Capital return: the engine returns cash above FY0's share of sales from FY2 (buybacks; debt repayment
  above 2.5x net debt / EBITDA). To hold the cash for a deal, explain under
  `thesis.explanations["cash build"]`.
- Thesis: answer the seven PM questions — `price_implies`, `variant_view` (we differ on X because Y; we
  will know by Z), scenarios on the thesis-critical variables (`scenario_probs`), dated `catalysts`,
  the `breaker`, `sensitivities`, `consensus_notes`. The rating is computed by rule on alpha (expected
  return with value growing at the cost of equity, less ke); override only with `thesis.rating_override`
  {rating, reason}. The rule (PM round 3): Long only at a probability-weighted value 15%+ above the price AND a
  long-run IRR 2pt+ above WACC with 2:1 skew and the price inside bear–bull; Short mirror-image with a dated
  catalyst, `thesis.short_evidence` (when the premium last compressed, on what), a crowding check and, for a
  low-volatility compounder, `thesis.pair`; else Neutral with a bias. The recommendation's call is generated
  from the rule: write `thesis.recommendation` as one supporting sentence with no call in it, and state no
  call or bias anywhere that the rule does not make (gate 25). Build the bull from what the reverse DCF says
  the price requires (gate 41: the price inside bear–bull). A claimed fix must be in the data (gate 42).
  Base case at consensus unless the variant view says why and when it is tested (a management target is the
  bull); the bear is the breaker made numerical, 15%+ below base (gate 33); acquired revenue annualizes as mna,
  not organic (gate 36); RONIC at or below the return including goodwill when growth was bought (gate 34);
  state debt-funded buybacks (gate 39). Republishing after a critique: build with `--previous OLD_overrides.json`
  so the Checks tab bridges value version to version; never raise value on a judgment input without new
  evidence. Write model numbers as placeholders (`{fv}`, `{dcf_pg}`, `{comps}`, `{bear}`, `{irr}` (long-run IRR,
  not a 3-year return), `{ret3y}`, `{gap}`, `{irr_spread}` ...),
  never typed (gate 28). Explain every warning gate and open check you leave in `thesis.explanations`
  under the key the Checks tab shows; an unexplained one holds the rating at Under review.
- After the balance sheet: enter or dismiss each 8-K the Checks tab lists (closings in `drivers.acq` above
  the year-to-date amount); an accelerated share repurchase settling after the 10-Q updates
  `market.basic_shares`. Pending acquisitions: give the target's `revenue` and `margin` (or `ebitda`).
- Margin guidance: set `guidance.margin_basis` (gaap / adjusted / segment). A GAAP operating-margin guide
  already includes restructuring: do not also deduct `repo_pct`. Never move a driver "to tie" a guide
  (gate 11 reads the notes); a working-capital or RONIC call is a thesis-level view stated in `variant_view`.
- Hidden value or liabilities: `valuation.other_stake` + `labels.other_stake`; `debt_like` +
  its labels.
- `seg_after_charges`: keep the pipeline's detection unless the segment note says otherwise.
- `adj_eps_ex_sbc = 1` when the company's non-GAAP EPS excludes stock comp (most software): never zero
  `sbc_pct` or other real costs to hit a non-GAAP number. One-off gains (investment gains, disposals) are
  non-recurring — keep them out of run-rate `other_nonop` and say so in `flags` if guidance includes them.
- The brief's "What the model computes" block shows FY1 sales / EPS / FCF vs guidance and every method's value
  before you start: use it to see which defaults are wrong.
- Model the business the way its industry is analysed — the brief's **Industry** section is the checklist
  (pharma: product lines, launch curves, loss of exclusivity; media: regions, members x ARM, content cash vs
  amortization; retail: units x sales per unit; software: clouds, cRPO, stock comp; captive-finance
  industrials: equipment vs finance arm; cyclicals: mid-cycle, not trough, in the outer years).
- Revenue lines: where the brief lists XBRL product / region lines, set `segments.<name>.lines.<line>.growth`
  (launch curves, patent expiries, regional pricing); lines you don't set follow the segment's path. Define
  lines yourself with `history` when a release discloses a split the XBRL lacks (must add up within 3%).
- KPIs: `segments.<name>.kpi` or `lines.<line>.kpi` = volume x price with three years of disclosed history
  (warehouses x sales per warehouse, members x ARM, production x realized price). The KPIs tab reconciles it
  to reported revenue; growth becomes (1 + volume g) x (1 + price g) - 1.
- Peers: give each peer's EPS with its source; prices are re-set to the market close on the valuation date
  (the pipeline replaces any price that is off by more than 2%).
- Margin basis: in a single-segment model segment profit = GAAP operating income + amortization + stock comp,
  so it still contains acquired IPR&D, restructuring and litigation. A company's non-GAAP margin guide is on a
  different basis; if you use it, put every excluded charge (including those booked this year) in repo_pct.
- The current year must fit the year to date (Interim tab, "Rest of year implied by the model"). One-offs
  already booked (stake gains, settlements, untagged other income) go in `drivers.nonrec` for FY1: in GAAP
  EPS, excluded from adjusted EPS.
- Long growth runways: `valuation.fade_years` (three-stage DCF) instead of a high terminal growth rate.
  Comps: `valuation.comps_pe_weight` (P/E vs EV / LTM EBITDA; peers' EBITDA comes from their filings).
  Captive finance: `valuation.captive_share` = finance-arm debt / total debt (10-K supplemental data).
- Thesis: specific, with numbers, honest about where the DCF disagrees; dated catalysts. The stance follows
  the valuation (long needs fair value above price or a 3-year return above the cost of equity).

## 4. Build and check, iterate

Shortcut: `python -m valuation build TICKER --ai --analyst --save-overrides --check` runs the in-app AI analyst
(OpenAI) first; then refine its `<out>_overrides.json` with your own research and rebuild with `--overrides`.

```bash
python -m valuation build TICKER --as-of DATE --peers ... --overrides TICKER_DATE_overrides.json --check
```

Fix what `--check` reports (integrity, Excel errors, methods far from price without a reason,
terminal value share, exit multiple vs perpetuity consistency, guidance misses, rest-of-year vs year to date,
stance vs valuation, methods that disagree, peer outliers, an implausible rest-of-year operating margin, full-year
other income below what is already booked, idle cash piling up, thesis figures that differ from the model,
balance-sheet lines turning negative, an exit multiple far below the peers', expensed IPR&D entered as an
acquisition). Rebuild until
clean or until each remaining item is a deliberate, explained call (put it in `flags`).

What the gates will not accept:
- **A refused input (gate 45).** It holds the rating until the field holds a value the guard accepts. Read the
  warnings `--check` prints.
- **Moved history.** `adj_eps_tax_hist` must stay within 3pt of the stated non-GAAP rate. If gate 1 fails, check
  the release's reconciliation line and the definition switches.
- **A higher interest rate for missing debt.** Gate 8 fails when interest is over twice the cost of debt on the
  debt read; find the notes instead.
- **A "fixed" explanation on a check that still fires.** It stays open.
- **A closed deal on the Deals tab.** A deal that closed before the balance sheet is already in net debt and the
  forecast.
- **The same risk in several places.** Disruption or competition goes in the bear and its probability. Beta,
  terminal growth (the house rate) and fade years don't move for it, and the base stays at consensus unless a
  distinct variant view says otherwise.
- **An input or explanation that argues from the output** (the skew, the gap, the rating).
- **Comps at peer multiples for a company 30%+ below them** (gate 48: comps ≤ 10%, the discount is the variant
  view). Use `{p_bear_implied}` and `{p_bear_long}` to frame the call as one bear probability.
- **Share issuance below stock comp ÷ the repurchase price** (gate 46), or text that contradicts the model
  (gate 28).

## 5. Deliver

Report: the workbook path, weighted fair value vs price and the four methods, the 3-5
judgments that move value most (with sources), and anything unresolved. The user can also
upload the JSON in the web app (step 2 → Agent hand-off) to rebuild there.
