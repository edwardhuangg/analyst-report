# Valuation Model Builder

Enter a ticker and a valuation date. The app pulls the company's SEC filings and market data and produces a
buy-side Excel model in the layout of `HON_Valuation_Model_2025-08-29.xlsx`. The website only collects inputs;
the deliverable is the `.xlsx`.

Every formula's result is computed when the file is written (`formulas` library) and stored with the formula, so
the numbers show in any viewer — Excel, Quick Look, previews, Google Sheets — not only after Excel recalculates.
If that computation ever fails, the Cover says so in red ("open it in Excel before reading the numbers").

```bash
pip install -r requirements.txt          # includes `formulas`: computes every formula so the file stores real values
streamlit run streamlit_app.py
```

## Two layers: filings in one click, judgment from an analyst

A model like the HON reference is two things:

1. **The machine**: 3 years of reported history, interim/LTM, a segment build, three linked statements that
   balance every year, DCF (two terminal methods, sensitivity grids), comps, SOTP, a scenario switch. The
   pipeline builds this deterministically for any US 10-K filer, point-in-time as of the valuation date.
2. **The judgment**: why each segment grows and earns what it does in bear/base/bull, announced M&A and
   spin-offs, guidance, debt-like liabilities (asbestos), hidden assets (Quantinuum), how to split a segment
   that is about to be spun (Solstice), each peer's own guidance, the discount the market applies, the thesis.
   That comes from reading documents and thinking, so it lives in an **overrides JSON**
   (`valuation/overrides.py`) that any analyst can produce:
   - the in-app **AI analyst agent** (OpenAI, `OPENAI_ANALYST_MODEL`, default `gpt-5.5`): reads the releases,
     10-K/10-Q MD&A and 8-Ks, sets the assumptions, builds and recalculates the model, fixes what the checker
     flags (guidance misses, inconsistent multiples) and calibrates FCF to guidance, in ~2-4 minutes at low effort,
   - an **external agent**: download the research brief in step 2 → *Agent hand-off*, or run
     `python -m valuation brief`, let Claude Code / ChatGPT research and return the JSON, upload it,
   - a person editing the JSON or the blue cells in Excel.

`samples/HON_2025-08-29_overrides.json` is the gold standard: applied to HON as of 2025-08-29 it reproduces
the reference model's valuation to the cent (weighted fair value $231.76, DCF $222.27 / $225.64, SOTP $236.15).
Without overrides the same build gives the filing-based defaults ($231.7 weighted, with mechanical drivers).

### Claude Code as the analyst

The repo ships a Claude Code skill (`.claude/skills/build-valuation-model`). In Claude Code, ask e.g.
*"build the valuation model for NFLX as of today"*: it writes the brief, reads the 10-K/10-Q/releases and peer
releases, writes the overrides with sourced notes, builds, runs the checks and iterates.

## Valuation framework (economic, the default)

Built to the buy-side review standard (the CRM / DE / ETN / GOOGL / NFLX / ROK / SHW / COST critique):

- **Cost of capital, set once per valuation date** (`valuation/capital.py`): the 10-yr Treasury on the date, one
  ERP for every name (the published start-of-year implied ERP, or `FA_ERP`), the filer's agency rating (read from
  the 10-Q / 10-K text: middle of three, lower of two) on one credit-spread table for the cost of debt — a synthetic
  rating from interest coverage only when the filings name none — and beta by method (bottom-up peer beta relevered at the company's D/E, averaged
  with its own Blume-adjusted beta). The AI analyst cannot change the risk-free rate or the ERP; beta moves at
  most ±0.10 and the cost of debt ±0.75pt, with a reason. Review passes that cite the share price are rejected.
- **Terminal value on economics**: NOPAT × (1 − g ÷ RONIC) ÷ (WACC − g), with RONIC an input (default: the last
  forecast year's return on invested capital including goodwill, DCF C93, floor WACC, cap 60%; about WACC for
  commoditizing industries; set lower only for a dated sector reason, gate 34). The fade stage grows NOPAT and
  reinvests g ÷ RONIC, with RONIC moving in equal steps from the return the last forecast year implies to the
  terminal RONIC, so free cash flow does not drop into the fade (gate 27). The exit multiple is derived from the same framework unless set, and a gate requires
  the two terminal methods to reconcile (±20% multiple, ±1pt growth). The DCF tab prints a terminal-year sanity
  table against 10 years of margin history.
- **What the price implies** (reverse DCF, DCF rows 103–108): implied IRR, terminal growth, terminal margin and
  years of above-terminal growth the price requires.
- **One discount framework**: the 3-year test runs exactly 3.0 years and derives its exit P/E from the model's
  own value at the exit date (the de-rating it implies is stated), next to the return at today's P/E and at the
  peer P/E. Bear / base / bull are recalculated end to end with probabilities and skew. The **rating is a rule**
  (PM round 3): Long only when the probability-weighted value is 15%+ above the price (the model's error band)
  AND the price offers a long-run IRR (reverse DCF) 2pt+ above WACC, with 2:1 skew and the price inside
  bear–bull; Short at −15% and −2pt with a dated catalyst, bull ≤ +25%, written evidence that the gap closes, a
  crowding check and, for a low-volatility compounder, a pair; else Neutral, with a positive / negative bias when
  the value is 5%+ away. The recommendation line is generated from the rule (an override is labeled and logged). The expected 12-month return lets value grow at the cost of equity and assumes the price
  closes a third of the gap a year (not all of it). Under review while a gate fails or any warning / open check
  is not explained in `thesis.explanations`; an analyst override only in writing. Summary row 59 shows the gap
  and the IRR spread; placeholders `{gap}` and `{irr_spread}`.
- **Bear is a real bear** (gate 33): at least 15% below base, its growth about one standard deviation of the
  company's 10-year revenue growth below base, built on the thesis breaker; skew above 3:1 needs a reason.
- **RONIC when growth was bought** (gate 34, guard `cap_ronic`): when goodwill and acquired intangibles are 20%+
  of invested capital or acquisitions ran 2%+ of sales over 10 years, terminal RONIC is held at the return
  including goodwill (DCF C93), floor WACC, unless recurring acquisition spend is forecast.
- **Base case at consensus** (gate 35): FY1 / FY2 revenue growth more than 0.75pt (or EPS 3%) from consensus, or
  growth notes anchored to a management target, needs a variant view; targets belong in the bull case.
- **Organic means organic** (gate 36): acquisitions paid in FY0 annualize as M&A in FY1, estimated at the
  company's own EV / sales when the target's revenue is not given.
- **Comps sanity** (gate 37): peer next-year growth from one source (consensus +1y ÷ 0y); the P/E regression on
  EPS growth and the EV / EBITDA regression on revenue growth are used only with 6+ peers, a positive slope and
  R² ≥ 0.2, else median PEG / median.
- **Deal inputs agree** (gate 38): standalone value from the price and premium vs EBITDA × multiple within 50%.
  **Debt-funded buybacks are stated** (gate 39): buybacks above FCF with net debt / EBITDA above 1x.
- **Gates that passed on paper now check the data**: a consensus snapshot after the date counts as point in time
  only if the peer filed no earnings 8-K in between (gate 5), and a "guidance" EPS equal to consensus is labeled
  consensus; a pending or undisclosed-price deal in the M&A note fails the "already paid" test (gate 18) and
  each named pending deal must be on the Deals tab (gate 32); a plug-table warning over 5% of revenue needs its
  own explanation with its number.
- **Found in the round-4 sector runs (all companies):**
  - **Perimeter (gate 40, FAIL):** a spin-off after the 10-K, or consensus implying a perimeter change, holds the model until it is rebuilt on the remaining company. HON was valued on its 10-K perimeter (Aerospace included) against the post-spin price: +105%.
  - **Scenarios in order (gate 33, FAIL):** bear ≤ base ≤ bull. NEE's bull was worth less than its base.
  - **Bear / bull spreads:** default spreads are ±0.75σ of 10-year revenue growth, 2–5pt, half that in FY1.
  - **Fade continuity:** when the last forecast year's reinvestment is negative (deferred-revenue float), the fade's reinvestment rate steps from that level to g ÷ RONIC, instead of jumping from an implied 100% RONIC (CRM's default build lost 9% of FCF into the fade).
  - **The non-GAAP tax rate is a tax line:** it drives operating EPS and so every multiple. A rate solved onto the EPS guide, or more than 1.5pt off the company's stated history, fails gate 11 (CRM: 24.3% "needed to reconcile to the guide").
  - **Deals:**
    - Synergies are capitalized at the lower of the deal multiple and the acquirer's own EV / EBITDA.
    - A target margin "set so the cross-check ties", or an acquisition with no price, is flagged under gate 38.
    - The Deals tab counts as judgment in the version bridge.
  - **Analyst reading pack:** it now includes the 10-Q / 10-K notes on business combinations, pending deals and share repurchases. CRM's 10-Q disclosed Contentful ($1.5bn) and Fin ($3.6bn), which the MD&A-only pack missed.
  - **AI analyst path:**
    - The analyst's peers get betas, splits and the point-in-time check (`fill_peer_ev`), as in the app.
    - An AI beta is held within ±0.10 of the method beta for the final peer set; an echoed method beta is not a judgment.
  - **Gate 8** benchmarks first-year interest against the year to date plus the rest of the year on the 10-Q debt, when debt was raised mid-year.
  - **Comps growth:** the company's growth is the two-year operating-EPS CAGR, not one year's (one-offs).
  - **Round trip:** `--save-overrides --check` rebuilds the saved JSON and prints MISMATCH unless it reproduces the fair value. The export now carries every driver (`drivers.adj_tax` had been dropped).
  - **Explanation keys** match on words, not numbers, so an explanation survives a number moving between builds.
- **Found in the round-4 PM review (all companies):**
  - **The text says what the rule says (gate 25):** the recommendation's call (Long / Short / Neutral and bias) is
    generated from the rule and prefixed to the analyst's sentence. A call stated anywhere in the thesis that the
    rule does not make ("Long, sized for…", "Underweight", "warrants ownership", a positive bias under a negative
    rule) holds the rating; an analyst override is labeled "Override" and logged on the Checks tab.
  - **The price sits inside bear–bull (gate 41):** a bull below the price of a liquid large cap means the scenarios
    are miscalibrated. At build, such a bull (or a bear above the price) is rebuilt from the reverse DCF: its
    margins move toward the terminal margin the price requires, scaled so it lands just past the price, with a note
    when that margin is beyond the 10-year high. A Long needs the price inside the range too (no "no downside").
  - **Shorts need more than a gap:** a dated catalyst, `thesis.short_evidence` (when the premium last compressed,
    on what), a crowding check (Yahoo short interest; 10% of float or 7 days to cover needs a plan) and, for a
    low-volatility compounder (beta < 0.85, revenue σ < 6%, ROIC > 15%), `thesis.pair`. Otherwise Neutral with a
    negative bias.
  - **RONIC defaults to ROIC including goodwill** (guard `default_ronic`), the fade stepping to it from the
    forecast's own incremental return; set lower only with a dated sector reason under 'gate 34' (two-sided).
  - **Missing data fails:** 10-Q debt read as zero against 10-K debt fails gate 7; no year-to-date interest needs
    the 10-Q figure quoted (gate 8). Filers that tag lines only by business (CAT: Machinery vs Financial Products
    columns) are summed across the business axis, and the finance arm's share of debt comes from those columns.
  - **Claims match the data (gate 42):** an explanation that says a peer's EPS was fixed or excluded is checked
    against the Comps table.
  - **Peers:** next-year EPS falling 20%+ (one-offs) or rising 100%+ (a charge), or a P/E under 0.4x / over 2.5x
    the others' median (a basis error), is excluded from the P/E statistics; for a cyclical, no growth regression
    (next-year growth measures the cycle); rebounds above 40% stay out of any fit.
  - **Symmetric mid-cycle:** cyclicals fade to the 10-year median margin from a peak as from a trough (defaults,
    comps EPS basis, gate 16); home-improvement retail is a housing cyclical.
  - **Fade reinvestment counts every line** between NOPAT and UFCF (NFLX's content spend above amortization sits in
    other operating items); gate 27 is two-sided (a jump above the fade year's growth + 5pt fails).
  - **Synergies:** 50% credit, run-rate in year 3, at the lower of the deal and acquirer multiples, integration
    costs after tax; every deal value-creating is flagged (gate 38).
  - **The rest of FY1 (gate 43)** — forecast less year to date — must look like last year's (LLY: an implied −$2bn
    Q4 free cash flow); **KPI builds tie** to modeled revenue within 1% (gate 44); the excess-cash policy is held
    off only by a named use of the cash.
  - **PM notes on a review:** `--review --notes critique.md` gives the review pass the PM's critique and the
    reading pack, answered in `thesis.explanations['pm review']`.
  - **Credit ratings:** the synthetic (coverage) rating tops out at AA; most filers name no agency, so an analyst
    `credit_rating` {"S&P", "Moody's", "source"} sets the spread, and only market evidence (a bond's coupon or
    yield) moves the cost of debt off it.
  - **Saved market data:** the overrides carry `market_data` (own beta, own P/E history, risk-free rate, short
    interest); a rebuild uses it when a Yahoo fetch comes back empty, so a rate limit cannot move the value.
  - **Peer EPS conflicts:** an analyst EPS far from consensus is kept when its P/E sits nearer the other peers'
    (NVO's DKK consensus vs a USD ADR price), and a normalized EPS re-bases the peer's growth (GOOGL ex-gains).
  - **Organic means organic:** release extraction leaves growth that includes acquired revenue out of the organic
    guide; a KPI build is compared with revenue net of the M&A contribution it does not cover (gate 44).
  - **AI time limits** count only the time the machine is awake (a sleeping laptop no longer cancels a paid
    response); a calibration note that is negated ("rather than calibrated") is not a plug.
- **Version bridge** (`--previous PREV_overrides.json`): the new version against the one the PM last read, each
  group of changes valued in turn on the Checks tab (judgment vs data). A rating change carried mainly by
  judgment inputs holds the rating until an independent AI reviewer keeps or reverts each change
  (`thesis.explanations['independent review']`).
- **Opening balance sheet**: cash flows count from the latest 10-Q (seasonal share of FY1); FY1 debt, buybacks
  and acquisitions already done this year are in FY1, and FY1 interest runs at the 10-Q's annualized rate when
  debt was raised or repaid during the year; the share count opens from the 10-Q (year-to-date weighted
  average, then today's count less the rest of the year's buybacks); interest income is earned on cash net of
  distributions not funded by debt; acquisitions whose growth is forecast are paid out of value; long-term
  investments on the latest balance sheet (CRM's strategic investments) are in every equity bridge after the
  haircut, and gate 19 checks the balance-sheet line, not the input. 8-Ks filed after the balance sheet and an
  unsettled accelerated share repurchase are listed for the analyst to enter or dismiss (gate 10).
- **Reading the filings (data audit of 20 fresh names, 2026-11-15)**: share counts tagged on the wrong scale are
  restated from net income ÷ EPS (MCD); operating income comes from pre-tax income when cost lines are missing
  (SLB); segments whose "profit" is after interest and tax fall back to one segment on operating income (DUK);
  capex comes from the other-productive-assets tag (VZ), the company's own tags (NEE: the roll-forward decides
  whether one tag is the total of the others) or the PP&E roll-forward, for every history year; interest
  expense is gross and positive, never net of interest income that is also tagged (DIS, NKE, NEE). Interest
  income counts the year's cash generation. The cash floor runs on every build (buybacks cut, then new debt
  where dividends or capex exceed cash generation). A method at or below zero gets no weight (gate 29); a
  first-year margin more than 5pt from last year's needs a reason (gate 30); cyclicals return to the 10-year
  median margin by the last forecast year; hyper-growth halves its excess growth each year; the sum of the parts
  deducts operating costs that sit outside segment profit (MRK's R&D); regulated utilities lean on P/E vs peers,
  and a utility built without peers fails gate 31 (its DCF-only value is not a valuation).
- **Terminal value on the right earnings**: the fade and terminal value start from NOPAT before acquired-
  intangible amortization (it is non-cash and runs off; the explicit years add it back), terminal growth never
  exceeds the forecast's last-year growth, and default growth fades toward the house terminal rate.
- **Capital return by policy**: from the second forecast year, cash above the company's FY0 share of sales is
  returned as buybacks, or repays debt above 2.5x net debt / EBITDA (VZ). Idle cash no longer earns its way into EPS
  (the defaults had AVGO's cash at $792bn by FY6). An analyst who holds cash for a deal says why under
  `thesis.explanations["cash build"]`.
- **Deals of any size**: a signed pending deal named anywhere in the notes but missing from the Deals tab fails
  to clear gate 32; acquired revenue down to 0.2% of sales needs its price (gate 18); gate 12 names the cash-flow
  lines behind an FCF drop (CRM: receivables days dipped in FY1 on a seasonal argument applied to a year-end).
- **Peer splits are not spin-offs**: a stock split keeps the peer's EV / EBITDA (share count restated if it
  predates the split); spin-offs and mergers are still excluded until pro forma.
- **PP&E rises with growth**: depreciation that runs PP&E down is capped at what the PP&E roll-forward shows
  (beginning + capex – ending; the rest of D&A amortizes leases and capitalized costs), and default capex is at
  least 1.1x depreciation.
- **No plugs**: guidance is met through operating drivers (organic growth, working-capital days, the engine's
  step fading out by the fourth year); tax, other income and generic "other" lines are flagged, and so is any
  driver whose own note says it was set to hit a guide; working-capital paths moved away from the defaults show
  their cash cost; FY2 continuity covers EPS and free cash flow. Margin guidance is compared like for like
  (GAAP, adjusted or segment, `guidance.margin_basis`).
- **Earnings definition contract**: four switches (stock comp, share of non-segment charges, one-off gains,
  share of amortization) reproduce the company's adjusted EPS; history must tie within $0.05. Multiples use
  operating EPS (without one-off gains).
- **Deals tab**: pending acquisitions and separations with probability, price, the target's own revenue and
  margin (without them: the price less a 30% control premium, never zero), synergies, integration and
  stranded costs, break fee — probability-weighted into fair value.
- **Comps and SOTP**: growth-adjusted peer P/E (regression or PEG) blended with the company's own 10-year P/E
  band (only where GAAP EPS is comparable with adjusted EPS), mid-cycle EPS for cyclicals, peers with a spin or
  split excluded from EV/EBITDA, trailing GAAP peer EPS kept out of the forward median, look-ahead flagged;
  corporate cost capitalized at the blended segment multiple, segment EBITDA after stock comp.
- **Checks tab**: the pre-publication gates (1–51) (definitions, data integrity, balance sheet and timing, plugs,
  valuation integrity, decision layer, FCF continuity into the fade, thesis numbers), the open warnings and
  checks with their explanations, consensus comparison, a ten-number final check, release figures verified
  against the release text, and an audit trail of every change with the fair value after each pass and the
  fair-value effect of each group of overrides (RONIC, working capital, stakes, margins...).
- **Thesis numbers are live**: thesis text uses placeholders ({fv}, {dcf_pg}, {comps}, {irr}, {bear} ...) that
  the file shows as linked numbers; a typed valuation figure that no longer matches fails gate 28.
- **Universe check**: `python -m valuation calibrate models/*.xlsx` compares the IRRs the prices imply with the
  costs of equity across built models. The ERP stays the published market-wide figure (Damodaran's start-of-year
  implied ERP, against the 10-yr Treasury); a median IRR well below the median cost of equity points at the
  terminal framework or the forecasts, never at backing the ERP out of the covered names' own prices.

- **Inputs on their own basis (round 6, the first website run: ADBE)**:
  - A margin guide moves margins by its change on its own basis (segment, adjusted operating or GAAP; the
    analyst's `guidance.margin_basis`, else the nearest). ADBE's 45% operating guide had cut gross-profit segments
    by 44pt.
  - A company whose latest 10-Q reports one reportable segment is built as one segment, with the 10-K segments
    as revenue lines.
  - `us-gaap:LongTermDebt` is read when it is the only long-term debt tag (ADBE's notes). Gate 8 fails when
    interest ÷ debt is over twice the cost of debt.
  - A guide in last year's adjusted-EPS slot is rejected, and five-column reconciliation lines are read.
  - Peer EPS is calendarized to the company's year (MSFT's June year against a November year). A peer whose EPS
    falls next year while revenue grows is a one-off.
- **Refusals are visible**:
  - An analyst input a guard refuses fails gate 45 until the field changes. The refusal is carried in the
    overrides file under `rejected`.
  - History is locked: a historical non-GAAP tax rate more than 3pt from the stated rate is refused.
  - An explanation that claims a fix on a check that still fires stays open.
  - A deal closed before the balance sheet is not a Deals-tab item.
  - RONIC above the 60% cap is held at the cap, which is also gate 34's reference.
  - Analyst passes that move fair value 30%+ from the defaults while gates fail have their override groups tried
    at the defaults. A group whose revert clears failed gates is reverted and flagged.
  - The website keeps every peer field through its table, keeps the engine's scenario paths, and changes the
    model only on an edit.
  - Betas use month-end closes from daily prices up to the valuation date. Yahoo's monthly bar for that month
    closes at the latest price: a look-ahead.
- **Each risk counted once, and the decision in one number (round 7, ADBE)**:
  - **Scenario risk lives in the bear and its probability.**
    - Terminal growth stays at the house rate for the AI analyst.
    - Beta and fade years move only for a measurement reason (leverage, mix), not for disruption or competition.
    - The bear's terminal growth is its own last-year growth, floored at 0% (DCF C74). A declining bear no longer
      re-accelerates to GDP, and fade reinvestment is never negative.
  - **No output targeting.** An input or explanation whose reason is the skew, the gap or the rating is rejected.
  - **One-number framing on the Summary:**
    - the bear probability the price implies (row 62);
    - where the call changes (row 63);
    - the discount to the peer median (row 64);
    - a probability-weighted three-year return (row 65).
  - **Comps:**
    - The PEG runs on operating net income growth, so buybacks are not growth.
    - Peers whose EBITDA margin is under half the company's leave the EV/EBITDA median.
    - Gate 48 asks for comps at 10% or less when the company trades 30%+ below the peers.
  - **Dilution:** shares issued default to stock comp ÷ the repurchase price; gate 46 flags less than half.
  - **Facts:**
    - First-year acquisitions above the cash paid need a named later closing (AI inputs).
    - Gate 18 accepts the price disclosed in the 10-Q.
    - Year-to-date adjusted EPS follows the current definition, or the full-year guide less the guided last
      quarter (gate 47).
  - **Text matches data (gate 28):** typed rates, premium/discount claims, buybacks "within FCF", unrendered
    placeholders.
  - **Gate 45** also records refused M&A and malformed inputs.
  - **Audit trail:** it travels in the overrides file and ends with a row reconciled to the published value.
  - **Website:**
    - An optional previous-version upload bridges to the version the PM last read. When judgment flips the
      rating, an independent AI review challenges each change.
    - Step 3 also downloads the assumptions JSON for next time.
- **The written rule decides the rating (round 8)**:
  - A Long needs four things: the probability-weighted value 15%+ above the price, an IRR 2pt+ over WACC, the
    price inside bear–bull, and probability-weighted skew of 2:1 (expected upside ÷ expected downside).
  - A likely, deep bear (25%+ chance of −40% or worse) is binary sizing, not a different rating: "Long, sized as a
    binary".
  - The Cover, Thesis and app lead with the probability-weighted value.
  - Re-dating: an overrides file applied on another date takes that date's prices for the company and its peers.
  - **Comps and EPS:**
    - The PEG puts peers on net-income growth (consensus EPS growth with their share-count change).
    - The EV/EBITDA leg is dropped when it is 1.5x away from the P/E leg.
    - Year-to-date adjusted EPS is the sum of the quarters the company reported, each from its own release, so
      gate 47 tests it independently.
  - Quarter margins replace partly tagged stock comp.
  - The D&A check excludes capitalized contract-cost amortization, which is matched by new contract costs.
- **Sector paths from the first hosted runs (round 9, MRK and DE)**:
  - **Captive finance:**
    - A captive arm's debt is operating in invested capital too (C93), as it is in net debt.
    - The cash policy's leverage test excludes it.
    - Gate 49 fails when RONIC sits under WACC, because growth then destroys value and the capital base is wrong.
  - **Mid-cycle:** cyclicals use the last five years' median margin when it is more than 3pt from the five
    before (a structural shift); otherwise the 10-year median. The defaults, gate 16 and the comps use the same
    helper.
  - **Scenarios:**
    - The engine no longer manufactures a scenario: when the price needs a terminal margin more than 5pt outside
      the 10-year range, the scenarios are left as built and gate 41 fails.
    - Gate 51 asks for a stated basis for the scenario probabilities.
  - **Pharma cliff:** a decline still running in FY6 is finished in the fade (`fade_path`, growth by fade year),
    and terminal growth stays at the house rate (gate 50).
  - **Guidance:**
    - An EPS guide far below last year stands when the release says charges are in it.
    - A guide in billions is net income (a sixth Interim check row).
    - The stated non-GAAP tax-rate reader covers "effective" and skips outlook ranges.
    - With no stated rate, history stays within 8pt of the GAAP rate and at least 8%.
  - **Refusals:** every engine floor that changes an analyst input fails gate 45, as do refused guidance and an
    input bent to fit another.
  - **Acquisitions:** cash expensed as acquired IPR&D stays out of `drivers.acq` when the analyst's note says so.
  - **Cost of debt:** the synthetic rating uses the higher of LTM and FY0 EBIT.
  - **Consensus:** retried once, with FY2 used when FY1 has ended unreported. A missing consensus for a $10bn+
    company is a data failure (gate 24).

`framework: "reference"` in an overrides file reproduces the mechanics of the hand-built HON model (used by the
regression test); everything else uses the framework above.

## Command line

```bash
python -m valuation build HON --as-of 2025-08-29 --peers GE,RTX,ETN,EMR,ROK,JCI \
    --overrides samples/HON_2025-08-29_overrides.json --check
python -m valuation brief NFLX                       # research brief + current assumptions JSON for an agent
python -m valuation releases GE --as-of 2025-08-29   # a company's earnings releases before a date
python -m valuation doc <sec.gov URL> --grep "guidance|outlook"
python -m valuation check model.xlsx                 # recalculate + sanity checks (needs `formulas`)
```

`--ai` uses OpenAI to read releases, pick peers and write the thesis; `--analyst` adds the analyst pass.
`--overrides X --review` runs only the AI review passes on a saved (or PM-edited) overrides file, with no new
analyst pass (about a third of the cost). `--previous OLD_overrides.json` (with `--check`) adds the version
bridge to the Checks tab; with `--ai`, a judgment-driven rating change also gets the independent review pass.

## What the workbook contains

Same 13 tabs, row order, colour coding (blue input / black formula / green link) and number formats as the
HON model. Every output is a live formula.

| Tab | Built from |
|---|---|
| Cover, Summary, Thesis | Links to the model; who set the assumptions; data notes; thesis |
| Assumptions | Bear / base / bull drivers by segment (scenario selector C4); C14 = segment profit already after amortization & stock comp |
| Interim | Latest 10-Q year-to-date, LTM, net debt, debt-like items, quarter segment results, FY guidance check |
| Segments, IS, BS, CF, Schedules | 10-K XBRL (3 years IS/CF, 2 years BS) + forecast; BS and cash checks = 0 every year |
| DCF, Comps, SOTP | WACC, two terminal methods, sensitivity grids; peer P/E; segment EV/EBITDA with optional sub-splits and stakes |

## Data sources (per ticker, point-in-time as of the chosen date)

- **10-K / latest 10-Q XBRL**: statements, segment note (incl. ASU 2023-07 aggregation members, gross-profit
  segment measures), loss-contingency accruals and recoveries, split ratios, discontinued operations.
- **Earnings-release 8-Ks** (EX-99.1): guidance, adjusted EPS, organic growth (read by AI when enabled).
- **Yahoo Finance**: the close that actually traded on the date (later split / spin-off back-adjustment undone),
  10-yr Treasury, 5-year monthly beta computed to the date. Consensus revenue/EPS only when the date is recent,
  and only for the fiscal year it refers to.
- **Peers**: price on the date; EPS = each peer's own current-FY consensus (recent dates), its guidance midpoint
  from its last release before the date (AI), or its LTM GAAP EPS from SEC filings made by then.

Lines the template doesn't break out are plugs, so reported totals always tie. Anything derived rather than
read directly, and anything suspicious (resegmentation, a business separated after the 10-K, a split, a
consensus perimeter mismatch), is listed under "Data notes" in step 2 and on the Cover. Banks and insurers are
refused (the template is an industrial EV model) unless forced under Advanced.

## Beyond industrials: what the model adapts per company

- **Industry profile** (`valuation/industry.py`): industrial, captive-finance industrial, software, internet,
  media, retail, staples, pharma, semis, energy, utility — detected from SIC / Yahoo industry. It sets the
  conventions (revenue lines on/off, adjusted EPS excluding stock comp, P/E vs EV/EBITDA comps, longest fade)
  and gives the analyst the industry checklist (brief + AI prompt).
- **Revenue lines**: products / regions from the 10-K's disaggregated revenue (LLY by drug, NFLX by region,
  CRM by cloud, GOOGL Search / YouTube / Network / Subscriptions inside Google Services). Subtotal members are
  removed via the taxonomy hierarchy. Each line has bear / base / bull growth; the segment is their sum.
- **KPIs tab**: an optional volume x price build (warehouses x sales per warehouse, members x ARM) that drives a
  line's growth and reconciles to reported revenue.
- **One-offs**: gains on equity stakes / debt extinguishment (tagged) and untagged excess other income are a
  separate IS line, kept in GAAP EPS and excluded from adjusted EPS; FY1 carries what is already booked.
- **Year to date**: FY1 defaults anchor to H1 / 9M growth and margin change when there is no guidance; the
  Interim tab shows the rest of the year the model implies vs the year-to-date trend.
- **Three-stage DCF**: `fade_years` after the six explicit years, growth fading to the terminal rate.
- **Comps**: P/E (forward) and EV / LTM EBITDA (peers' EBITDA and net debt from their filings), blended.
- **Captive finance** (DE, CAT): the finance arm's interest is an operating cost inside its segment profit and
  its debt (funding receivables) is excluded from net debt and the WACC.
- **Checks the AI review pass must answer**: stance vs valuation, methods that disagree, fair value far from the
  price (mid-cycle for cyclicals, runway for compounders), rest of year vs year to date (incl. an implausible
  rest-of-year operating margin), full-year other income below what is already booked, idle cash piling up,
  thesis figures that differ from the model, peer outliers, balance-sheet lines that turn negative, an exit
  multiple far below the peers' (or implying implausible growth at the model's WACC: both sides are reconciled),
  expensed acquired IPR&D entered again as an acquisition.
- **Balance-sheet guards** (any assumptions source): acquisitions are cash paid (>= 0); amortization stops at the
  intangibles left; debt repayment stops at the debt outstanding; buybacks (then repayment) are trimmed so
  ending cash stays above a working minimum (the smaller of FY0 cash and 3% of sales). Each change is a data note.
- **Data safeguards**: peers are always priced at the market close on the date (a model's prices are replaced);
  peer EV/EBITDA outside 2-80x is treated as a data error; stale companyfacts tags are ignored; a cover share
  count that includes shares not outstanding for EPS (employee benefit trusts) is replaced by the EPS count;
  interest income is estimated when a company does not report it separately; the AI cannot move the market
  risk-free rate; KPI history tagged in the filings (store / warehouse counts) is handed to the analyst;
  lessors (DE) carry equipment leased to customers in PP&E with its net purchases in capex (its depreciation is in
  D&A); payables reported with accrued liabilities are picked up; segment profit reported before tax (SHW) is put
  on an operating basis; a resegmented 10-Q (ETN) feeds year-to-date segment profit from its own segments; release
  adjusted EPS read from the wrong columns is replaced by the analyst's figures.

## Configuration (`.streamlit/secrets.toml` or environment)

| Key | Purpose |
|---|---|
| `APP_PASSWORD` | The app's password. With an OpenAI key set and no password, the AI is switched off so a public host cannot spend the key. Keep it in secrets, never in the code |
| `SEC_USER_AGENT` | SEC asks for a contact in the User-Agent, e.g. `"Jane Doe jane@firm.com"` |
| `OPENAI_API_KEY` | Optional: release extraction, peers, peer guidance, analyst pass, thesis |
| `OPENAI_MODEL` | Optional, extraction model, default `gpt-4.1-mini` |
| `OPENAI_ANALYST_MODEL` | Optional, analyst model, default `gpt-5.5` (falls back to `gpt-4.1` if unavailable) |
| `OPENAI_REASONING_EFFORT` | Optional, analyst and review effort: `medium` (default, ~$2 a company), `high` (~$3.30, no better on the CRM test) or `low` (barely reasons: the critiqued v1 demos) |
| `OPENAI_MAX_COST_PER_RUN` | Optional, $ cap per build; optional AI steps are skipped once spent (default by effort: low $1, medium $2.50, high $4) |
| `FA_ERP`, `FA_ERP_SOURCE` | Optional, the house equity risk premium for every name (default: the published start-of-year implied ERP in `valuation/capital.py`) |

**Hosting (Streamlit Community Cloud):** the app runs from `streamlit_app.py` on the `master` branch. In the
app's Settings → Secrets, set `APP_PASSWORD`, `OPENAI_API_KEY` and `SEC_USER_AGENT`. Builds take 5–15 minutes
with the AI analyst; the free tier's memory is enough for one build at a time.

## Code layout

```
streamlit_app.py        3-step UI: ticker -> review inputs / agent hand-off -> download .xlsx
valuation/sec.py        EDGAR: ticker->CIK, filing selection by date, documents, release exhibits
valuation/xbrl.py       XBRL instance reader (contexts, dimensions, facts)
valuation/financials.py XBRL -> standardized IS/BS/CF, segments, interim (YTD/LTM), debt-like, perimeter checks
valuation/facts.py      SEC companyfacts: point-in-time LTM EPS / EBITDA / net debt (peers)
valuation/market.py     Yahoo prices (split-unadjusted), rates, beta to date, FY-matched consensus, peers
valuation/spec.py       ModelSpec + default drivers (with source notes)
valuation/industry.py   Industry profiles: conventions + analyst checklists
valuation/overrides.py  Analyst overrides: export / apply / schema
valuation/brief.py      Research brief for an analyst agent
valuation/ai.py         Optional OpenAI helpers incl. the analyst pass
valuation/workbook.py   XlsxWriter renderer for the 13-tab template
valuation/check.py      Recalculate + sanity checks (optional `formulas`)
valuation/pipeline.py   fetch -> spec (-> AI) (-> overrides) -> workbook
valuation/__main__.py   CLI
legacy/                 previous dashboard app (kept for reference)
```

Tests: `PYTHONPATH=. python tests/regression.py` (11 companies + the HON reference reproduction, needs network
and `formulas`), `tests/test_accounting.py` (parsing fixes: DE interest, PP&E, leased fleet and payables, SHW pre-tax segments, CRM fiscal years, lines, one-offs), `tests/apptest_flow.py` (Streamlit flow headless), `tests/test_ai_mock.py` (AI path, mocked).
