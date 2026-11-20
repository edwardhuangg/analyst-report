# demo/v5: round 5 (your round-4 critique, applied to every company)

These are the same six names as v4. Each started from its v4 overrides. A gpt-5.5 review pass (medium effort) read your critique and the source documents, then the files were rebuilt on the round-5 engine. CRM is valued as of Nov 12, 2026; the others as of Nov 13, 2026.

In all six files:
- every formula value is stored;
- the saved overrides JSON rebuilds the workbook to the cent;
- the balance sheet and cash ties hold;
- no open item is left unexplained.

AI cost for the round was about $7.

| | Price | Fair value | Prob.-wtd vs price | Implied IRR vs WACC | Bear / bull | Rating (rule) |
|---|---|---|---|---|---|---|
| CRM | $217.57 | $215.48 | −3.3% | 9.5% vs 9.3% (+0.2pt) | $97 / $313 | Neutral |
| COST | $1,106.21 | $542.43 | −37.0% | 5.9% vs 8.1% (−2.3pt) | $413 / $1,290 | Neutral, negative bias |
| LLY | $1,443.50 | $981.96 | −30.9% | 5.8% vs 7.2% (−1.5pt) | $537 / $1,491 | Neutral, negative bias |
| CAT | $600.20 | $390.58 | −29.4% | 7.5% vs 10.1% (−2.6pt) | $245 / $669 | Neutral, negative bias |
| NFLX | $62.15 | $61.45 | −4.6% | 10.0% vs 10.3% (−0.2pt) | $36 / $78 | Neutral |
| HD | $348.69 | $295.06 | −16.0% | 8.2% vs 8.9% (−0.7pt) | $190 / $391 | Neutral, negative bias |

Across the six:
- **No Shorts.** COST and CAT were Shorts in v4. Each now fails the short tests you set out (below).
- **Recommendation text matches the rule.** The recommendation line on the Thesis tab is generated from the rule in every file, followed by the analyst's supporting sentence.
- **Price inside bear–bull.** It now sits inside the range for every name.

## Your three headline problems

**1. The recommendation text contradicted the rule (four of six).**
- The call is now generated from the rule, for example "Neutral, negative bias: …".
- Any call the rule does not make, stated anywhere in the thesis, fails gate 25 and holds the rating. Examples: "Long, sized for…", "Underweight", "warrants ownership", a positive bias under a negative rule. The check tells a stated call apart from a mention ("hurdles for a Long", "Long-term" do not count).
- A divergent call has to be a `rating_override` with its reason. It is labeled "Override" and logged on the Checks tab. None of the six uses one.

**2. The price sat outside the scenario range (COST, CAT, HD).**
- Gate 41 requires the price inside bear–bull.
- At build, a bull below the price is rebuilt from the reverse DCF: its margins move toward the terminal margin the price requires, scaled so the bull lands just past the price. A bear above the price is rebuilt the same way.
- The note says when that margin is beyond the 10-year high. COST: the price requires a 9.4% terminal operating margin against a 10-year high of 3.9%, and that is the finding.
- "Bull at or below the price" no longer supports a Short, and "no downside" no longer supports a Long.

**3. The systematic downward bias.**
- **RONIC** now defaults to the last forecast year's ROIC including goodwill (DCF C93). The fade steps to it from the forecast's own incremental return.
- Gate 34 is two-sided: setting RONIC lower needs a dated sector reason, with the year.
- **Universe check** (`universe_calibration.md`): 42 default builds, no AI, as of Nov 13.
  - Median implied IRR − cost of equity is −0.6pt (−0.4pt against WACC), against −1.2pt for your six names in round 4. That is inside your ±1pt test, so no global setting was moved.
  - By group, the result is not uniform:
    - tech, semis and pharma: −0.3 to −0.4pt;
    - staples: −1.7pt;
    - cyclical industrials: −2.4pt.
  - These are framework questions for you, not something to tune name by name toward prices (see the end).

## The other seven points (all companies)

4. **Shorts need more than a gap.**
   - Beyond the gap, a dated catalyst, the IRR spread and bull ≤ +25%, a Short now needs:
     - `thesis.short_evidence`: when the premium last compressed, by how much, on what;
     - a crowding check: Yahoo short interest; 10%+ of float or 7+ days to cover needs a borrow / squeeze plan;
     - for a low-volatility compounder, a pair (`thesis.pair`). Low volatility means beta < 0.85, 10-year revenue σ < 6% and ROIC > 15%.
   - COST fails on evidence and on the pair. CAT fails on evidence.
5. **Missing data fails.**
   - **10-Q debt:** read as zero against 10-K debt, it now fails gate 7.
   - **Interest not tagged:** a 10-Q that does not tag year-to-date interest (LLY) is a warning that needs the 10-Q figure quoted. It is no longer N/A.
   - **Claims vs data (gate 42):** an explanation that says a peer's EPS was fixed or excluded is checked against the Comps table. It caught LLY again this round (see below).
   - **CRM gate 36:** the threshold is tightened (M&A must reach 80% of the acquired-revenue estimate).
6. **Fade reinvestment.**
   - The reinvestment rate (DCF C94) counts every line between NOPAT and UFCF, including other operating items (NFLX's content spend above amortization).
   - Gate 27 is two-sided: a jump above the fade year's growth + 5pt fails as a drop of 5% does.
   - NFLX no longer jumps into 2032.
7. **Mid-cycle runs both ways.**
   - A cyclical fades to its 10-year median margin from a peak as from a trough, in three places:
     - the defaults;
     - the comps EPS basis;
     - gate 16, where a terminal margin more than 2pt above the median needs the structural argument with its pt effect.
   - Home-improvement retail is now a housing cyclical, so HD gets the housing playbook: recovery in the base case or with an explicit probability.
8. **Synergies.**
   - Synergies now get 50% credit, with the run-rate reached in year 3 (deferred two years at WACC), at the lower of the deal and acquirer multiples. Integration costs are after tax.
   - Every deal coming out value-creating is flagged (gate 38).
9. **Peer EPS with one-offs.** A peer's P/E is excluded from the statistics, with the reason in the cell note, when:
   - next-year EPS falls 20%+ (one-offs: GOOGL's gains);
   - next-year EPS rises 100%+ (a charge: MRK);
   - its P/E is outside 0.4–2.5x the others' median (a basis error: NVO's DKK EPS against a USD ADR price).

   If the analyst enters a normalized EPS, growth is measured from it, so the peer comes back.
   - For a cyclical, the P/E-on-growth regression is not used. Next-year growth measures the cycle; DE's trough rebound had set CAT's slope.
10. **Cost of debt.**
    - The coverage-based rating now tops out at AA. AAA is an agency verdict, held by MSFT and JNJ.
    - Agency ratings come from the 10-Q / 10-K text where the filer names them (CAT: A).
    - **Analyst input:** five of the six filers name no agency at all; CRM's 10-K never mentions one. For those, the ratings are an analyst input (`credit_rating`), as known in mid-2026, labeled "verify against S&P / Moody's":

      | | S&P | Moody's |
      |---|---|---|
      | CRM | A+ | A2 |
      | COST | AA | Aa2 |
      | LLY | A+ | A1 |
      | NFLX | A | A3 |
      | HD | A | A2 |

    - The rating sets the cost of debt. Only market evidence (a bond's coupon or yield) moves it: NFLX's 5.61% rests on its July 2026 5.25% notes due 2036.

## Model by model: v4 → v5, and what to challenge

The Checks tab of each file carries the version bridge. Its starting point is v4 revalued on the round-5 engine, so the engine's own effect is separated from the v5 changes.

**CRM: Neutral. $255.28 (v4) → $253.12 (v4 on this engine) → $215.48.**

Your points:
- **The rule's call:** the text now says Neutral.
- **"Organic":** the release-extraction step had recorded "up 11–12% Y/Y" as organic guidance, and the AI passes cannot remove a release figure. The Aug 26, 2026 release says FY27 revenue is "up 11% - 12% Y/Y and 11% Y/Y in CC, including slightly above 3pts Informatica contribution", plus "$200 million from the pending Contentful and Fin acquisitions". So:
  - the organic guide is removed;
  - FY27 M&A + FX is 4.1pt;
  - every revenue line's FY27 growth is 3.5pt lower;
  - organic constant-currency growth is 7.5%, and sales stay inside the $46.1–46.4bn guide.

  I (Claude Code, as the analyst) made this edit, and it is noted in the M&A note and `pm review`. The extractor now treats growth that includes acquired revenue as not organic.
- **Contentful and Fin:** treated as closed, $5.1bn of cash paid in FY27 acquisitions, revenue in the forecast. Fin is identified from the 10-Q as Intercom, Inc. at about $3.6bn.
- **FY27 interest:** about $1.8bn: H1's $790mm plus the rest of the year at about 5% on the $39.3bn 10-Q debt. The AI's first attempt set a 7.7% rate "to bring adjusted EPS back inside guidance". The plug check now catches that wording, and gate 8's tolerance is ±10%.
- **Synergies:** zero for all three deals. The documents quantify none.

The bridge to v4 on this engine (−$37.64; judgment carries 83% of the gross move):

| Change | $/share |
|---|---|
| Margins: the v4 hold near the FY27 guide reverted to a fade | **−21.19** |
| Deals (Contentful and Fin paid; no synergies) | −7.29 |
| Capital return (later buybacks cut so cash stays positive) | −6.15 |
| Working capital | −5.04 |
| Revenue growth and M&A | −3.11 |
| Cost of debt (A rating) | −2.67 |
| Share count / ASR (data) | +9.10 |

**Challenge:** the margin path. v4's +$20 margin hold had no new evidence, so it went back.

**COST: Neutral, negative bias (was Short). $511.83 → $509.36 → $542.43.**
- RONIC is 32%, the return including goodwill; goodwill is negligible (+$11.36).
- Margins no longer fade below the 10-year uptrend (+$21.77).
- The bull is built from the reverse DCF.
- It fails the short tests: no evidence the premium compresses, and it is a low-volatility compounder without a pair. The useful output is your sentence: the price implies a 5.9% IRR against an 8.1% WACC.

**Challenge:** the membership-fee cycle is still not a discrete event in the model.

**LLY: Neutral, negative bias. $1,215.44 → $981.96 (−$233).**
- RONIC is 43.5% (from 22%). Business development and acquired IPR&D are already expensed in EBIT, so the double penalty is gone.
- The loss of exclusivity is in the terminal:
  - terminal growth is 2.5% (from 3.75%) for the mid-2030s incretin LOE and drug-price negotiation;
  - margins normalize in the late decade (−$49.82);
  - the net terminal effect is −$162.95, with the RONIC increase inside it.
- The implied −$2bn Q4 FCF is gone: inventory days now sit on a run-rate basis (gate 43).
- **NVO and MRK EPS (−$17 from peers).**
  - The review entered NVO at $3.55 per ADR (USD) and MRK at $8.98 (company-adjusted). An engine bug then kept Yahoo's figures instead: an EPS far from consensus was treated as pre-split.
  - Gate 42 caught the claim against the data. The engine now keeps whichever EPS gives a P/E nearer the other peers'.
  - The two figures are restored from the AI's own output, noted in each peer's source.

**Challenge:**
- The AI says the patent table was not in its documents, so the LOE year is an estimate, not a cited date.
- 2.5% terminal growth is the lever: check it against product-level erosion.

**CAT: Neutral, negative bias (was Short). $435.99 → $437.62 → $390.58.**
- **The data errors:**
  - Debt is now read from CAT's own columns: $45.2bn at Sep 30, 2026, 75% of it Financial Products. The bridge deducts the industrial debt.
  - Interest is on the industrial debt only, $509mm a year against $2,028mm.
  - The P/E is the peer median (23.6x) on mid-cycle EPS. No regression: DE's trough had set its slope.
  - The cost of debt is at CAT's A rating.
- **The judgments:**
  - RONIC 19% (from 14%) and 5 fade years (from 3): +$15.29.
  - Beta back to the method's 1.21: −$7.69.
- **Mid-cycle:** the first review pass cut margins to the 10-year median (14.3%) while growing sales 52% to $102.6bn, a peak-volume path. A second pass made volume and margin consistent: the 2031 operating margin is 18.3% (80th percentile), with the structural argument stated (Power & Energy mix).
- **Remaining judgment moves:** margins −$17.58; method weights −$16.27 (less SOTP); working capital −$11.47; the adjusted-EPS definition −$6.15 (only restructuring is excluded, per the Q3 reconciliation).
- **Challenge:** you expected 15–20% below the price after the data fixes. The model is 35% below.
  - The difference is the judgments above, all in the bridge.
  - The base also keeps 2027 growth at 7.8% against 11.6% consensus, a stated variant view tested at the January 2027 outlook (gate 35).

**NFLX: Neutral (was Neutral, positive-bias text). $56.72 → $54.43 → $61.45.**
- The text follows the rule.
- **GOOGL:** normalized ex-gains to $10.82 (about 27x) and back in the peer set, +$4.62. The figure is an estimate from your $127bn: the AI labeled the arithmetic, and no GOOGL document was read.
- The fade reinvestment counts the content spend.
- RONIC is 43%.
- **Peer set:** the AI dropped WBD and SPOT, leaving DIS, CMCSA, ROKU and GOOGL. DIS's $6.93 is the Nov 16 snapshot, before its Nov 17 release.

**Challenge:**
- The terminal operating margin of 32.8% is still above all 10 years of history (explained as ad tier and scale).
- Content is not yet a full schedule.

**HD: Neutral, negative bias. $271.15 → $273.77 → $295.06.**
- **Housing:** a measured housing / Pro recovery is now in the base case, and the 2031 operating margin is at the 10-year median (14.3%).
- **RONIC:** 30%, +$11.84.
- **Buybacks:** excess cash now goes to buybacks, $4.5–11.0bn a year from FY27. The v4 "cash build" explanation had switched the policy off without naming a use of the cash; that now takes a named use.
- **KPI build:** it ties. The −2.8% was GMS's acquired revenue sitting outside the organic stores × sales-per-store build. Gate 44 nets it out and the KPI tab says so.

## Framework questions for you

1. **Cyclical industrials price 2.4pt below their cost of equity on default builds** (CAT, DE, ETN, LIN, SHW, RTX).
   - The mid-cycle margin is the 10-year median. It is measured at lower volumes, and it includes structural breaks: LIN's pre-merger years, CAT's 2016 trough at 3% on $38.5bn of sales.
   - Should mid-cycle be the margin at mid-cycle volume, or the last full cycle (7 years), rather than a 10-year median?
2. **Staples price 1.7pt below.** This is the low-beta premium again (KO, WMT, COST, MCD).
   - Accept that the DCF says they are rich?
   - Or does the house want a minimum ERP / beta for defensives?
3. **Credit ratings.** Most filers name none. Should we buy a ratings feed, or keep the analyst input with its verification note?

## Process notes

- **AI spend:** about $7 of the $13.93.
  - Review pass with your critique and the reading pack: CRM $1.12, CAT $1.34, COST $0.52, LLY $0.90, NFLX $0.63.
  - Second, targeted passes: CRM $1.23, CAT $0.55, HD $0.60.
  - HD's first gpt-5.5 pass was cancelled at its 10-minute limit while the laptop slept with the lid closed. The limit now counts only time the machine is awake.
- **Market data:**
  - A Yahoo rate limit in one build dropped COST's and LLY's own beta and P/E history. COST's comps went from $611 to $343 on that alone.
  - The saved overrides now carry the market series (own beta, P/E history, risk-free rate, short interest), and a rebuild uses them when a fetch comes back empty.
- **Hand edits, each noted in its source field:**
  - CRM's organic / M&A split;
  - LLY's NVO / MRK EPS restored from the AI's output;
  - NFLX's DIS snapshot date;
  - the five analyst credit ratings.

## Files

- `<TICKER>_Valuation_Model.xlsx` and `_overrides.json`: the six models.
- `universe_calibration.md` and `universe/`: the 42 default builds behind the calibration.
- `notes/`: your critique as given to each review pass (`_common.md` plus the name), and the targeted second-pass notes.
- `runs/`:
  - each pass as it ran (`*_as_run.xlsx` and its overrides);
  - the metered logs (`*_live_run.log`);
  - the final no-AI builds with their bridges (`*_final_build.log`);
  - the inputs to the final builds (`*_final_input2_overrides.json`).
