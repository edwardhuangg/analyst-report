Systemic points from the PM (apply to this model):
- The recommendation text must follow the rule-based rating. Write thesis.recommendation as one supporting sentence with no call in it; never write a different call or bias anywhere in the thesis ("Long", "Underweight", "avoid adding", "warrants ownership", a positive bias under a negative rule). Disagreeing needs rating_override {rating, reason}.
- The price must lie inside bear–bull. For a liquid large cap, a bull below the price means the scenarios are miscalibrated. Build the bull from what the reverse DCF says the price requires (the terminal margin or years of growth it implies).
- The framework has a downward bias. RONIC defaults to ROIC including goodwill (DCF C93). Lower it only for a dated sector reason (loss of exclusivity with product and year, competitive entry, recurring acquisitions not otherwise modeled), explained under 'gate 34' with the year.
- A short needs more than a valuation gap:
  - a dated catalyst;
  - evidence that the gap closes (thesis.short_evidence: when the premium last compressed, by how much, on what);
  - a borrow and crowding check;
  - for a low-volatility compounder, a pair (thesis.pair) rather than an outright short.
- Claims in explanations must match the data (gate 42). Missing or zero required data is a failure, not N/A.
- Mid-cycle normalization is symmetric for cyclicals: a peak is brought down just as a trough is brought up.
- Deal synergies get 50% credit, are phased in over about 3 years, and are capitalized at the acquirer's multiple. If every deal creates value, recheck the inputs.
- Peer EPS with one-offs (next-year EPS growth below −20%) or a basis error is excluded. Fix the EPS to bring the peer back.
- The cost of debt follows the actual agency rating, not a synthetic AAA.
PM on NFLX (a credible Neutral; rule rating Neutral with a negative bias, but the text said a positive bias):
- Fix the bias wording to match the rule.
- GOOGL's peer EPS ($19.60) includes about $127bn of securities gains, so it shows 15x instead of about 27x. The engine now excludes it as one-off EPS. Bring GOOGL back only with an EPS that excludes the gains; that alone lifts the comps value toward the mid-$60s.
- Fade reinvestment: content overspend is booked in "other operating", so the fade began at a RONIC of 1.0 and free cash flow jumped 24% into 2032. The engine now counts other operating items in the reinvestment rate (DCF C94), and gate 27 is two-sided.
- Content should be a real schedule: content assets are held flat while the overspend builds "other assets". At least make the content cash spend vs amortization path explicit and consistent, with its note.
- The terminal operating margin of 32.8% is above all 10 years of history. It is explained as a structural ad-tier and scale argument; keep it only with numbers.
