"""Standardize 10-K / 10-Q XBRL into the line items the model template uses.

Every subtotal on the template ties to the filing: lines the template does not
break out are computed as plugs (e.g. "other current assets" = reported total
current assets less the lines shown), so reported totals always reconcile.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from .xbrl import Context, Instance, humanize_member

MM = 1e6

# Candidate us-gaap concepts per line item, in priority order.
C = {
    "sales": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
              "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
              "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold",
             "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization", "CostOfServices"],
    "rnd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
            "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost"],
    "sga": ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"],
    "selling": ["SellingAndMarketingExpense", "SellingExpense", "MarketingAndAdvertisingExpense"],
    "opinc": ["OperatingIncomeLoss"],
    "nonop": ["NonoperatingIncomeExpense", "OtherNonoperatingIncomeExpense", "OtherNonoperatingIncome",
              "OtherNonoperatingIncomeExpenseNet"],
    "costs_total": ["CostsAndExpenses", "OperatingExpenses"],
    "interest_exp": ["InterestExpense", "InterestExpenseNonoperating", "InterestAndDebtExpense",
                     "InterestExpenseDebt", "InterestExpenseNet", "InterestIncomeExpenseNonoperatingNet"],
    "interest_inc": ["InvestmentIncomeInterest", "InterestIncomeOther", "InvestmentIncomeInterestAndDividend",
                     "InterestAndOtherIncome", "InterestIncomeOperating"],
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomesticAndForeign"],
    "tax": ["IncomeTaxExpenseBenefit", "IncomeTaxExpenseBenefitContinuingOperations"],
    "ni_total": ["ProfitLoss"],
    "ni_attr": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "nci": ["NetIncomeLossAttributableToNoncontrollingInterest", "MinorityInterestInNetIncomeLossOfConsolidatedEntities"],
    "dil_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"],
    "basic_shares": ["WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"],
    "eps_dil": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "amort": ["AmortizationOfIntangibleAssets", "AmortizationOfAcquiredIntangibleAssets",
              "AmortizationOfIntangibleAssetsExcludingGoodwill"],
    # commissions capitalized and amortized: inside cash-flow D&A, matched by new contract costs in operating cash
    # flow, so the model carries neither (ADBE: $282mm of its $818mm FY25 D&A)
    "contract_amort": ["CapitalizedContractCostAmortization"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense",
            "ShareBasedCompensationExpense",
            # equity statement: the expense recognized in paid-in capital (ETN tags no cash-flow SBC line)
            "AdjustmentsToAdditionalPaidInCapitalSharebasedCompensationRequisiteServicePeriodRecognitionValue",
            "StockIssuedDuringPeriodValueShareBasedCompensation"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
           "DepreciationAndAmortization", "DepreciationAmortizationAndOther"],
    "dep": ["Depreciation", "DepreciationNonproduction", "CostDepreciationAmortizationAndDepletion"],
    "dps": ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"],
    # cash flow
    "pension_nc": ["PensionAndOtherPostretirementBenefitsExpenseReversalOfExpenseNoncash"],
    "d_ar": ["IncreaseDecreaseInAccountsReceivable", "IncreaseDecreaseInReceivables",
             "IncreaseDecreaseInAccountsAndOtherReceivables", "IncreaseDecreaseInAccountsReceivableAndOtherOperatingAssets"],
    "d_inv": ["IncreaseDecreaseInInventories", "IncreaseDecreaseInRetailRelatedInventories"],
    "d_oca": ["IncreaseDecreaseInOtherCurrentAssets", "IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets",
              "IncreaseDecreaseInPrepaidExpense", "IncreaseDecreaseInOtherOperatingAssets"],
    "d_ap": ["IncreaseDecreaseInAccountsPayable", "IncreaseDecreaseInAccountsPayableTrade",
             "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities"],
    "d_accr": ["IncreaseDecreaseInAccruedLiabilities", "IncreaseDecreaseInOtherAccruedLiabilities",
               "IncreaseDecreaseInAccruedLiabilitiesAndOtherOperatingLiabilities",
               "IncreaseDecreaseInOtherCurrentLiabilities", "IncreaseDecreaseInOtherOperatingLiabilities"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment"],
    "acq": ["PaymentsToAcquireBusinessesNetOfCashAcquired", "PaymentsToAcquireBusinessesGross",
            "PaymentsToAcquireBusinessesAndInterestInAffiliates"],
    "divest": ["ProceedsFromDivestitureOfBusinesses", "ProceedsFromDivestitureOfBusinessesNetOfCashDivested",
               "ProceedsFromSaleOfBusinesses"],
    "cfi": ["NetCashProvidedByUsedInInvestingActivities",
            "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    "stock_proc": ["ProceedsFromIssuanceOfCommonStock", "ProceedsFromStockOptionsExercised",
                   "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions",
                   "ProceedsFromIssuanceOrSaleOfEquity",
                   "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlans"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"],
    "divs": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "PaymentsOfOrdinaryDividends"],
    "cff": ["NetCashProvidedByUsedInFinancingActivities",
            "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    "cash_cf": ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
                "CashAndCashEquivalentsAtCarryingValue", "Cash"],
    # balance sheet
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments", "Cash"],
    "sti": ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
            "OtherShortTermInvestments"],
    "ar": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent",
           "AccountsReceivableNet"],
    "inv": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves", "RetailRelatedInventoryMerchandise"],
    "ahfs": ["AssetsHeldForSaleNotPartOfDisposalGroupCurrent", "DisposalGroupIncludingDiscontinuedOperationAssetsCurrent",
             "AssetsOfDisposalGroupIncludingDiscontinuedOperationCurrent", "AssetsHeldForSaleCurrent"],
    "ca": ["AssetsCurrent"],
    "lt_inv": ["LongTermInvestmentsAndReceivablesNet", "LongTermInvestments", "EquityMethodInvestments",
               "MarketableSecuritiesNoncurrent", "InvestmentsInAffiliatesSubsidiariesAssociatesAndJointVentures",
               "OtherLongTermInvestments"],  # CRM's strategic investments ($11.3bn at Jul 31, 2026)
    "ppe": ["PropertyPlantAndEquipmentNet",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
            # DE (extension tag): own property, with equipment leased to customers on its own line
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetExcludingLessorAssetUnderOperatingLease"
            "AfterAccumulatedDepreciationAndAmortization"],
    "goodwill": ["Goodwill"],
    "intang": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet",
               "IntangibleAssetsNetIncludingGoodwill"],
    "rou": ["OperatingLeaseRightOfUseAsset"],
    "dta": ["DeferredIncomeTaxAssetsNet", "DeferredTaxAssetsNetNoncurrent"],
    "assets": ["Assets"],
    "ap": ["AccountsPayableCurrent", "AccountsPayableTradeCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent",
           "AccountsPayableAndAccruedLiabilitiesCurrentAndNoncurrent"],  # unclassified balance sheet (DE)
    "st_debt": ["ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings", "LinesOfCreditCurrent"],
    "cur_ltd": ["LongTermDebtAndCapitalLeaseObligationsCurrent", "LongTermDebtCurrent",
                "LongTermDebtAndFinanceLeasesObligationsCurrent" ,"SeniorNotesCurrent"],
    "debt_cur_total": ["DebtCurrent"],
    "lhfs": ["LiabilitiesOfDisposalGroupIncludingDiscontinuedOperationCurrent",
             "DisposalGroupIncludingDiscontinuedOperationLiabilitiesCurrent", "LiabilitiesHeldForSaleCurrent"],
    "cl": ["LiabilitiesCurrent"],
    "ltd": ["LongTermDebtAndCapitalLeaseObligations", "LongTermDebtNoncurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent", "LongTermDebtAndFinanceLeasesObligations",
            "LongTermDebtAndFinanceLeasesNoncurrent",
            "SeniorLongTermNotes", "OtherLongTermDebtNoncurrent", "LongTermNotesPayable"],
    "dtl": ["DeferredIncomeTaxLiabilitiesNet", "DeferredTaxLiabilitiesNoncurrent"],
    "postret": ["PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
                "DefinedBenefitPensionPlanLiabilitiesNoncurrent", "PostemploymentBenefitsLiabilityNoncurrent",
                "OtherPostretirementDefinedBenefitPlanLiabilitiesNoncurrent",
                "LiabilityForPensionAndOtherPostretirementBenefitsNoncurrent"],
    "lease_liab": ["OperatingLeaseLiabilityNoncurrent"],
    "redeem_nci": ["RedeemableNoncontrollingInterestEquityCarryingAmount",
                   "RedeemableNoncontrollingInterestEquityCommonCarryingAmount",
                   "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests",
                   "TemporaryEquityCarryingAmountAttributableToParent"],
    "equity": ["StockholdersEquity"],
    "equity_total": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "nci_bs": ["MinorityInterest"],
    "le_total": ["LiabilitiesAndStockholdersEquity"],
}

# debt cash flows: (concepts, sign); within a group the first concept found wins
DEBT_GROUPS = [
    (["ProceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet", "ProceedsFromIssuanceOfLongTermDebt",
      "ProceedsFromIssuanceOfSeniorLongTermDebt", "ProceedsFromIssuanceOfUnsecuredDebt",
      "ProceedsFromIssuanceOfDebt", "ProceedsFromDebtNetOfIssuanceCosts"], 1),
    (["RepaymentsOfLongTermDebtAndCapitalSecurities", "RepaymentsOfLongTermDebt", "RepaymentsOfSeniorDebt",
      "RepaymentsOfUnsecuredDebt", "RepaymentsOfDebt"], -1),
    (["ProceedsFromRepaymentsOfShortTermDebt"], 1),
    (["ProceedsFromRepaymentsOfCommercialPaper"], 1),
    (["ProceedsFromShortTermDebtMaturingInThreeMonthsOrLess",
      "ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess"], 1),
    (["ProceedsFromShortTermDebt", "ProceedsFromShortTermDebtMaturingInMoreThanThreeMonths"], 1),
    (["RepaymentsOfShortTermDebt", "RepaymentsOfShortTermDebtMaturingInMoreThanThreeMonths"], -1),
    (["ProceedsFromIssuanceOfCommercialPaper"], 1),
    (["RepaymentsOfCommercialPaper"], -1),
    (["ProceedsFromLinesOfCredit", "ProceedsFromLongTermLinesOfCredit"], 1),
    (["RepaymentsOfLinesOfCredit", "RepaymentsOfLongTermLinesOfCredit"], -1),
]

SEG_AXIS = "us-gaap:StatementBusinessSegmentsAxis"
CONS_AXIS = "srt:ConsolidationItemsAxis"
OPSEG = "us-gaap:OperatingSegmentsMember"
CORP_RX = re.compile(r"(?i)corporate|reconcil|unallocated|elimination|nonsegment")
# subtotals of the reportable segments (ASU 2023-07 filers tag them on the segment axis)
AGG_RX = re.compile(r"(?i)aggregation|ReportableSegmentsMember$|TotalSegments|SegmentsTotal")
ALL_OTHER_RX = re.compile(r"(?i)allother|othersegment|otheroperatingsegment")
PROFIT_RX = re.compile(r"(?i)(segment.*(profit|operatingincome|operatingearnings|income))|^us-gaap:OperatingIncomeLoss$"
                       r"|operatingprofit|SegmentReportingInformationOperatingIncomeLoss|EarningsBeforeInterestAndTaxes"
                       r"|^us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes|^us-gaap:GrossProfit$")
# debt-like items outside net debt: litigation / asbestos accruals and the insurance recoveries against them
DEBT_LIKE = {"liab": ["LossContingencyAccrualCarryingValueNoncurrent", "LossContingencyAccrualAtCarryingValue"],
             "offset": ["LossContingencyReceivableNoncurrent", "LossContingencyReceivable"]}
DISC_OPS = ["IncomeLossFromDiscontinuedOperationsNetOfTax",
            "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
            "DisposalGroupIncludingDiscontinuedOperationRevenue"]
# non-operating one-offs analysts strip from adjusted EPS (mark-to-market gains on equity stakes, debt
# extinguishment); the first concept found in each group counts. Kept to items that sit below operating
# income for every filer, so they can come out of "other income" without touching EBIT.
NONREC_GROUPS = [
    ["DebtAndEquitySecuritiesGainLoss", "EquitySecuritiesFvNiGainLoss", "GainLossOnInvestments",
     "MarketableSecuritiesGainLoss", "EquitySecuritiesFvNiUnrealizedGainLoss"],
    ["GainsLossesOnExtinguishmentOfDebt"],
]
PRODUCT_AXIS = "srt:ProductOrServiceAxis"
GEO_AXIS = "srt:StatementGeographicalAxis"
CAPTIVE_RX = re.compile(r"(?i)financ|credit|capital")


@dataclass
class Segment:
    name: str
    member: str
    sales: list
    profit: list
    lines: list = field(default_factory=list)   # [(name, [sales x3])]: products / regions inside the segment


@dataclass
class History:
    years: list               # fiscal years, oldest -> newest, e.g. [2022, 2023, 2024]
    period_ends: list         # date per year
    period_starts: list
    IS: dict = field(default_factory=dict)   # line -> [y0, y1, y2] in $mm (None if missing)
    BS: dict = field(default_factory=dict)   # line -> [y0(None), y1, y2]
    CF: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)
    corp_sales: list = field(default_factory=list)
    corp_profit: list = field(default_factory=list)
    segments_reported: bool = False
    seg_profit_concept: str = ""
    seg_measure: str = "operating"  # "operating" or "gross profit"
    corp_members: list = field(default_factory=list)
    cover_shares: Optional[float] = None
    dps: list = field(default_factory=list)
    debt_like: dict = field(default_factory=dict)  # liab / offset at the two balance sheet dates
    captive: dict = field(default_factory=dict)    # captive finance arm: segment, interest [x3], share of interest
    lines_axis: str = ""                           # "product" / "geography": what the revenue lines split by
    content: dict = field(default_factory=dict)    # content businesses: {"amort": [x3], "additions": [x3]} ($mm)
    flags: list = field(default_factory=list)  # human-readable data notes


@dataclass
class Interim:
    available: bool
    fiscal_period: str = ""   # Q1 / Q2 / Q3
    label_prior: str = ""
    label_cur: str = ""
    ltm_label: str = ""
    period_end: Optional[date] = None
    quarter_label: str = ""
    ytd_prior: dict = field(default_factory=dict)
    ytd_cur: dict = field(default_factory=dict)
    bs_cur: dict = field(default_factory=dict)
    bs_fy: dict = field(default_factory=dict)
    q_segments: list = field(default_factory=list)  # dicts: name, sales_cur, sales_prior, profit_cur, profit_prior
    q_corp: dict = field(default_factory=dict)
    cover_shares: Optional[float] = None
    q_basic: Optional[float] = None
    q_diluted: Optional[float] = None
    ytd_basic: Optional[float] = None      # weighted average basic shares over the year to date (CRM: the ASR's
    ytd_diluted: Optional[float] = None    # 103mm shares retired early in the year)
    q_dps: Optional[float] = None
    split_ratio: Optional[float] = None    # StockholdersEquityNoteStockSplitConversionRatio1, if tagged
    debt_like: dict = field(default_factory=dict)
    disc_ops_ytd: Optional[float] = None   # discontinued operations in the YTD period ($mm)
    missing_segments: list = field(default_factory=list)  # 10-K segments the 10-Q no longer reports
    new_segments: list = field(default_factory=list)      # 10-Q segments the 10-K did not have (resegmentation)
    captive_share: Optional[float] = None  # finance arm's share of the 10-Q debt, when presented in its own columns


# ----------------------------------------------------------------- helpers
def _mm(v):
    return None if v is None else v / MM


def _nz(v):
    return 0.0 if v is None else v


def _annual_periods(inst: Instance, n: int = 3) -> list[Context]:
    ctxs = inst.durations(C["sales"] + C["ni_attr"] + C["cfo"], 340, 380)
    out, ends = [], set()
    for c in ctxs:
        if c.end not in ends:
            out.append(c)
            ends.add(c.end)
    return sorted(out[:n], key=lambda c: c.end)


IS_KEYS = {"sales", "cogs", "rnd", "sga", "interest_exp", "interest_inc", "pretax", "tax", "ni_total", "ni_attr",
           "nci", "dil_shares", "basic_shares", "eps_dil"}
_IS_CONCEPTS: set = set()  # set per parse call: concepts presented on the income statement


def _cands(key: str) -> list:
    cands = C[key]
    if key in IS_KEYS and _IS_CONCEPTS:
        cands = [c for c in cands if "us-gaap:" + c in _IS_CONCEPTS] + \
                [c for c in cands if "us-gaap:" + c not in _IS_CONCEPTS]
    return cands


CAPEX_RX = re.compile(r"(?i)^[a-z0-9]+:\w*CapitalExpenditure")  # company tags (NEE, VZ: no us-gaap capex)
CAPEX_EXCL = re.compile(r"(?i)Incurred|NotYetPaid|Planned|Commitment|Accru|Noncash|Unpaid|Future|Percent|Ratio|Budget")
GROSS_INT = ["InterestExpense", "InterestExpenseNonoperating", "InterestAndDebtExpense", "InterestExpenseDebt"]
NET_INT = ["InterestIncomeExpenseNonoperatingNet", "InterestIncomeExpenseNet", "InterestExpenseNet"]
CUSTOM_INT_RX = re.compile(r"(?i)^[a-z0-9]+:Interest(?:Expense|AndDebtExpense|Charges)(?!.*(?:Capitaliz|Tax|Penalt|"
                           r"Benefit|Accrued|Paid|Receiv))")


def _interest_exp(inst: Instance, ctx: Context) -> Optional[float]:
    """Gross interest expense, positive ($mm). DIS tags it negative; NKE tags only the net of its interest income
    (and the income separately, so it was counted twice); NEE uses its own tags (it read as 0)."""
    g = inst.value(GROSS_INT, start=ctx.start, end=ctx.end)
    if g is not None:
        return abs(g) / MM
    inc = inst.value(C["interest_inc"], start=ctx.start, end=ctx.end)
    inc = abs(inc) / MM if inc is not None else 0.0
    for c in NET_INT:
        n = inst.value([c], start=ctx.start, end=ctx.end)
        if n is None:
            continue
        n /= MM
        if c == "InterestExpenseNet":  # debit element: positive = net expense
            return max(0.0, n + inc)
        return -n + inc if n < 0 else max(0.0, inc - n)  # credit elements: positive = net interest income
    sp = business_split(inst, GROSS_INT, start=ctx.start, end=ctx.end)  # CAT: 'interest expense excluding
    if sp:  # Financial Products' (the finance arm's interest is an operating cost on the face statement)
        return abs(sp[0] - sp[1])
    custom = [c for c in inst.facts if CUSTOM_INT_RX.match(c)]
    vals = [inst.value([c], start=ctx.start, end=ctx.end) for c in custom]
    vals = [abs(v) for v in vals if v is not None]
    return max(vals) / MM if vals else None


def _dur(inst: Instance, key: str, ctx: Context, scale=MM):
    if key == "interest_exp" and scale == MM:
        return _interest_exp(inst, ctx)
    v = inst.value(_cands(key), start=ctx.start, end=ctx.end)
    return None if v is None else v / scale


def share_scale(x: Optional[float], ref: Optional[float]) -> Optional[float]:
    """MCD tags weighted shares in millions (709.9 'shares'), read as 0.0007mm: put a count back on the scale of
    the reference (net income / EPS, or the cover) when x 1e6 or x 1e3 lands on it."""
    if x is None or not ref or ref <= 0 or x <= 0 or 0.5 < x / ref < 2:
        return x
    for f in (1e6, 1e3):
        if 0.5 < x * f / ref < 2:
            return x * f
    return x


def _inst(inst: Instance, key: str, d: date):
    v = inst.value(C[key], end=d)
    if v is None:  # company-extension tag with a us-gaap local name (DE: de:LongTermDebtAndFinanceLeasesNoncurrent)
        names = set(C[key])
        custom = [c for c in inst.facts if not c.startswith("us-gaap:") and c.split(":")[-1] in names]
        custom.sort(key=lambda c: C[key].index(c.split(":")[-1]))
        v = inst.value(custom, end=d) if custom else None
    if v is None and key in DEBT_KEYS:
        sp = business_split(inst, C[key], end=d)
        if sp:
            return sp[0]
        return _ltd_generic(inst, d) if key == "ltd" else None
    return _mm(v)


def to_one_segment(h: History) -> None:
    """The latest 10-Q reports one reportable segment where the 10-K had several (ADBE, Q1 FY26): the company is the
    segment, on the single-segment profit definition (operating income + amortization + stock comp + other operating
    expense), and the 10-K segments stay as revenue lines with their history. The old segments' profit measure (ADBE:
    gross profit) has no 10-Q actuals to check against, and its basis is not the one the company guides."""
    IS = h.IS
    lines = [(s.name, list(s.sales)) for s in h.segments if s.member != "other" and _nz(s.sales[2]) > 0]
    other = [None if IS["sales"][i] is None else IS["sales"][i] - sum(_nz(v[i]) for _, v in lines) for i in range(3)]
    if lines and other[2] is not None and abs(other[2]) > 0.005 * _nz(IS["sales"][2]):
        lines.append(("Other", other))
    elif lines and other[2] is not None:  # rounding: fold into the largest line so the lines tie
        name, v = max(lines, key=lambda x: _nz(x[1][2]))
        k = lines.index((name, v))
        lines[k] = (name, [None if v[j] is None else v[j] + _nz(other[j]) for j in range(3)])
    prof = [None if IS["opinc"][i] is None else
            IS["opinc"][i] + _nz(IS["amort"][i]) + _nz(IS["sbc"][i]) + _other_opex({k: IS[k][i] for k in IS})
            for i in range(3)]
    h.segments = [Segment("Total company", "total", list(IS["sales"]), prof, lines=lines if len(lines) > 1 else [])]
    h.corp_sales = [0.0 if v is not None else None for v in IS["sales"]]
    h.corp_profit = [0.0 if v is not None else None for v in IS["sales"]]
    h.segments_reported = False
    h.seg_profit_concept, h.seg_measure, h.corp_members = "", "operating", []
    h.lines_axis = "former segment"


def _ltd_generic(inst: Instance, d: date) -> Optional[float]:
    """us-gaap:LongTermDebt as the noncurrent line, when no noncurrent tag exists. The taxonomy defines it as total
    long-term debt, but some filers put it on the face as the noncurrent line (ADBE: 6,210mm next to DebtCurrent 0;
    the model read no debt, so interest implied 35% and the bridge missed $6bn). It is total debt when it sits nearer
    the carrying amount alone than with the current portion added; then the current portion is netted out."""
    v = inst.value(["LongTermDebt"], end=d)
    if v is None:
        return None
    cur = inst.value(["LongTermDebtCurrent", "DebtCurrent"], end=d) or 0.0
    carrying = inst.value(["DebtInstrumentCarryingAmount", "LongTermDebtGross"], end=d)
    if cur and carrying and abs(v - carrying) < abs(v + cur - carrying):
        v -= cur
    return _mm(v)


DEBT_KEYS = ("st_debt", "cur_ltd", "ltd")


def _captive_member(m: str) -> bool:
    name = humanize_member(m)
    return bool(CAPTIVE_RX.search(name)) and not re.search(r"(?i)exclud|other than|\bex\b", name)


def business_split(inst: Instance, concepts, *, end: date, start: Optional[date] = None) -> Optional[tuple]:
    """(total, captive part) in $mm for a line the filer tags only by business on its face statements, with no
    consolidated total (CAT: 'Machinery, Power & Energy' and 'Financial Products' columns on
    srt:ProductOrServiceAxis — its debt read as $0 and its interest as 0, PM round 4). Only an axis with a
    finance-arm member counts, so a product breakdown is never summed."""
    for concept in concepts:
        q = concept if ":" in concept else "us-gaap:" + concept
        by_axis = defaultdict(dict)
        for ctx, v in inst.facts.get(q, {}).items():
            if len(ctx.dims) != 1 or abs((ctx.end - end).days) > 4 or (start is None) != (ctx.start is None):
                continue
            if start is not None and abs((ctx.start - start).days) > 4:
                continue
            (ax, m), = ctx.dims
            by_axis[ax][m] = v
        for ax, ms in by_axis.items():
            # 'Financial Products' and 'Financial Products Segment' on one axis are one business (CAT)
            stem = lambda m: re.sub(r"(?i)(segment)?member$", "", m.split(":")[-1])  # noqa: E731
            ms = {m: v for m, v in sorted(ms.items(), key=lambda x: "egment" in x[0])
                  if m == next(k for k in sorted(ms, key=lambda k: "egment" in k) if stem(k) == stem(m))}
            if ax == PRODUCT_AXIS or "Consolidat" in ax or "LegalEntity" in ax:
                if any(_captive_member(m) for m in ms) or any(re.search(r"(?i)exclud\w*financ", m) for m in ms):
                    return (sum(ms.values()) / MM, sum(v for m, v in ms.items() if _captive_member(m)) / MM)
    return None


def captive_from_split(inst: Instance, periods: list, ends: list) -> dict:
    """A finance arm the filer presents in its own columns (CAT): its debt share from the face balance sheet,
    its interest (inside operating costs already, so nothing is reclassified)."""
    tot = cap = 0.0
    member = None
    for key in DEBT_KEYS:
        sp = business_split(inst, C[key], end=ends[-1])
        if sp:
            tot, cap = tot + sp[0], cap + sp[1]
    if tot <= 0 or cap < 0.3 * tot:
        return {}
    for concept in inst.facts:
        if re.search(r"(?i)Debt|Borrowings", concept):
            for ctx in inst.facts[concept]:
                if len(ctx.dims) == 1 and _captive_member(next(iter(ctx.dims))[1]):
                    member = next(iter(ctx.dims))[1]
                    break
        if member:
            break
    ints = []
    for p in periods:
        v = None
        for c in ("FinancingInterestExpense", "InterestExpenseOperating", "InterestExpense"):
            q = "us-gaap:" + c
            v = next((x for cx, x in inst.facts.get(q, {}).items() if len(cx.dims) == 1 and
                      _captive_member(next(iter(cx.dims))[1]) and cx.start and
                      abs((cx.end - p.end).days) <= 4 and abs((cx.start - p.start).days) <= 4), None)
            if v is not None:
                break
        ints.append(_nz(_mm(v)))
    return {"segment": humanize_member(member or "Financial Products"), "member": member or "",
            "interest": ints, "share": round(min(cap / tot, 0.98), 3), "on_face": True,
            "debt": round(cap), "debt_total": round(tot)}


LESSOR_FLEET = ["PropertySubjectToOrAvailableForOperatingLeaseNet"]
LESSOR_BUY = ["PaymentsToAcquireEquipmentOnLease", "PaymentsToAcquirePropertySubjectToOrAvailableForOperatingLease"]
LESSOR_DEP = ["OperatingLeasesIncomeStatementDepreciationExpenseOnPropertySubjectToOrHeldForLease"]


def _lessor_capex(inst: Instance, ctx) -> Optional[float]:
    """Equipment the company leases to customers (captive finance: DE, CAT) is depreciated inside D&A, but its
    purchases sit in "other investing". Net purchases (bought less sold off lease), $mm, so capex is on the same
    basis as D&A; None when the company is not a lessor."""
    if inst.value(LESSOR_DEP, start=ctx.start, end=ctx.end) is None:
        return None
    buy = inst.value(LESSOR_BUY, start=ctx.start, end=ctx.end)
    if buy is None:
        return None
    sold = [c for c in inst.facts
            if re.search(r"ProceedsFromSales?Of(Equipment|Property)\w*OperatingLease", c.split(":")[-1])]
    return (buy - _nz(inst.value(sold, start=ctx.start, end=ctx.end) if sold else None)) / MM


def _pretax_basis(seg_total, d: dict) -> float:
    """Segment measures reported before tax (SHW) put interest and other non-operating items in "Corporate &
    All Other", so segments + corporate = pre-tax income. Returns what moves out of corporate (operating income
    – pre-tax income) to put total segment profit on the operating basis the forecast uses; 0 otherwise."""
    sales, pre, oi = d.get("sales"), d.get("pretax"), d.get("opinc")
    if None in (seg_total, sales, pre, oi) or not sales:
        return 0.0
    if abs(seg_total - pre) <= 0.002 * abs(sales) and abs(oi - pre) > 0.005 * abs(sales):
        return oi - pre
    return 0.0


def _cogs_from_total(d: dict) -> tuple[float, bool]:
    """Cost of sales derived from total costs and expenses (no cost-of-sales tag).

    Some filers (DE, CAT) put interest expense inside "costs and expenses", so sales – total costs
    = pre-tax income. Interest must then come out of the derived cost of sales, or it is deducted
    twice (once in COGS, once on the interest line). Returns (cogs, interest was inside)."""
    cogs = d["costs_total"] - _nz(d.get("sga")) - _nz(d.get("rnd"))
    ie = _nz(d.get("interest_exp"))
    if ie and d.get("pretax") is not None and d.get("sales") is not None:
        gap_in = abs(d["sales"] - d["costs_total"] - d["pretax"])
        gap_out = abs(d["sales"] - d["costs_total"] - ie - d["pretax"])
        if gap_in < gap_out:
            return cogs - ie, True
    return cogs, False


def _other_opex(d: dict) -> float:
    """Operating expense outside cost of sales, R&D and SG&A (acquired IPR&D, impairments, restructuring,
    litigation): the items a company's performance / adjusted operating margin excludes. 0 if not computable."""
    if any(d.get(k) is None for k in ("sales", "cogs", "opinc")):
        return 0.0
    x = d["sales"] - d["cogs"] - _nz(d.get("rnd")) - _nz(d.get("sga")) - d["opinc"]
    return x if x > 0.5 else 0.0


def _opinc(inst: Instance, ctx: Context, d: dict, is_concepts: set) -> tuple[Optional[float], str]:
    """GAAP operating income. Uses the OperatingIncomeLoss tag only when the income
    statement presents it; otherwise derives EBIT from pre-tax income (some filers tag
    total segment profit as OperatingIncomeLoss in the segment note)."""
    tag = d.get("opinc")
    if not is_concepts or ("us-gaap:OperatingIncomeLoss" in is_concepts and tag is not None):
        if tag is None and not is_concepts:
            if d.get("pretax") is not None:  # SLB: no OperatingIncomeLoss and no cost-of-sales tag (a 97% margin)
                nv = inst.value(C["nonop"], start=ctx.start, end=ctx.end)
                return (d["pretax"] + _nz(d.get("interest_exp")) - _nz(d.get("interest_inc")) - _nz(_mm(nv)),
                        "derived")
            if d.get("sales") is not None and d.get("cogs") is not None:
                return d["sales"] - d["cogs"] - _nz(d.get("rnd")) - _nz(d.get("sga")), "derived"
        return tag, "tag"
    nonop = None
    for c in C["nonop"]:
        if "us-gaap:" + c in is_concepts:
            nonop = inst.value([c], start=ctx.start, end=ctx.end)
            if nonop is not None:
                nonop /= MM
                break
    interest_on_is = any("us-gaap:" + c in is_concepts for c in C["interest_exp"])
    int_inc_on_is = [c for c in C["interest_inc"] if "us-gaap:" + c in is_concepts]
    s_ = d.get("sales")
    # the cost route needs a cost of sales and a plausible margin (SLB: no cost of sales tag -> a 97% margin)
    sales_ok = s_ is not None and _nz(d.get("cogs")) > 0 and \
        s_ - _nz(d.get("cogs")) - _nz(d.get("rnd")) - _nz(d.get("sga")) < 0.75 * s_
    if d.get("pretax") is not None and (nonop is not None or interest_on_is or not sales_ok):
        v = d["pretax"] + (_nz(d.get("interest_exp")) if interest_on_is or not sales_ok else 0) - _nz(nonop)
        if int_inc_on_is:
            v -= _nz(_mm(inst.value(int_inc_on_is, start=ctx.start, end=ctx.end)))
        elif not sales_ok:
            v -= _nz(d.get("interest_inc"))
        return v, "derived"
    if d.get("sales") is not None:
        return d["sales"] - _nz(d.get("cogs")) - _nz(d.get("rnd")) - _nz(d.get("sga")), "derived"
    return tag, "tag"


def _debt_flow(inst: Instance, ctx: Context) -> Optional[float]:
    total, found = 0.0, False
    for concepts, sign in DEBT_GROUPS:
        for c in concepts:
            v = inst.value([c], start=ctx.start, end=ctx.end)
            if v is not None:
                total += sign * v / MM
                found = True
                break
    return total if found else None


AMORT_RX = re.compile(r"(?i)amortization.*(intangible|acquired|purchased)")
AMORT_SKIP = re.compile(r"(?i)accumulated|nexttwelve|year(two|three|four|five)|afteryear|expected|future|retirement|"
                        r"impairment|deferred|tax")


DA_ALL = ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
          "DepreciationAndAmortization", "DepreciationAmortizationAndOther"]


def fix_da_amort(d: dict, inst: Instance, periods: list, flags: Optional[list] = None) -> None:
    """D&A must contain acquired-intangible amortization and depreciation must be positive (critique C9).
    CRM: the DepreciationDepletionAndAmortization tag (1,200) is below amortization (1,687); the cash-flow D&A is
    DepreciationAndAmortization (3,631) -> use the largest D&A tag. ETN: amortization untagged -> D&A less
    depreciation, else the 3.7% 'depreciation' rate double-counts it in EBITDA and the DCF."""
    keyed = isinstance(d.get("da"), list)
    rows = range(len(periods)) if keyed else [0]
    for j in rows:
        p = periods[j]
        get = (lambda k: d[k][j]) if keyed else (lambda k: d.get(k))  # noqa: E731
        da, am, dep = get("da"), get("amort"), get("dep")
        if da is not None and am and da < am:
            alt = [inst.value([c], start=p.start, end=p.end) for c in DA_ALL]
            best = max([v / MM for v in alt if v is not None] or [da])
            if best >= am:
                if dep is None:  # the smaller tag is depreciation of PP&E (CRM: 1,200 of 3,631)
                    if keyed:
                        d["dep"][j] = da
                    else:
                        d["dep"] = da
                    dep = da
                if keyed:
                    d["da"][j] = best
                else:
                    d["da"] = best
                if flags is not None and j == len(periods) - 1:
                    flags.append(f"D&A tag below acquired amortization ({da:,.0f} < {am:,.0f}mm): the largest D&A "
                                 f"tag ({best:,.0f}mm, cash-flow statement) is used")
        if not am and dep and da and da > dep * 1.02:
            am2 = da - dep
            if keyed:
                d["amort"][j] = am2
            else:
                d["amort"] = am2
            if flags is not None and j == len(periods) - 1:
                flags.append(f"Acquired-intangible amortization not tagged: D&A {da:,.0f} less depreciation "
                             f"{dep:,.0f} = {am2:,.0f}mm used, so depreciation is not double-counted")


CONTENT_AMORT_RX = re.compile(r"(?i)Amortization\w*(Streaming)?Content|ContentAssetsAmortization")
CONTENT_ADD_RX = re.compile(r"(?i)(Additions|Payments)\w*(Streaming)?Content(Assets)?$")


def content_history(inst: Instance, periods: list) -> dict:
    """Content businesses (NFLX): amortization of content assets and cash additions to them, so the forecast can
    charge the cash spent above amortization (critique D5: ~$1.7bn a year missing when content is held flat)."""
    out = {}
    for key, rx in (("amort", CONTENT_AMORT_RX), ("additions", CONTENT_ADD_RX)):
        cands = [c for c in inst.facts if rx.search(c.split(":")[-1]) and "Liabilit" not in c]
        vals = [_mm(inst.value(cands, start=p.start, end=p.end)) if cands else None for p in periods]
        if any(v for v in vals):
            out[key] = [abs(v) if v is not None else None for v in vals]
    return out if out.get("amort") and out.get("additions") else {}


def amort_fallback(inst: Instance, ctx: Context) -> Optional[float]:
    """Acquired-intangible amortization under a company-specific tag, e.g. CRM's
    AmortizationOfPurchasedIntangiblesFromBusinessCombinations, which is only reported split by
    income-statement line (cost of revenues / marketing): undimensioned value or the sum of the lines."""
    for c in sorted(inst.facts):
        if not AMORT_RX.search(c) or AMORT_SKIP.search(c):
            continue
        total, lines = None, []
        for cx, v in inst.facts[c].items():
            if not cx.start or abs((cx.end - ctx.end).days) > 4 or abs((cx.start - ctx.start).days) > 4:
                continue
            axes = {ax for ax, _ in cx.dims}
            if not axes:
                total = v
            elif axes == {"us-gaap:IncomeStatementLocationAxis"}:
                lines.append(v)
        if total is not None:
            return total / MM
        if lines:
            return sum(lines) / MM
    return None


# --------------------------------------------------------------- segments
def _segment_value(inst: Instance, concept: str, member, start, end) -> Optional[float]:
    """Value for a segment member, or for a corporate (axis, member) pair."""
    if isinstance(member, tuple):
        options = (frozenset({member}),)
        if member[0] == SEG_AXIS:  # an "all other" segment can carry the operating-segments member too
            options += (frozenset({member, (CONS_AXIS, OPSEG)}),)
    else:
        options = (frozenset({(SEG_AXIS, member)}), frozenset({(SEG_AXIS, member), (CONS_AXIS, OPSEG)}))
    for dims in options:
        v = inst.value([concept], start=start, end=end, dims=dims)
        if v is not None:
            return v / MM
    return None


def _segment_members(inst: Instance, concept: str, end: date) -> tuple[list[str], list[tuple]]:
    """(operating segment members, corporate/other (axis, member) pairs)."""
    ops, others = [], []
    for ctx in inst.dimensional(concept):
        if abs((ctx.end - end).days) > 4 or not ctx.start:
            continue
        axes = dict(ctx.dims)
        if set(axes) - {SEG_AXIS, CONS_AXIS}:
            continue
        m, cons = axes.get(SEG_AXIS), axes.get(CONS_AXIS)
        if m and AGG_RX.search(m):
            continue  # subtotal of the reportable segments: counting it would double the total
        if m and cons in (OPSEG, None):
            if CORP_RX.search(m) or ALL_OTHER_RX.search(m):
                others.append((SEG_AXIS, m))  # folded into "Corporate & All Other"
            else:
                ops.append(m)
        elif m is None and cons and cons != OPSEG and CORP_RX.search(cons):
            others.append((CONS_AXIS, cons))
    ops = list(dict.fromkeys(ops))
    others = [o for o in dict.fromkeys(others) if o[1] not in ops]
    seg_corp = [o for o in others if o[0] == SEG_AXIS and CORP_RX.search(o[1])]
    seg_other = [o for o in others if o[0] == SEG_AXIS and not CORP_RX.search(o[1])]
    cons_items = [o for o in others if o[0] == CONS_AXIS]
    if seg_corp:
        return ops, seg_other + seg_corp
    # consolidation axis: a "reconciling items" total would double count its children
    if len(cons_items) > 1:
        cons_items = [o for o in cons_items if not re.search(r"ReconcilingItems", o[1])] or cons_items
    return ops, seg_other + cons_items


def extract_segments(inst: Instance, periods: list[Context], total_sales: list) -> dict:
    end = periods[-1].end
    for rev_c in C["sales"]:
        ops, others = _segment_members(inst, rev_c, end)
        if len(ops) >= 2:
            break
    else:
        return {}
    # candidate profit concepts that cover every segment in the latest year
    # operating measures first, then gross profit (e.g. TSLA), then pre-tax
    cands = sorted({c for c in inst.facts if PROFIT_RX.search(c)},
                   key=lambda c: (c != "us-gaap:OperatingIncomeLoss", c == "us-gaap:GrossProfit",
                                  "BeforeIncomeTaxes" in c, c))
    profit_c = None
    for pc in cands:
        if "EBITDA" in pc.upper():
            continue
        if all(_segment_value(inst, pc, m, periods[-1].start, end) is not None for m in ops):
            profit_c = pc
            break
    if profit_c is None:
        return {}
    segs = []
    for m in ops:
        sales = [_segment_value(inst, rev_c, m, p.start, p.end) for p in periods]
        profit = [_segment_value(inst, profit_c, m, p.start, p.end) for p in periods]
        if sales[-1] is None:
            continue
        segs.append(Segment(humanize_member(m), m, sales, profit))
    if len(segs) < 2:
        return {}
    seg_total = sum(s.sales[-1] for s in segs)
    if not (0.6 * total_sales[-1] <= seg_total <= 1.25 * total_sales[-1]):
        return {}
    segs.sort(key=lambda s: -s.sales[-1])
    if len(segs) > 6:  # keep the template readable: top five plus the rest combined
        rest = segs[5:]
        comb = Segment("Other segments", "other",
                       [sum(_nz(s.sales[i]) for s in rest) for i in range(len(periods))],
                       [sum(_nz(s.profit[i]) for s in rest) for i in range(len(periods))])
        segs = segs[:5] + [comb]
    others = _segment_members(inst, profit_c, end)[1]
    corp_profit = []
    for p in periods:
        vals = [_segment_value(inst, profit_c, m, p.start, p.end) for m in others]
        vals = [v for v in vals if v is not None]
        corp_profit.append(sum(vals) if vals else None)
    seg_profit_sum = sum(_nz(s.profit[-1]) for s in segs)
    if corp_profit[-1] is not None and abs(corp_profit[-1]) > 0.6 * abs(seg_profit_sum):
        corp_profit = [None] * len(periods)  # looks like a total, not corporate costs
    return {"segments": segs, "profit_concept": profit_c, "revenue_concept": rev_c,
            "corp_members": others, "corp_profit": corp_profit,
            "measure": "gross profit" if profit_c.endswith(":GrossProfit") else "operating"}


def debt_like_items(inst: Instance, ends: list) -> dict:
    """Litigation / asbestos accruals and their insurance recoveries at each balance sheet date."""
    out = {}
    for k, concepts in DEBT_LIKE.items():
        for c in concepts:
            vals = [_mm(inst.value([c], end=d)) for d in ends]
            if any(v for v in vals):
                out[k] = [_nz(v) for v in vals]
                out[k + "_concept"] = c
                break
    return out


def nonrec_items(inst: Instance, ctx: Context) -> Optional[float]:
    """Non-operating one-off gains (+) / losses (–) in pre-tax income for a period ($mm), or None."""
    total, found = 0.0, False
    for grp in NONREC_GROUPS:
        v = inst.value(grp, start=ctx.start, end=ctx.end)
        if v is not None:
            total += v / MM
            found = True
    return total if found else None


def captive_finance(inst: Instance, periods: list, segs: list, interest: list) -> dict:
    """A captive finance arm (DE, CAT, PCAR, GM): a segment whose interest expense is operating, tagged on
    the segment axis. Its interest sits inside segment profit and its debt is funded by financing
    receivables, so neither belongs in EBIT-to-net-debt valuation as ordinary corporate debt."""
    for sg in segs:
        if sg.member in ("total", "other") or not CAPTIVE_RX.search(sg.name):
            continue
        ints = []
        for p in periods:
            v = None
            for c in ("InterestExpenseOperating", "InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense"):
                v = _segment_value(inst, "us-gaap:" + c, sg.member, p.start, p.end)
                if v is not None:
                    break
            ints.append(v)
        if ints[-1] and interest[-1] and ints[-1] >= 0.3 * interest[-1]:
            return {"segment": sg.name, "member": sg.member, "interest": [_nz(v) for v in ints],
                    "share": round(min(ints[-1] / interest[-1], 0.98), 3)}
    return {}


def _captive_interest(inst: Instance, member: str, ctx: Context) -> float:
    for c in ("InterestExpenseOperating", "InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense"):
        v = _segment_value(inst, "us-gaap:" + c, member, ctx.start, ctx.end)
        if v is not None:
            return v
    return 0.0


def _children(children: dict, members: set) -> set:
    """Members (of those present) that are the parent of another present member: subtotals."""
    return {m for m in members if children.get(m, set()) & members}


def _axis_values(inst: Instance, concept: str, axis: str, ctx: Context, children: dict,
                 seg: Optional[str] = None) -> list[dict]:
    """Candidate {member: value} maps on `axis` for one period, optionally inside one segment: facts tagged
    on the axis alone, and facts split further by a second axis (LLY: drug x US / non-US; NFLX: streaming x
    region) summed over that axis's leaf members. Filers often tag a partial breakdown on the axis alone
    (NFLX: US only), so both are returned for the caller to test against the total."""
    direct, split = {}, defaultdict(dict)
    for cx, v in inst.facts.get(concept, {}).items():
        if not cx.start or abs((cx.end - ctx.end).days) > 4 or abs((cx.start - ctx.start).days) > 4:
            continue
        d = dict(cx.dims)
        if d.get(CONS_AXIS) == OPSEG:
            del d[CONS_AXIS]
        if seg is not None:
            if d.pop(SEG_AXIS, None) != seg:
                continue
        elif SEG_AXIS in d:
            continue
        m = d.pop(axis, None)
        if m is None:
            continue
        if not d:
            direct[m] = v / MM
        elif len(d) == 1:
            (ax2, m2), = d.items()
            split[(ax2, m)][m2] = v / MM
    agg = {}
    for (_, m), vals in split.items():
        leaves = set(vals) - _children(children, set(vals))
        agg[m] = sum(vals[x] for x in leaves)
    return [x for x in (direct, agg) if x]


def _finest_partition(vals: dict, total: float, tol: float = 0.01, max_n: int = 16) -> Optional[set]:
    """The largest set of members whose values add up to the total (±tol): subtotal members (GOOGL
    'Google advertising' = Search + YouTube + Network) drop out because the finer split also adds up."""
    items = sorted(((m, v) for m, v in vals.items() if v > 0), key=lambda kv: -kv[1])[:max_n]
    n, best = len(items), 0
    sums, cnt = [0.0] * (1 << n), [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        i = low.bit_length() - 1
        sums[mask] = sums[mask ^ low] + items[i][1]
        cnt[mask] = cnt[mask ^ low] + 1
        if abs(sums[mask] - total) <= tol * total and cnt[mask] > cnt[best]:
            best = mask
    return {items[i][0] for i in range(n) if best >> i & 1} if best else None


def _pick_members(cand: dict, total: float, children: dict) -> Optional[set]:
    members = {m for m, v in cand.items() if v > 0 and v <= 0.97 * total}
    leaves = members - _children(children, members)
    s = sum(cand[m] for m in leaves)
    if abs(s - total) <= 0.02 * total:
        return leaves
    part = _finest_partition({m: cand[m] for m in leaves}, total) or _finest_partition(
        {m: cand[m] for m in members}, total)
    if part:
        return part
    return leaves if 0.5 * total <= s <= total else None  # no overlap possible: the rest is "Other"


BRANDS = {"You Tube": "YouTube", "Linked In": "LinkedIn", "I Phone": "iPhone", "I Pad": "iPad", "I Cloud": "iCloud",
          "Power Point": "PowerPoint",
          "Pay Pal": "PayPal", "Face Book": "Facebook", "Insta Gram": "Instagram", "Whats App": "WhatsApp"}


def _line_name(member: str) -> str:
    """'goog:YouTubeAdvertisingRevenueMember' -> 'YouTube Advertising'; 'GoogleSearchOtherMember' -> 'Google Search
    & Other'."""
    n = humanize_member(member)
    for k, v in BRANDS.items():
        n = n.replace(k, v)
    n = re.sub(r"\s+Revenues?$", "", n)
    n = re.sub(r"\b(\w+) Other$", r"\1 & Other", n) if not n.endswith("& Other") and n != "Other" else n
    return n


def revenue_lines(inst: Instance, periods: list, total: list, children: dict, seg: Optional[str] = None,
                  axes=(PRODUCT_AXIS, GEO_AXIS)) -> tuple[list, str]:
    """Products or regions that add up to a segment's (or the company's) revenue: [(name, [3 years])].

    Subtotal members (LLY 'Cardiometabolic Health' over Mounjaro / Zepbound) are dropped using the
    taxonomy's member hierarchy, or by finding the finest split that adds up to the total. Top five
    lines by latest revenue plus an 'Other' residual, so the lines always sum to the reported total."""
    if not total or total[-1] is None or total[-1] <= 0:
        return [], ""
    T = total[-1]
    for axis in axes:
        for rc in C["sales"]:
            concept = "us-gaap:" + rc
            for k, cand in enumerate(_axis_values(inst, concept, axis, periods[-1], children, seg)):
                leaves = _pick_members(cand, T, children)
                if not leaves or len(leaves) < 2:
                    continue
                hist = {m: [None, None, cand[m]] for m in leaves}
                for j in (0, 1):
                    older = _axis_values(inst, concept, axis, periods[j], children, seg)
                    vals = older[k] if k < len(older) else (older[0] if older else {})
                    for m in leaves:
                        hist[m][j] = vals.get(m)
                named = [m for m in sorted(leaves, key=lambda m: -cand[m])
                         if _line_name(m).lower() not in ("other", "all other", "other revenue")]
                top = named[:5]
                lines = [(_line_name(m), [hist[m][j] if hist[m][j] is not None else (0.0 if total[j] else None)
                                              for j in range(3)]) for m in top]
                other = [None if total[j] is None else total[j] - sum(_nz(v[j]) for _, v in lines) for j in range(3)]
                if other[-1] is not None and abs(other[-1]) > 0.01 * T:
                    lines.append(("Other", other))
                elif other[-1] is not None and lines:  # rounding: fold into the largest line so lines tie
                    name, v = lines[0]
                    lines[0] = (name, [None if v[j] is None else v[j] + _nz(other[j]) for j in range(3)])
                big = max(v[-1] for _, v in lines)
                if len(top) >= 2 and big < 0.9 * T and all(v[1] is not None and v[1] >= 0 for _, v in lines):
                    return lines, "product" if axis == PRODUCT_AXIS else "geography"
    return [], ""


# ------------------------------------------------------------------ 10-K
def parse_annual(inst: Instance, is_concepts: set = frozenset(), children: Optional[dict] = None,
                 bs_concepts: set = frozenset()) -> History:
    global _IS_CONCEPTS
    _IS_CONCEPTS = set(is_concepts)
    periods = _annual_periods(inst, 3)
    if len(periods) < 2:
        raise ValueError("Could not find annual periods in the 10-K XBRL")
    while len(periods) < 3:  # young filers: pad the oldest year as empty
        p = periods[0]
        periods.insert(0, Context(p.start - timedelta(days=365), p.start - timedelta(days=1), frozenset()))
    fy = inst.dei("DocumentFiscalYearFocus")
    fy_last = int(fy) if fy and fy.isdigit() else periods[-1].end.year
    h = History(years=[fy_last - 2, fy_last - 1, fy_last],
                period_ends=[p.end for p in periods], period_starts=[p.start for p in periods])

    def row(key, scale=MM):
        return [_dur(inst, key, p, scale) for p in periods]

    IS = {k: row(k) for k in ["sales", "cogs", "rnd", "sga", "selling", "opinc", "costs_total", "interest_exp",
                              "interest_inc", "pretax", "tax", "ni_total", "ni_attr", "nci", "amort", "sbc",
                              "da", "dep", "contract_amort"]}
    if all(v is None for v in IS["amort"]):
        IS["amort"] = [amort_fallback(inst, p) for p in periods]
    fix_da_amort(IS, inst, periods, h.flags)
    IS["dil_shares"] = row("dil_shares")
    IS["basic_shares"] = row("basic_shares")
    IS["eps_dil"] = row("eps_dil", 1)
    for k in ("dil_shares", "basic_shares"):
        for i in range(3):
            ni_, eps_ = IS["ni_attr"][i], IS["eps_dil"][i]
            fixed = share_scale(IS[k][i], ni_ / eps_ if ni_ and eps_ else None)
            if fixed != IS[k][i]:
                if k == "dil_shares" and i == 2:
                    h.flags.append(f"Share counts are tagged on the wrong scale ({IS[k][i]:g}mm): restated to "
                                   f"{fixed:,.1f}mm from net income / EPS")
                IS[k][i] = fixed
    if all(v is None for v in IS["sales"]):
        raise ValueError("No revenue line found in the 10-K XBRL")
    for i in range(3):
        if IS["sga"][i] is None and IS["selling"][i] is not None:
            IS["sga"][i] = IS["selling"][i]
        elif IS["sga"][i] is not None and IS["selling"][i] is not None and \
                inst.value(["SellingGeneralAndAdministrativeExpense"], start=periods[i].start, end=periods[i].end) is None:
            IS["sga"][i] += IS["selling"][i]  # G&A and S&M reported separately
        if IS["ni_total"][i] is None and IS["ni_attr"][i] is not None:
            IS["ni_total"][i] = IS["ni_attr"][i] + _nz(IS["nci"][i])
        if IS["nci"][i] is None and IS["ni_total"][i] is not None and IS["ni_attr"][i] is not None:
            IS["nci"][i] = IS["ni_total"][i] - IS["ni_attr"][i]
        if IS["pretax"][i] is None and IS["ni_total"][i] is not None:
            IS["pretax"][i] = IS["ni_total"][i] + _nz(IS["tax"][i])
        if IS["cogs"][i] is None and IS["costs_total"][i] is not None and IS["sales"][i] is not None:
            IS["cogs"][i], inside = _cogs_from_total({k: IS[k][i] for k in IS})
            h.flags.append(f"FY{h.years[i]} cost of sales derived from total costs less SG&A and R&D"
                           + (" and interest expense (reported inside 'costs and expenses')" if inside else ""))
        seg_tag = IS["opinc"][i]
        IS["opinc"][i], how = _opinc(inst, periods[i], {k: IS[k][i] for k in IS}, is_concepts)
        if how == "derived" and i == 2:
            h.flags.append("Operating income is not a line on the income statement; EBIT derived as "
                           "pre-tax income + interest expense – other (income)"
                           + (f" (the {seg_tag:,.0f}mm OperatingIncomeLoss tag appears only outside the income statement)" if seg_tag else ""))
        if not IS["dil_shares"][i] and IS["ni_attr"][i] and IS["eps_dil"][i]:
            IS["dil_shares"][i] = IS["ni_attr"][i] / IS["eps_dil"][i]
            h.flags.append(f"FY{h.years[i]} diluted shares not tagged: derived as net income / diluted EPS")
        if IS["da"][i] is None and IS["dep"][i] is not None:
            IS["da"][i] = IS["dep"][i] + _nz(IS["amort"][i])
    IS["disc"] = [None if IS["ni_total"][i] is None or IS["pretax"][i] is None else
                  IS["ni_total"][i] - (IS["pretax"][i] - _nz(IS["tax"][i])) for i in range(3)]
    if any(v is not None and abs(v) >= 0.5 for v in IS["disc"]):
        h.flags.append("Net income includes discontinued operations / other after-tax items: shown on their own IS line")
    h.IS = IS
    for k in ("amort", "sbc", "interest_inc", "interest_exp", "nci", "cogs", "rnd"):
        IS[k] = [_nz(v) if IS["sales"][i] is not None else None for i, v in enumerate(IS[k])]

    # ---- balance sheet: two year-ends
    BS = {}
    ends = h.period_ends
    for k in ["cash", "cash_cf", "sti", "ar", "inv", "ahfs", "ca", "lt_inv", "ppe", "goodwill", "intang", "rou",
              "dta", "assets", "ap", "st_debt", "cur_ltd", "debt_cur_total", "lhfs", "cl", "ltd", "dtl", "postret",
              "lease_liab", "redeem_nci", "equity", "equity_total", "nci_bs", "le_total"]:
        BS[k] = [None] + [_inst(inst, k, d) for d in ends[1:]]
    for i in (1, 2):
        if BS["st_debt"][i] is None and BS["cur_ltd"][i] is None:
            BS["st_debt"][i] = BS["debt_cur_total"][i]
        if BS["ap"][i] is not None and inst.value(["AccountsPayableCurrent", "AccountsPayableTradeCurrent"], end=ends[i]) is None:
            if "Accounts payable reported combined with accrued liabilities" not in h.flags:
                h.flags.append("Accounts payable reported combined with accrued liabilities")
        if BS["equity"][i] is None and BS["equity_total"][i] is not None:
            BS["equity"][i] = BS["equity_total"][i] - _nz(BS["nci_bs"][i])
        if BS["nci_bs"][i] is None:
            BS["nci_bs"][i] = (BS["equity_total"][i] - _nz(BS["equity"][i])) if BS["equity_total"][i] is not None else 0.0
        # the model's total equity is equity + NCI, so tie the plugs to that sum
        BS["equity"][i] = _nz(BS["equity"][i])
        BS["equity_total"][i] = BS["equity"][i] + BS["nci_bs"][i]
        if BS["le_total"][i] is None:
            BS["le_total"][i] = BS["assets"][i]
        if BS["cash_cf"][i] is None:
            BS["cash_cf"][i] = BS["cash"][i]
        # model cash = cash that ties to the cash flow statement (incl. restricted cash)
        if BS["cash"][i] is not None and BS["cash_cf"][i] is not None and abs(BS["cash_cf"][i] - BS["cash"][i]) > 0.5:
            h.flags.append(f"FY{h.years[i]} cash includes {BS['cash_cf'][i] - BS['cash'][i]:,.0f}mm restricted cash "
                           "(ties to the cash flow statement); other current assets reduced to match")
        BS["cash"][i] = BS["cash_cf"][i] if BS["cash_cf"][i] is not None else BS["cash"][i]
        for k in ("sti", "ar", "inv", "ahfs", "lt_inv", "ppe", "goodwill", "intang", "rou", "dta",
                  "ap", "st_debt", "cur_ltd", "lhfs", "ltd", "dtl", "postret", "lease_liab", "redeem_nci"):
            BS[k][i] = _nz(BS[k][i])
        if BS["ca"][i] is None:  # unclassified balance sheet
            BS["ca"][i] = BS["cash"][i] + BS["sti"][i] + BS["ar"][i] + BS["inv"][i] + BS["ahfs"][i]
            h.flags.append("Balance sheet is unclassified; current assets = listed current items")
        if BS["cl"][i] is None:
            BS["cl"][i] = BS["ap"][i] + BS["st_debt"][i] + BS["cur_ltd"][i] + BS["lhfs"][i]
    # lessors: equipment on operating leases is depreciated in D&A, so it belongs with PP&E when it is its own
    # balance-sheet line (DE); when PP&E already includes it (CAT), only capex needs the purchases
    fleet = [None] + [_mm(inst.value(LESSOR_FLEET, end=d)) for d in ends[1:]]
    lessor_capex = [_lessor_capex(inst, p) for p in periods]
    own_line = "us-gaap:" + LESSOR_FLEET[0] in bs_concepts or any(
        "ExcludingLessorAsset" in c and "AfterAccumulated" in c for c in inst.facts)
    if any(lessor_capex) and own_line and any(fleet[1:]):
        for i in (1, 2):
            BS["ppe"][i] += _nz(fleet[i])
        h.flags.append(f"PP&E includes equipment leased to customers ({_nz(fleet[2]):,.0f}mm at FY{h.years[2]}): its "
                       "depreciation is in D&A, so capex includes net purchases of leased equipment")
    elif any(lessor_capex):
        h.flags.append("Capex includes net purchases of equipment leased to customers (its depreciation is in D&A)")
    BS["oca"] = [None] + [BS["ca"][i] - BS["cash"][i] - BS["sti"][i] - BS["ar"][i] - BS["inv"][i] - BS["ahfs"][i]
                          for i in (1, 2)]
    BS["other_assets"] = [None] + [BS["assets"][i] - BS["ca"][i] - BS["lt_inv"][i] - BS["ppe"][i] - BS["goodwill"][i]
                                   - BS["intang"][i] - BS["rou"][i] - BS["dta"][i] for i in (1, 2)]
    for i in (1, 2):  # restricted cash held outside current assets: take it from other assets instead
        if BS["oca"][i] < 0:
            BS["other_assets"][i] += BS["oca"][i]
            BS["oca"][i] = 0.0
    BS["accrued"] = [None] + [BS["cl"][i] - BS["ap"][i] - BS["st_debt"][i] - BS["cur_ltd"][i] - BS["lhfs"][i]
                              for i in (1, 2)]
    BS["other_liab"] = [None] + [BS["le_total"][i] - BS["equity_total"][i] - BS["redeem_nci"][i] - BS["cl"][i]
                                 - BS["ltd"][i] - BS["dtl"][i] - BS["postret"][i] - BS["lease_liab"][i]
                                 for i in (1, 2)]
    h.BS = BS

    # ---- cash flow
    CF = {k: row(k) for k in ["pension_nc", "d_ar", "d_inv", "d_oca", "d_ap", "d_accr", "cfo", "capex", "acq",
                              "divest", "cfi", "stock_proc", "buyback", "divs", "cff"]}
    CF["debt"] = [_debt_flow(inst, p) for p in periods]
    CF["capex"] = [v if x is None else _nz(v) + x for v, x in zip(CF["capex"], lessor_capex)]
    if not any(CF["capex"]):
        # VZ: "capital expenditures (including capitalized software)" on the other-productive-assets tag; checked
        # against the PP&E roll-forward below like a company tag (elsewhere it can be a small side line)
        other = [_mm(inst.value(["PaymentsToAcquireOtherProductiveAssets"], start=p.start, end=p.end)) for p in periods]
        custom = (["us-gaap:PaymentsToAcquireOtherProductiveAssets"] if any(other) else
                  sorted(c for c in inst.facts if CAPEX_RX.match(c) and not CAPEX_EXCL.search(c)))
        dep0 = IS["dep"][2] if IS["dep"][2] else _nz(IS["da"][2]) - _nz(IS["amort"][2])
        p1, p0 = BS.get("ppe", [None, None, None])[2], BS.get("ppe", [None, None, None])[1]
        rf = (p1 - p0 + dep0) if p1 and p0 else None
        yv = []
        for i in range(3):
            vals = [inst.value([c], start=periods[i].start, end=periods[i].end) for c in custom]
            yv.append(list({round(abs(v) / MM, 1): None for v in vals if v is not None}))  # one value, one figure
        # parts add up, but one tag can be the total of the others (NEE: the segment-table total beside FPL's own
        # line): the PP&E roll-forward picks which reading fits, the same for every year
        pick = sum
        if rf and rf > 0 and len(yv[2]) > 1:
            pick = min((sum, max), key=lambda fn: abs(math.log(fn(yv[2]) / rf)))
        for i in range(3):
            if yv[i]:
                CF["capex"][i] = pick(yv[i])
        if rf and rf > 0 and not 0.75 * rf <= _nz(CF["capex"][2]) <= 1.33 * rf:  # missing, partial or double-counted
            CF["capex"][2] = rf
            h.flags.append(f"Capital expenditures are not on a standard tag: FY{h.years[2]} capex {rf:,.0f}mm from the "
                           f"PP&E roll-forward (PP&E {p0:,.0f} → {p1:,.0f} + depreciation {dep0:,.0f}); earlier years "
                           + ("from the company's own capex tags" if any(CF["capex"][:2]) else "not tagged"))
        elif any(CF["capex"]):
            h.flags.append("Capital expenditures read from a non-standard tag (" + ", ".join(
                c.split(":")[-1] for c in custom[:3]) + ")" + ("; the largest is the total of the others"
                                                              if pick is max else "") +
                (f", within 25% of the PP&E roll-forward ({rf:,.0f}mm)"
                                                              if rf else ": no PP&E roll-forward to check it against"))
    begin = [_mm(inst.value(C["cash_cf"], end=p.start - timedelta(days=1), tol=8)) for p in periods]
    end_ = [_mm(inst.value(C["cash_cf"], end=p.end)) for p in periods]
    for i in range(3):
        if begin[i] is None and i > 0:
            begin[i] = end_[i - 1]
    out = {"ni": IS["ni_total"], "da": [_nz(v) for v in IS["da"]], "sbc": IS["sbc"],
           "pension": [_nz(v) for v in CF["pension_nc"]],
           "ar": [-_nz(v) for v in CF["d_ar"]], "inv": [-_nz(v) for v in CF["d_inv"]],
           "oca": [-_nz(v) for v in CF["d_oca"]], "ap": [_nz(v) for v in CF["d_ap"]],
           "accr": [_nz(v) for v in CF["d_accr"]], "cfo": CF["cfo"],
           "capex": [-_nz(v) for v in CF["capex"]], "acq": [-_nz(v) for v in CF["acq"]],
           "divest": [_nz(v) for v in CF["divest"]], "cfi": CF["cfi"],
           "debt": [_nz(v) for v in CF["debt"]], "stock": [_nz(v) for v in CF["stock_proc"]],
           "buyback": [-_nz(v) for v in CF["buyback"]], "divs": [-_nz(v) for v in CF["divs"]],
           "cff": CF["cff"], "begin": begin, "end": end_}
    out["other_op"], out["other_inv"], out["nci_other"], out["fx"] = [], [], [], []
    for i in range(3):
        if out["cfo"][i] is None or out["ni"][i] is None:
            for k in ("other_op", "other_inv", "nci_other", "fx"):
                out[k].append(None)
            continue
        op_lines = (out["ni"][i] + out["da"][i] + _nz(out["sbc"][i]) + out["pension"][i] + out["ar"][i]
                    + out["inv"][i] + out["oca"][i] + out["ap"][i] + out["accr"][i])
        out["other_op"].append(out["cfo"][i] - op_lines)
        out["other_inv"].append(_nz(out["cfi"][i]) - out["capex"][i] - out["acq"][i] - out["divest"][i])
        out["nci_other"].append(_nz(out["cff"][i]) - out["debt"][i] - out["stock"][i] - out["buyback"][i] - out["divs"][i])
        if begin[i] is not None and end_[i] is not None:
            out["fx"].append(end_[i] - begin[i] - out["cfo"][i] - _nz(out["cfi"][i]) - _nz(out["cff"][i]))
        else:
            out["fx"].append(0.0)
            if begin[i] is None and end_[i] is not None:
                out["begin"][i] = end_[i] - out["cfo"][i] - _nz(out["cfi"][i]) - _nz(out["cff"][i])
    h.CF = out
    h.content = content_history(inst, periods)

    IS["nonrec"] = [nonrec_items(inst, p) for p in periods]

    # ---- segments
    seg = extract_segments(inst, periods, IS["sales"])
    if seg:
        sp = sum(_nz(s_.profit[2]) for s_ in seg["segments"]) + _nz(seg["corp_profit"][2])
        # not 'other operating expense': AMZN's fulfillment and technology costs land there ($269bn)
        op = _nz(IS["opinc"][2]) + _nz(IS["amort"][2]) + _nz(IS["sbc"][2])
        if op > 0 and sp < 0.4 * op:
            h.flags.append(f"Segment profit as reported ({sp:,.0f}mm in FY{h.years[2]}) is far below operating "
                           f"profit ({op:,.0f}mm): the segment measure is after interest and tax (utilities report "
                           "segment net income), so the model is built as one segment on operating income")
            seg = None
    if seg:
        h.segments = seg["segments"]
        h.segments_reported = True
        h.seg_profit_concept = seg["profit_concept"]
        h.seg_measure = seg["measure"]
        h.corp_profit = seg["corp_profit"]
        h.corp_members = seg["corp_members"]
        moved = [_pretax_basis(sum(_nz(s.profit[i]) for s in h.segments) + _nz(h.corp_profit[i]),
                               {k: IS[k][i] for k in ("sales", "pretax", "opinc")})
                 if h.corp_profit[i] is not None else 0.0 for i in range(3)]
        if moved[2]:
            h.corp_profit = [None if v is None else v + m for v, m in zip(h.corp_profit, moved)]
            h.flags.append(f"Segment profit is reported before tax: interest and other non-operating items "
                           f"({moved[2]:,.0f}mm in FY{h.years[2]}) moved out of Corporate & All Other, so total "
                           "segment profit is on the operating basis the forecast uses")
        h.corp_sales = [None if IS["sales"][i] is None else
                        IS["sales"][i] - sum(_nz(s.sales[i]) for s in h.segments) for i in range(3)]
    else:
        prof = [None if IS["opinc"][i] is None else
                IS["opinc"][i] + _nz(IS["amort"][i]) + _nz(IS["sbc"][i]) + _other_opex({k: IS[k][i] for k in IS})
                for i in range(3)]
        h.segments = [Segment("Total company", "total", list(IS["sales"]), prof)]
        h.corp_sales = [0.0 if v is not None else None for v in IS["sales"]]
        h.corp_profit = [0.0 if v is not None else None for v in IS["sales"]]
        h.flags.append("No usable segment disclosure in XBRL: modelled as one segment (segment profit = operating "
                       "income + intangible amortization + stock comp + other operating expense such as acquired "
                       "IPR&D, impairments and restructuring, which sit in 'non-segment charges')")
    children = children or {}
    if h.segments[0].member == "total":
        h.segments[0].lines, h.lines_axis = revenue_lines(inst, periods, IS["sales"], children)
    else:
        for sg in h.segments:
            if sg.member != "other" and not CAPTIVE_RX.search(sg.name):
                sg.lines, ax = revenue_lines(inst, periods, sg.sales, children, seg=sg.member, axes=(PRODUCT_AXIS,))
                h.lines_axis = h.lines_axis or ax
    h.captive = captive_finance(inst, periods, h.segments, IS["interest_exp"])
    if not h.captive:
        h.captive = captive_from_split(inst, periods, h.period_ends)
        if h.captive:
            h.flags.append(f"Captive finance arm ({h.captive['segment']}) presented in its own columns: "
                           f"{h.captive['debt']:,.0f}mm of the {h.captive['debt_total']:,.0f}mm debt "
                           f"({h.captive['share']:.0%}) funds its receivables and is excluded from net debt; its "
                           f"interest ({h.captive['interest'][-1]:,.0f}mm in FY{h.years[-1]}) is already an "
                           "operating cost, so the interest line carries the industrial debt only")
    if h.captive and not h.captive.get("on_face"):
        # the finance arm's interest is its cost of goods sold: move it above EBIT (it is inside its segment
        # profit), leaving only corporate interest on the interest line
        for i, fi in enumerate(h.captive["interest"]):
            if IS["opinc"][i] is not None:
                IS["opinc"][i] -= fi
            if IS["interest_exp"][i] is not None:
                IS["interest_exp"][i] -= fi
        h.flags.append(f"Captive finance arm ({h.captive['segment']}): its interest expense "
                       f"({h.captive['interest'][-1]:,.0f}mm in FY{h.years[-1]}, {h.captive['share']:.0%} of the total) "
                       "is treated as an operating cost inside EBIT, and the debt that funds its receivables is "
                       "excluded from net debt (Assumptions: captive-finance share of debt)")
    vals = [v for c, v in inst.facts.get("dei:EntityCommonStockSharesOutstanding", {}).items()]
    h.cover_shares = sum(vals) / MM if vals else None  # multiple share classes are summed
    h.dps = [_dur(inst, "dps", p, 1) for p in periods]
    h.debt_like = debt_like_items(inst, h.period_ends[1:])
    return h


# ------------------------------------------------------------------ 10-Q
def _find_duration(inst: Instance, end: date, approx_days: int, tol_days: int = 20) -> Optional[Context]:
    ctxs = inst.durations(C["sales"] + C["ni_attr"] + C["cfo"], approx_days - tol_days, approx_days + tol_days)
    ctxs = [c for c in ctxs if abs((c.end - end).days) <= 6]
    return min(ctxs, key=lambda c: abs(c.days - approx_days)) if ctxs else None


def _ytd_block(inst: Instance, ctx: Context, segs: list, profit_c: str, corp_members: list,
               is_concepts: set = frozenset()) -> dict:
    g = lambda k, s=MM: _dur(inst, k, ctx, s)
    d = {k: g(k) for k in ["sales", "cogs", "rnd", "sga", "selling", "opinc", "interest_exp", "pretax", "tax",
                          "ni_total", "ni_attr", "nci", "da", "cfo", "capex", "amort", "sbc", "costs_total",
                          "buyback", "acq", "divs", "dep", "contract_amort"]}
    if d["amort"] is None:
        d["amort"] = amort_fallback(inst, ctx)
    fix_da_amort(d, inst, [ctx])
    d["eps"] = g("eps_dil", 1)
    lc = _lessor_capex(inst, ctx)
    if lc is not None:  # same basis as the 10-K columns
        d["capex"] = _nz(d["capex"]) + lc
    if d["sga"] is None and d["selling"] is not None:
        d["sga"] = d["selling"]
    elif d["sga"] is not None and d["selling"] is not None and \
            inst.value(["SellingGeneralAndAdministrativeExpense"], start=ctx.start, end=ctx.end) is None:
        d["sga"] += d["selling"]  # G&A and S&M reported separately: same definition as the 10-K columns
    if d["pretax"] is None and d["ni_total"] is not None:
        d["pretax"] = d["ni_total"] + _nz(d["tax"])
    if d["cogs"] is None and d["costs_total"] is not None:
        d["cogs"], _ = _cogs_from_total(d)
    d["opinc"], _ = _opinc(inst, ctx, d, is_concepts)
    if d["nci"] is None and d["ni_total"] is not None and d["ni_attr"] is not None:
        d["nci"] = d["ni_total"] - d["ni_attr"]
    if d["ni_total"] is None and d["ni_attr"] is not None:
        d["ni_total"] = d["ni_attr"] + _nz(d["nci"])
    sp = None
    if profit_c and segs and segs[0].member != "total":
        vals = [_segment_value(inst, profit_c, s.member, ctx.start, ctx.end) for s in segs if s.member != "other"]
        if all(v is not None for v in vals):
            corp = [_segment_value(inst, profit_c, m, ctx.start, ctx.end) for m in corp_members]
            sp = sum(vals) + sum(v for v in corp if v is not None)
            sp += _pretax_basis(sp, d)  # same operating basis as the 10-K columns
        else:  # resegmented 10-Q (ETN: Mobility instead of Vehicle + eMobility): its own segments, same measure
            own = [_segment_value(inst, profit_c, m, ctx.start, ctx.end)
                   for m in _segment_members(inst, profit_c, ctx.end)[0]]
            own = [v for v in own if v is not None]
            if own:
                corp = [_segment_value(inst, profit_c, m, ctx.start, ctx.end) for m in corp_members]
                sp = sum(own) + sum(v for v in corp if v is not None)
                sp += _pretax_basis(sp, d)
    if sp is None and d["opinc"] is not None:
        sp = d["opinc"] + _nz(d["amort"]) + _nz(d["sbc"]) + _other_opex(d)
    d["seg_profit"] = sp
    d["nonrec"] = nonrec_items(inst, ctx)
    return d


def parse_quarterly(inst: Instance, hist: History, is_concepts: set = frozenset()) -> Interim:
    global _IS_CONCEPTS
    _IS_CONCEPTS = set(is_concepts)
    profit_c, corp_members = hist.seg_profit_concept, hist.corp_members
    end = date.fromisoformat(inst.dei("DocumentPeriodEndDate")[:10]) if inst.dei("DocumentPeriodEndDate") else None
    fp = (inst.dei("DocumentFiscalPeriodFocus") or "").upper()
    fy_start = hist.period_ends[-1] + timedelta(days=1)
    if end is None:
        return Interim(False)
    ytd_days = (end - fy_start).days
    ytd = _find_duration(inst, end, ytd_days)
    ytd_prior = _find_duration(inst, end - timedelta(days=365), ytd_days) or \
        _find_duration(inst, end - timedelta(days=364), ytd_days)
    if ytd is None or ytd_prior is None:
        return Interim(False)
    q = _find_duration(inst, end, 91, 12)
    qp = _find_duration(inst, end - timedelta(days=365), 91, 12) or _find_duration(inst, end - timedelta(days=364), 91, 12)
    segs = hist.segments
    it = Interim(True, fiscal_period=fp, period_end=end)
    n = {"Q1": "Q1", "Q2": "H1", "Q3": "9M"}.get(fp, "YTD")
    fy_cur, fy_prev = hist.years[-1] + 1, hist.years[-1]
    it.label_prior, it.label_cur = f"{n} {fy_prev}", f"{n} {fy_cur}"
    it.ltm_label = f"LTM {end:%b-%y}"
    it.quarter_label = f"{fp or 'Q'} {fy_cur}"
    it.ytd_cur = _ytd_block(inst, ytd, segs, profit_c, corp_members, is_concepts)
    it.ytd_prior = _ytd_block(inst, ytd_prior, segs, profit_c, corp_members, is_concepts)
    if hist.captive and not hist.captive.get("on_face"):  # same reclassification as the 10-K columns
        for d, c in ((it.ytd_cur, ytd), (it.ytd_prior, ytd_prior)):
            fi = _captive_interest(inst, hist.captive["member"], c)
            if d.get("opinc") is not None:
                d["opinc"] -= fi
            if d.get("interest_exp") is not None:
                d["interest_exp"] -= fi
    for k, keys in {"cur": end, "fy": hist.period_ends[-1]}.items():
        bs = {key: _inst(inst, key, keys) for key in ["cash", "sti", "st_debt", "cur_ltd", "ltd", "equity_total",
                                                        "equity", "nci_bs", "debt_cur_total", "cash_cf", "lt_inv"]}
        if bs["cash"] is None:
            bs["cash"] = bs["cash_cf"]
        if bs["st_debt"] is None and bs["cur_ltd"] is None:
            bs["st_debt"] = bs["debt_cur_total"]
        if bs["equity_total"] is None:
            bs["equity_total"] = _nz(bs["equity"]) + _nz(bs["nci_bs"])
        if bs["equity"] is None:
            bs["equity"] = bs["equity_total"] - _nz(bs["nci_bs"])
        setattr(it, "bs_" + k, {kk: _nz(v) for kk, v in bs.items()})
    if hist.captive.get("on_face"):  # the finance arm's share of the 10-Q debt (CAT: 76% at Sep 30, 2026)
        parts = [business_split(inst, C[key], end=end) for key in DEBT_KEYS]
        tot, cap = sum(p_[0] for p_ in parts if p_), sum(p_[1] for p_ in parts if p_)
        if tot > 0:
            it.captive_share = round(min(cap / tot, 0.98), 3)
    if q is not None and segs and segs[0].member != "total":
        for s in segs:
            if s.member == "other":
                continue
            row = {"name": s.name}
            for tag, c in (("cur", q), ("prior", qp)):
                row["sales_" + tag] = _segment_sales(inst, s.member, c.start, c.end) if c else None
                row["profit_" + tag] = _segment_value(inst, profit_c, s.member, c.start, c.end) if c else None
            it.q_segments.append(row)
        it.q_corp = {}
        for tag, c in (("cur", q), ("prior", qp)):
            sales = _dur(inst, "sales", c) if c else None
            it.q_corp["sales_" + tag] = (None if sales is None else
                                         sales - sum(_nz(r["sales_" + tag]) for r in it.q_segments))
            vals = [_segment_value(inst, profit_c, m, c.start, c.end) for m in corp_members] if c else []
            vals = [v for v in vals if v is not None]
            it.q_corp["profit_" + tag] = sum(vals) if vals else None
            if vals and all(r["profit_" + tag] is not None for r in it.q_segments):
                blk = _ytd_block(inst, c, [], None, [], is_concepts)  # the quarter's P&L, for the basis test
                it.q_corp["profit_" + tag] += _pretax_basis(
                    sum(r["profit_" + tag] for r in it.q_segments) + it.q_corp["profit_" + tag], blk)
    elif q is not None and segs and segs[0].member == "total":
        # single segment: the quarter row is the company (same definition as the 10-K column:
        # operating income + acquired-intangible amortization + stock comp)
        row = {"name": segs[0].name}
        for tag, c in (("cur", q), ("prior", qp)):
            blk = _ytd_block(inst, c, segs, profit_c, corp_members, is_concepts) if c else {}
            ytd_ = it.ytd_cur if tag == "cur" else it.ytd_prior
            for k_ in ("sbc", "amort"):  # cash-flow items a 10-Q tags for the year to date only: the quarter's share
                # by sales (PM round 6, ADBE: the Q3 margin read 36.9% without stock comp against 44.8% for 9M)
                if blk.get("seg_profit") is not None and ytd_.get(k_) and ytd_.get("sales") and blk.get("sales"):
                    est = ytd_[k_] * blk["sales"] / ytd_["sales"]
                    if blk.get(k_) is None or blk[k_] < 0.5 * est:  # missing, or a partial tag (ADBE Q3: $35mm of
                        # stock comp against $1,582mm for nine months)
                        blk["seg_profit"] += est - _nz(blk.get(k_))
            row["sales_" + tag] = blk.get("sales")
            row["profit_" + tag] = blk.get("seg_profit")
            row["ebit_" + tag] = (None if blk.get("opinc") is None else  # segment profit basis when C14 = 1
                                  blk["opinc"] + _other_opex(blk))
            row["full_" + tag] = blk.get("seg_profit")
        it.q_segments.append(row)
        it.q_corp = {"sales_cur": 0.0, "sales_prior": 0.0, "profit_cur": 0.0, "profit_prior": 0.0}
    vals = list(inst.facts.get("dei:EntityCommonStockSharesOutstanding", {}).values())
    it.cover_shares = sum(vals) / MM if vals else None
    ratios = [v for c, v in inst.facts.get("us-gaap:StockholdersEquityNoteStockSplitConversionRatio1", {}).items()
              if not c.dims]
    it.split_ratio = ratios[0] if ratios else None
    it.debt_like = debt_like_items(inst, [end, hist.period_ends[-1]])
    disc = inst.value(DISC_OPS, start=ytd.start, end=ytd.end)
    it.disc_ops_ytd = _mm(disc) if disc else None
    if segs and segs[0].member != "total":
        it.missing_segments = [s.name for s in segs if s.member != "other" and
                               _segment_sales(inst, s.member, ytd.start, ytd.end) is None]
        if it.missing_segments:
            known = {s.member for s in segs}
            q_ops = [m for c in C["sales"] for m in _segment_members(inst, c, end)[0]]
            it.new_segments = list(dict.fromkeys(humanize_member(m) for m in q_ops if m not in known))
    ref = it.cover_shares or hist.cover_shares
    it.ytd_basic = share_scale(_dur(inst, "basic_shares", ytd), ref)
    it.ytd_diluted = share_scale(_dur(inst, "dil_shares", ytd), ref)
    if q is not None:
        it.q_basic = share_scale(_dur(inst, "basic_shares", q), ref)
        it.q_diluted = share_scale(_dur(inst, "dil_shares", q), ref)
        it.q_dps = _dur(inst, "dps", q, 1)
    return it


def _segment_sales(inst: Instance, member: str, start, end) -> Optional[float]:
    """Segment revenue under whichever revenue concept the filing uses for it (10-Qs often tag segment
    revenue with a different concept than the product / geography breakdowns)."""
    for c in C["sales"]:
        v = _segment_value(inst, c, member, start, end)
        if v is not None:
            return v
    return None


def empty_interim(hist: History) -> Interim:
    """No 10-Q since the 10-K: LTM = last fiscal year, YTD columns empty."""
    fy = hist.years[-1]
    it = Interim(False, fiscal_period="", period_end=hist.period_ends[-1],
                 label_prior="n/a", label_cur="n/a", ltm_label=f"LTM {hist.period_ends[-1]:%b-%y}",
                 quarter_label=f"FY{fy}")
    keys = ["sales", "cogs", "rnd", "sga", "opinc", "interest_exp", "pretax", "tax", "nci", "eps", "seg_profit",
            "da", "cfo", "capex"]
    it.ytd_cur = {k: None for k in keys}   # no 10-Q yet: the year-to-date columns stay blank
    it.ytd_prior = {k: None for k in keys}
    bs = hist.BS
    snap = {"cash": bs["cash"][2], "sti": bs["sti"][2], "st_debt": bs["st_debt"][2], "cur_ltd": bs["cur_ltd"][2],
            "ltd": bs["ltd"][2], "equity_total": bs["equity_total"][2]}
    it.bs_cur = dict(snap)
    it.bs_fy = dict(snap)
    it.cover_shares = hist.cover_shares
    return it
