"""Leakage guard for the Lending Club credit-default model: which columns may reach the model, and why.

The model scores a loan APPLICATION. Anything that is only known after the loan was issued (payments,
balances, recoveries, hardship plans, settlements, refreshed credit pulls, the loan status itself) would
let the model "see the future" and make test metrics look far better than any real decision could be.

Every column of the Lending Club accepted-loans file (accepted_2007_to_2018Q4.csv, 151 columns) gets
exactly one decision in COLUMN_DECISIONS below, with a one-line reason. docs/leakage.md presents the same
table for reviewers, and tests/risk/test_leakage.py checks that the two agree.

Two groups are rejected by assert_no_leakage:

- BANNED_POST_ORIGINATION_COLUMNS: known only after origination, or the outcome itself. Always rejected.
- LENDER_OUTPUT_COLUMNS: Lending Club's own decisions about the application (grade, sub_grade, interest
  rate, installment, funded amounts, listing status). They are known at origination, but they are the
  output of Lending Club's underwriting and pricing model. Feeding them to our model would mostly copy that
  model and hide what our own features can do, so they are rejected by default. sub_grade is used only as
  a separate benchmark score (how well does Lending Club's own grade rank the same test loans?).

The other excluded groups (identifiers and free text, geography, joint-application fields, fields added
late in the sample, constant or redundant fields) are not leakage, so the guard does not reject them; they
are simply not on the model's allow-list (uwcopilot.risk.credit.CREDIT_FEATURES).

Caveat: decisions follow the Lending Club data dictionary (LCDataDictionary.xlsx, shipped with the public
loan data). We cannot independently verify that no bureau field was refreshed after origination.
"""

from collections.abc import Iterable
from dataclasses import dataclass

POST_ORIGINATION = "banned_post_origination"
LENDER_OUTPUT = "lender_output"
FEATURE = "feature"
FEATURE_SOURCE = "feature_source"
SPLIT_FILTER_OR_TARGET = "split_filter_or_target"
IDENTIFIER_OR_TEXT = "excluded_identifier_or_free_text"
GEOGRAPHY = "excluded_geography_fair_lending"
JOINT_APPLICATION = "excluded_joint_application"
LATE_INTRODUCED = "excluded_added_late_in_sample"
OPERATIONAL = "excluded_operational"
CONSTANT_OR_REDUNDANT = "excluded_constant_or_redundant"

CATEGORY_DESCRIPTIONS = {
    POST_ORIGINATION: "Known only after the loan was issued (or the outcome itself). Rejected by the leakage guard.",
    LENDER_OUTPUT: (
        "Lending Club's own underwriting or pricing output. Rejected by the leakage guard for the main model; "
        "sub_grade is reported only as a benchmark."
    ),
    FEATURE: "Application-time field used as a model feature unchanged.",
    FEATURE_SOURCE: "Application-time field turned into a model feature (parsed or derived; see reason).",
    SPLIT_FILTER_OR_TARGET: "Used to define the target, the out-of-time split or the population, never as a feature.",
    IDENTIFIER_OR_TEXT: "Identifier or free text: no stable signal, and free text can carry personal data.",
    GEOGRAPHY: "Geography: a well-known proxy for protected characteristics (redlining risk).",
    JOINT_APPLICATION: "Joint-application field. Joint applications are out of scope (individual applications only).",
    LATE_INTRODUCED: (
        "Added late in the sample, so missing for nearly all training loans (the loader measures the missing "
        "share per issue year to confirm this)."
    ),
    OPERATIONAL: "Operational detail of the loan, not a credit attribute of the applicant.",
    CONSTANT_OR_REDUNDANT: "Constant or a duplicate of another field.",
}

# Name of the derived target column added by the loader; it must never be a feature either.
TARGET_COLUMN = "is_bad"


@dataclass(frozen=True)
class ColumnDecision:
    """What happens to one raw column, and why."""

    column: str
    category: str
    reason: str


def _decisions(category: str, entries: dict[str, str]) -> list[ColumnDecision]:
    """Build ColumnDecision objects for one category from {column: reason}."""
    return [ColumnDecision(column=column, category=category, reason=reason) for column, reason in entries.items()]


_POST_ORIGINATION = {
    "loan_status": "The outcome (Fully Paid, Charged Off, Current, ...); the target is derived from it.",
    "pymnt_plan": "Whether a payment plan was put in place for the loan, which happens after issuance.",
    "out_prncp": "Outstanding principal remaining; changes every month after issuance.",
    "out_prncp_inv": "Outstanding principal held by investors; post-issuance balance.",
    "total_pymnt": "Payments received to date; a defaulted loan has paid less, so it encodes the outcome.",
    "total_pymnt_inv": "Payments received to date on the investor-funded portion.",
    "total_rec_prncp": "Principal received to date.",
    "total_rec_int": "Interest received to date.",
    "total_rec_late_fee": "Late fees received to date; only exists if the borrower paid late.",
    "recoveries": "Post-charge-off gross recovery; non-zero almost only for charged-off loans.",
    "collection_recovery_fee": "Post-charge-off collection fee; only exists after a charge-off.",
    "last_pymnt_d": "Date of the last payment received; an early date reveals an early default or payoff.",
    "last_pymnt_amnt": "Amount of the last payment; a large final payment reveals a payoff.",
    "next_pymnt_d": "Next scheduled payment date; empty once a loan is paid off or charged off.",
    "last_credit_pull_d": "Most recent month Lending Club pulled credit for the loan; refreshed after issuance.",
    "last_fico_range_high": "FICO from the most recent credit pull (after issuance); falls when a borrower defaults.",
    "last_fico_range_low": "FICO from the most recent credit pull (after issuance); falls when a borrower defaults.",
    "hardship_flag": "Whether the borrower is on a hardship plan (after issuance).",
    "hardship_type": "Hardship plan type (after issuance).",
    "hardship_reason": "Reason the borrower requested a hardship plan (after issuance).",
    "hardship_status": "Hardship plan status (after issuance).",
    "deferral_term": "Months of payment deferral under a hardship plan (after issuance).",
    "hardship_amount": "Interest payment under a hardship plan (after issuance).",
    "hardship_start_date": "Hardship plan start date (after issuance).",
    "hardship_end_date": "Hardship plan end date (after issuance).",
    "payment_plan_start_date": "First payment date of a hardship plan (after issuance).",
    "hardship_length": "Hardship plan length in months (after issuance).",
    "hardship_dpd": "Days past due when the hardship plan started (after issuance).",
    "hardship_loan_status": "Loan status when the hardship plan started (after issuance).",
    "orig_projected_additional_accrued_interest": "Projected extra interest from a hardship plan (after issuance).",
    "hardship_payoff_balance_amount": "Payoff balance when the hardship plan started (after issuance).",
    "hardship_last_payment_amount": "Last payment amount when the hardship plan started (after issuance).",
    "debt_settlement_flag": "Whether a charged-off borrower works with a debt-settlement company (after charge-off).",
    "debt_settlement_flag_date": "Date the debt-settlement flag was set (after charge-off).",
    "settlement_status": "Status of a settlement plan (after charge-off).",
    "settlement_date": "Date the settlement plan was agreed (after charge-off).",
    "settlement_amount": "Amount agreed in the settlement (after charge-off).",
    "settlement_percentage": "Settlement amount as a share of the balance (after charge-off).",
    "settlement_term": "Number of months of the settlement plan (after charge-off).",
}

_LENDER_OUTPUT = {
    "grade": "Lending Club's risk grade (A-G) from its own model; using it would copy that model.",
    "sub_grade": "Lending Club's risk sub-grade (A1-G5); used only as a benchmark score, never as a feature.",
    "int_rate": "Interest rate set by Lending Club from the sub-grade; a pricing output, not an applicant fact.",
    "installment": "Monthly payment computed from amount, term and int_rate, so it reveals int_rate.",
    "funded_amnt": "Amount Lending Club committed, which can be less than requested after its review.",
    "funded_amnt_inv": "Amount investors funded during the listing period; investors react to the grade.",
    "initial_list_status": "Whole vs fractional listing, chosen by Lending Club for investors after approval.",
}

_FEATURES = {
    "loan_amnt": "Amount the applicant asked for.",
    "annual_inc": "Self-reported annual income (also used for loan_to_income).",
    "dti": "Debt-to-income ratio at application, excluding the new loan and mortgage.",
    "fico_range_low": "Lower bound of the FICO range at application.",
    "delinq_2yrs": "30+ days past-due incidents in the past 2 years (bureau, at application).",
    "inq_last_6mths": "Credit inquiries in the past 6 months (bureau, at application).",
    "mths_since_last_delinq": "Months since the last delinquency (empty = none on file).",
    "mths_since_last_record": "Months since the last public record (empty = none on file).",
    "open_acc": "Number of open credit lines.",
    "pub_rec": "Number of derogatory public records.",
    "revol_bal": "Total revolving balance.",
    "revol_util": "Revolving utilisation rate (percent strings are converted to numbers).",
    "total_acc": "Total number of credit lines on file.",
    "collections_12_mths_ex_med": "Collections in the past 12 months, excluding medical collections.",
    "mths_since_last_major_derog": "Months since the most recent 90-day or worse rating.",
    "acc_now_delinq": "Accounts currently delinquent.",
    "tot_coll_amt": "Total collection amounts ever owed.",
    "tot_cur_bal": "Total current balance of all accounts.",
    "total_rev_hi_lim": "Total revolving credit limit.",
    "acc_open_past_24mths": "Trades opened in the past 24 months.",
    "avg_cur_bal": "Average current balance of all accounts.",
    "bc_open_to_buy": "Total open-to-buy on revolving bank cards.",
    "bc_util": "Balance to credit limit on bank cards.",
    "chargeoff_within_12_mths": "Charge-offs within the past 12 months (on other credit, at application).",
    "delinq_amnt": "Past-due amount owed on accounts now delinquent.",
    "mo_sin_old_il_acct": "Months since the oldest installment account was opened.",
    "mo_sin_old_rev_tl_op": "Months since the oldest revolving account was opened.",
    "mo_sin_rcnt_rev_tl_op": "Months since the most recent revolving account was opened.",
    "mo_sin_rcnt_tl": "Months since the most recent account was opened.",
    "mort_acc": "Number of mortgage accounts.",
    "mths_since_recent_bc": "Months since the most recent bank card was opened.",
    "mths_since_recent_bc_dlq": "Months since the most recent bank card delinquency.",
    "mths_since_recent_inq": "Months since the most recent inquiry.",
    "mths_since_recent_revol_delinq": "Months since the most recent revolving delinquency.",
    "num_accts_ever_120_pd": "Accounts ever 120 or more days past due.",
    "num_actv_bc_tl": "Active bank card accounts.",
    "num_actv_rev_tl": "Active revolving trades.",
    "num_bc_sats": "Satisfactory bank card accounts.",
    "num_bc_tl": "Bank card accounts.",
    "num_il_tl": "Installment accounts.",
    "num_op_rev_tl": "Open revolving accounts.",
    "num_rev_accts": "Revolving accounts.",
    "num_rev_tl_bal_gt_0": "Revolving trades with a balance above 0.",
    "num_sats": "Satisfactory accounts.",
    "num_tl_120dpd_2m": "Accounts currently 120 days past due (updated in the 2 months before application).",
    "num_tl_30dpd": "Accounts currently 30 days past due (updated in the 2 months before application).",
    "num_tl_90g_dpd_24m": "Accounts 90+ days past due in the past 24 months.",
    "num_tl_op_past_12m": "Accounts opened in the past 12 months.",
    "pct_tl_nvr_dlq": "Percent of trades never delinquent.",
    "percent_bc_gt_75": "Percent of bank cards above 75% of their limit.",
    "pub_rec_bankruptcies": "Public-record bankruptcies.",
    "tax_liens": "Tax liens.",
    "tot_hi_cred_lim": "Total high credit / credit limit.",
    "total_bal_ex_mort": "Total credit balance excluding mortgage.",
    "total_bc_limit": "Total bank card credit limit.",
    "total_il_high_credit_limit": "Total installment high credit / credit limit.",
    "home_ownership": "Housing status from the application (ANY, NONE and OTHER are grouped as OTHER).",
    "verification_status": "Whether Lending Club verified the income before issuance (set during underwriting).",
    "purpose": "Loan purpose chosen by the applicant.",
}

_FEATURE_SOURCES = {
    "term": "Parsed to term_months (36 or 60), chosen by the applicant.",
    "emp_length": "Parsed to emp_length_years (0 to 10; '10+ years' becomes 10, '< 1 year' becomes 0).",
    "earliest_cr_line": "Turned into credit_history_months = months from this date to the issue date.",
}

_SPLIT_FILTER_OR_TARGET = {
    "issue_d": "Issue month: defines the out-of-time split and vintages; not a feature (it is the time axis).",
    "application_type": "Population filter: only 'Individual' applications are modelled.",
}

_IDENTIFIER_OR_TEXT = {
    "id": "Loan identifier (also carries footer text rows in the Kaggle file).",
    "member_id": "Borrower identifier (empty in the public file).",
    "url": "Link to the loan listing.",
    "desc": "Free-text loan description written by the borrower (rarely filled after 2014).",
    "emp_title": "Free-text job title: high cardinality and can contain personal data.",
    "title": "Free-text loan title written by the borrower.",
}

_GEOGRAPHY = {
    "zip_code": "First 3 digits of the ZIP code: a known proxy for race and ethnicity (redlining risk).",
    "addr_state": "State: excluded with zip_code to avoid geographic proxies; Lending Club has no protected "
    "attributes to test proxy effects directly.",
}

_JOINT_APPLICATION = {
    "annual_inc_joint": "Combined income of co-borrowers.",
    "dti_joint": "Combined debt-to-income of co-borrowers.",
    "verification_status_joint": "Verification of the co-borrowers' joint income.",
    "revol_bal_joint": "Combined revolving balance of co-borrowers.",
    "sec_app_fico_range_low": "Secondary applicant FICO (low).",
    "sec_app_fico_range_high": "Secondary applicant FICO (high).",
    "sec_app_earliest_cr_line": "Secondary applicant earliest credit line.",
    "sec_app_inq_last_6mths": "Secondary applicant inquiries in 6 months.",
    "sec_app_mort_acc": "Secondary applicant mortgage accounts.",
    "sec_app_open_acc": "Secondary applicant open accounts.",
    "sec_app_revol_util": "Secondary applicant revolving utilisation.",
    "sec_app_open_act_il": "Secondary applicant active installment trades.",
    "sec_app_num_rev_accts": "Secondary applicant revolving accounts.",
    "sec_app_chargeoff_within_12_mths": "Secondary applicant charge-offs in 12 months.",
    "sec_app_collections_12_mths_ex_med": "Secondary applicant collections in 12 months.",
    "sec_app_mths_since_last_major_derog": "Secondary applicant months since major derogatory.",
}

_LATE_INTRODUCED = {
    "open_acc_6m": "Open trades in the last 6 months.",
    "open_act_il": "Currently active installment trades.",
    "open_il_12m": "Installment accounts opened in the past 12 months.",
    "open_il_24m": "Installment accounts opened in the past 24 months.",
    "mths_since_rcnt_il": "Months since the most recent installment account was opened.",
    "total_bal_il": "Total current balance of installment accounts.",
    "il_util": "Balance to credit limit on installment accounts.",
    "open_rv_12m": "Revolving trades opened in the past 12 months.",
    "open_rv_24m": "Revolving trades opened in the past 24 months.",
    "max_bal_bc": "Maximum current balance owed on revolving accounts.",
    "all_util": "Balance to credit limit on all trades.",
    "inq_fi": "Personal finance inquiries.",
    "total_cu_tl": "Finance trades.",
    "inq_last_12m": "Credit inquiries in the past 12 months.",
}

_OPERATIONAL = {
    "disbursement_method": "How the borrower receives the funds (Cash or DirectPay), not a credit attribute.",
}

_CONSTANT_OR_REDUNDANT = {
    "policy_code": "Always 1 in the accepted-loans file.",
    "fico_range_high": "Always fico_range_low + 4 (except at the 850 cap), so it adds nothing.",
}

COLUMN_DECISIONS: tuple[ColumnDecision, ...] = tuple(
    _decisions(POST_ORIGINATION, _POST_ORIGINATION)
    + _decisions(LENDER_OUTPUT, _LENDER_OUTPUT)
    + _decisions(FEATURE, _FEATURES)
    + _decisions(FEATURE_SOURCE, _FEATURE_SOURCES)
    + _decisions(SPLIT_FILTER_OR_TARGET, _SPLIT_FILTER_OR_TARGET)
    + _decisions(IDENTIFIER_OR_TEXT, _IDENTIFIER_OR_TEXT)
    + _decisions(GEOGRAPHY, _GEOGRAPHY)
    + _decisions(JOINT_APPLICATION, _JOINT_APPLICATION)
    + _decisions(LATE_INTRODUCED, _LATE_INTRODUCED)
    + _decisions(OPERATIONAL, _OPERATIONAL)
    + _decisions(CONSTANT_OR_REDUNDANT, _CONSTANT_OR_REDUNDANT)
)

DOCUMENTED_COLUMNS: tuple[str, ...] = tuple(decision.column for decision in COLUMN_DECISIONS)

# The derived target is banned too: a column named is_bad must never reach the model.
BANNED_POST_ORIGINATION_COLUMNS: frozenset[str] = frozenset(_POST_ORIGINATION) | {TARGET_COLUMN}
LENDER_OUTPUT_COLUMNS: frozenset[str] = frozenset(_LENDER_OUTPUT)
LATE_INTRODUCED_COLUMNS: tuple[str, ...] = tuple(_LATE_INTRODUCED)

# Whole families of post-origination fields, so a renamed or newly added field in the same family is still
# caught (for example a future "hardship_..." or "settlement_..." column).
BANNED_PREFIXES: tuple[str, ...] = (
    "hardship_",
    "settlement_",
    "debt_settlement_",
    "total_rec_",
    "last_pymnt",
    "last_fico_range_",
    "out_prncp",
    "total_pymnt",
)

# sklearn's ColumnTransformer names outputs "<transformer>__<feature>", and SimpleImputer(add_indicator=True)
# adds "missingindicator_<feature>". These are stripped before checking, so encoded names are checked too.
_TRANSFORMER_SEPARATOR = "__"
_INDICATOR_PREFIX = "missingindicator_"


class LeakageError(ValueError):
    """Raised when a banned column (post-origination or lender output) would reach the model."""


def base_feature_name(name: str) -> str:
    """Strip sklearn transformer prefixes ("num__", "cat__") and "missingindicator_" from an encoded name."""
    base = str(name)
    if _TRANSFORMER_SEPARATOR in base:
        base = base.split(_TRANSFORMER_SEPARATOR, 1)[1]
    if base.startswith(_INDICATOR_PREFIX):
        base = base[len(_INDICATOR_PREFIX) :]
    return base


def _matches(name: str, banned: Iterable[str]) -> str | None:
    """Return the banned column that name refers to (exact or one-hot "<column>_<level>"), or None."""
    for column in banned:
        if name == column or name.startswith(column + "_"):
            return column
    return None


def leakage_problems(feature_names: Iterable[str], allow_lender_outputs: bool = False) -> list[str]:
    """Describe every feature name that is banned. An empty list means the names are safe.

    Checks each name after stripping encoder prefixes: exact match or one-hot match against
    BANNED_POST_ORIGINATION_COLUMNS, a match against BANNED_PREFIXES, and (unless allow_lender_outputs)
    a match against LENDER_OUTPUT_COLUMNS.
    """
    problems = []
    for name in feature_names:
        base = base_feature_name(name)
        post_origination = _matches(base, BANNED_POST_ORIGINATION_COLUMNS)
        if post_origination is None and base.startswith(BANNED_PREFIXES):
            post_origination = base
        if post_origination is not None:
            problems.append(f"'{name}' is post-origination information ({post_origination})")
            continue
        lender_output = _matches(base, LENDER_OUTPUT_COLUMNS)
        if lender_output is not None and not allow_lender_outputs:
            problems.append(f"'{name}' is a Lending Club underwriting/pricing output ({lender_output})")
    return problems


def assert_no_leakage(feature_names: Iterable[str], allow_lender_outputs: bool = False) -> None:
    """Raise LeakageError listing every banned feature name; return None if all names are allowed.

    Call this on the exact column names that go into model fitting and scoring (raw or encoded).
    """
    problems = leakage_problems(list(feature_names), allow_lender_outputs=allow_lender_outputs)
    if problems:
        raise LeakageError("Leakage guard: banned columns would reach the model: " + "; ".join(problems))


def decisions_by_category() -> dict[str, list[ColumnDecision]]:
    """Column decisions grouped by category, in the order of CATEGORY_DESCRIPTIONS."""
    grouped: dict[str, list[ColumnDecision]] = {category: [] for category in CATEGORY_DESCRIPTIONS}
    for decision in COLUMN_DECISIONS:
        grouped[decision.category].append(decision)
    return grouped


def decision_for(column: str) -> ColumnDecision | None:
    """The documented decision for a raw column, or None if the column is not documented."""
    for decision in COLUMN_DECISIONS:
        if decision.column == column:
            return decision
    return None


def undocumented_columns(columns: Iterable[str]) -> list[str]:
    """Columns of a file that have no documented decision (a reviewer should classify them)."""
    documented = set(DOCUMENTED_COLUMNS)
    return [column for column in columns if column not in documented]


def markdown_table(category: str) -> str:
    """Markdown table (column, reason) for one category, used to write docs/leakage.md."""
    lines = ["| Column | Reason |", "| --- | --- |"]
    for decision in decisions_by_category()[category]:
        lines.append(f"| `{decision.column}` | {decision.reason} |")
    return "\n".join(lines)
