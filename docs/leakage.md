# Leakage controls: Lending Club credit-default model

The credit model scores a loan **application**. A column may be a feature only if its value is known when
the application is decided. Anything recorded later (payments, balances, recoveries, hardship plans,
settlements, refreshed credit pulls, the loan status itself) would let the model "see the future": test
metrics would look excellent and mean nothing for a real decision.

The rules live in code, in `src/uwcopilot/risk/leakage.py`. This page explains them. Every one of the
151 columns in the Kaggle accepted-loans file (`accepted_2007_to_2018Q4.csv`) has exactly one decision
below. The tables are generated from `COLUMN_DECISIONS` in that module, and
`tests/risk/test_leakage.py` fails if this page and the code disagree.

| Decision | Columns |
| --- | --- |
| banned_post_origination | 39 |
| lender_output | 7 |
| feature | 59 |
| feature_source | 3 |
| split_filter_or_target | 2 |
| excluded_identifier_or_free_text | 6 |
| excluded_geography_fair_lending | 2 |
| excluded_joint_application | 16 |
| excluded_added_late_in_sample | 14 |
| excluded_operational | 1 |
| excluded_constant_or_redundant | 2 |

The model uses 63 features: 59 columns used as they are, plus
4 derived from application-time fields (`term_months`, `emp_length_years`,
`credit_history_months`, `loan_to_income`). 3 of them are categorical
(`home_ownership`, `verification_status`, `purpose`).

## How the guard is enforced

1. **Allow-list, not block-list.** The model only ever sees `CREDIT_FEATURES` (defined in
   `src/uwcopilot/risk/data/lending_club.py`). `prepare_loans` builds the model table from that list plus
   a few metadata columns, so every other raw column is dropped when the data is loaded.
2. **`assert_no_leakage(feature_names)`** raises `LeakageError` listing every banned name. It runs:
   - when the loader prepares the data (a self-check of the allow-list itself);
   - in `fit_logistic_scorecard`, on the raw feature names **and** on the encoded column names the
     regression sees (one-hot levels, missing-value indicators);
   - in `fit_lightgbm`, on the raw names, the encoded frame and the trained booster's feature names;
   - in `CreditModels.score`, on every input and every scenario override, so the serving path cannot be
     handed a post-origination field either.
3. **Encoded names are caught.** sklearn prefixes (`num__`, `cat__`), `missingindicator_` and one-hot
   suffixes (`pymnt_plan_y`, `grade_B`) are recognised. So are whole families: any future `hardship_*`,
   `settlement_*`, `total_rec_*`, `last_pymnt*`, `last_fico_range_*`, `out_prncp*` or `total_pymnt*`
   column is rejected.
4. **Tests.** `tests/risk/test_credit.py::test_leakage_guard_fails_when_banned_column_reaches_the_model`
   adds `recoveries`, `total_pymnt`, `last_fico_range_low`, `int_rate` or the target `is_bad` to the
   training features and checks that both models refuse to fit. `test_banned_column_would_leak_the_outcome`
   shows why this matters: in the SYNTHETIC test data, `recoveries` alone separates bad from good loans
   perfectly (AUC 1.0, because recoveries only exist after a charge-off). That number describes the
   synthetic fixture, not Lending Club.

## Decision: Lending Club's own risk outputs (grade, sub_grade, int_rate)

`grade`, `sub_grade`, `int_rate` and `installment` are known at origination, so strictly they are not
post-origination leakage. They are, however, **outputs of Lending Club's own underwriting and pricing
model**. `int_rate` is set from the sub-grade, and `installment` is computed from the amount, the term and
`int_rate`, so it reveals the rate.

**Decision: excluded from the main model** (rejected by the guard unless `allow_lender_outputs=True`).

- With them, our model would mostly re-learn Lending Club's grade. Its metrics would then say little about
  what our own application features can do, and the reason codes would mostly say "your grade".
- In a real underwriting flow the risk model comes before pricing: the price is set from the model's
  output, so a model that needs the price as an input cannot be used to set it.
- `sub_grade` is still reported as a **benchmark**: `benchmark_lending_club_sub_grade` in
  `results/risk_metrics.json` is the test AUC of the sub-grade used as a score (A1 = 1 ... G5 = 35). This
  is a reference point, not a fair contest: Lending Club set it with data we do not have.

`funded_amnt`, `funded_amnt_inv` and `initial_list_status` sit in the same group. They are decided by
Lending Club or its investors after the application (funded amounts can be cut after review; investors fund
during the listing period; the listing type is chosen for investors).

## Other exclusions (not leakage)

The guard does not reject these columns: they are not leakage. They are simply not on the allow-list.

- **Geography (`zip_code`, `addr_state`).** A well-known proxy for race and ethnicity (redlining risk).
  Lending Club has no protected attributes, so proxy effects cannot be tested here. Fairness testing is
  demonstrated separately on UCI German Credit (`fairness_demo` in `results/risk_metrics.json`).
- **Joint applications.** Joint applications appear only in the later years (the loader counts them per
  issue year in `data_quality.joint_applications_by_issue_year`), so the model covers individual
  applications only. `application_type` is used as a population filter, and the co-borrower fields
  (`*_joint`, `sec_app_*`) are excluded.
- **Fields added late in the sample** (`open_acc_6m`, `il_util`, `all_util`, `inq_last_12m`, ...). They are
  missing for nearly all training loans (2015 and earlier). A model would learn the vintage rather than the
  borrower. The loader **measures** the missing share of each of these fields per issue year
  (`data_quality.missing_share_by_issue_year`), so this claim is checked whenever the real data is run.
- **Identifiers and free text** (`id`, `url`, `desc`, `emp_title`, `title`). They carry no stable signal,
  and free text can contain personal data.
- **Operational, constant or redundant** (`disbursement_method`, `policy_code`, `fico_range_high`).

## Target and population

- `loan_status` defines the target and is banned as a feature. Finished loans only:
  - Fully Paid → 0, and Charged Off / Default → 1.
  - The older "Does not meet the credit policy. Status:..." variants map like the plain status. The loader
    counts them (`old_credit_policy`). That flag lives inside `loan_status`, so it is never a feature either.
  - Current, In Grace Period and Late loans have no final outcome. They are excluded from training and
    testing, but kept for the population-stability (PSI) checks, which need no outcome.
  - An unknown status raises an error instead of being mapped silently.
- `issue_d` defines the out-of-time split (train ≤ 2015, validation 2016, test 2017-2018) and the
  vintages. It is the time axis, not a feature.
- **Censoring caveat.** Loans issued in 2017-2018 had at most about two years to finish before the end of
  the data. The finished ones are therefore mostly early payoffs and early charge-offs. The per-vintage
  table (`data_quality.vintages`) shows the finished share and bad rate for each year, so this is visible
  rather than hidden.

## Residual risks

- Decisions follow Lending Club's data dictionary (`LCDataDictionary.xlsx`, published with the loan data).
  We cannot independently verify that no bureau field (for example `mths_since_recent_inq`) was refreshed
  after origination.
- `verification_status` is kept. It is set by Lending Club during underwriting, before issuance, so it is
  available at decision time, but it partly reflects Lending Club's own risk process.
- A feature can be known at application time and still be a proxy for a protected characteristic.
  Excluding geography reduces this risk but does not remove it.

## Every column, by decision

The tables below are generated from `src/uwcopilot/risk/leakage.py`.

### banned_post_origination (39)

Known only after the loan was issued (or the outcome itself). Rejected by the leakage guard.

| Column | Reason |
| --- | --- |
| `loan_status` | The outcome (Fully Paid, Charged Off, Current, ...); the target is derived from it. |
| `pymnt_plan` | Whether a payment plan was put in place for the loan, which happens after issuance. |
| `out_prncp` | Outstanding principal remaining; changes every month after issuance. |
| `out_prncp_inv` | Outstanding principal held by investors; post-issuance balance. |
| `total_pymnt` | Payments received to date; a defaulted loan has paid less, so it encodes the outcome. |
| `total_pymnt_inv` | Payments received to date on the investor-funded portion. |
| `total_rec_prncp` | Principal received to date. |
| `total_rec_int` | Interest received to date. |
| `total_rec_late_fee` | Late fees received to date; only exists if the borrower paid late. |
| `recoveries` | Post-charge-off gross recovery; non-zero almost only for charged-off loans. |
| `collection_recovery_fee` | Post-charge-off collection fee; only exists after a charge-off. |
| `last_pymnt_d` | Date of the last payment received; an early date reveals an early default or payoff. |
| `last_pymnt_amnt` | Amount of the last payment; a large final payment reveals a payoff. |
| `next_pymnt_d` | Next scheduled payment date; empty once a loan is paid off or charged off. |
| `last_credit_pull_d` | Most recent month Lending Club pulled credit for the loan; refreshed after issuance. |
| `last_fico_range_high` | FICO from the most recent credit pull (after issuance); falls when a borrower defaults. |
| `last_fico_range_low` | FICO from the most recent credit pull (after issuance); falls when a borrower defaults. |
| `hardship_flag` | Whether the borrower is on a hardship plan (after issuance). |
| `hardship_type` | Hardship plan type (after issuance). |
| `hardship_reason` | Reason the borrower requested a hardship plan (after issuance). |
| `hardship_status` | Hardship plan status (after issuance). |
| `deferral_term` | Months of payment deferral under a hardship plan (after issuance). |
| `hardship_amount` | Interest payment under a hardship plan (after issuance). |
| `hardship_start_date` | Hardship plan start date (after issuance). |
| `hardship_end_date` | Hardship plan end date (after issuance). |
| `payment_plan_start_date` | First payment date of a hardship plan (after issuance). |
| `hardship_length` | Hardship plan length in months (after issuance). |
| `hardship_dpd` | Days past due when the hardship plan started (after issuance). |
| `hardship_loan_status` | Loan status when the hardship plan started (after issuance). |
| `orig_projected_additional_accrued_interest` | Projected extra interest from a hardship plan (after issuance). |
| `hardship_payoff_balance_amount` | Payoff balance when the hardship plan started (after issuance). |
| `hardship_last_payment_amount` | Last payment amount when the hardship plan started (after issuance). |
| `debt_settlement_flag` | Whether a charged-off borrower works with a debt-settlement company (after charge-off). |
| `debt_settlement_flag_date` | Date the debt-settlement flag was set (after charge-off). |
| `settlement_status` | Status of a settlement plan (after charge-off). |
| `settlement_date` | Date the settlement plan was agreed (after charge-off). |
| `settlement_amount` | Amount agreed in the settlement (after charge-off). |
| `settlement_percentage` | Settlement amount as a share of the balance (after charge-off). |
| `settlement_term` | Number of months of the settlement plan (after charge-off). |

### lender_output (7)

Lending Club's own underwriting or pricing output. Rejected by the leakage guard for the main model; sub_grade is reported only as a benchmark.

| Column | Reason |
| --- | --- |
| `grade` | Lending Club's risk grade (A-G) from its own model; using it would copy that model. |
| `sub_grade` | Lending Club's risk sub-grade (A1-G5); used only as a benchmark score, never as a feature. |
| `int_rate` | Interest rate set by Lending Club from the sub-grade; a pricing output, not an applicant fact. |
| `installment` | Monthly payment computed from amount, term and int_rate, so it reveals int_rate. |
| `funded_amnt` | Amount Lending Club committed, which can be less than requested after its review. |
| `funded_amnt_inv` | Amount investors funded during the listing period; investors react to the grade. |
| `initial_list_status` | Whole vs fractional listing, chosen by Lending Club for investors after approval. |

### feature (59)

Application-time field used as a model feature unchanged.

| Column | Reason |
| --- | --- |
| `loan_amnt` | Amount the applicant asked for. |
| `annual_inc` | Self-reported annual income (also used for loan_to_income). |
| `dti` | Debt-to-income ratio at application, excluding the new loan and mortgage. |
| `fico_range_low` | Lower bound of the FICO range at application. |
| `delinq_2yrs` | 30+ days past-due incidents in the past 2 years (bureau, at application). |
| `inq_last_6mths` | Credit inquiries in the past 6 months (bureau, at application). |
| `mths_since_last_delinq` | Months since the last delinquency (empty = none on file). |
| `mths_since_last_record` | Months since the last public record (empty = none on file). |
| `open_acc` | Number of open credit lines. |
| `pub_rec` | Number of derogatory public records. |
| `revol_bal` | Total revolving balance. |
| `revol_util` | Revolving utilisation rate (percent strings are converted to numbers). |
| `total_acc` | Total number of credit lines on file. |
| `collections_12_mths_ex_med` | Collections in the past 12 months, excluding medical collections. |
| `mths_since_last_major_derog` | Months since the most recent 90-day or worse rating. |
| `acc_now_delinq` | Accounts currently delinquent. |
| `tot_coll_amt` | Total collection amounts ever owed. |
| `tot_cur_bal` | Total current balance of all accounts. |
| `total_rev_hi_lim` | Total revolving credit limit. |
| `acc_open_past_24mths` | Trades opened in the past 24 months. |
| `avg_cur_bal` | Average current balance of all accounts. |
| `bc_open_to_buy` | Total open-to-buy on revolving bank cards. |
| `bc_util` | Balance to credit limit on bank cards. |
| `chargeoff_within_12_mths` | Charge-offs within the past 12 months (on other credit, at application). |
| `delinq_amnt` | Past-due amount owed on accounts now delinquent. |
| `mo_sin_old_il_acct` | Months since the oldest installment account was opened. |
| `mo_sin_old_rev_tl_op` | Months since the oldest revolving account was opened. |
| `mo_sin_rcnt_rev_tl_op` | Months since the most recent revolving account was opened. |
| `mo_sin_rcnt_tl` | Months since the most recent account was opened. |
| `mort_acc` | Number of mortgage accounts. |
| `mths_since_recent_bc` | Months since the most recent bank card was opened. |
| `mths_since_recent_bc_dlq` | Months since the most recent bank card delinquency. |
| `mths_since_recent_inq` | Months since the most recent inquiry. |
| `mths_since_recent_revol_delinq` | Months since the most recent revolving delinquency. |
| `num_accts_ever_120_pd` | Accounts ever 120 or more days past due. |
| `num_actv_bc_tl` | Active bank card accounts. |
| `num_actv_rev_tl` | Active revolving trades. |
| `num_bc_sats` | Satisfactory bank card accounts. |
| `num_bc_tl` | Bank card accounts. |
| `num_il_tl` | Installment accounts. |
| `num_op_rev_tl` | Open revolving accounts. |
| `num_rev_accts` | Revolving accounts. |
| `num_rev_tl_bal_gt_0` | Revolving trades with a balance above 0. |
| `num_sats` | Satisfactory accounts. |
| `num_tl_120dpd_2m` | Accounts currently 120 days past due (updated in the 2 months before application). |
| `num_tl_30dpd` | Accounts currently 30 days past due (updated in the 2 months before application). |
| `num_tl_90g_dpd_24m` | Accounts 90+ days past due in the past 24 months. |
| `num_tl_op_past_12m` | Accounts opened in the past 12 months. |
| `pct_tl_nvr_dlq` | Percent of trades never delinquent. |
| `percent_bc_gt_75` | Percent of bank cards above 75% of their limit. |
| `pub_rec_bankruptcies` | Public-record bankruptcies. |
| `tax_liens` | Tax liens. |
| `tot_hi_cred_lim` | Total high credit / credit limit. |
| `total_bal_ex_mort` | Total credit balance excluding mortgage. |
| `total_bc_limit` | Total bank card credit limit. |
| `total_il_high_credit_limit` | Total installment high credit / credit limit. |
| `home_ownership` | Housing status from the application (ANY, NONE and OTHER are grouped as OTHER). |
| `verification_status` | Whether Lending Club verified the income before issuance (set during underwriting). |
| `purpose` | Loan purpose chosen by the applicant. |

### feature_source (3)

Application-time field turned into a model feature (parsed or derived; see reason).

| Column | Reason |
| --- | --- |
| `term` | Parsed to term_months (36 or 60), chosen by the applicant. |
| `emp_length` | Parsed to emp_length_years (0 to 10; '10+ years' becomes 10, '< 1 year' becomes 0). |
| `earliest_cr_line` | Turned into credit_history_months = months from this date to the issue date. |

### split_filter_or_target (2)

Used to define the target, the out-of-time split or the population, never as a feature.

| Column | Reason |
| --- | --- |
| `issue_d` | Issue month: defines the out-of-time split and vintages; not a feature (it is the time axis). |
| `application_type` | Population filter: only 'Individual' applications are modelled. |

### excluded_identifier_or_free_text (6)

Identifier or free text: no stable signal, and free text can carry personal data.

| Column | Reason |
| --- | --- |
| `id` | Loan identifier (also carries footer text rows in the Kaggle file). |
| `member_id` | Borrower identifier (empty in the public file). |
| `url` | Link to the loan listing. |
| `desc` | Free-text loan description written by the borrower (rarely filled after 2014). |
| `emp_title` | Free-text job title: high cardinality and can contain personal data. |
| `title` | Free-text loan title written by the borrower. |

### excluded_geography_fair_lending (2)

Geography: a well-known proxy for protected characteristics (redlining risk).

| Column | Reason |
| --- | --- |
| `zip_code` | First 3 digits of the ZIP code: a known proxy for race and ethnicity (redlining risk). |
| `addr_state` | State: excluded with zip_code to avoid geographic proxies; Lending Club has no protected attributes to test proxy effects directly. |

### excluded_joint_application (16)

Joint-application field. Joint applications are out of scope (individual applications only).

| Column | Reason |
| --- | --- |
| `annual_inc_joint` | Combined income of co-borrowers. |
| `dti_joint` | Combined debt-to-income of co-borrowers. |
| `verification_status_joint` | Verification of the co-borrowers' joint income. |
| `revol_bal_joint` | Combined revolving balance of co-borrowers. |
| `sec_app_fico_range_low` | Secondary applicant FICO (low). |
| `sec_app_fico_range_high` | Secondary applicant FICO (high). |
| `sec_app_earliest_cr_line` | Secondary applicant earliest credit line. |
| `sec_app_inq_last_6mths` | Secondary applicant inquiries in 6 months. |
| `sec_app_mort_acc` | Secondary applicant mortgage accounts. |
| `sec_app_open_acc` | Secondary applicant open accounts. |
| `sec_app_revol_util` | Secondary applicant revolving utilisation. |
| `sec_app_open_act_il` | Secondary applicant active installment trades. |
| `sec_app_num_rev_accts` | Secondary applicant revolving accounts. |
| `sec_app_chargeoff_within_12_mths` | Secondary applicant charge-offs in 12 months. |
| `sec_app_collections_12_mths_ex_med` | Secondary applicant collections in 12 months. |
| `sec_app_mths_since_last_major_derog` | Secondary applicant months since major derogatory. |

### excluded_added_late_in_sample (14)

Added late in the sample, so missing for nearly all training loans (the loader measures the missing share per issue year to confirm this).

| Column | Reason |
| --- | --- |
| `open_acc_6m` | Open trades in the last 6 months. |
| `open_act_il` | Currently active installment trades. |
| `open_il_12m` | Installment accounts opened in the past 12 months. |
| `open_il_24m` | Installment accounts opened in the past 24 months. |
| `mths_since_rcnt_il` | Months since the most recent installment account was opened. |
| `total_bal_il` | Total current balance of installment accounts. |
| `il_util` | Balance to credit limit on installment accounts. |
| `open_rv_12m` | Revolving trades opened in the past 12 months. |
| `open_rv_24m` | Revolving trades opened in the past 24 months. |
| `max_bal_bc` | Maximum current balance owed on revolving accounts. |
| `all_util` | Balance to credit limit on all trades. |
| `inq_fi` | Personal finance inquiries. |
| `total_cu_tl` | Finance trades. |
| `inq_last_12m` | Credit inquiries in the past 12 months. |

### excluded_operational (1)

Operational detail of the loan, not a credit attribute of the applicant.

| Column | Reason |
| --- | --- |
| `disbursement_method` | How the borrower receives the funds (Cash or DirectPay), not a credit attribute. |

### excluded_constant_or_redundant (2)

Constant or a duplicate of another field.

| Column | Reason |
| --- | --- |
| `policy_code` | Always 1 in the accepted-loans file. |
| `fico_range_high` | Always fico_range_low + 4 (except at the 850 cap), so it adds nothing. |
