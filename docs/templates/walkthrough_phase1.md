<!-- template-only: This is the source of docs/walkthrough_phase1.md. `make reports` replaces every
{{ path | format }} placeholder with the value at that path in results/risk_metrics.json (or "not measured"),
so no result in the walkthrough is typed by hand. A {{ derived.name }} placeholder is a sentence or value
from walkthrough_statements() in scripts/render_reports.py, whose wording follows the intervals: put any
claim that depends on the data there, never in this prose. Edit the prose here, never the rendered file. -->
# Phase 1 walkthrough: risk models, stress-test interface, model-risk pack

> Rendered by `scripts/render_reports.py` from `docs/templates/walkthrough_phase1.md`. Every result below is
> filled in from `results/risk_metrics.json` when `make reports` runs; a result that does not exist yet reads
> "not measured". Edit the template, not this file.

Phase 1 builds one reusable risk-modelling framework and applies it to three problems: credit default
(Lending Club), motor insurance claim frequency and severity (freMTPL2), and a fairness demonstration (UCI German
Credit). It adds a scenario (what-if) interface used later by the Phase 4 stress tests, a scoring API, and a
model-risk pack (model cards and a validation report) generated from the results file.

## Honest status

- **Insurance (freMTPL2): measured** on the real public data. Model bundle `{{ insurance_frequency.model_version.label }}`,
  results generated {{ insurance_frequency.metadata.generated_at_utc }}.
- **Fairness demo (German Credit): measured** on the real public data, with a documented data-quality problem
  in the dataset's own documentation (see Q&A 10 and the validation report).
- {{ derived.credit_status }}
- Nothing here has been independently validated, and nothing claims compliance with any regulation.

## How to run it

```bash
make phase1   # train-insurance, fairness, train-credit (writes "not measured" without the Kaggle zip), reports
make serve    # scoring API on http://127.0.0.1:8000 (interactive docs at /docs)
make test     # every test runs on synthetic data, so CI needs no downloads
```

Example request (the feature values are an illustrative policy, not data from any person):

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "product": "insurance_frequency",
  "features": {"VehPower": 6, "VehAge": 2, "DrivAge": 45, "BonusMalus": 50, "Density": 1200,
               "Area": "C", "VehGas": "Regular", "VehBrand": "B12", "Region": "R24"},
  "overrides": {"BonusMalus": 80}}'
```

The response holds the baseline prediction (claims per policy-year), its risk band, the top-4 reason codes, the
model version (here {{ derived.served_frequency_version }}) and a `scenario` block with the baseline and the
scenario side by side. `GET /health` lists which models are loaded, their versions and their risk-band cut-offs. Every
input is checked against the training data: a missing feature, an out-of-range value, an unseen category or a
fraction of a whole-number feature gets HTTP 422 naming the problem.{{ derived.credit_api }}

## Results at a glance

| Model | Test metric | Value [95% bootstrap CI] |
|---|---|---|
| Frequency, Poisson GLM | Mean Poisson deviance | {{ insurance_frequency.test_metrics.glm.mean_poisson_deviance | ci }} |
| Frequency, LightGBM | Mean Poisson deviance | {{ insurance_frequency.test_metrics.lgbm.mean_poisson_deviance | ci }} |
| Frequency, LightGBM minus GLM | Paired difference | {{ insurance_frequency.test_metrics.lgbm_minus_glm.mean_poisson_deviance | ci }} |
| Frequency, GLM / LightGBM | Lorenz Gini | {{ insurance_frequency.test_metrics.glm.lorenz_gini | ci }} / {{ insurance_frequency.test_metrics.lgbm.lorenz_gini | ci }} |
| Severity, Gamma GLM / LightGBM | Mean Gamma deviance | {{ insurance_severity.test_metrics.glm.mean_gamma_deviance | ci }} / {{ insurance_severity.test_metrics.lgbm.mean_gamma_deviance | ci }} |
| Severity, LightGBM minus GLM | Paired difference | {{ insurance_severity.test_metrics.lgbm_minus_glm.mean_gamma_deviance | ci }} |
| Expected loss, GLM pair / LightGBM pair | Observed / predicted | {{ insurance_severity.expected_loss.glm.observed_to_predicted | ci }} / {{ insurance_severity.expected_loss.lgbm.observed_to_predicted | ci }} |
| Expected loss, served pair ({{ insurance_severity.expected_loss.served_pair.frequency_model | model }} frequency x {{ insurance_severity.expected_loss.served_pair.severity_model | model }} severity) | Observed / predicted | {{ insurance_severity.expected_loss.served_pair.observed_to_predicted | ci }} |
| Frequency, split grouped on rating features | LightGBM minus GLM deviance | {{ insurance_frequency.grouped_split_sensitivity.lgbm_minus_glm.mean_poisson_deviance | ci }} |
| German Credit, logistic / LightGBM | AUC | {{ fairness_demo.models.logistic_regression.test_auc | ci3 }} / {{ fairness_demo.models.lightgbm.test_auc | ci3 }} |
| Credit PD (Lending Club) | AUC, KS, Gini, Brier, ECE | not measured (see status) |

Bootstrap settings: {{ insurance_frequency.bootstrap.n_resamples | int }} resamples, seed
{{ insurance_frequency.bootstrap.seed }}, percentile intervals; paired differences resample the same rows for both
models.

## Every file, what it does and why

### Framework (`src/uwcopilot/risk/`)

| File | What | Why it is built this way |
|---|---|---|
| `metrics.py` | AUC, KS, Gini, Brier, ECE, calibration table, Poisson and Gamma deviance, Lorenz curve and Gini | Written out plainly and tested against scikit-learn and SciPy, so every metric can be explained line by line. NaN input raises instead of producing a silent number. |
| `bootstrap.py` | `bootstrap_ci` and `paired_bootstrap_diff` (percentile method) | Ground rule 1 asks for CIs on headline metrics. Paired resampling answers "is model A better than B on the same cases?", which two separate intervals cannot. |
| `stability.py` | PSI for numeric and categorical variables, bands and flag | Quantile bins of the reference period, a floor for empty bins, NaN as its own bin. Thresholds 0.10 / 0.25 are the usual industry convention and are constants in one place. |
| `fairness.py` | Selection rates, adverse impact ratio, equal-opportunity gap | Small functions on arrays, reused by the German Credit demo and reusable for any model with group labels. |
| `scenario.py` | Feature ranges learned from training data, override and input validation, `monotonic_check` | Out-of-range overrides are rejected, never clipped (Q&A 10); the base applicant or policy gets the same checks, plus "no missing key" and "null only where training had missing values". The monotonic check reports the share of violations instead of asserting zero. |
| `explain.py` | SHAP for tree models, reason codes, global importance, a reusable explainer | Reason codes are the top contributions in one direction, mapped to plain language; an unmapped feature gets a visible generic reason. |
| `bands.py` | Risk-band cut-offs from training-score quantiles | Cut-offs are computed, stored with the model and shown by the API, never typed by hand. |
| `results.py` | Writes `results/*.json` sections with provenance (git commit, dirty flag, package versions, dataset SHA-256) | Every number in every document traces back to one file and one run. "not measured" is a first-class value. |
| `tracking.py` | MLflow helpers (local SQLite store, telemetry off) | Runs, metrics, artifacts and registered model versions are kept locally, so they can be inspected with `mlflow ui`. |
| `data/fremtpl2.py` | Load, clean, link and split freMTPL2; synthetic generator for tests | Every cleaning step counts what it changed and reports it (see `docs/insurance_data_notes.md`). |
| `data/german_credit.py` | Parse UCI German Credit, derive age groups and sex mappings | Keeps both the documented and the corrected sex coding so the problem stays visible. |
| `data/lending_club.py` | Stream the Kaggle zip in chunks, map loan status to the target, data-quality report | Fails loudly on a missing column or an unknown status instead of guessing. |
| `leakage.py` | One decision and reason for each Lending Club column; `assert_no_leakage` | The guard runs on raw names, encoded names and booster inputs, at fit time and at score time. `docs/leakage.md` holds the same table and a test keeps them in sync. |
| `insurance.py` | Poisson / Gamma GLMs, LightGBM models, metrics, expected loss, SHAP, contributions, save / load | GLMs are banded so coefficients read as relativities; GLM and SHAP contributions both add up to the log prediction, so either model can give reason codes. |
| `credit.py` | Scorecard and LightGBM, out-of-time split, champion/challenger rule, PSI by vintage, reason codes, `score()` | The model choice is fixed before the test set is scored; the scorecard wins unless LightGBM is clearly better. |
| `fairness_demo.py` | The German Credit fairness pipeline: models, approval threshold, AIR, equal opportunity, proxy check | A separate, clearly labelled demonstration because Lending Club has no protected attributes. |
| `service.py` | FastAPI `/score` and `/health` | Serves the model preferred on validation for each product, validates every input, and answers 503 with a reason when a model is missing. |

### Scripts, docs and outputs

| File | What | Why |
|---|---|---|
| `scripts/train_insurance.py` | Trains the four insurance models, writes metrics, figures, models, MLflow runs and the data notes | One command reproduces every insurance number. |
| `scripts/fairness_german.py` | Runs the fairness demo and writes its section | Same pattern: one script, one results section. |
| `scripts/train_credit.py` | Runs the credit pipeline, or writes "not measured" with the reason when the zip is missing | A missing download is recorded honestly instead of skipped silently. |
| `scripts/render_reports.py` | Renders the model cards, validation report, this walkthrough and the README table from the results JSON | No number in a document is typed by hand, and data-dependent claims are worded by code from the intervals. Tests check that every number exists in the JSON and that the committed documents equal a fresh render of the committed results. |
| `docs/model_cards/credit_model_card.md` | Credit model card aligned to the structure of SR 26-2 / OCC Bulletin 2026-13 | Shows the documentation a bank model inventory expects, with "not measured" where it applies. |
| `docs/model_cards/insurance_model_card.md` | Insurance model card referencing the NAIC AI Model Bulletin and NY DFS Circular Letter No. 7 | Same sections, insurance-specific governance and fairness-testing references. |
| `docs/model_cards/validation_report.md` | Tests, backtests, sensitivity analyses and numbered findings | The "effective challenge" record: what was checked, what failed, what was done about it. |
| `docs/leakage.md`, `docs/insurance_data_notes.md` | Column decisions; freMTPL2 cleaning decisions | Reviewers can check every exclusion and data fix without reading code. |
| `results/risk_metrics.json`, `results/figures/` | All Phase 1 numbers with provenance; Lorenz and calibration plots | The single source of truth for every document. |
| `models/`, `mlruns.db`, `mlruns/` | Saved models and MLflow store (gitignored) | Rebuilt by the training scripts. |
| `tests/risk/*.py`, `tests/test_render_reports.py` | One test module per source module, plus the API and the renderer | Every test builds small synthetic data itself, so CI needs no downloads and no saved models. |
| `Makefile` | `train-insurance`, `fairness`, `train-credit`, `reports`, `phase1`, `serve` | One command per step (ground rule 3). |

## Design choices worth defending

1. **One framework, three models.** Metrics, bootstrap, PSI, fairness, scenarios, explanations and results
   writing are shared, so the credit, insurance and fairness code only contains what is specific to each problem.
2. **Champion/challenger rules fixed in advance.** Insurance: the model with the lower validation deviance is
   preferred ({{ insurance_frequency.preferred_model.name | model }} for frequency,
   {{ insurance_severity.preferred_model.name | model }} for severity). Credit: LightGBM only if its validation AUC
   gain is large enough and its paired interval excludes zero.
3. **The API serves what the results report.** Frequency is served by
   {{ insurance_frequency.preferred_model.name | model }}, severity by the
   {{ insurance_severity.preferred_model.name | model }}. Serving a model other than the validated one would be a
   governance gap, so the GLM got its own contribution-based reason codes instead.
4. **Risk bands from training quantiles.** The 50th, 80th and 95th percentiles of the served model's training
   predictions. For frequency the cut-offs are {{ insurance_frequency.risk_bands.cutoffs.0 | f4 }},
   {{ insurance_frequency.risk_bands.cutoffs.1 | f4 }} and {{ insurance_frequency.risk_bands.cutoffs.2 | f4 }} claims
   per policy-year. A band says where a case sits in the training portfolio; it is not a decision rule.
5. **Data problems change the model, and are documented.** The freMTPL2 claims without amounts and the
   German Credit coding error were both found by checking the data against itself, and both changed what was built.
6. **"Not measured" over a guess.** The credit section says why it is empty; the insurance PSI and out-of-time
   test say why they cannot exist for freMTPL2.

## Interview questions and answers

**1. Why an out-of-time split?**
A lender scores applications today with a model trained on the past. A random split lets the model learn from
loans issued after the ones it is tested on, so it looks better than it can ever be in use. The credit pipeline
trains on loans issued in 2015 and earlier, chooses and recalibrates on 2016, and touches 2017-2018 once.
{{ derived.credit_results }} freMTPL2 has no date field, so the insurance split is
random and stratified, and the results file says so: "{{ insurance_frequency.out_of_time_test.reason }}" That
limitation is also why insurance PSI is not measured.

**2. Why KS as well as AUC?**
AUC is the probability that a random bad loan scores riskier than a random good one: a ranking measure over all
cut-offs. KS is the largest gap between the cumulative score distributions of good and bad loans, which is the
best separation at a single cut-off; credit scorecard teams have reported it for decades, so validators expect
it. Neither says anything about calibration. Credit KS on Lending Club (scorecard):
{{ credit.models.logistic_scorecard.test.ks | ci }}{{ derived.credit_not_run }}.

**3. Why PSI?**
Population stability compares the distribution of a score or feature now with the training population, and needs
no outcomes, so it warns months before defaults arrive. The bins are quantiles of the reference period; PSI below
0.10 is read as stable, 0.10 to 0.25 as a moderate shift, above 0.25 as significant and flagged (the thresholds
are constants in `uwcopilot.risk.stability`). For credit, PSI runs per issue-year vintage on the score and the
top-10 features (flag threshold in the results: {{ credit.stability.threshold }}{{ derived.credit_not_run }}). For
freMTPL2 it is {{ insurance_frequency.stability | text }}.

**4. Give examples of leakage you guarded against.**
`recoveries`, `total_pymnt`, `last_pymnt_d`, `out_prncp` and `last_fico_range_low` are only known after the loan
was issued; `loan_status` is the outcome itself. `int_rate`, `grade` and `sub_grade` are Lending Club's own
underwriting outputs: known at origination, but using them would mostly copy Lending Club's model. On the
synthetic test data, `recoveries` alone separates bad from good loans perfectly, which is what a leak looks like.
The guard runs at fit time, at score time, and in the API before a model is even loaded; a test fails if any
banned column reaches a model. `sub_grade` is reported only as a benchmark (test AUC
{{ credit.benchmark_lending_club_sub_grade.test_auc | ci }}{{ derived.credit_not_run }}).

**5. Calibration vs discrimination?**
Discrimination is ranking: does the model put riskier cases above safer ones (AUC, Gini, Lorenz Gini).
Calibration is level: do predicted rates match observed rates. On the freMTPL2 test split,
{{ derived.frequency_ranking }} (Lorenz Gini {{ insurance_frequency.test_metrics.lgbm.lorenz_gini | ci }} for
LightGBM vs {{ insurance_frequency.test_metrics.glm.lorenz_gini | ci }} for the GLM; paired difference
{{ insurance_frequency.test_metrics.lgbm_minus_glm.lorenz_gini | ci }}), {{ derived.frequency_total_calibration }}
(observed / predicted claims {{ insurance_frequency.test_metrics.lgbm.observed_to_predicted_claims | ci }} for
LightGBM and {{ insurance_frequency.test_metrics.glm.observed_to_predicted_claims | ci }} for the GLM). A model can
rank perfectly and still be off by a factor of two in level. For credit, class weighting inflates LightGBM's
probabilities, so they are Platt-recalibrated on 2016 without changing the ranking.

**6. What does the Poisson offset do?**
Claim counts grow with exposure: a policy on cover for half a year should expect half the claims. The GLM models
log(expected claims) = log(exposure) + Xβ with the log(exposure) coefficient fixed at 1, so Xβ is a rate per
policy-year. A test checks that this equals a Poisson regression of claims / exposure weighted by exposure. In
LightGBM the offset goes in as `init_score` = log(exposure) + log(training claim rate) (base
{{ insurance_frequency.models.lgbm.base_log_frequency | f4 }}); LightGBM does not store `init_score`, so prediction
adds the base back and returns claims per year. Early stopping kept
{{ insurance_frequency.models.lgbm.best_iteration | int }} trees.

**7. Why split frequency and severity, and how is expected loss built?**
They have different drivers and different noise. Frequency uses every policy
({{ insurance_frequency.split.sizes.train.n_policies | int }} in training); severity only uses policies with claim
amounts ({{ insurance_severity.split.severity_rows.train | int }} training rows), and amounts are heavy-tailed.
Claims are capped at {{ insurance_severity.large_loss_cap.cap_value | amount }} (the
{{ insurance_severity.large_loss_cap.quantile | text }} quantile of training claims), which caps
{{ insurance_severity.large_loss_cap.train.share_of_claims_capped | pct2 }} of claims but removes
{{ insurance_severity.large_loss_cap.train.share_of_amount_removed_by_cap | pct }} of the training amount, so
expected loss is on a capped basis and pricing would need a large-loss loading. Expected loss per policy-year is
frequency x severity. On the test split the GLM pair predicts
{{ insurance_severity.expected_loss.glm.total_predicted_loss | int }} against
{{ insurance_severity.expected_loss.glm.total_observed_loss_capped | int }} observed (capped); observed / predicted
{{ insurance_severity.expected_loss.glm.observed_to_predicted | ci }}. The API serves
{{ insurance_severity.expected_loss.served_pair.frequency_model | model }} frequency with
{{ insurance_severity.expected_loss.served_pair.severity_model | model }} severity, and that exact pair is validated
too: observed / predicted {{ insurance_severity.expected_loss.served_pair.observed_to_predicted | ci }}.
{{ derived.severity_vs_frequency }} (severity Lorenz Gini {{ derived.served_severity_gini | ci }} for the served
{{ insurance_severity.preferred_model.name | model }}, frequency {{ derived.served_frequency_gini | ci }} for the
served {{ insurance_frequency.preferred_model.name | model }}), {{ derived.pure_premium_driver }} (pure-premium
Lorenz Gini {{ insurance_severity.expected_loss.served_pair.pure_premium_lorenz_gini | ci }} for the served pair).

**8. What are the caveats of SHAP?**
SHAP splits one prediction into additive contributions; for trees it is exact (largest additivity error here
{{ insurance_frequency.explainability.max_additivity_error | sci }}). It describes how the model uses a feature, not
what would happen if the feature changed: correlated features share or swap credit, and the baseline is the
average training case, so "increased the risk" means "relative to that average".
{{ insurance_frequency.explainability.global_importance.0.feature }} has the largest mean absolute SHAP
({{ insurance_frequency.explainability.global_importance.0.mean_abs_shap | f4 }} on the log scale).
The scenario check shows the difference between explanation and behaviour: adding
{{ insurance_frequency.scenario_sanity_checks.checks.0.step | int }} to
{{ insurance_frequency.scenario_sanity_checks.checks.0.feature }} lowers LightGBM's predicted frequency for
{{ insurance_frequency.scenario_sanity_checks.checks.0.lgbm.share_violations | pct2 }} of test policies, and the
GLM's for {{ insurance_frequency.scenario_sanity_checks.checks.0.glm.share_violations | pct2 }}. A monotone
constraint would fix that if the business requires it.

**9. SR 26-2 vs SR 11-7: what changed, and why is generative AI out of scope?**
On April 17, 2026 the Federal Reserve, OCC and FDIC issued SR 26-2 / OCC Bulletin 2026-13 / FDIC FIL-15-2026,
which supersedes SR 11-7 (2011) and SR 21-8. It keeps the core (model inventory, documentation, effective
challenge, outcomes analysis, ongoing monitoring, vendor-model oversight) but is explicitly risk-based and
proportionate, says it is expected to be most relevant to banking organizations with over $30 billion in assets,
and excludes simple arithmetic and deterministic rule-based processes from the model definition. Footnote 3 says
generative AI and agentic AI models are not within its scope because they are novel and evolving. So the credit
card is "aligned to the structure of SR 26-2" (never "SR 11-7-style", never "compliant"), and the LLM extraction
and agent in later phases get their own evaluation, monitoring, release gate and human approval rather than a
stretched SR 26-2.

**10. Why validate scenario overrides instead of clipping them?**
A stress test answers "what if this value were X?". If X is outside what the model saw in training, clipping
silently answers a different question ("what if it were the maximum?") and reports it as X. The API instead
returns HTTP 422 naming the feature and its training range, so the user knows the scenario is outside the model's
evidence. The same idea runs through the project: the German Credit documentation (german.doc) labels the
personal-status-and-sex codes wrongly, as the corrected code table published with the South German Credit data
shows. The demo neither quietly "fixes" nor quietly uses the documented mapping: it reports the documented
mapping with a warning recorded in the results file (`fairness_demo.data_quality_warning`) plus a
corrected-code sensitivity check.

## Talking about Phase 1 with a financial-services consultancy

These points connect the work to what banks and insurers ask advisers for. They describe this repo only; they
make no claim about any firm's practice.

- **Model risk management deliverables.** A model card per model with purpose, materiality, data, methodology,
  validation results, outcomes analysis, monitoring thresholds, owner and version, plus a validation report with
  numbered findings. All generated from the results file, so documentation never drifts from the evidence.
- **Effective challenge with evidence.** Two data problems were found and acted on: freMTPL2 claims without
  amounts ({{ insurance_frequency.data_quality.linkage.share_of_claims_without_amounts | pct }} of claims,
  concentrated in one IDpol block; a GLM fitted on the recorded claim counts (capped at
  {{ insurance_frequency.data_quality.frequency_file.claim_count_cap }}, claims without amounts kept) showed observed / predicted claims of
  {{ insurance_frequency.recorded_claim_count_sensitivity.glm.observed_to_predicted_claims_outside_block | f3 }}
  outside that block, against
  {{ insurance_frequency.recorded_claim_count_sensitivity.glm.primary_model_observed_to_predicted_claims_outside_block | f3 }}
  for the chosen treatment), and the German Credit documentation's sex coding.
- **Fair lending and insurance fairness testing.** Adverse impact ratios with intervals (age under 25, logistic
  regression: {{ fairness_demo.models.logistic_regression.fairness.age_lt25.adverse_impact_ratio | ci3 }}), equal
  opportunity, and {{ derived.proxy_age_lt25 }}. Every interval that excludes "no difference" is raised as a
  validation finding with a multiple-comparisons caveat, rather than explained away.
- **Challenging the split, not only the model.** freMTPL2 has no customer identifier, so
  {{ insurance_frequency.grouped_split_sensitivity.share_of_test_policies_whose_rating_features_appear_in_train.random_split | pct }}
  of test policies share every rating feature with a training policy. Refitting on a split grouped on those
  features gives a LightGBM minus GLM deviance difference of
  {{ insurance_frequency.grouped_split_sensitivity.lgbm_minus_glm.mean_poisson_deviance | ci }}, against
  {{ insurance_frequency.test_metrics.lgbm_minus_glm.mean_poisson_deviance | ci }} on the headline split.
- **Explainability for adverse action.** Top-4 reason codes in Regulation B style for credit, and per-policy
  reasons for insurance, from SHAP or from GLM / scorecard contributions.
- **Scenario analysis through an API.** Validated what-if overrides with baseline vs scenario, risk bands, model
  versions and 503 / 422 answers that say why.
- **Generative AI governance gap.** SR 26-2 leaves generative and agentic AI out of scope, so those systems need
  their own evaluation suites, release gates and human approval: the subject of Phases 5 and 6.
