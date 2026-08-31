"""Recourse logic for the credit risk affordability project.

Used by both the Phase 4 notebook and the Phase 6 Streamlit dashboard, so the
"what changes would help, and by how much" logic only lives in one place.

Design principle: we only ever suggest changes to features a real applicant
can actually act on. Age, dependents, historical delinquency counts, and our
own pipeline's data-quality flags are all excluded on purpose - see
EXCLUDED_FEATURES for the reasoning behind each one.
"""
import pandas as pd

# Features we WILL suggest changes to - genuinely actionable by a real applicant
ACTIONABLE_FEATURES = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']

# Features we will NEVER suggest changing, and why - kept explicit and visible
# rather than silently dropped, so this is easy to defend if questioned
EXCLUDED_FEATURES = {
    'age': 'immutable',
    'NumberOfDependents': 'not appropriate to suggest changing',
    'total_past_delinquencies': 'historical - cannot be undone',
    'NumberOfTimes90DaysLate': 'historical - cannot be undone',
    'NumberOfTime30-59DaysPastDueNotWorse': 'historical - cannot be undone',
    'NumberOfTime60-89DaysPastDueNotWorse': 'historical - cannot be undone',
    'income_missing': 'pipeline artifact, not a real applicant attribute',
    'dependents_missing': 'pipeline artifact, not a real applicant attribute',
    'has_delinquency_data_error': 'pipeline artifact, not a real applicant attribute',
}


def simulate_change(row, utilization_mult=1.0, debt_ratio_mult=1.0):
    """Return a copy of `row` with actionable features scaled, and the
    downstream derived features recomputed consistently, ready to feed
    back into the model. Changing DebtRatio alone, without also updating
    estimated_monthly_debt_payment and disposable_income_estimate, would
    hand the model an internally inconsistent, impossible row.
    """
    new_row = row.copy()
    new_row['RevolvingUtilizationOfUnsecuredLines'] *= utilization_mult
    new_row['DebtRatio'] *= debt_ratio_mult
    new_row['estimated_monthly_debt_payment'] = new_row['DebtRatio'] * new_row['MonthlyIncome']
    new_row['disposable_income_estimate'] = new_row['MonthlyIncome'] - new_row['estimated_monthly_debt_payment']
    return new_row


def _predict_one(model, row):
    """Predict on a single row, selecting only the columns the model was
    actually trained on. This keeps the recourse logic working whether the
    model includes `age` or not - the row itself can carry extra columns
    (e.g. for display), they're just ignored at prediction time."""
    row_df = pd.DataFrame([row])
    if hasattr(model, 'feature_names_in_'):
        row_df = row_df[model.feature_names_in_]
    return model.predict_proba(row_df)[:, 1][0]


def get_recourse_scenarios(row, model, utilization_cuts=(0.5, 0.75), debt_ratio_cuts=(0.7, 0.85)):
    """Try a small set of realistic changes and return the baseline score
    plus each scenario's estimated new score, sorted best (lowest risk) first.

    This re-runs the actual model on each hypothetical row rather than adding
    up SHAP values by hand - SHAP values are a local, approximate explanation,
    and re-predicting directly is the more trustworthy way to estimate the
    effect of a genuinely different, non-local scenario on a nonlinear model.
    """
    baseline_score = _predict_one(model, row)

    scenarios = {}
    for u in utilization_cuts:
        label = f"Reduce revolving utilisation by {int((1 - u) * 100)}%"
        new_row = simulate_change(row, utilization_mult=u)
        scenarios[label] = _predict_one(model, new_row)

    for d in debt_ratio_cuts:
        label = f"Reduce debt ratio by {int((1 - d) * 100)}%"
        new_row = simulate_change(row, debt_ratio_mult=d)
        scenarios[label] = _predict_one(model, new_row)

    both_row = simulate_change(row, utilization_mult=min(utilization_cuts), debt_ratio_mult=min(debt_ratio_cuts))
    both_label = (f"Reduce revolving utilisation by {int((1 - min(utilization_cuts)) * 100)}% "
                  f"AND debt ratio by {int((1 - min(debt_ratio_cuts)) * 100)}%")
    scenarios[both_label] = _predict_one(model, both_row)

    results = sorted(scenarios.items(), key=lambda kv: kv[1])
    return baseline_score, results


# Illustrative bands only - not an official lending cutoff, just a plain-English
# translation of the raw probability so a score means something on first read.
RISK_BANDS = [
    (0.10, "Low risk", "🟢"),
    (0.25, "Moderate risk", "🟡"),
    (0.50, "Elevated risk", "🟠"),
    (1.01, "High risk", "🔴"),
]


def risk_band(score):
    """Translate a raw 0-1 risk score into a plain-English band and an icon."""
    for threshold, label, icon in RISK_BANDS:
        if score < threshold:
            return label, icon
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]
