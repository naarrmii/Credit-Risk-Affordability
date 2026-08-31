"""Turns plain, human-entered financial details into the exact feature
vector the trained model expects.

Used by the Phase 6 dashboard's consumer-facing form, so a real person's
everyday numbers (income, debts, credit lines) map onto the same feature
engineering the historical training data went through in Phase 2 - without
asking anyone to type in an "estimated_monthly_debt_payment" themselves.
"""
import pandas as pd

# The exact feature set the deployed model expects. Age is intentionally
# absent - see Phase 5's decision to exclude it, retained only for monitoring.
MODEL_FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    'income_missing',
    'dependents_missing',
    'has_delinquency_data_error',
    'estimated_monthly_debt_payment',
    'disposable_income_estimate',
    'total_past_delinquencies',
]

# Same caps chosen in Phase 2, applied here too so an extreme typed input
# can't push the model outside the range it was actually trained on.
UTILIZATION_CAP = 2
DEBT_RATIO_CAP = 5


def build_feature_row(monthly_income, monthly_debt_payments, revolving_utilization,
                       num_open_credit_lines, num_real_estate_loans, num_dependents,
                       late_30_59, late_60_89, late_90_plus):
    """Build a single model-ready feature row from plain, human-entered inputs.

    This mirrors the Phase 2 cleaning/engineering logic, but for a live input
    rather than historical data: there's no "missing" income here (a real
    person just typed a number), and no sentinel/placeholder values to catch,
    so those flags are always 0.
    """
    debt_ratio = (monthly_debt_payments / monthly_income) if monthly_income > 0 else DEBT_RATIO_CAP
    debt_ratio = min(debt_ratio, DEBT_RATIO_CAP)
    utilization = min(revolving_utilization, UTILIZATION_CAP)

    # Use the real entered debt payment directly, not debt_ratio * income.
    # That derivation only existed because the historical training data never
    # had a raw payment figure, only a ratio. For a live person we already
    # know the true number precisely, so re-deriving it from a ratio that may
    # have been capped would silently understate it, and would report it as
    # exactly £0 whenever income is £0, regardless of any real debt entered.
    estimated_monthly_debt_payment = monthly_debt_payments
    disposable_income_estimate = monthly_income - estimated_monthly_debt_payment
    total_past_delinquencies = late_30_59 + late_60_89 + late_90_plus

    row = pd.Series({
        'RevolvingUtilizationOfUnsecuredLines': utilization,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': num_open_credit_lines,
        'NumberOfTimes90DaysLate': late_90_plus,
        'NumberRealEstateLoansOrLines': num_real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfDependents': num_dependents,
        'income_missing': 0,
        'dependents_missing': 0,
        'has_delinquency_data_error': 0,
        'estimated_monthly_debt_payment': estimated_monthly_debt_payment,
        'disposable_income_estimate': disposable_income_estimate,
        'total_past_delinquencies': total_past_delinquencies,
    })
    return row[MODEL_FEATURES]
