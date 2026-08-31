"""Credit Risk & Affordability - two-view dashboard.

Run with: streamlit run app.py  (from the project's root folder)
"""
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from src.recourse import get_recourse_scenarios, risk_band
from src.features import build_feature_row

st.set_page_config(page_title="Credit Risk and Affordability, Imraan Morolong", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#0F172A"
SURFACE = "#FFFFFF"
BG = "#F8FAFC"
ACCENT = "#4F46E5"
ACCENT_2 = "#7C3AED"
MUTED = "#64748B"
BORDER = "#E2E8F0"
SIDEBAR_BG = "#0B0F19"

GRADIENT = f"linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%)"

RISK_COLORS = {
    "Low risk": "#10B981",
    "Moderate risk": "#F59E0B",
    "Elevated risk": "#F97316",
    "High risk": "#EF4444",
}

CHART_FONT = dict(family="Inter, sans-serif", color=INK)
TRANSPARENT_BG = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# Plain English descriptions used only on the consumer side, so someone with
# no background in this at all can see why they got their result, not just
# what they could change about it.
FEATURE_PLAIN_ENGLISH = {
    'RevolvingUtilizationOfUnsecuredLines': 'how much of your credit limit you are using',
    'NumberOfTime30-59DaysPastDueNotWorse': 'being 30 to 59 days late on a payment in the past',
    'DebtRatio': 'how much of your income goes toward debt',
    'MonthlyIncome': 'your monthly income',
    'NumberOfOpenCreditLinesAndLoans': 'how many credit accounts and loans you have open',
    'NumberOfTimes90DaysLate': 'being 90 or more days late on a payment in the past',
    'NumberRealEstateLoansOrLines': 'your mortgages and property loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'being 60 to 89 days late on a payment in the past',
    'NumberOfDependents': 'the number of people who depend on you financially',
    'estimated_monthly_debt_payment': 'your estimated monthly debt payments',
    'disposable_income_estimate': 'how much money you have left after your debts are paid',
    'total_past_delinquencies': 'your overall history of missed payments',
}


def afford_verdict(dti_pct, num_dependents, total_delinquencies, utilization_pct, model_score, threshold):
    """A single plain verdict built from more than one figure, since real
    affordability is never just a debt to income ratio. Each reason states
    its own number and what that number means, so nobody reading it has to
    do the interpretation themselves.

    Severity matters, not just how many categories are triggered - a debt to
    income ratio of 44 per cent and one of 400 per cent must not produce the
    same verdict just because neither one happens to trip a second concern.
    """
    reasons = []
    concerns = 0

    if dti_pct >= 100:
        reasons.append(
            f"your debt to income ratio is {dti_pct:.0f} per cent, meaning these payments alone "
            f"would cost more than your entire income"
        )
        return "This is not affordable. The payments would exceed your income entirely.", reasons, "error"

    if dti_pct >= 70:
        reasons.append(f"your debt to income ratio is {dti_pct:.0f} per cent, far beyond what almost any lender would consider workable")
        concerns += 2
    elif dti_pct >= 43:
        reasons.append(f"your debt to income ratio is {dti_pct:.0f} per cent, which many lenders would treat as risky")
        concerns += 1
    elif dti_pct >= 36:
        reasons.append(f"your debt to income ratio is {dti_pct:.0f} per cent, a little higher than lenders usually prefer")
        concerns += 1
    else:
        reasons.append(f"your debt to income ratio is {dti_pct:.0f} per cent, in a range lenders generally see as healthy")

    if num_dependents >= 2:
        reasons.append(f"you have {num_dependents} dependents, which raises your real cost of living beyond the debts listed here")
        concerns += 1

    if total_delinquencies > 0:
        reasons.append("you have a history of missed payments")
        concerns += 1

    if utilization_pct >= 75:
        reasons.append(f"you are using {utilization_pct:.0f} per cent of your available credit")
        concerns += 1

    if model_score >= threshold:
        reasons.append("the model's own risk score also flags this as high risk")
        concerns += 1

    if concerns == 0:
        return "This looks affordable based on what you have entered.", reasons, "success"
    if concerns == 1:
        return "This looks broadly affordable, though one thing is worth knowing.", reasons, "warning"
    return "This looks like it could be a stretch, based on more than one factor.", reasons, "error"


def metrics_at_threshold(y_true, probs, threshold):
    """Precision, recall, and the share of applications flagged, at one
    specific decision threshold, computed directly rather than read off the
    precision-recall curve, so the number shown always matches exactly."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    selection_rate = preds.mean()
    return precision, recall, selection_rate


def explain_single_result(model, row):
    """Work out, in plain terms, which pieces of information mattered most
    for one specific person's score. Kept separate from the bank view's SHAP
    chart deliberately, since this needs to read as an outcome, not a method."""
    explainer = shap.TreeExplainer(model)
    row_df = pd.DataFrame([row])[model.feature_names_in_]
    raw = explainer.shap_values(row_df)
    if isinstance(raw, list):
        values = np.array(raw[1])[0]
    elif raw.ndim == 3:
        values = raw[0, :, 1]
    else:
        values = raw[0]
    pairs = [(f, v) for f, v in zip(row_df.columns, values) if f in FEATURE_PLAIN_ENGLISH]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:3]


def hex_to_rgba(hex_color, alpha):
    """Plotly's colour validator wants rgba(r,g,b,a), not eight digit hex with alpha."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def inject_custom_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], p, li, span {{
        font-family: 'Inter', sans-serif;
        color: {INK};
    }}
    .stApp {{ background-color: {BG}; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .block-container {{
        padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 1400px;
    }}

    .eyebrow {{
        text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem;
        font-weight: 700; color: {ACCENT}; margin-bottom: 4px;
    }}
    .tab-accent-bar {{ height: 3px; width: 44px; border-radius: 3px; background: {GRADIENT}; margin: 6px 0 14px 0; }}

    [data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG}; border-right: 1px solid #1E293B; }}
    [data-testid="stSidebar"] * {{ color: #94A3B8 !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: #F8FAFC !important; font-weight: 600; }}
    [data-testid="stSidebar"] hr {{ border-color: #1E293B; }}

    .author-card {{
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-top: 20px;
    }}
    .author-label {{ color: #64748B !important; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 4px; }}
    .author-name {{ color: #F8FAFC !important; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; }}
    .author-degree {{ color: #94A3B8 !important; font-size: 0.8rem; margin: 2px 0 0 0; line-height: 1.5; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; background-color: transparent; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 8px 8px 0 0; padding: 12px 24px; font-weight: 600; color: {MUTED}; border-bottom: 2px solid transparent; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; background-color: transparent !important; }}

    [data-testid="stMetric"] {{
        background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    [data-testid="stMetric"]:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(15,23,42,0.08); }}
    [data-testid="stMetricValue"] {{ font-weight: 700 !important; color: {INK} !important; }}

    [data-testid="stFormSubmitButton"] button, .stButton > button {{
        background: {GRADIENT} !important; color: #FFFFFF !important; border-radius: 8px !important; border: none !important;
        padding: 0.6rem 1rem !important; font-weight: 600 !important; transition: all 0.2s; width: 100%;
    }}
    [data-testid="stFormSubmitButton"] button:hover, .stButton > button:hover {{
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35); transform: translateY(-1px);
    }}

    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {SURFACE}; border-radius: 12px !important; border: 1px solid {BORDER} !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03) !important;
        padding: 1.5rem !important;
    }}
    .stNumberInput input, .stTextInput input {{ border-radius: 6px !important; border: 1px solid {BORDER} !important; }}

    .action-card {{
        background-color: {SURFACE}; border-left: 4px solid {ACCENT}; border-radius: 8px;
        padding: 16px 20px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        display: flex; justify-content: space-between; align-items: center;
        border-right: 1px solid {BORDER}; border-top: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .action-card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 24px rgba(15,23,42,0.10); }}
    .action-title {{ font-weight: 600; color: {INK}; margin-bottom: 4px; font-size: 1rem; }}
    .action-impact {{ color: {MUTED}; font-size: 0.9rem; }}
    .action-badge {{ background-color: #EEF2FF; color: {ACCENT}; padding: 6px 12px; border-radius: 99px; font-weight: 700; font-size: 0.9rem; }}

    .gradient-text {{
        background: {GRADIENT}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }}

    .comparison-chip {{
        background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 14px 12px; text-align: center; height: 100%;
    }}
    .comparison-chip .chip-label {{ font-size: 0.7rem; color: {MUTED}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
    .comparison-chip .chip-value {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin-top: 6px; }}
    .comparison-chip .chip-arrow {{ color: {ACCENT}; font-weight: 700; }}
    </style>
    """, unsafe_allow_html=True)


def render_action_card(title, new_score, point_drop, band_label):
    st.markdown(f"""
        <div class="action-card">
            <div>
                <div class="action-title">{title}</div>
                <div class="action-impact">Estimated risk falls to {new_score:.0%}, {band_label}.</div>
            </div>
            <div class="action-badge">Down {point_drop:.0f} points</div>
        </div>
    """, unsafe_allow_html=True)


def render_comparison_chip(label, before, after):
    st.markdown(f"""
        <div class="comparison-chip">
            <div class="chip-label">{label}</div>
            <div class="chip-value">{before} <span class="chip-arrow">to</span> {after}</div>
        </div>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    return joblib.load('data/processed/credit_risk_model.joblib')


@st.cache_data(show_spinner="Loading the data...")
def load_split():
    df = pd.read_csv('data/processed/cleaned_features.csv', index_col=0)
    X = df.drop(columns=['SeriousDlqin2yrs'])
    y = df['SeriousDlqin2yrs']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


@st.cache_data
def compute_shap_values(_model, X_model):
    explainer = shap.TreeExplainer(_model)
    raw = explainer.shap_values(X_model)
    if isinstance(raw, list):
        return np.array(raw[1])
    if raw.ndim == 3:
        return raw[:, :, 1]
    return raw


@st.cache_data
def compute_age_fairness(_model, X_full, X_model, y_true):
    bins = [17, 30, 45, 60, 120]
    labels = ['18 to 30', '31 to 45', '46 to 60', '61 and over']
    age_group = pd.cut(X_full['age'], bins=bins, labels=labels)
    y_pred = _model.predict(X_model)
    results = pd.DataFrame({'age_group': age_group.values, 'y_true': y_true.values, 'y_pred': y_pred})

    def group_metrics(g):
        negatives = g[g['y_true'] == 0]
        positives = g[g['y_true'] == 1]
        return pd.Series({
            'n': len(g),
            'Flagged high risk': g['y_pred'].mean(),
            'Wrongly flagged': negatives['y_pred'].mean() if len(negatives) else np.nan,
            'Missed real risk': (1 - positives['y_pred']).mean() if len(positives) else np.nan,
        })
    return results.groupby('age_group', observed=True).apply(group_metrics, include_groups=False)


inject_custom_css()
model = load_model()
X_train, X_test, y_train, y_test = load_split()
X_test_model = X_test[list(model.feature_names_in_)]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Credit Risk and Affordability")
    st.markdown("A project built around one idea. A lending decision should be explainable and fair, not just accurate.")
    st.divider()
    st.markdown("**Navigation**")
    st.markdown("Bank view: how the model performs, what drives its decisions, and whether it treats people fairly.")
    st.markdown("Consumer view: check an estimated score and see what could improve it.")

    st.markdown(f"""
    <div class="author-card">
        <div class="author-label">DEVELOPED BY</div>
        <div class="author-name">Imraan Morolong</div>
        <div class="author-degree">MSc Data Science</div>
        <div class="author-degree">BSc Accounting and Finance</div>
    </div>
    """, unsafe_allow_html=True)

bank_tab, consumer_tab = st.tabs(["Bank view", "Consumer view"])

# ============================== BANK VIEW ==============================
with bank_tab:
    st.markdown('<div class="eyebrow">FOR THE BANK</div>', unsafe_allow_html=True)
    st.markdown("#### How the model performs")
    st.markdown('<div class="tab-accent-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        "This is the view a bank's risk team would look at. It shows how well the model performs, "
        "what actually drives its decisions, and whether it treats people fairly. Nothing here is "
        "dressed up. That includes the parts that are not fully solved yet."
    )
    st.write("")

    probs = model.predict_proba(X_test_model)[:, 1]
    auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    random_pr_baseline = y_test.mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC", f"{auc:.3f}", delta=f"+{auc - 0.5:.3f} vs random")
    m2.metric("PR-AUC", f"{pr_auc:.3f}", delta=f"+{pr_auc - random_pr_baseline:.3f} vs random")
    m3.metric("Baseline default rate", f"{random_pr_baseline:.2%}", delta="Highly imbalanced", delta_color="off")

    with st.expander("What do ROC-AUC and PR-AUC actually mean?"):
        st.markdown(
            f"ROC-AUC measures how well the model can tell a risky borrower from a safe one. Picture "
            f"picking one person who defaulted and one who did not, at random. ROC-AUC is roughly how "
            f"often the model correctly guesses which of the two was riskier. A score of 0.5 means it "
            f"is guessing. A score of 1.0 means it always gets it right. This model scores {auc:.2f}, "
            f"clearly better than guessing, and a realistic number rather than a suspiciously perfect "
            f"one. A model that looks almost perfect usually has a data leak somewhere, not genuinely "
            f"outstanding performance.\n\n"
            f"PR-AUC matters more in this particular case. Only about {random_pr_baseline:.1%} of "
            f"borrowers in this data actually defaulted, so a model that guessed at random would score "
            f"close to that same rate. This model scores {pr_auc:.2f}, comfortably above that baseline. "
            f"When an event is this rare, PR-AUC gives a more honest picture than ROC-AUC on its own."
        )

    st.write("")
    st.subheader("The trade off this model makes")
    precision, recall, _ = precision_recall_curve(y_test, probs)

    st.markdown(
        "Every point on the line below is a different decision threshold. In plain terms, if the "
        "bank wants to catch nearly every real defaulter, it will also end up wrongly flagging a lot "
        "of good borrowers as risky. If it wants to avoid wrongly flagging good borrowers, it will "
        "miss more real defaulters. There is no single setting that solves this perfectly, so the "
        "slider below lets you set where the bank actually draws that line. This is the same line "
        "used in the consumer view, so moving it here changes what counts as high risk there too."
    )

    threshold_pct = st.slider(
        "Flag an application as high risk above this estimated score", 0, 100, 50, format="%d%%",
        key="decision_threshold_pct"
    )
    threshold = threshold_pct / 100

    thresh_precision, thresh_recall, thresh_selection = metrics_at_threshold(y_test.values, probs, threshold)
    tcol1, tcol2, tcol3 = st.columns(3)
    tcol1.metric("Applications flagged", f"{thresh_selection:.1%}")
    tcol2.metric("Real defaulters caught", f"{thresh_recall:.1%}")
    tcol3.metric("Accuracy of a flag", f"{thresh_precision:.1%}")

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines', fill='tozeroy', name='Model',
        line=dict(color=ACCENT, width=3), fillcolor=hex_to_rgba(ACCENT, 0.12),
        hovertemplate="Recall: %{x:.0%}<br>Precision: %{y:.0%}<extra></extra>"
    ))
    fig_pr.add_trace(go.Scatter(
        x=[thresh_recall], y=[thresh_precision], mode='markers', name='Current setting',
        marker=dict(size=14, color=INK, symbol='circle'),
        hovertemplate="Current setting<br>Recall: %{x:.0%}<br>Precision: %{y:.0%}<extra></extra>"
    ))
    fig_pr.update_layout(
        xaxis_title="Recall: the share of real defaulters actually caught",
        yaxis_title="Precision: how accurate a high risk flag actually is",
        height=380, margin=dict(l=10, r=10, t=20, b=10), font=CHART_FONT,
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER), showlegend=False,
        hoverlabel=dict(bgcolor=SURFACE, font=CHART_FONT, bordercolor=BORDER), **TRANSPARENT_BG
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.write("")
    st.subheader("What drives the model's decisions")
    st.markdown(
        "The bars below show how much each piece of information moves the model's prediction, on "
        "average. This comes from a method called SHAP, which works out how much each feature "
        "contributed to a decision, rather than leaving the model as a black box nobody can explain."
    )
    shap_values = compute_shap_values(model, X_test_model)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X_test_model.columns, 'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)

    fig_shap = go.Figure(go.Bar(
        x=importance_df['importance'], y=importance_df['feature'], orientation='h',
        marker_color=ACCENT, marker_line_width=0,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"
    ))
    fig_shap.update_layout(
        xaxis_title="Average impact on predicted risk",
        height=520, margin=dict(l=10, r=10, t=10, b=10), font=CHART_FONT,
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        hoverlabel=dict(bgcolor=SURFACE, font=CHART_FONT, bordercolor=BORDER), **TRANSPARENT_BG
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.markdown(
        "Past payment history and credit utilisation matter most here. That is exactly the kind of "
        "signal you would expect from a credit model, not a random pattern. Age does not appear at "
        "all. It has been deliberately left out, and the reason why is explained just below."
    )
    st.markdown(
        "One real limitation follows directly from this. Payment history only shows up after someone "
        "has already struggled for a while, while a change in income shows up immediately. Because "
        "this model leans so heavily on the former, it can understate the risk for someone whose "
        "income has just dropped sharply but who has not yet missed a payment. That is a genuine gap, "
        "not a corner case worth ignoring."
    )

    st.write("")
    st.subheader("How this connects to affordability")
    st.markdown(
        "Most credit models only ask whether someone is likely to default. This one also tries to "
        "capture whether a loan looks genuinely affordable for someone's real financial life, not "
        "just statistically low risk on paper. Two features do this directly: an estimate of "
        "someone's monthly debt payments in pounds, and an estimate of what they would have left "
        "over each month once those payments are made. Neither is the single biggest driver in the "
        "chart above, since raw payment history and credit use matter more to the model's accuracy. "
        "They matter for a different reason. They turn a raw risk score into something a person can "
        "actually understand and act on, which is why the same two figures appear again in the "
        "consumer view."
    )

    st.write("")
    st.subheader("Why age is not part of this model")
    st.markdown(
        "While building this model, age turned out to be one of the strongest predictors of risk. Age "
        "is also a protected characteristic under the UK Equality Act, so rather than assume that was "
        "fine, it was tested directly. Eighteen to thirty year olds were being flagged as high risk "
        "roughly five times as often as people over sixty one, a gap far too large to be a coincidence."
    )

    chip1, chip2, chip3 = st.columns(3)
    with chip1:
        render_comparison_chip("Selection rate gap", "34.3%", "26.7%")
    with chip2:
        render_comparison_chip("False positive gap", "29.3%", "21.7%")
    with chip3:
        render_comparison_chip("False negative gap", "29.1%", "18.8%")
    st.caption("Gap across age groups, before and after age was removed from training.")

    st.write("")
    st.markdown(
        "Age was removed from the model entirely. This cost almost nothing in accuracy, which "
        "suggests the model did not really need it once it already knew someone's payment history and "
        "credit use. Removing age narrowed the gap shown above but did not close it. The age figures "
        "shown above come from the historical dataset this model was tested against, which already "
        "included age, and that is the only reason this comparison could be made at all. Nobody using "
        "the consumer view is ever asked for their age, and the live model cannot see it under any "
        "circumstances. This mirrors how FICO scores work in practice, since they leave age out too "
        "and rely on credit history instead."
    )

    fairness_df = compute_age_fairness(model, X_test, X_test_model, y_test)
    fairness_colors = [ACCENT, "#F59E0B", "#EF4444"]
    fig_fair = go.Figure()
    for col_name, color in zip(
        ['Flagged high risk', 'Wrongly flagged', 'Missed real risk'], fairness_colors
    ):
        fig_fair.add_trace(go.Bar(
            name=col_name, x=fairness_df.index.astype(str), y=fairness_df[col_name], marker_color=color,
            hovertemplate=f"{col_name}: " + "%{y:.1%}<extra></extra>"
        ))
    fig_fair.update_layout(
        barmode='group', yaxis_tickformat='.0%', height=380, font=CHART_FONT,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(gridcolor='rgba(0,0,0,0)'), yaxis=dict(gridcolor=BORDER),
        hoverlabel=dict(bgcolor=SURFACE, font=CHART_FONT, bordercolor=BORDER), **TRANSPARENT_BG
    )
    st.plotly_chart(fig_fair, use_container_width=True)
    st.markdown(
        "Each group of bars is one age band. The three colours show, for that age band, how often "
        "the model flagged someone as high risk, how often that flag turned out to be wrong, and how "
        "often a real default was missed entirely. The pattern to look for is how much those bars "
        "change from the youngest group to the oldest, which is exactly the gap described above."
    )

# ============================== CONSUMER VIEW ==============================
with consumer_tab:
    st.markdown('<div class="eyebrow">FOR YOU</div>', unsafe_allow_html=True)
    st.markdown("#### Affordability check")
    st.markdown('<div class="tab-accent-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        "Fill in a few details about your finances below. This estimates how a lending model might "
        "assess your application. It uses the same kind of model shown in the bank view, just from "
        "your side of the table."
    )
    st.info(
        "This is a demonstration model trained on historical, anonymised data. It is not a real "
        "credit decision, not financial advice, and does not reflect any specific real lender's "
        "criteria."
    )

    with st.form("consumer_inputs"):
        st.markdown("**Your financial details**")
        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_income = st.number_input(
                "Monthly income", min_value=0.0, value=3000.0, step=100.0,
                help="Your total take home pay each month, after tax, in pounds."
            )
            num_open_credit_lines = st.number_input("Open credit lines and loans", min_value=0, value=4, step=1)
        with col2:
            monthly_debt_payments = st.number_input(
                "Monthly debt payments", min_value=0.0, value=500.0, step=50.0,
                help="Everything you pay each month toward existing debts, such as cards and loans, in pounds."
            )
            num_real_estate_loans = st.number_input("Mortgages and property loans", min_value=0, value=0, step=1)
        with col3:
            revolving_utilization_pct = st.slider(
                "Credit card utilisation", 0, 100, 30, format="%d%%",
                help="How much of your available credit limit you are currently using."
            )
            num_dependents = st.number_input("Number of dependents", min_value=0, value=0, step=1)

        st.markdown("<br>**Missed payments in the past**", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        late_30_59 = c1.number_input("30 to 59 days late", min_value=0, value=0, step=1)
        late_60_89 = c2.number_input("60 to 89 days late", min_value=0, value=0, step=1)
        late_90_plus = c3.number_input("90 or more days late", min_value=0, value=0, step=1)

        st.write("")
        submitted = st.form_submit_button("Check my estimated score")

    if submitted:
        st.session_state['consumer_submitted'] = True

    if st.session_state.get('consumer_submitted', False):
        row = build_feature_row(
            monthly_income=monthly_income, monthly_debt_payments=monthly_debt_payments,
            revolving_utilization=revolving_utilization_pct / 100, num_open_credit_lines=num_open_credit_lines,
            num_real_estate_loans=num_real_estate_loans, num_dependents=num_dependents,
            late_30_59=late_30_59, late_60_89=late_60_89, late_90_plus=late_90_plus,
        )
        baseline_score, results = get_recourse_scenarios(row, model)
        band_label, _ = risk_band(baseline_score)

        st.write("")
        st.markdown("#### Your result")
        st.markdown(
            "This score estimates how likely someone with this financial profile is to fall seriously "
            "behind on payments within the next two years, meaning ninety or more days late, based on "
            "patterns in past data. Lower is better."
        )

        if monthly_income < 200:
            st.warning(
                "Your income is very low or zero. This model leans heavily on payment history rather "
                "than income, which is a real limitation worth knowing: a sudden loss of income has "
                "not yet shown up as a missed payment, so this score may understate the risk here. "
                "Treat the number below with real caution rather than at face value."
            )

        with st.container(border=True):
            gauge_col, text_col = st.columns([1, 1])
            with gauge_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=baseline_score * 100,
                    number={'suffix': '%', 'font': {'family': 'Inter, sans-serif', 'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': INK},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'steps': [
                            {'range': [0, 10], 'color': hex_to_rgba(RISK_COLORS["Low risk"], 0.2)},
                            {'range': [10, 25], 'color': hex_to_rgba(RISK_COLORS["Moderate risk"], 0.2)},
                            {'range': [25, 50], 'color': hex_to_rgba(RISK_COLORS["Elevated risk"], 0.2)},
                            {'range': [50, 100], 'color': hex_to_rgba(RISK_COLORS["High risk"], 0.2)},
                        ],
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=10), font=CHART_FONT, **TRANSPARENT_BG)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with text_col:
                st.markdown(
                    f'<div class="gradient-text" style="font-size:3.2rem;font-weight:800;line-height:1;">{baseline_score:.0%}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"### {band_label}")
                st.caption("These bands are for illustration. They are not an official lending cutoff.")

        current_dti = (monthly_debt_payments / monthly_income * 100) if monthly_income > 0 else 100.0
        total_delinquencies = row['total_past_delinquencies']
        threshold = st.session_state.get('decision_threshold_pct', 50) / 100
        headline, reasons, style = afford_verdict(
            current_dti, num_dependents, total_delinquencies, revolving_utilization_pct, baseline_score, threshold
        )

        model_flagged = baseline_score >= threshold
        affordability_flagged = (style == "error")

        if model_flagged or affordability_flagged:
            if model_flagged and affordability_flagged:
                reason_text = "both the model's score and the affordability check below"
            elif model_flagged:
                reason_text = "the model's score"
            else:
                reason_text = (
                    "the affordability check below, even though the model's own score did not "
                    "cross the bank's threshold on its own"
                )
            st.warning(
                f"This application would be flagged for extra review, based on {reason_text}. "
                f"Your estimated risk score is {baseline_score:.0%}, against the bank's current "
                f"threshold of {threshold:.0%}."
            )
        else:
            st.success(
                f"This application would not be flagged for extra review. Your estimated risk "
                f"score of {baseline_score:.0%} is below the bank's current threshold of "
                f"{threshold:.0%}, and the affordability check below did not raise a serious "
                f"concern either."
            )

        st.write("")
        st.markdown("### Your affordability at a glance")
        st.markdown(
            "Affordability here means whether taking on this level of credit looks sustainable for "
            "you, not just whether the model predicts you might default. That depends on more than "
            "one thing: how your existing debts compare to your income, how much of your available "
            "credit you are already using, how many people rely on you financially, and whether you "
            "have kept up with payments in the past."
        )
        getattr(st, style)(headline)
        for reason in reasons:
            st.markdown(f"- {reason[0].upper()}{reason[1:]}")

        st.write("")
        st.markdown("**Thinking about taking on more credit**")
        extra_payment = st.number_input(
            "Extra monthly payment you are considering, in pounds", min_value=0.0, value=0.0, step=25.0,
            key="extra_payment_input",
            help="Enter the expected monthly cost of a new loan, card or payment plan. Leave at 0 if you are not considering anything new."
        )
        if extra_payment > 0:
            new_dti = ((monthly_debt_payments + extra_payment) / monthly_income * 100) if monthly_income > 0 else 100.0
            hypothetical_row = build_feature_row(
                monthly_income=monthly_income, monthly_debt_payments=monthly_debt_payments + extra_payment,
                revolving_utilization=revolving_utilization_pct / 100, num_open_credit_lines=num_open_credit_lines,
                num_real_estate_loans=num_real_estate_loans, num_dependents=num_dependents,
                late_30_59=late_30_59, late_60_89=late_60_89, late_90_plus=late_90_plus,
            )
            hyp_row_df = pd.DataFrame([hypothetical_row])[model.feature_names_in_]
            hypothetical_score = model.predict_proba(hyp_row_df)[:, 1][0]

            st.write("")
            st.markdown("With that extra payment added:")
            new_headline, new_reasons, new_style = afford_verdict(
                new_dti, num_dependents, total_delinquencies, revolving_utilization_pct, hypothetical_score, threshold
            )
            getattr(st, new_style)(new_headline)
            for reason in new_reasons:
                st.markdown(f"- {reason[0].upper()}{reason[1:]}")

        st.write("")
        top_factors = explain_single_result(model, row)
        st.markdown("### Why you got this score")
        st.markdown("In plain terms, here is what mattered most for your result, based on what you entered above.")
        for feature, value in top_factors:
            direction = "pushed your risk up" if value > 0 else "pushed your risk down"
            st.markdown(f"- {FEATURE_PLAIN_ENGLISH[feature]}, which {direction}")

        st.write("")
        st.markdown("### What could help")
        st.markdown(
            "These are the changes most likely to help, based on this model, starting with the one "
            "that would help most. They show the change in your estimated risk, not a promise about a "
            "real application."
        )

        any_shown = False
        for label, new_score in results[:3]:
            point_drop = (baseline_score - new_score) * 100
            if point_drop <= 0.5:
                continue
            any_shown = True
            new_band_label, _ = risk_band(new_score)
            render_action_card(label, new_score, point_drop, new_band_label)

        if not any_shown:
            if baseline_score < 0.25:
                st.markdown("""
                <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; padding: 20px; text-align: center; border-radius: 8px;">
                    <p style="color: #64748B; margin: 0; font-weight: 500;">Your profile already looks stable.</p>
                    <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">None of the changes we tried moved your score by very much, which is a good sign here.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #FEF2F2; border: 1px dashed #FCA5A5; padding: 20px; text-align: center; border-radius: 8px;">
                    <p style="color: #991B1B; margin: 0; font-weight: 500;">Reducing your credit use or your debt would not meaningfully change your score here.</p>
                    <p style="color: #B91C1C; font-size: 0.85rem; margin: 0;">That usually means the biggest factors behind your result sit elsewhere, most often a history of missed payments, which cannot be changed after the fact. See "Why you got this score" above for the specific factors driving your result.</p>
                </div>
                """, unsafe_allow_html=True)

        st.caption(
            "These are estimates based on patterns in past data. They are not a guarantee of "
            "approval, and not a promise about how a real lender would respond."
        )
