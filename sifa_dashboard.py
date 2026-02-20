"""
Sifa-Score AI Dashboard v2.0
-----------------------------
Explainable AI for Kenyan credit risk assessment.
SDG #1: No Poverty

Author: Derick Gikonyo | February 2026
"""

import os
import matplotlib                          # ← must come BEFORE pyplot
matplotlib.use('Agg')                      # ← thread-safe backend for Streamlit
import matplotlib.pyplot as plt

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sifa-Score AI",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — theme-aware (works in both light and dark mode) ────────────────────
st.markdown("""
<style>
/* Metric cards: no hardcoded background so dark/light theme both work */
[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    opacity: 0.65;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
}

/* Decision badges */
.badge-approved {
    background: #d1fae5; color: #065f46;
    padding: 5px 14px; border-radius: 20px; font-weight: 700;
}
.badge-declined {
    background: #fee2e2; color: #991b1b;
    padding: 5px 14px; border-radius: 20px; font-weight: 700;
}

/* Advisory letter */
.advisory-box {
    border-left: 5px solid #2563eb;
    padding: 24px 28px;
    border-radius: 0 12px 12px 0;
    line-height: 1.8;
    white-space: pre-wrap;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "sifa_model.joblib"
META_PATH    = "sifa_meta.joblib"
DATA_PATH    = "kiva_loans.csv"
FEATURE_COLS = [
    'loan_amount', 'term_in_months', 'lender_count',
    'borrower_count', 'female_ratio', 'loan_per_borrower',
    'month', 'day_of_week', 'sector', 'activity'
]
SECTORS = sorted([
    'Agriculture', 'Arts', 'Clothing', 'Construction', 'Education',
    'Entertainment', 'Financial Services', 'Food', 'Health', 'Housing',
    'Manufacturing', 'Personal Use', 'Retail', 'Services',
    'Transportation', 'Wholesale'
])

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def _parse_gender(s):
    if pd.isna(s) or str(s).strip() == '':
        return 1, 0.5
    parts = [g.strip().lower() for g in str(s).split(',')]
    n = len(parts)
    return n, sum(1 for g in parts if 'female' in g) / n


def engineer_features(df, top_activities=None):
    d = df.copy()
    gs = d['borrower_genders'].apply(_parse_gender)
    d['borrower_count']    = gs.apply(lambda x: x[0])
    d['female_ratio']      = gs.apply(lambda x: x[1])
    d['loan_per_borrower'] = d['loan_amount'] / d['borrower_count'].clip(lower=1)
    d['posted_time']       = pd.to_datetime(d['posted_time'], errors='coerce')
    d['month']             = d['posted_time'].dt.month.fillna(6).astype(int)
    d['day_of_week']       = d['posted_time'].dt.dayofweek.fillna(0).astype(int)

    if top_activities is None:
        top_activities = d['activity'].value_counts().head(50).index.tolist()
    d['activity'] = d['activity'].apply(
        lambda x: x if x in top_activities else 'Other'
    )
    X = pd.get_dummies(d[FEATURE_COLS], drop_first=True)
    return X, top_activities


# ── MODEL: LOAD OR TRAIN ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading Sifa-Score model…")
def load_sifa_stack():
    """Load persisted model+metadata, or train fresh if not found."""
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        return joblib.load(MODEL_PATH), joblib.load(META_PATH)

    if not os.path.exists(DATA_PATH):
        st.error(f"`{DATA_PATH}` not found. Add it to the app folder and restart.")
        st.stop()

    df_k = pd.read_csv(DATA_PATH).query("country == 'Kenya'").copy()
    X, top_acts = engineer_features(df_k)
    y = df_k['repayment_interval'].apply(lambda x: 1 if x == 'monthly' else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mdl = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        early_stopping_rounds=50, tree_method='hist',
        eval_metric='logloss', random_state=42
    )
    mdl.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds  = mdl.predict(X_test)
    probas = mdl.predict_proba(X_test)[:, 1]
    report = classification_report(
        y_test, preds, target_names=['Declined', 'Approved'], output_dict=True
    )

    meta = {
        'X_full':         X,
        'X_test':         X_test,
        'y_test':         y_test,
        'top_activities': top_acts,
        'column_order':   X.columns.tolist(),
        'metrics': {
            'accuracy':   accuracy_score(y_test, preds),
            'f1':         f1_score(y_test, preds),
            'auc':        roc_auc_score(y_test, probas),
            'train_size': len(X_train),
            'test_size':  len(X_test),
            'report':     report,
        }
    }
    joblib.dump(mdl,  MODEL_PATH)
    joblib.dump(meta, META_PATH)
    return mdl, meta


# ── SHAP EXPLAINER — cached separately so it survives page reruns ─────────────
@st.cache_resource(show_spinner="🧠 Building SHAP explainer…")
def get_explainer(_model):
    """
    Cached independently from the model so a page rerun doesn't recompute it.
    The leading underscore tells Streamlit not to hash _model itself.
    """
    return shap.Explainer(_model)


# ── SHAP PLOT HELPERS ────────────────────────────────────────────────────────
# Sifa brand palette
_GREEN = (0.063, 0.725, 0.506)   # #10b981 — pushes toward Approved
_ROSE  = (0.957, 0.247, 0.369)   # #f43f5e — pushes toward Declined

_BASE_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.grid":          True,
    "grid.color":         "#e5e7eb",
    "grid.linewidth":     0.7,
    "grid.linestyle":     "--",
    "xtick.color":        "#6b7280",
    "ytick.color":        "#374151",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "figure.facecolor":   "#f9fafb",
    "axes.facecolor":     "#ffffff",
    "axes.edgecolor":     "#e5e7eb",
    "axes.labelcolor":    "#374151",
    "text.color":         "#111827",
}


def _apply_shap_colors():
    """Override SHAP's default red/blue with Sifa brand colors."""
    import shap.plots.colors as sc
    sc.red_rgb  = _GREEN   # positive SHAP → pushes toward Approved → green
    sc.blue_rgb = _ROSE    # negative SHAP → pushes toward Declined → rose


def plot_waterfall(sv_single, pred: int) -> plt.Figure:
    """
    Styled SHAP waterfall for a single loan.
    Green bars = features pushing toward Approved.
    Rose bars  = features pushing toward Declined.
    """
    _apply_shap_colors()
    decision_color = "#065f46" if pred == 1 else "#991b1b"
    decision_label = "Approved ✅" if pred == 1 else "Declined ❌"

    with plt.rc_context(_BASE_RC):
        fig = plt.figure(figsize=(11, 5.5))
        shap.plots.waterfall(sv_single, show=False, max_display=12)

        ax = plt.gca()
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#f9fafb")

        ax.set_title(
            f"Loan Decision: {decision_label}",
            fontsize=14, fontweight="bold",
            color=decision_color, pad=16, loc="left"
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='y', labelsize=10, colors="#374151")

        fig.tight_layout(pad=2.0)
    return fig


def plot_summary(sv_global, X_sample: pd.DataFrame) -> plt.Figure:
    """
    Styled SHAP bar summary (global feature importance).
    """
    _apply_shap_colors()

    with plt.rc_context(_BASE_RC):
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(
            sv_global, X_sample,
            plot_type="bar",
            show=False,
            max_display=12,
            color=_GREEN,
        )

        ax = plt.gca()
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#f9fafb")

        ax.set_title(
            "Global Feature Importance  (mean |SHAP value|)",
            fontsize=13, fontweight="bold", color="#111827", pad=14, loc="left"
        )
        ax.set_xlabel("Mean |SHAP value|", fontsize=10, color="#6b7280")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Value labels on bars
        for bar in ax.patches:
            w = bar.get_width()
            if w > 0:
                ax.text(
                    w + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", ha="left",
                    fontsize=9, color="#374151"
                )

        fig.tight_layout(pad=2.0)
    return fig


# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────
model, meta = load_sifa_stack()
explainer   = get_explainer(model)            # ← cached, not recomputed each run

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇰🇪 Sifa-Score AI")
    st.caption("Explainable Credit Risk · Kenya")
    st.divider()
    page = st.radio(
        "Navigate to",
        ["📊 Overview", "🔍 Analyze Loan", "📬 AI Advisor"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**⚙️ Model**")
    if st.button("🔄 Retrain from scratch", use_container_width=True):
        for p in [MODEL_PATH, META_PATH]:
            if os.path.exists(p):
                os.remove(p)
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.caption(
        "XGBoost · SHAP · LangChain · Ollama  \n"
        "SDG #1 — No Poverty"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Model Performance Overview")
    st.caption(
        "Global explainability and evaluation metrics across the held-out test set."
    )

    m = meta['metrics']
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",       f"{m['accuracy']:.2%}")
    c2.metric("F1-Score",       f"{m['f1']:.2%}")
    c3.metric("ROC-AUC",        f"{m['auc']:.3f}")
    c4.metric("Training Loans", f"{m['train_size']:,}")
    c5.metric("Test Loans",     f"{m['test_size']:,}")

    st.divider()
    col_shap, col_report = st.columns([1.6, 1])

    with col_shap:
        st.subheader("🌍 Global Feature Importance (SHAP)")
        st.caption("Mean |SHAP value| across 200 sampled test loans.")
        with st.spinner("Computing…"):
            X_s      = meta['X_test'].sample(min(200, len(meta['X_test'])), random_state=42)
            sv_global = explainer(X_s)
        fig = plot_summary(sv_global, X_s)
        st.pyplot(fig)
        plt.close()

    with col_report:
        st.subheader("📋 Classification Report")
        r = meta['metrics']['report']
        df_r = pd.DataFrame(r).T.loc[
            ['Declined', 'Approved'], ['precision', 'recall', 'f1-score']
        ]
        st.dataframe(df_r.style.format("{:.3f}"), use_container_width=True)

        st.divider()
        n_app = int((meta['y_test'] == 1).sum())
        n_dec = int((meta['y_test'] == 0).sum())
        st.subheader("🎯 Class Distribution")
        st.bar_chart(
            pd.DataFrame({'Count': [n_app, n_dec]},
                         index=['Approved', 'Declined'])
        )

    st.divider()
    st.info(
        "📌 **Target variable note:** The model predicts *monthly* repayment "
        "schedule as a proxy for structured, lower-risk loans — not actual default. "
        "Replace with a real default label (`status == 'defaulted'`) for production use."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · ANALYZE LOAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Loan":
    st.title("🔍 Analyze a Loan")

    tab_form, tab_index = st.tabs(["📝 Manual Input", "📂 From Test Set"])

    # ── Manual Input ──────────────────────────────────────────────────────────
    with tab_form:
        st.markdown("Score a brand-new loan application by filling in the fields below.")
        st.markdown("")
        ca, cb, cc = st.columns(3)

        with ca:
            loan_amount  = st.number_input("Loan Amount (KES)", 1_000, 500_000, 25_000, 1_000)
            term_months  = st.slider("Term (months)", 1, 60, 12)
            lender_count = st.number_input("Number of Lenders", 1, 500, 10)

        with cb:
            borrower_count = st.number_input("Borrowers in Group", 1, 50, 1)
            female_ratio   = st.slider("Female Ratio in Group", 0.0, 1.0, 0.5, 0.05)
            month = st.selectbox(
                "Month of Application", range(1, 13),
                format_func=lambda m: [
                    '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                ][m]
            )

        with cc:
            sector   = st.selectbox("Business Sector", SECTORS)
            top_acts = meta['top_activities']
            activity = st.selectbox(
                "Business Activity",
                ['Other'] + sorted([a for a in top_acts if a != 'Other'])
            )

        st.markdown("")
        if st.button("🔮 Score This Loan", type="primary",
                     use_container_width=True, key="btn_form"):

            row = pd.DataFrame([{
                'loan_amount':       loan_amount,
                'term_in_months':    term_months,
                'lender_count':      lender_count,
                'borrower_count':    borrower_count,
                'female_ratio':      female_ratio,
                'loan_per_borrower': loan_amount / max(borrower_count, 1),
                'month':             month,
                'day_of_week':       0,
                'sector':            sector,
                'activity':          activity if activity in top_acts else 'Other'
            }])
            enc = pd.get_dummies(row[FEATURE_COLS], drop_first=True)
            # Align columns to training schema — fill missing dummies with 0
            enc = enc.reindex(columns=meta['column_order'], fill_value=0)

            pred = int(model.predict(enc)[0])
            prob = float(model.predict_proba(enc)[0, 1])
            sv   = explainer(enc)

            st.session_state['analysis'] = {
                'pred': pred, 'prob': prob, 'sv': sv,
                'feature_names': meta['column_order']   # ← always from training schema
            }

    # ── Test Set Index ────────────────────────────────────────────────────────
    with tab_index:
        st.markdown("Pick a row from the held-out test set to inspect.")
        idx = st.number_input("Row index", 0, len(meta['X_test']) - 1, 0)

        if st.button("🔮 Score This Loan", type="primary",
                     use_container_width=True, key="btn_idx"):
            sample = meta['X_test'].iloc[idx:idx + 1]
            pred   = int(model.predict(sample)[0])
            prob   = float(model.predict_proba(sample)[0, 1])
            sv     = explainer(sample)

            st.session_state['analysis'] = {
                'pred': pred, 'prob': prob, 'sv': sv,
                'feature_names': meta['column_order']   # ← consistent key
            }

    # ── Results ───────────────────────────────────────────────────────────────
    if 'analysis' in st.session_state:
        a = st.session_state['analysis']
        st.divider()

        badge = (
            '<span class="badge-approved">✅ Approved</span>'
            if a['pred'] == 1
            else '<span class="badge-declined">❌ Declined</span>'
        )
        r1, r2, r3 = st.columns(3)
        r1.markdown(f"**Decision**<br>{badge}", unsafe_allow_html=True)
        r2.metric("Confidence",  f"{a['prob']:.1%}")
        r3.metric("Risk Level",  "Low" if a['pred'] == 1 else "High")

        st.markdown("#### Why did the model decide this?")
        st.caption(
            "Positive bars pushed the score toward Approved. "
            "Negative bars pushed it toward Declined."
        )
        fig = plot_waterfall(a['sv'][0], a['pred'])
        st.pyplot(fig)
        plt.close()

        st.info("💡 Head to **📬 AI Advisor** to generate a borrower letter.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · AI ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📬 AI Advisor":
    st.title("📬 AI Advisor")
    st.caption("Generate a personalised borrower letter via Llama 3, running locally.")

    if 'analysis' not in st.session_state:
        st.info("💡 Score a loan first in **🔍 Analyze Loan**, then return here.")
        st.stop()

    a        = st.session_state['analysis']
    decision = "Approved" if a['pred'] == 1 else "Declined"

    # Build SHAP evidence using the consistent feature_names key
    shap_df = pd.DataFrame({
        'feature':      a['feature_names'],          # ← from training schema
        'contribution': a['sv'].values[0]
    })

    if a['pred'] == 1:
        top_factors = shap_df.nlargest(3, 'contribution')
        tone        = "warm, celebratory, and professional"
    else:
        top_factors = shap_df.nsmallest(3, 'contribution')
        tone        = "empathetic, advisory, and constructive"

    evidence = "\n".join([
        f"- {row['feature'].replace('_', ' ').title()}: "
        f"SHAP impact {row['contribution']:+.4f}"
        for _, row in top_factors.iterrows()
    ])

    # Preview
    badge = (
        '<span class="badge-approved">✅ Approved</span>'
        if a['pred'] == 1
        else '<span class="badge-declined">❌ Declined</span>'
    )
    st.markdown(f"**Decision:** {badge}", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**Key SHAP factors:**")
    for _, row in top_factors.iterrows():
        icon  = "⬆️" if row['contribution'] > 0 else "⬇️"
        label = row['feature'].replace('_', ' ').title()
        st.write(f"{icon} **{label}** — `{row['contribution']:+.4f}`")

    st.divider()

    TEMPLATE = """
You are a Senior Loan Officer at Sifa-Score in Nairobi, Kenya.
The AI model has {status} this loan application.

SHAP Evidence:
{evidence}

Instructions:
- Write a {tone} letter to the borrower.
- Weave the SHAP factors naturally into the narrative — do not just list them.
- For approvals: explain the strengths and outline next steps.
- For declines: give one concrete, actionable improvement tip per risk factor,
  grounded in Kenyan context (e.g. M-Pesa repayment cycles, chama group savings,
  agricultural seasons, market days).
- Do NOT use placeholder text like "[Name]" — write as a real letter.
- Length: 200–280 words. Professional but warm.
- End with: Sifa-Score Team, Nairobi.
"""

    if st.button("✉️ Generate Advisory Letter", type="primary", use_container_width=True):
        with st.spinner("Llama 3 is composing your letter locally…"):
            try:
                llm    = ChatOllama(model="llama3", temperature=0.3)
                prompt = ChatPromptTemplate.from_template(TEMPLATE)
                letter = (prompt | llm).invoke({
                    "status":   decision,
                    "evidence": evidence,
                    "tone":     tone,
                }).content

                st.markdown(
                    f'<div class="advisory-box">{letter}</div>',
                    unsafe_allow_html=True
                )
                st.markdown("")
                st.download_button(
                    "⬇️ Download Letter (.txt)",
                    data=letter,
                    file_name=f"sifa_{decision.lower()}_letter.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Ollama connection failed: {e}")
                st.warning(
                    "Make sure Ollama is running:  \n"
                    "`ollama serve` then `ollama pull llama3`"
                )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🇰🇪 Sifa-Score AI v2.0 · XGBoost · SHAP · LangChain · Ollama · "
    "SDG #1 No Poverty"
)