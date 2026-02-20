# 🇰🇪 Sifa-Score AI
### Explainable Credit Risk for Kenyan Entrepreneurs

> *"No" is not an answer. Every credit decision deserves a reason.*

Sifa-Score is a full-stack, explainable AI (XAI) system for predicting loan repayment risk among Kenyan entrepreneurs. Built on XGBoost and SHAP, it provides a mathematical receipt for every decision — then uses a local Llama 3 agent to translate that into plain, actionable language for the borrower.

Inspired by **UN SDG #1 — No Poverty**.

---

## ✨ What's New in v2.0

| Area | Improvement |
|---|---|
| Dashboard | Rebuilt as a 3-page app — Overview · Analyze Loan · AI Advisor |
| Scoring | Manual loan input form — score any new application directly |
| Pipeline | Model persists to disk via `joblib` — no retraining on every load |
| Evaluation | Added F1-Score, ROC-AUC, and Classification Report alongside accuracy |
| Explainability | Global SHAP summary on Overview page + fully restyled waterfall plots |
| AI Advisor | Richer Kenyan context in prompts, no placeholder bugs, `.txt` download |

See the [v2.0.0 release notes](../../releases/tag/v2.0.0) for the full changelog.

---

## 🧠 Core Tenets

**Privacy First**
The entire stack — XGBoost model, SHAP explainer, and Llama 3 LLM — runs locally via Ollama. Borrower data never leaves your machine.

**Explainability as a Right**
Every decision comes with a SHAP waterfall plot: a feature-by-feature breakdown of exactly what drove the score, in green (toward approval) and rose (toward decline).

**Contextual Fairness**
The model looks beyond raw financials. It captures social dynamics (`female_ratio` in borrower groups), economic density (`loan_per_borrower`), and seasonal cycles (`month`, `day_of_week`) to reflect the reality of Kenya's economy.

---

## 📊 Performance

| Metric | Score |
|---|---|
| Accuracy | 89.74% |
| F1-Score | — |
| ROC-AUC | — |
| Model | XGBoost (Early Stopping, Stratified Split) |
| Test Set | 20% held-out, stratified |

> **Note on the target variable:** The model currently predicts *monthly* repayment schedule as a proxy for structured, lower-risk lending — not an actual default label. For production use, replace with a `status == 'defaulted'` column from your platform's repayment data.

---

## 🗂️ Project Structure

```
sifa-score-ai/
├── sifa_dashboard.py      # Streamlit app (3-page)
├── sifa_score.ipynb       # Model training + SHAP notebook
├── requirements.txt       # Python dependencies
├── kiva_loans.csv         # Dataset (not tracked — add locally)
├── sifa_model.joblib      # Persisted model (generated on first run)
└── sifa_meta.joblib       # Persisted metadata (generated on first run)
```

---

## 🚀 How to Run

**1. Install Ollama and pull Llama 3**
```bash
ollama serve
ollama pull llama3
```

**2. Add the dataset**

Download `kiva_loans.csv` from [Kiva on Kaggle](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) and place it in the project root.

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the dashboard**
```bash
streamlit run sifa_dashboard.py
```

The model trains and saves to disk on first run. Subsequent launches load instantly.

---

## 🖥️ Dashboard Pages

**📊 Overview**
Model performance metrics (Accuracy, F1, ROC-AUC), a global SHAP feature importance chart, and the classification report across the held-out test set.

**🔍 Analyze Loan**
Two modes: score a new application via a manual input form, or inspect any row from the test set. Outputs a styled SHAP waterfall showing exactly what drove the decision.

**📬 AI Advisor**
Generates a personalised borrower letter via Llama 3 — celebratory for approvals, empathetic and actionable for declines — grounded in Kenyan economic context (M-Pesa cycles, chama savings, agricultural seasons). Downloadable as `.txt`.

---

## 🛠️ Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost |
| Explainability | SHAP |
| AI Agent | LangChain + Ollama (Llama 3) |
| UI | Streamlit |
| Persistence | joblib |
| Language | Python 3.10+ |

---

## 🌍 SDG Alignment

This project was built around **UN Sustainable Development Goal #1 — No Poverty**. By making credit systems transparent and accessible, Sifa-Score aims to empower small business owners in Kenya to understand and navigate the financial landscape on their own terms.

---
**Images**
<img width="1633" height="657" alt="image" src="https://github.com/user-attachments/assets/2a411052-bd75-42db-a119-87f9fe95798c" />
<img width="1907" height="894" alt="image" src="https://github.com/user-attachments/assets/79ff8b96-9460-4322-81d3-f55faf8b9e0d" />



## 👤 Author

**Derick Gikonyo** — February 2026

---

*Built with purpose. Powered by open-source AI.*
