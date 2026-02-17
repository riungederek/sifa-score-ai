🇰🇪 Sifa-Score: Kenya Loan Risk & Explainability
Sifa-Score is an end-to-end ML pipeline designed to predict loan repayment risk for Kenyan entrepreneurs. While most credit models act as "black boxes," this project focuses on Explainable AI (XAI)—using SHAP to decode the math and Llama 3 to translate those stats into clear, actionable advice for the borrower.

 Core Tenets
Privacy First: The entire stack—from the XGBoost model to the Llama 3 LLM—runs locally via Ollama. Borrower data never leaves your machine, ensuring complete financial privacy.

Explainability as a Right: We believe "No" isn't an answer. By integrating SHAP, we provide a mathematical "receipt" for every decision, showing the exact features that influenced the score.

Contextual Fairness: The model doesn't just look at debt; it looks at Social Dynamics (female ratio in groups), Economic Density (loan-per-borrower), and Seasonal Cycles (month/day) to capture the reality of the Kenyan economy.

 Performance & Tech
Accuracy: 89%

Engine: XGBoost with Stratified Splitting to ensure balanced risk distribution.

Feature Engineering: Custom features like female_ratio, loan_per_borrower, and seasonal_indices to improve predictive power.

AI Agent: A dynamic LangChain agent that automatically switches between "Approval" and "Decline" tones based on SHAP evidence.

 Stack
ML: Python, XGBoost, SHAP

AI: LangChain, Ollama (Llama 3)

UI: Streamlit

 How to run it
Model Setup: Install Ollama and run ollama pull llama3.

Dependencies:

Bash
pip install -r requirements.txt
Launch:

Bash
streamlit run sifa_dashboard.py

🌍SDG Inspiration
This project was inspired by UN Sustainable Development Goal #1 (No Poverty). By making credit systems more transparent and accessible, we aim to empower small business owners in the developing world to better understand and navigate the financial landscape.