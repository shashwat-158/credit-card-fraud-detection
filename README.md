# Credit Card Fraud Detection

An end-to-end, production-grade machine learning project to detect fraudulent credit card transactions. This project emphasizes rigorous, leakage-free validation and business-driven financial metrics over standard statistical accuracy.

## Project Highlights
* **Out-of-Time (OOT) Validation:** Chronological data splitting (train on the past, validate/test on the future) to strictly prevent data leakage and mirror real-world deployment.
* **Business-Cost Optimization:** Custom decision thresholds tuned to real transaction amounts to minimize actual dollar loss and manage alert budgets.
* **Imbalance Ablation:** Comparative analysis of class weights, SMOTE, and undersampling on a highly skewed dataset (0.17% fraud rate).
* **Advanced Evaluation:** Robust scoring using PR-AUC, Matthews Correlation Coefficient (MCC), and bootstrap confidence intervals.
* **Explainability & Calibration:** SHAP (SHapley Additive exPlanations) for transparent transaction flagging and probability calibration for reliable risk scoring.

## Project Structure
- `app/`: Contains the Streamlit web application files.
- `data/`: Store datasets here (ignored by git).
- `images/`: Visualizations, SHAP plots, and EDA charts.
- `model/`: Saved trained models and calibration pipelines (e.g., `.joblib`).
- `notebook/`: Jupyter notebook detailing the full EDA, feature engineering, ablation study, and model training workflow.

## Setup Instructions

1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage
*Instructions on how to run the Streamlit app will go here.*