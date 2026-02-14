# Credit Card Default Prediction: A Behavioral Time-Series Approach

📌 Project Overview
This project focuses on identifying early warning signals of financial distress to improve the underwriting and pricing of Payment Protection Insurance (PPI). Instead of treating credit snapshots in isolation, this model analyzes 13 months of transaction history to detect trends in consumer behavior.

🚀 Key Results
Predictive Power: Achieved a 94.8% AUC using Gradient Boosting (XGBoost/LightGBM).

Early Intervention: Identified a 6-month early warning window where defaults can be predicted with high reliability.

Business Impact: Implementing a proactive intervention strategy for high-risk customers shows a potential 650% return on investment by preventing claims.

🛠️ Technical Implementation
I developed dynamic features that capture the "trajectory" of a customer's financial health:

Behavioral Trajectories: 3-month rolling averages of payment amounts and credit utilization growth rates.

Consistency Metrics: Standard deviation of payment timing to detect irregular financial habits.

Model Comparison: Evaluated Logistic Regression, Random Forests, and Gradient Boosting to find the optimal trade-off between lead time and accuracy.

📂 Repository Structure
AmexBehavioralAnalysis.ipynb: The core data science workflow, including feature engineering and model evaluation.

Credit Card Default Prediction.pdf: The full research paper detailing the methodology and insurance applications.

requirements.txt: List of Python libraries (Pandas, Scikit-Learn, XGBoost) required to run the analysis.

📊 Data Source
The analysis uses the American Express - Default Prediction dataset from Kaggle, covering ~450,000 customers. Due to the large file size, the raw data is not included here but can be accessed here.
