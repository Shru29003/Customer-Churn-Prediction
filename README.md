Customer Churn Prediction

Intern: Shruti Mandvakar | Institute: TechnoSignia Software Training Institute | Project: Project 2, Customer Churn Prediction (Days 38-50)

1. Project Overview

An end-to-end machine learning project that predicts which telecom customers are likely to leave (churn). It covers data cleaning, exploratory analysis, statistical analysis, feature engineering, preprocessing, train-test splitting, three classification models (Logistic Regression, Decision Tree, Random Forest), evaluation, hyperparameter tuning, feature importance and customer risk scoring.

2. Business Problem

Winning a new customer costs more than keeping an existing one. About 18% of customers in this dataset churn, and the company cannot contact everyone with a retention offer. It needs to know which customers are most at risk, why, and how reliable that prediction is, so that offers go to the right people.

3. Objectives
Clean and understand the customer data.
Find the factors associated with churn.
Build and compare three models on a held-out test set.
Select a final model and explain its predictions.
Produce a ranked list of at-risk customers that supports retention decisions.
4. Dataset Description
Item	Value
File	Customer_Churn_Prediction_8000_Records.csv (raw), cleaned_customer_churn.csv (processed)
Size	8,000 customers, 27 columns
Target	Churn (Yes/No): 1,440 churned (18.0%), 6,560 retained (82.0%)
Identifier	CustomerID (unique; not used as a feature)
Customer profile	Gender, SeniorCitizen, Partner, Dependents, TenureMonths
Services	PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Account	ContractType, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
Behaviour	SupportCallsLast90Days, ComplaintRaised, AvgMonthlyDataUsageGB, AvgMonthlyCalls, SatisfactionScore, DaysSinceLastContact

The raw data is not included in this repository (customer data must not be uploaded). Place your copy at data/raw/customer_churn.csv.

5. Tools and Technologies

Python 3, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, GridSearchCV, RandomizedSearchCV), SciPy, Matplotlib, Seaborn, joblib, Jupyter Notebook, Git/GitHub.

6. Data Cleaning Process
Missing values: 464 in four columns, all under 3% of rows. OnlineBackup (62) was filled with the mode ("No"); AvgMonthlyDataUsageGB (101, median 37.8), SatisfactionScore (202, median 3.0) and DaysSinceLastContact (99, median 19.0) with medians. 0 missing values remain.
Duplicates: none (rows and CustomerID).
Consistency: whitespace stripped; category labels checked. "No phone service" / "No internet service" are valid structural categories, not errors.
Outliers: checked with the IQR rule and kept. The flagged values (heavy data users, long tenure, many support calls) are plausible customer behaviour.
Result: 8,000 rows × 27 columns, saved as cleaned_customer_churn.csv.
7. Exploratory Data Analysis
Segment	Churn rate
Month-to-month contract	27.8%
One-year contract	7.4%
Two-year contract	3.7%
Complaint raised	49.4% (vs 15.8% with no complaint)
New customers (≤ 12 months)	27.1%
Long-term (> 36 months)	11.3%
Electronic-check payers	22.1% (vs 15.0% for bank transfer)
Fiber optic internet	21.9%

Key observations: month-to-month customers are 56% of the base but 86% of churners; complaints are a strong warning sign; the first year is the riskiest period. Satisfaction score has the strongest numeric correlation with churn (-0.29); tenure and total charges are highly correlated with each other (0.91). These are associations, not proven causes, and no formal hypothesis tests were run.

8. Feature Engineering

New columns: tenure bands (Tenure_Group: New / Medium / Long-term), Spending_Category (Low/Medium/High by median and 75th percentile of monthly charges), Usage_Category and Call_Usage_Category (Low/High around the median), and Engagement_Score (0 to 2, number of usage measures at or above the median). Thresholds are fixed constants in src/preprocessing.py so new customers are bucketed exactly like training customers. CustomerID, Churn and any column derived from the target are removed before modelling.

9. Machine Learning Workflow
Separate features and target (Churn: No = 0, Yes = 1).
Split 80/20, stratified, random_state=42 (6,400 train / 1,600 test; churn 18.0% in both).
Preprocess inside a scikit-learn pipeline, fitted on training data only: median imputation + StandardScaler for numeric columns; most-frequent imputation + one-hot encoding for categorical columns (about 69 columns after encoding).
Train Logistic Regression (baseline), Decision Tree and Random Forest.
Tune with 5-fold stratified cross-validation on the training set (F1 scoring).
Evaluate on the untouched test set; interpret with feature importance.
10. Model Comparison

Test-set results (1,600 customers, 288 churners):

Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.8606	0.7372	0.3507	0.4753	0.8569
Decision Tree (tuned)	0.8256	0.5260	0.3160	0.3948	0.7314
Random Forest (tuned)	0.8338	0.5377	0.5451	0.5414	0.8242
Random Forest (final configuration)	0.8338	0.5390	0.5278	0.5333	0.8173

Full table in reports/Model_Comparison.csv. See Limitations for how these figures were obtained.

11. Evaluation Metrics

Accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC. Accuracy alone is misleading here: predicting "stay" for everyone gives 82% accuracy and finds no churners. Recall (churners found) and F1 matter most for retention; precision shows how many offers would be wasted.

12. Final Model

Random Forest: 100 trees, max_depth=20, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', inside one pipeline with the preprocessing (models/final_model.joblib).

Why: it finds far more churners than the alternatives (recall 0.53 vs 0.35 for Logistic Regression) with the best F1, and its ROC-AUC (0.82) is close to Logistic Regression's (0.86). Under illustrative costs (a missed churner 500, an unnecessary offer 50) it has the lowest total error cost. Logistic Regression remains the better choice if precision and interpretability matter more.

Final model, confusion matrix on the test set: TN 1,182, FP 130, FN 136, TP 152.

13. Important Features
Rank	Feature	Importance
1	Contract: month-to-month	0.094
2	SatisfactionScore	0.081
3	TotalCharges	0.061
4	MonthlyCharges	0.054
5	TenureMonths	0.054
6	AvgMonthlyCalls	0.047
7	DaysSinceLastContact	0.047

Logistic Regression agrees: month-to-month contracts (+1.25), complaints (+0.62) and support calls (+0.37) raise churn odds; two-year contracts (-0.99), satisfaction (-0.70) and tenure (-0.57) lower them. These match the EDA findings.

14. Results and Business Interpretation

The model produces a churn probability per customer. Using cut-offs of 0.30 and 0.60, the 1,600 test customers fall into Low risk 923 (57.7%), Medium 501 (31.3%) and High 176 (11.0%). A retention team could contact the High group first (for example CUST007307, probability 0.905), then Medium as budget allows, focusing on month-to-month customers, those who have complained, and low-satisfaction customers. The model supports prioritising; it does not say why an individual leaves or whether an offer will work.

15. Limitations
Target leakage was found and fixed. A 0/1 copy of the target (Churn_Binary) stayed in the feature set from Day 44 to Day 48, giving perfect (1.0) scores in those live notebook runs. It was caught on Day 49 and removed. Only the final Random Forest figures come from a live leak-free run.
Some leak-free numbers are hard-coded. The Logistic Regression and tuned Decision Tree / Random Forest figures in the comparison table are recorded values from the notebook, not live output. Re-run the notebooks after the fix to confirm them.
The Day 48 search was affected. The tree and forest searches ran with the leak present, and the Logistic Regression search returned NaN (F1 scorer without pos_label), so the reported best hyperparameters are not reliable. The final forest used hand-set parameters.
Test data influenced model choice. The three models were compared on the test set to pick the final model. A stricter approach selects on cross-validation and uses the test set once.
Risk bands are not validated. The 0.30 / 0.60 cut-offs were not justified with validation data, and class_weight='balanced' inflates probabilities (a 0.60 score is not a 60% chance).
Feature importance and coefficients show association, not causation; correlated features (tenure, total charges, usage measures) share importance.
Recall is about 53%: roughly half of churners are still missed, and about half of flagged customers do not churn.
One dataset, one split; performance on future customers may differ. Medians used in cleaning and feature thresholds were computed on the full data.
16. How to Run the Project
bash
git clone <your-repository-url>
cd Customer_Churn_Prediction
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. put the raw data at data/raw/customer_churn.csv, then:
python src/clean_data.py --input data/raw/customer_churn.csv --output data/processed/cleaned_customer_churn.csv

# 2. train, evaluate and save the final model (writes models/ and reports/ files)
python src/train_final_model.py

# 3. check the model and score customers
python src/predict_churn.py --validate
python src/predict_churn.py --input data/processed/cleaned_customer_churn.csv --output reports/new_predictions.csv

Run the notebooks in notebooks/ in numeric order (01 to 13) with jupyter notebook.

Validation (Day 50 Task 2). --validate checks that the model loads, the pipeline scores records, probabilities lie between 0 and 1, and the saved test metrics are reproduced. The last rule (no test-set use in model selection) is a process rule; see Limitations.

Note: the scripts were smoke-tested on synthetic data with the same columns, not on the original dataset. After running train_final_model.py on the real data, the printed metrics (also saved in reports/final_model_metrics.csv) may differ slightly from the figures above, because the script drops the redundant duplicate columns (TenureGroup, Average_Usage) used in the notebook. Update the tables in this README with the values it prints.

17. Future Scope
Re-run Days 44-48 with the leak removed and select the model by cross-validation.
Tune the decision threshold using the real cost of a retention offer versus a lost customer; validate risk bands against observed churn.
Try gradient boosting (XGBoost / LightGBM) and probability calibration.
Add permutation importance and SHAP for individual explanations.
Add formal hypothesis tests (t-test, chi-square) and survival analysis on tenure.
Deploy as a simple API or dashboard and monitor performance on new customers.
18. Day-wise Task Details (Days 38-50)

Summary of each daily task: deliverables, key results and notes.

Leakage note (applies to Days 44-48): from Day 44 to the Day 48 searches, a 0/1 copy of the target (Churn_Binary) was left in the model inputs, so live notebook runs of those cells show 1.0 on every metric. It was created on Day 40 to compute correlations, found on Day 49 (importance 0.668) and removed. The Logistic Regression, Decision Tree and Random Forest figures in this README are the leak-free values recorded in the notebook (typed in, not live output); only the final Day 49 Random Forest was re-run live on clean inputs. Re-run Days 44-48 to confirm.

Day	Task	Deliverables
Day 38	Data Cleaning	02_Data_Cleaning.ipynb, cleaned_customer_churn.csv, Data_Cleaning_Report.docx
Day 39	Exploratory Data Analysis	03_Churn_EDA.ipynb, EDA_Report.docx, (chart images saved from the notebook)
Day 40	Statistical Analysis	04_Statistical_Analysis.ipynb, Statistical_Analysis_Report.docx
Day 41	Feature Engineering	05_Feature_Engineering.ipynb, Feature_Engineering_Report.docx
Day 42	Encoding and Scaling	06_Preprocessing.ipynb, preprocessing.py, preprocessing_documentation.docx
Day 43	Train-Test Split	07_Train_Test_Split.ipynb, split_summary.csv, Data_Splitting_Notes.docx
Day 44	Logistic Regression (baseline)	08_Logistic_Regression.ipynb, baseline_metrics.csv, Baseline_Model_Analysis.docx
Day 45	Decision Tree	09_Decision_Tree.ipynb, model_comparison_day45.csv, Decision_Tree_Analysis.docx
Day 46	Random Forest	10_Random_Forest.ipynb, model_comparison.csv, Random_Forest_Analysis.docx
Day 47	Model Evaluation	11_Model_Evaluation.ipynb, confusion_matrices/, model_evaluation.csv, Model_Evaluation_Report.docx
Day 48	Hyperparameter Tuning	12_Hyperparameter_Tuning.ipynb, tuning_results.csv, Before_After_Comparison.docx
Day 49	Model Interpretation and Feature Importance	13_Feature_Importance.ipynb, feature_importance.png, customer_churn_predictions.csv, Model_Interpretation_Report.docx
Day 38 - Data Cleaning

Objective: Inspect the raw churn data, treat missing values, duplicates and outliers, and save a clean file.

Deliverables: 02_Data_Cleaning.ipynb, cleaned_customer_churn.csv, Data_Cleaning_Report.docx

Results:

Raw data: 8,000 rows x 27 columns; 0 duplicate rows or CustomerIDs.
Missing values: 464 in four columns, all under 3% (OnlineBackup 62, AvgMonthlyDataUsageGB 101, SatisfactionScore 202, DaysSinceLastContact 99).
Treatment: OnlineBackup filled with the mode ("No"); the three numeric columns with medians (37.8, 3.0, 19.0). 0 missing values remain.
Whitespace stripped; category labels checked ("No phone service" / "No internet service" are valid structural categories).
Outliers checked with the IQR rule and kept (plausible customer behaviour, not errors).

Notes:

About 7 of the 62 missing OnlineBackup rows probably belong to no-internet customers but were filled with "No".
Fill values were computed on the full dataset before the split (small effect).
The imputation cell runs twice (cells 16 and 21); remove the duplicate.
Day 39 - Exploratory Data Analysis

Objective: Explore how churn relates to tenure, contract, charges, payment method and complaints.

Deliverables: 03_Churn_EDA.ipynb, EDA_Report.docx, (chart images saved from the notebook)

Results:

Overall churn: 1,440 of 8,000 (18.0%).
Contract: month-to-month 27.8%, one-year 7.4%, two-year 3.7%. Month-to-month customers are 56% of the base but 86% of churners.
Complaints: 49.4% churn with a complaint vs 15.8% without.
Tenure: New (<= 12 months) 27.1%, Medium 18.3%, Long-term 11.3%.
Payment: electronic check 22.1% vs bank transfer 15.0%.
Churners pay slightly more per month (mean 88.09 vs 84.07).

Notes:

These are associations, not causes; segments overlap.
No significance tests were run in this section.
The tenure chart plotted TenureMonths on the y-axis; plot churn rate instead.
Day 40 - Statistical Analysis

Objective: Describe the numeric variables, compare churners with non-churners, and measure correlations.

Deliverables: 04_Statistical_Analysis.ipynb, Statistical_Analysis_Report.docx

Results:

Skewness: TenureMonths +0.76, MonthlyCharges -1.22, TotalCharges +0.96.
Retained vs churned (means): tenure 30.2 vs 23.3, monthly charges 84.07 vs 88.09, total charges 2,542 vs 2,053, satisfaction 3.56 vs 2.85.
Correlation with churn: SatisfactionScore -0.287, SupportCalls +0.160, Tenure -0.145, TotalCharges -0.109, MonthlyCharges +0.075; calls and days-since-contact about 0.
Tenure and TotalCharges correlate at 0.912 (redundant).
Fiber-optic customers churn 21.9%; senior citizens 21.3% vs 17.4%; gender differences under 2 points.

Notes:

scipy.stats was imported but no hypothesis tests were run; the findings are descriptive.
Churn_Binary was created here (cell 61) just to compute correlations and was never dropped. It later leaked into the models. Compute correlations on a temporary copy instead.
Day 41 - Feature Engineering

Objective: Create features that express customer behaviour more usefully for models.

Deliverables: 05_Feature_Engineering.ipynb, Feature_Engineering_Report.docx

Results:

Tenure_Group: New (<= 12) 1,603, Medium (13-36) 4,032, Long-term 2,365.
Spending_Category (monthly-charge bands), Usage_Category and Call_Usage_Category (around the median), Engagement_Score (0 to 2).
Churn rises with spending (fixed-cut-off version): Low 11.9%, Medium 18.4%, High 21.2%.

Notes:

Remove Churn_Binary (target copy) before modelling.
TenureGroup and Tenure_Group are duplicates; Average_Usage copies AvgMonthlyDataUsageGB.
Spending_Category was created twice; re-count the final column before quoting numbers.
Thresholds were computed on the full data; use training data or fixed business values.
Day 42 - Encoding and Scaling

Objective: Convert the cleaned data into model-ready numeric input with a reusable pipeline.

Deliverables: 06_Preprocessing.ipynb, preprocessing.py, preprocessing_documentation.docx

Results:

Target: Churn mapped No = 0, Yes = 1. CustomerID dropped.
32 input features (11 numeric, 21 categorical) -> 69 columns after encoding.
Pipeline: median imputation + StandardScaler (numeric); most-frequent imputation + OneHotEncoder(handle_unknown="ignore") (categorical); combined with ColumnTransformer and fitted on training data only.
Trees do not need scaling; Logistic Regression does.

Notes:

An early run left CustomerID in X (about 6,458 one-hot columns); fixed by dropping it.
Churn_Binary must also be dropped from X (see the Day 40 and Day 44 notes).
Day 43 - Train-Test Split

Objective: Hold out unseen customers for evaluation; compare class balance.

Deliverables: 07_Train_Test_Split.ipynb, split_summary.csv, Data_Splitting_Notes.docx

Results:

80/20 split, stratified on Churn, random_state = 42.
Original 8,000 (1,440 churners, 18.0%); training 6,400 (1,152, 18.0%); test 1,600 (288, 18.0%).
Stratification kept the churn rate identical in all three sets.
Leakage rules: split first, fit scaler/encoder on training data only, never choose features on the test set.

Notes:

The split was correct, but the features still contained the target copy Churn_Binary (see Day 44).
Day 44 - Logistic Regression (baseline)

Objective: Train the baseline classifier and record its metrics.

Deliverables: 08_Logistic_Regression.ipynb, baseline_metrics.csv, Baseline_Model_Analysis.docx

Results:

Leak-free baseline (test set, 1,600 customers, 288 churners):

Accuracy	Precision	Recall	F1	ROC-AUC
0.8606	0.7372	0.3507	0.4753	0.8567

Confusion matrix (derived from the metrics): TN 1,276, FP 36, FN 187, TP 101. The model finds only 35% of churners, so accuracy (86%, versus 82% for always predicting "stay") overstates it.

Notes:

See the leakage note at the top of this section.

Day 45 - Decision Tree

Objective: Train a Decision Tree and compare it with Logistic Regression.

Deliverables: 09_Decision_Tree.ipynb, model_comparison_day45.csv, Decision_Tree_Analysis.docx

Results:

Concepts covered: root / decision / leaf nodes, Gini impurity, entropy, overfitting.
Untuned tree ROC-AUC about 0.60 (from the ROC plot legend) vs 0.857 for Logistic Regression: a classic overfitting gap. Other default-tree metrics were not recorded.
Tuned tree (Day 48): accuracy 0.8256, precision 0.5260, recall 0.3160, F1 0.3948, AUC 0.7314.

Notes:

See the leakage note at the top of this section.

The saved model_comparison_day45.csv compares leak-free Logistic Regression numbers against a leaked tree, so it is not a fair comparison.

Day 46 - Random Forest

Objective: Train a Random Forest and compare it with the earlier models.

Deliverables: 10_Random_Forest.ipynb, model_comparison.csv, Random_Forest_Analysis.docx

Results:

Ensemble of trees on bootstrap samples with random feature subsets; class_weight="balanced" used.
Untuned forest ROC-AUC about 0.825 (ROC plot legend); other default metrics not recorded.
Most churners found: Random Forest. Fewest false alarms and best AUC: Logistic Regression. Easiest to interpret: Logistic Regression.

Notes:

See the leakage note at the top of this section.

In the live run all three models scored 1.0 with identical confusion matrices, which is a warning sign, not a result.

Day 47 - Model Evaluation

Objective: Evaluate the three models with accuracy, precision, recall, F1, confusion matrix and ROC-AUC.

Deliverables: 11_Model_Evaluation.ipynb, confusion_matrices/, model_evaluation.csv, Model_Evaluation_Report.docx

Results:

Metric	Logistic Regression	Decision Tree (tuned)	Random Forest (tuned)
Accuracy	0.8606	0.8256	0.8338
Precision	0.7372	0.5260	0.5377
Recall	0.3507	0.3160	0.5451
F1	0.4753	0.3948	0.5414
ROC-AUC	0.8569	0.7314	0.8242

Confusion matrices (TN/FP/FN/TP, derived): LR 1,276/36/187/101; DT 1,230/82/197/91; RF 1,177/135/131/157. Under assumed costs (missed churner 500, unnecessary offer 50; illustration only) total error cost is lowest for the Random Forest.

Notes:

See the leakage note at the top of this section.

Replace the assumed costs with the company's real figures.

Day 48 - Hyperparameter Tuning

Objective: Tune each model with cross-validation and compare before and after.

Deliverables: 12_Hyperparameter_Tuning.ipynb, tuning_results.csv, Before_After_Comparison.docx

Results:

5-fold StratifiedKFold on the training data, F1 scoring, preprocessing inside the pipeline.
Logistic Regression: GridSearchCV over C. Decision Tree: GridSearchCV (120 fits). Random Forest: RandomizedSearchCV (10 candidates, 50 fits).
Test results after tuning: LR F1 0.4753 (AUC 0.8569, no real change); Decision Tree F1 0.3948 (AUC 0.60 -> 0.73); Random Forest F1 0.5414 (AUC 0.8253 -> 0.8242).
Tuning improved the tree but not every metric for every model.

Notes:

See the leakage note at the top of this section.

The tree and forest searches reported CV F1 = 1.0 (a leak symptom), so their best parameters are unreliable. The Logistic Regression search returned NaN for every C (F1 scorer without pos_label on Yes/No labels), so C = 0.01 is not a tuned choice. Fix the scorer and re-run.

Day 49 - Model Interpretation and Feature Importance

Objective: Identify the features behind the predictions and the customers the model considers high risk.

Deliverables: 13_Feature_Importance.ipynb, feature_importance.png, customer_churn_predictions.csv, Model_Interpretation_Report.docx

Results:

The leak was found here: Churn_Binary had importance 0.668. It was removed and the forest re-fitted.
Final Random Forest (live, leak-free): accuracy 0.8338, precision 0.5390, recall 0.5278, F1 0.5333, ROC-AUC 0.8173.
Top features: month-to-month contract (0.094), SatisfactionScore (0.081), TotalCharges (0.061), MonthlyCharges (0.054), TenureMonths (0.054).
Logistic Regression coefficients agree: month-to-month +1.25, complaint +0.62, two-year contract -0.99, satisfaction -0.70.
Risk bands (<0.30 / 0.30-0.60 / >=0.60): Low 923, Medium 501, High 176 of 1,600 test customers. Highest risk: CUST007307 (0.905).

Notes:

The 0.30 / 0.60 cut-offs were not validated; class_weight="balanced" inflates probabilities.
Importance and coefficients show association, not causation; correlated features share credit.

