# Bank Marketing Classification — Five ML Models

## 1. Problem statement
Predict whether a bank customer subscribes to a term deposit (`y = yes/no`) after a direct-marketing campaign. This is a binary classification task.

## 2. Dataset description
- **Source:** UCI Machine Learning Repository, Bank Marketing (dataset ID 222).
- **Selected file:** `bank-full.csv`.
- **Size:** 45,211 instances and 16 input features plus the target column `y` (17 columns total).
- **Target:** `y` (`yes` or `no`).
- **License:** CC BY 4.0. Cite: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- Download the official ZIP from: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- Extract `bank-full.csv` and put it in the project root, beside `train_models.py`.

## 3. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (LinearSVC)

The same stratified 80/20 train-test split and the same input dataset are used for all models. Categorical variables are one-hot encoded; numeric values are imputed and standardized. `test_data.csv` is held out from training and includes the true target so the app can calculate evaluation metrics.

## 4. Evaluation metrics
- **Accuracy:** proportion of all predictions that are correct.
- **AUC:** area under the ROC curve, computed from probabilities or decision scores.
- **Precision:** proportion of predicted positives that are actually positive.
- **Recall:** proportion of actual positives correctly identified.
- **F1:** harmonic mean of precision and recall.
- **MCC:** Matthews correlation coefficient, a balanced measure based on TP, TN, FP, and FN.

## 5. Model comparison table
Run `python train_models.py` to generate actual results in `training_metrics.csv`. Copy those values into the report/PDF. Do not enter estimated values.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8457 | 0.9079 | 0.4182 | 0.8147 | 0.5527 | 0.5092 |
| Decision Tree | 0.8474 | 0.8407 | 0.4196 | 0.7940 | 0.5490 | 0.5020 |
| KNN | 0.8996 | 0.8513 | 0.6298 | 0.3440 | 0.4450 | 0.4169 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Support Vector Machine | 0.8505 | 0.9077 | 0.4266 | 0.8072 | 0.5582 | 0.5136 |

## 6. Observations on model performance
Complete this section after reviewing `training_metrics.csv`:
- **Logistic Regression:** Logistic Regression provides strong class discrimination and good recall, making it a useful baseline for this dataset.
- **Decision Tree:** The Decision Tree performs reasonably well, but its ability to model nonlinear patterns does not translate into the best overall evaluation scores in this experiment.
- **KNN:** KNN achieves the highest accuracy, but its low recall and F1-score make it less effective when identifying customers likely to subscribe is important.
- **Naive Bayes:** Naive Bayes provides a relatively simple and efficient classification approach, but its predictive performance is weaker than that of Logistic Regression and SVM on this dataset.
- **Support Vector Machine:** SVM offers the strongest overall balance between precision and recall among the five models in this run. It is a strong candidate for the overall winner.
- **Overall winner:** SVM is the best overall choice according to F1-score and MCC, while Logistic Regression is a close alternative with slightly better AUC and recall.

## 7. Run locally
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## 8. Streamlit Community Cloud deployment
1. Push this project and the generated `models/*.joblib`, `test_data.csv`, `requirements.txt`, `README.md`, and `app.py` to GitHub.
2. Open https://streamlit.io/cloud and sign in with GitHub.
3. Create a new app, choose the repository, branch (`main`), and `app.py`.
4. Deploy. Open the live URL and confirm the interactive app loads.
5. Test with the included `test_data.csv` and take a screenshot of the assignment execution on BITS Virtual Lab.

## 9. Required submission PDF
Include, in the order specified by your assignment:
1. GitHub repository link.
2. Live Streamlit app link.
3. Screenshot of assignment execution on BITS Virtual Lab.
4. This README content, including the completed comparison table and observations.

## 10. Notes
- The app requires labeled test data with all 16 input columns and `y`. The app evaluates saved models; it does not retrain models on uploaded test data.
- `duration` is included because it is part of the official dataset, but UCI notes it is not known before the call ends. For real pre-call prediction, consider excluding it and retraining all models.
