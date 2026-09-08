# Bank Term Deposit Subscription Prediction

This project builds a logistic regression model to predict whether a bank customer will subscribe to a term deposit, using the UCI Bank Marketing dataset (bank-full.csv, 45,211 rows, 17 columns). The goal is to help a bank prioritize which customers to target in future marketing campaigns.

## Project Structure

```
bank_subscription_prediction/
├── source/
│   └── Solution_2.ipynb
├── Diagrams/
│   ├── confusion_matrix.jpeg
│   ├── roc_curve.jpeg
│   ├── precision_recall_curve.jpeg
│   ├── top10_feature_coefficients.jpeg
│   └── top10_misclassified_profiles.jpeg
└── README.md
```

## Dataset

The data comes from a bank's direct marketing campaigns (phone calls). Each row is a customer contacted during a campaign, described by:

- Demographics: age, job, marital status, education
- Financial info: average yearly balance, housing loan, personal loan
- Campaign info: contact type, month and day of last contact, call duration, number of contacts this campaign, number of days since last contact in a previous campaign, outcome of the previous campaign
- Target: whether the customer subscribed to a term deposit (yes/no), mapped to 1/0

## Approach and Methodology

1. Data loading: the raw CSV was extracted from a zipped archive and loaded with a semicolon delimiter.
2. Preprocessing: the target column was mapped to binary values, categorical features were one-hot encoded, and the data was split 80/20 into train and test sets using stratified sampling to preserve the class ratio. Numeric features were then standardized.
3. Handling class imbalance: only about 11.7% of customers in the dataset actually subscribed. To stop the model from simply predicting "no" for everyone, it was trained with balanced class weights, which pushes it to pay more attention to the minority (subscribed) class.
4. Model: a logistic regression classifier was trained on the scaled, encoded features.
5. Evaluation: the model was assessed using accuracy, ROC-AUC, a full classification report, a confusion matrix, an ROC curve, a precision-recall curve, the top feature coefficients, and a review of the customers it misclassified with the highest confidence.

## Results

| Metric | Score |
|---|---|
| Accuracy | 84.60% |
| ROC-AUC | 0.9079 |
| Precision (Subscribed) | 0.42 |
| Recall (Subscribed) | 0.81 |
| F1-score (Subscribed) | 0.55 |

## Confusion Matrix
![Confusion Matrix](Diagrams/confusion_matrix.jpeg)

## ROC Curve
![ROC Curve](Diagrams/roc_curve.jpeg)

## Precision-Recall Curve
![Precision-Recall Curve](Diagrams/precision_recall_curve.jpeg)

## Top 10 Feature Coefficients
![Top 10 Feature Coefficients](Diagrams/top10_feature_coefficients.jpeg)

## Top 10 Misclassified Customer Profiles
![Top 10 Misclassified Customer Profiles](Diagrams/top10_misclassified_profiles.jpeg)

## Findings

The model performs well overall, with an accuracy of 84.6% and a strong ROC-AUC of 0.91, meaning it does a good job separating customers who are likely to subscribe from those who aren't. Because it was trained with balanced class weights, it leans toward catching as many real subscribers as possible: recall for subscribers is high at 81%, but this comes at the cost of precision (42%), meaning a fair number of the customers it flags as likely subscribers will not actually subscribe. The precision-recall curve shows this trade-off clearly, with precision dropping steadily as recall increases, a natural consequence of subscribers being a small minority of the dataset.

Looking at the feature coefficients, call duration stands out as by far the strongest positive predictor, followed by having succeeded in a previous campaign. On the negative side, having an unknown contact method is the strongest signal against subscribing, and a few months, having a housing loan, and a higher number of contacts during the campaign are all associated with lower odds of subscribing, suggesting that repeated contact attempts don't necessarily help.

The misclassified customer analysis reinforces this: nearly all of the model's most confident errors are customers who had unusually long calls but still did not subscribe. This shows that call duration, while the single most useful predictor, is also the main source of confident mistakes, and since the duration of a call is only known after it happens, it should be used carefully if the model is ever applied to real-time targeting rather than after-the-fact analysis.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Google Colab

## How to Run

1. Open source/Solution_2.ipynb in Colab or Jupyter.
2. Point the dataset path to your copy of bank-data.zip.
3. Run all cells to reproduce the preprocessing, training, results, and plots.
