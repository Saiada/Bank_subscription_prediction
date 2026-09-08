# Bank Term Deposit Subscription Prediction

A Logistic Regression model that predicts whether a bank customer will subscribe to a **term deposit**, based on the classic UCI Bank Marketing dataset (`bank-full.csv`, 45,211 rows × 17 columns).

---

## 📁 Project Structure

```
bank_subscription_prediction/
│
├── source/
│   └── Solution_2.ipynb        # Full notebook: preprocessing, training, evaluation
│
├── Diagrams/
│   ├── confusion_matrix.jpeg
│   ├── roc_curve.jpeg
│   ├── precision_recall_curve.jpeg
│   ├── top10_feature_coefficients.jpeg
│   └── top10_misclassified_profiles.jpeg
│
└── README.md
```

---

## 🧾 Dataset

- **Source:** Bank Marketing dataset (`bank-full.csv`), 45,211 records, 17 columns.
- **Target variable:** `y` — whether the client subscribed to a term deposit (`yes` → 1, `no` → 0).
- **Features:** Customer demographics (age, job, marital status, education), financial info (balance, housing/personal loan), and campaign contact details (contact type, month, duration, campaign, previous outcome).

---

## ⚙️ Methodology

1. **Data loading** — CSV extracted from a zipped archive, loaded with `;` delimiter.
2. **Preprocessing**
   - Target mapped to binary (`yes`/`no` → `1`/`0`).
   - Categorical features one-hot encoded (`pd.get_dummies`, `drop_first=True`).
   - 80/20 stratified train-test split.
   - Features scaled with `StandardScaler`.
3. **Model** — `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`, trained to handle the class imbalance between subscribers and non-subscribers.
4. **Evaluation** — Accuracy, ROC-AUC, classification report, confusion matrix, ROC curve, precision-recall curve, feature coefficients, and misclassified-customer analysis.

---

## 📊 Results Summary

| Metric | Score |
|---|---|
| Accuracy | **84.60%** |
| ROC-AUC | **0.9079** |
| Precision (Subscribed) | 0.42 |
| Recall (Subscribed) | 0.81 |
| F1-score (Subscribed) | 0.55 |

---

## Confusion Matrix

![Confusion Matrix](Diagrams/confusion_matrix.jpeg)

**Observation:** Out of 9,043 test customers, the model correctly identified 861 of the 1,058 actual subscribers (recall ≈ 81%), but at the cost of 1,196 false positives — customers predicted to subscribe who did not. This is expected given `class_weight="balanced"`: the model was tuned to catch more true subscribers, trading precision for recall, which suits a marketing use case where missing a potential subscriber is costlier than a wasted call.

---

## ROC Curve

![ROC Curve](Diagrams/roc_curve.jpeg)

**Observation:** The ROC-AUC of **0.908** indicates strong overall separability between subscribers and non-subscribers — the model ranks positive cases far above negative cases most of the time. The curve rises steeply toward the top-left, well above the random-chance diagonal.

---

## Precision-Recall Curve

![Precision-Recall Curve](Diagrams/precision_recall_curve.jpeg)

**Observation:** Precision holds around 0.6–0.65 for recall values up to ~0.4, then declines steadily as recall increases further. This reflects the class imbalance (only ~11.7% of customers subscribed) — pushing recall higher unavoidably pulls in more false positives, so the optimal operating threshold depends on the campaign's tolerance for wasted contacts vs. missed leads.

---

## Top 10 Feature Coefficients

![Top 10 Feature Coefficients](Diagrams/top10_feature_coefficients.jpeg)

**Observation:**
- **`duration`** (call length) is by far the strongest positive predictor — longer calls strongly correlate with subscription, though this is a post-outcome signal (a long call often *implies* interest) so it should be interpreted carefully for real-time targeting.
- **`poutcome_success`** (success in a previous campaign) also boosts odds notably — past converters are good targets again.
- **`contact_unknown`** is the strongest negative predictor — customers with an unknown contact method are far less likely to subscribe.
- Certain months (`jul`, `may`, `aug`, `nov`, `jan`) and having a housing loan (`housing_yes`) are associated with lower subscription odds, while a high number of campaign contacts (`campaign`) also reduces the odds — suggesting diminishing (or even negative) returns from repeated contact attempts.

---

## Top 10 Misclassified Customer Profiles

![Top 10 Misclassified Customer Profiles](Diagrams/top10_misclassified_profiles.jpeg)

**Observation:** The model's most confident errors (all false positives, confidence ≈ 0.9998–1.0) share a clear pattern: unusually **long call durations** (1,700–3,284 seconds) despite an actual outcome of "No." This confirms `duration` is being weighted very heavily by the model — when a call runs long but doesn't convert, the model is almost certain to (wrongly) predict a subscription. These cases point to `duration` as a double-edged feature: highly predictive on average, but a source of high-confidence mistakes on the tail end.

---

## 🔑 Key Takeaways

- Call **duration** and **prior campaign success** are the dominant drivers of subscription likelihood.
- The model favors **recall over precision** by design (`class_weight="balanced"`), making it well-suited for lead prioritization where missing a likely subscriber is costlier than an extra call.
- Repeated contact attempts (`campaign`) and unknown contact methods show diminishing or negative returns — worth revisiting outreach strategy.
- Because `duration` is only known *after* a call happens, a production/real-time model should consider excluding or de-weighting it to avoid data leakage, and instead focus on pre-call features for targeting.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `Google Colab`

---

## 📌 How to Run

1. Open `source/Solution_2.ipynb` in Google Colab (or Jupyter).
2. Update the dataset path (`ZIP_PATH_IN_DRIVE`) to point to your copy of `bank-data.zip`.
3. Run all cells sequentially to reproduce preprocessing, training, and the visualizations in `Diagrams/`.

---

*This README is editable — update the results table, observations, or structure as the analysis evolves.*
