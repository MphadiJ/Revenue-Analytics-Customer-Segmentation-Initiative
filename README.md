
# Revenue Analytics & Customer Segmentation Initiative 🛒

[![Live App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://revenueanalytics-customer-segmentation-initiative.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/cloud)

> An end-to-end customer intelligence system — from raw retail transactions to live, interactive segment and churn risk predictions.

---

## 🔗 Live Demo

**[→ Open the App](https://revenueanalytics-customer-segmentation-initiative.streamlit.app/)**

##Download the churn data from segmentation app and upload here (https://revenue-analytics-customer-segmentation-initiative-wvh7mhrjgci.streamlit.app) for kpi dashboard.

---

## 📌 Problem Statement

Businesses running generic marketing campaigns lose revenue by treating all customers the same. This project answers three questions from raw transaction data:

1. **Who are my customers?** — Segment them by behaviour
2. **Which customers are about to leave?** — Score churn risk per customer
3. **What does a new customer look like?** — Predict segment and churn risk in real time

---

## 🧠 Solution Overview

The pipeline runs in three stages:

```
Raw Transactions → RFM Feature Engineering → K-Means Segmentation
                                           → Churn Risk Scoring
                                           → Streamlit Dashboard
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | Synthetic retail transaction dataset |
| Shape | 197,316 rows × 13 columns |
| Key raw fields | `CustomerID`, `InvoiceDate`, `Quantity`, `UnitPrice`, `TotalPrice` |

**Engineered RFM features:**

| Feature | Description |
|---|---|
| `Recency` | Days since last purchase |
| `Frequency` | Total number of purchases |
| `Monetary` | Total spend (£) |
| `Tenure` | Days since first purchase |
| `AvgOrderValue` | Monetary ÷ Frequency |

---

## 🧩 Customer Segments

K-Means clustering (optimal K selected via Elbow Method + Silhouette Score) produces four business-labelled segments:

| Segment | Profile |
|---|---|
| 🟢 High-Value Customers | High spend, high frequency, recent buyers |
| 🔵 Loyal Customers | Consistent purchasers with long tenure |
| 🟡 At-Risk Customers | Previously active, now showing declining engagement |
| ⚪ Occasional Buyers | Low frequency, low spend, infrequent visits |

---

## ⚠️ Churn Risk Analysis

Each customer receives a **Churn Score (0–1)** computed from their RFM profile — no labelled churn data required.

**Scoring formula:**

```
Churn Score = 0.50 × Recency_norm + 0.30 × (1 - Frequency_norm) + 0.20 × (1 - Monetary_norm)
```

| Risk Band | Score Range | Meaning |
|---|---|---|
| 🔴 High Risk | ≥ 0.65 | Likely churning — immediate retention action needed |
| 🟡 Medium Risk | 0.35 – 0.64 | Showing warning signs — monitor closely |
| 🟢 Low Risk | < 0.35 | Engaged and active |

---

## 💻 App Features

The Streamlit app has three tabs:

### 📊 Tab 1 — Segmentation
- Upload a CSV with RFM features
- Get instant K-Means segment predictions
- Download segmented results

### ⚠️ Tab 2 — Churn Analysis
- Runs automatically on segmented data from Tab 1 (or upload separately)
- KPI cards: total customers, High / Medium / Low risk counts
- Churn score distribution histogram
- Average churn score per segment (bar chart)
- Risk breakdown per segment (stacked bar)
- Recency vs Monetary scatter coloured by churn risk
- Full customer table with download

### 👤 Tab 3 — Single Customer
- Manual input for any customer's RFM values
- Predicts segment and churn risk simultaneously
- Score breakdown table showing what is driving the risk
- Visual gauge bar for churn score

---

## 🗂️ Project Structure

```
├── raw data/
│   └── rt_data.csv                  # Raw retail transaction data
├── src/
│   ├── features/
│   │   └── build_features.py        # RFM feature engineering
│   ├── Transformer/
│   │   └── Preprocessing.py         # Scaling and preprocessing
│   ├── trainer/
│   │   └── train_pipeline.py        # KMeansTrainer class
│   ├── inference/
│   │   └── inference.py             # InferencePipeline class
│   └── churn/
│       ├── __init__.py
│       └── churn_analysis.py        # Churn scoring module
├── models/
│   ├── kmeans_best.pkl              # Trained K-Means model
│   └── preprocessor.pkl             # Fitted scaler/preprocessor
├── notebooks/                       # Exploratory analysis
├── streamlit_app/
│   └── app.py                       # Streamlit dashboard
├── reports/
├── predictions/
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| ML | scikit-learn (KMeans, Silhouette Score) |
| Visualisation | Plotly, Seaborn, Matplotlib |
| App | Streamlit |
| Serialisation | Joblib |
| Version Control | Git & GitHub |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/MphadiJ/Revenue-Analytics-Customer-Segmentation-Initiative.git
cd Revenue-Analytics-Customer-Segmentation-Initiative

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run streamlit_app/app.py
```

---

## 👤 Author

**Selowa Mphadi John**
Data Science Practitioner | BSc Mathematical Sciences (Statistics & Operations Research)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-selowamj-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/selowamj)
[![GitHub](https://img.shields.io/badge/GitHub-MphadiJ-181717?style=flat&logo=github)](https://github.com/MphadiJ)
