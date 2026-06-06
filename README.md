# 🛒 Revenue Analytics & Customer Segmentation Initiative

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?logo=streamlit)](https://revenueanalytics-customer-segmentation-initiative.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means-orange?logo=scikit-learn)](https://scikit-learn.org/)

> **🔗 Try the live app:** [revenueanalytics-customer-segmentation-initiative.streamlit.app](https://revenueanalytics-customer-segmentation-initiative.streamlit.app)

---

## 1. What Problem Does This Solve?

Businesses often struggle to understand diverse customer behaviours, leading to generic marketing campaigns and missed revenue opportunities. One-size-fits-all strategies result in wasted marketing spend and poor customer retention.

This project solves the **Customer Understanding** problem. By applying unsupervised machine learning to transactional data, we automatically group customers into meaningful segments — enabling personalised marketing, targeted retention strategies, and data-informed revenue decisions.

---

## 2. Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-Learn | K-Means clustering & preprocessing |
| Pandas & NumPy | Data manipulation & RFM engineering |
| Streamlit | Interactive web app deployment |
| Joblib | Model serialization |
| Plotly & Seaborn | Visualizations |

---

## 3. What Data Was Used?

A synthetic dataset simulating real-world retail transactions. Key raw features:

- **CustomerID** — Unique customer identifier
- **InvoiceNo** — Transaction identifier
- **InvoiceDate** — Transaction timestamp
- **Quantity & UnitPrice** — Purchase metrics
- **Country** — Customer location

**Derived RFM Features (engineered):**

| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Total number of purchases |
| Monetary | Total spend amount |
| Tenure | Days since first purchase |
| AvgOrderValue | Monetary ÷ Frequency |

---

## 4. Modeling Approach

**Algorithm:** K-Means Clustering

**Why K-Means?** It is fast, interpretable, and well-suited for RFM-based segmentation where the goal is to find distinct, actionable customer groups rather than predict a label.

**Cluster Selection:** Optimal number of clusters determined using the **Elbow Method**.

**Evaluation Metrics:**
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

**Outcome — 4 Business-Aligned Segments:**

| Segment | Profile |
|---|---|
| 🏆 High Value | High spend, high frequency, purchased recently |
| 🔄 Frequent Buyer | Buys often but lower average spend |
| ⚠️ At Risk | Haven't purchased recently, may churn |
| 📦 Regular | Moderate activity across all RFM dimensions |

---

## 5. Project Structure

```
├── raw data/              # Source dataset
├── notebooks/             # EDA, feature engineering, model development
├── src/
│   ├── features/          # RFM feature engineering pipeline
│   ├── Transformer/       # Custom preprocessing (outliers, skew, scaling)
│   └── inference/         # Prediction & segment naming pipeline
├── models/
│   ├── preprocessor.pkl   # Fitted preprocessor
│   └── kmeans_best.pkl    # Trained K-Means model
├── streamlit_app/         # Web application
├── predictions/           # Output predictions
├── reports/               # Analysis reports
└── requirements.txt
```

---

## 6. How to Run Locally

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

## 7. App Features

- **CSV Upload** — Upload a file with RFM columns for bulk segmentation
- **Manual Input** — Enter a single customer's details and get an instant segment prediction
- **Segment Output** — Returns both the cluster number and a human-readable segment name

**Required CSV columns:**
```
Recency, Tenure, Frequency, Monetary, AvgOrderValue
```

---

## 8. Why Should You Care?

This project demonstrates end-to-end unsupervised ML product thinking:

- **Feature Engineering** — RFM metrics derived from raw transactional data, a standard industry framework used by retailers and banks
- **Custom Preprocessing Pipeline** — Handles outlier capping, skewness transformation, and standard scaling in a reusable, modular class
- **Business-Aligned Output** — Cluster numbers are automatically mapped to interpretable labels (High Value, At Risk, etc.) making results actionable without data science expertise
- **Modular Architecture** — Separate modules for ingestion, feature engineering, preprocessing, and inference — following professional software engineering patterns
- **Live Deployment** — End-to-end from raw data to a publicly accessible web app
