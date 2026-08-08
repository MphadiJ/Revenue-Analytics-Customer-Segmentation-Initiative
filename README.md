# Customer Intelligence & Revenue Analytics Platform 🛒

[![Live App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://revenue-analytics-customer-segmentation-initiative-fwtdszq3pgl.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/cloud)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3D4DB7?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

> An end-to-end customer intelligence system — from raw retail transactions to live segmentation, churn risk scoring, revenue KPI reporting, and executive decision support. One app. One data source. Everything computed automatically.

---

## 🔗 Live Demo

**[→ Open the Customer Intelligence Platform](https://revenue-analytics-customer-segmentation-initiative-fwtdszq3pgl.streamlit.app/)**

---

## 📌 Problem Statement

Businesses running generic marketing campaigns lose revenue by treating all customers the same. This project answers five questions from raw transaction data:

1. **Who are my customers?** — Segment them by behaviour
2. **Which customers are about to leave?** — Score churn risk per customer with a binary decision
3. **What does a new customer look like?** — Predict segment and churn risk in real time
4. **How is the business performing?** — Track revenue, products, and geography in a live KPI dashboard
5. **What should leadership do about it?** — Automated health scoring, win-back simulation, and strategic action playbook

---

## 🧠 Solution Architecture

Everything flows from a single raw data source — no manual uploads or data transfers between tabs:

```
raw data/rt_data.csv
        │
        ▼
load_raw_data()          ← Clean, remove returns, engineer time columns
        │
        ▼
compute_rfm()            ← Recency, Frequency, Monetary, Tenure, AvgOrderValue
        │
        ▼
InferencePipeline        ← Preprocess → K-Means predict → name_segments()
        │
        ▼
add_churn_analysis()     ← Weighted RFM churn score (0–1) → risk band → decision → action
        │
        ▼
5-Tab Streamlit App      ← Segmentation · Churn · Single Customer · KPIs · Executive Summary
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | Retail transaction dataset |
| Shape | 197,316 rows × 13 columns |
| Transaction Value | R5.9 billion |
| Key raw fields | `CustomerID`, `InvoiceDate`, `Quantity`, `UnitPrice`, `TotalPrice`, `Country`, `Description` |

**Engineered RFM features fed into the model:**

| Feature | Description |
|---|---|
| `Recency` | Days since last purchase |
| `Frequency` | Total number of unique orders |
| `Monetary` | Total spend (£) |
| `Tenure` | Days since first purchase |
| `AvgOrderValue` | Monetary ÷ Frequency |

---

## 🧩 Customer Segments

K-Means clustering — optimal K selected via Elbow Method and Silhouette Score:

| Segment | Profile |
|---|---|
| 🟢 High-Value Customers | High spend, high frequency, recent buyers |
| 🔵 Loyal Customers | Consistent purchasers with long tenure |
| 🟡 At-Risk Customers | Previously active, now showing declining engagement |
| ⚪ Occasional Buyers | Low frequency, low spend, infrequent visits |

---

## ⚠️ Churn Risk Scoring

Each customer receives a **Churn Score (0–1)** computed from their RFM profile — no labelled churn data required. Fully vectorised. Runs automatically on startup.

**Scoring formula:**

```
Churn Score = 0.50 × Recency_norm + 0.30 × (1 - Frequency_norm) + 0.20 × (1 - Monetary_norm)
```

| Risk Band | Score | Decision | Action |
|---|---|---|---|
| 🔴 High Risk | ≥ 0.65 | Will Churn | Immediate retention offer |
| 🟡 Medium Risk | 0.35 – 0.64 | At Risk | Re-engagement campaign |
| 🟢 Low Risk | < 0.35 | Retained | Standard loyalty programme |

---

## 💻 App — 5 Tabs

### 🎯 Tab 1 — Customer Segmentation
- Loads and displays auto-segmented data from `rt_data.csv` on startup — no upload required
- Customer count, revenue, avg frequency and recency KPI cards
- Segment distribution bar chart and revenue contribution donut
- RFM landscape scatter — Recency vs Monetary coloured by segment
- Segment statistics table with avg revenue, frequency, recency, tenure
- Optional CSV upload to override with custom RFM data
- Download segmentation results

### ⚠️ Tab 2 — Churn Risk Analysis
- Auto-scores every customer on load — no button click required
- 5 KPI cards: total customers, Will Churn count, At Risk count, Retained count, Revenue at Risk (£)
- Churn score distribution histogram with threshold lines at 0.35 and 0.65
- Stacked bar: churn risk breakdown per segment
- Avg churn score per segment bar chart
- Recency vs Monetary scatter coloured by churn risk
- Top 10 customers most likely to churn — ranked by score with recommended actions
- Full churn table with risk tier filter and download

### 👤 Tab 3 — Single Customer Intelligence
- **Look Up mode** — search any existing customer by ID, view segment, churn decision, gauge chart, RFM metrics, recommended action, and comparison vs segment average
- **Manual Prediction mode** — enter any RFM values and get instant segment prediction, churn risk score, score breakdown table, and per-driver progress bars
- Churn gauge correctly scaled 0–1 with colour-coded threshold bands

### 📈 Tab 4 — Revenue KPI Dashboard
- 7 headline KPI cards with period-over-period delta indicators (Revenue, Orders, Customers, AOV, Rev/Customer, Units Sold, Repeat Rate)
- Monthly revenue trend + MoM growth overlay
- Year × Month revenue heatmap
- Orders and revenue by day of week
- Top 10 products by revenue and by volume
- Revenue vs units sold scatter (top 30 products)
- New vs returning customer revenue split
- Customer order frequency distribution
- Top 10 customers by lifetime revenue
- Revenue, orders, and AOV by country

### 🧠 Tab 5 — Executive Intelligence Summary
- **Business Health Score** (0–10) — automatically computed from churn exposure and revenue risk
- Colour-coded health banner: Strong / Stable — Monitor / Critical — Action Required
- Monthly revenue trend + revenue split by risk tier
- Revenue by segment + avg churn score per segment
- **Win-Back Revenue Simulator** — adjust a conversion rate slider and see projected revenue reclaimed, customers rescued, and simulated post-campaign health score in real time
- Dynamic Strategic Action Playbook — conditional recommendations based on live data thresholds
- Two exports: executive report `.txt` + ranked High Risk customer list `.csv`

---

## 🗂️ Project Structure

```
├── raw data/
│   └── rt_data.csv                        # Raw retail transaction data
├── src/
│   ├── features/
│   │   └── build_features.py              # RFM feature engineering
│   ├── Transformer/
│   │   └── Preprocessing.py               # Scaling and preprocessing
│   ├── trainer/
│   │   └── train_pipeline.py              # KMeansTrainer class
│   ├── inference/
│   │   └── inference.py                   # InferencePipeline class
│   ├── churn/
│   │   ├── __init__.py
│   │   └── churn_analysis.py              # Churn scoring module
│   └── kpi_engine.py                      # All KPI computation functions
├── models/
│   ├── kmeans_best.pkl                    # Trained K-Means model
│   └── preprocessor.pkl                   # Fitted scaler/preprocessor
├── notebooks/                             # Exploratory analysis
├── streamlit_app/
│   ├── app2.py                            # ✅ Main combined app (5 tabs)
│   ├── app.py                             # Legacy segmentation + churn app
│   └── tabs/
│       ├── segmentation.py                # Tab 1 — Customer Segmentation
│       ├── churn.py                       # Tab 2 — Churn Risk Analysis
│       ├── single_customer.py             # Tab 3 — Single Customer Intelligence
│       ├── dashboard.py                   # Tab 4 — Revenue KPI Dashboard
│       └── executive_summary.py           # Tab 5 — Executive Intelligence Summary
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
| ML | scikit-learn (KMeans, StandardScaler, Silhouette Score) |
| Visualisation | Plotly |
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

# Launch the combined intelligence app
streamlit run streamlit_app/app2.py
```

---

## 📥 CSV Upload Formats

The app auto-loads `raw data/rt_data.csv` on startup. Two optional upload paths exist:

**Sidebar — raw transaction CSV:**
```
InvoiceNo, InvoiceDate, CustomerID, Quantity, UnitPrice
```
Runs the full pipeline: clean → RFM → segment → churn.

**Segmentation tab — pre-computed RFM CSV:**
```
Recency, Tenure, Frequency, Monetary, AvgOrderValue
```
Goes straight to the model — skips feature engineering.

---

## 👤 Author

**Selowa Mphadi John**
Data Science Practitioner (Workplace-Based Learning) | Isazi Consulting × Mindworx Academy
BSc Mathematical Sciences (Statistics & Operations Research) — University of Limpopo

[![LinkedIn](https://img.shields.io/badge/LinkedIn-selowamj-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/selowamj)
[![GitHub](https://img.shields.io/badge/GitHub-MphadiJ-181717?style=flat&logo=github)](https://github.com/MphadiJ)
