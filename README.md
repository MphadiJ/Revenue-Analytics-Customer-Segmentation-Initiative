Customer Segmentation Project
🚀 Overview

This project delivers a data-driven customer segmentation solution that identifies distinct customer groups to enhance marketing strategies, improve retention, and maximize revenue. It demonstrates end-to-end data science expertise, from data ingestion and feature engineering to model deployment and interactive visualization.

🧩 Problem Statement

Businesses often struggle to understand diverse customer behaviors, leading to generic marketing campaigns and missed revenue opportunities. This project aims to segment customers based on purchasing behavior and other key metrics, enabling personalized marketing and data-informed decision-making.

📊 Dataset
Source: Synthetic dataset simulating real-world retail transactions
Key features:
CustomerID – Unique customer identifier
InvoiceNo – Transaction identifier
InvoiceDate – Transaction timestamp
Quantity & UnitPrice – Purchase metrics
Country – Customer location
Derived metrics (RFM Analysis):
Recency – Days since last purchase
Frequency – Total number of purchases
Monetary – Total purchase amount
Tenure
AvgOrderValue

⚙️ Features Engineered
RFM metrics for customer behavior profiling
Categorical encodings for transactional features
Aggregated metrics per customer to feed clustering models
Segment labels derived from K-Means clustering for actionable insights

🧠 Modeling Approach
Algorithm: K-Means clustering
Evaluation Metrics: Silhouette score, Calinski-Harabasz Index, Davies-Bouldin Index
Hyperparameter Tuning: Optimal cluster number selected using the Elbow Method
Outcome: Clear, interpretable customer segments with business-aligned labels:
High-Value Customers
Loyal Customers
At-Risk Customers
Occasional Buyers

💻 Deployment
Framework: Streamlit web application
Functionality:
Upload raw transactional data
Automatic feature engineering & preprocessing
Real-time segment prediction

📈 Visualizations
Distribution plots for numeric & categorical features
Boxplots to detect outliers and insights
Customer segment dashboards with interactive filtering
Count plots for categorical feature analysis

📦 Tech Stack
Python | Pandas | NumPy | Scikit-Learn | Plotly | Streamlit | seaborn | matplotlib
Git & GitHub for version control

How to run the code
# Clone the repo
git clone <repo_url>

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit dashboard
streamlit run app.py

