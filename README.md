# Customer Behavior Analysis — PCA & K-Means Segmentation

## Project Overview

This project analyzes customer purchasing and engagement behavior using Principal Component Analysis (PCA) and K-Means clustering.

The goal was to reduce multiple customer behavior variables into a smaller number of dimensions, identify behavioral customer segments, and translate the machine-learning results into actionable business insights.

## Dataset

- Customers: 1,000
- Behavioral variables: 8
- Missing values: 5
- Missing values were handled using median imputation.

## Features

- Total Spending
- Total Orders
- Average Order Value
- Website Visits
- Purchase Frequency
- Discount Usage %
- Returns
- Engagement Score

## Methodology

1. Data quality inspection
2. Missing-value treatment
3. Feature selection
4. Feature standardization
5. PCA dimensionality reduction
6. Explained variance analysis
7. PCA loading interpretation
8. Outlier investigation
9. K-Means clustering
10. Elbow method evaluation
11. Silhouette score evaluation
12. Customer segment profiling
13. Business recommendations
14. Dashboard development

## PCA Results

PCA reduced the original 8 behavioral variables to 3 principal components.

- PC1: 56.68%
- PC2: 12.68%
- PC3: 12.39%
- Cumulative variance explained: 81.75%

PC1 primarily represented overall customer activity and value, while PC2 and PC3 captured differences involving discount usage and average order value.

## K-Means Segmentation

Several cluster counts were evaluated using silhouette scores.

| K | Silhouette Score |
|---:|---:|
| 2 | 0.3991 |
| 3 | 0.2948 |
| 4 | 0.2467 |
| 5 | 0.2479 |
| 6 | 0.2525 |

K=2 produced the highest silhouette score. However, K=5 was selected for business segmentation because it provided more detailed and interpretable customer behavior profiles.

## Final Customer Segments

| Segment | Customers | Key Characteristic |
|---|---:|---|
| High-Value Occasional | 205 | Low frequency, highest AOV |
| Active Loyal | 244 | Strong purchasing and engagement |
| Regular Moderate | 258 | Moderate activity |
| Discount-Driven | 205 | Highest discount usage |
| Highly Engaged VIP | 88 | Highest spending and engagement |

## Key Business Insights

- Highly Engaged VIP customers have the highest average spending at $2,824.74.
- Active Loyal customers show strong repeat purchasing and engagement.
- High-Value Occasional customers have the highest average order value but low purchase frequency.
- Discount-Driven customers have the highest discount usage at 44.57%.
- PCA visualization shows meaningful behavioral patterns but also substantial overlap between segments.

## Business Recommendations

### Highly Engaged VIP

Develop VIP benefits, personalized experiences, early access, and loyalty rewards.

### Active Loyal

Use loyalty programs, cross-selling, and personalized product recommendations to increase customer value.

### High-Value Occasional

Use re-engagement campaigns, personalized reminders, and relevant product recommendations to increase purchase frequency.

### Discount-Driven

Test targeted promotions and measure whether discounts generate incremental purchases rather than simply reducing revenue.

### Regular Moderate

Use personalized offers and targeted campaigns to gradually increase engagement and purchasing frequency.

## Dashboard

The dashboard includes:

- Customer KPI cards
- PCA explained variance
- Customer segment distribution
- Average customer spending by segment
- PCA customer segmentation scatter plot
- Business insights and recommendations

## Tools

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Standardization
- PCA
- Dimensionality Reduction
- K-Means Clustering
- Elbow Method
- Silhouette Analysis
- Customer Segmentation
- Data Visualization
- Business Analysis
- Dashboard Development

## Project Outcome

This project demonstrates how unsupervised machine learning can be combined with business analysis to transform customer behavior data into interpretable customer segments and actionable business strategies.
