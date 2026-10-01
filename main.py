import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df=pd.read_csv('data/pca_cleaned.csv')


pca_features = ['Total_Spending', 'Total_Orders', 'Avg_Order_Value', 'Website_Visits', 'Purchase_Frequency', 'Discount_Usage_Pct', 'Returns', 'Engagement_Score']

####Step5
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[pca_features])

# Keep the result as a DataFrame with the original index and column names
df_scaled = pd.DataFrame(scaled_array, columns=pca_features, index=df.index)
#print(df_scaled.describe().loc[['mean','std']].round(2).stack())


#####Step6
pca = PCA(n_components=None)  
pca_components = pca.fit_transform(df_scaled)
#print(pca_components)

explained_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(pca.n_components_)],
    'Eigenvalue': pca.explained_variance_,
    'Explained_Variance_Pct': pca.explained_variance_ratio_ * 100,
    'Cumulative_Variance_Pct': np.cumsum(pca.explained_variance_ratio_) * 100
})

#print(df_scaled.shape)             # expect (n_rows, 8)         
#print(list(df_scaled.columns))     # expect all 8 feature names
#print(pca.n_components_)
#print(explained_df.round(2).stack())

#####Step7
# Loading matrix: rows = original features, columns = principal components
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=pca_features
)
#print(loadings.round(3))

#####Step8
pca_df = pd.DataFrame(
    pca_components[:, :3],
    columns=['PC1', 'PC2', 'PC3'],
    index=df.index
)
pca_df.insert(0, 'Customer_ID', df['Customer_ID'].values)
print(f"PCA df {pca_df.shape}")
#print(pca_df.head())

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, s=20, color='steelblue', edgecolor='none')
# Reference lines through the origin
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
# Put the variance share of each component in the axis labels
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Customers in PCA Space (PC1 vs PC2)')
plt.tight_layout()
plt.show()

#####Step9
top_pc1 = pca_df.loc[pca_df['PC1'].idxmax()]
#print(top_pc1)

#####Step10
idx = pca_df['PC1'].idxmax()
z_scores = df_scaled.loc[idx].round(2)
#print(f"Customer_ID: {pca_df.loc[idx, 'Customer_ID']}")
#print(z_scores.sort_values(ascending=False))

#####Step11
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(explained_df['Component'], explained_df['Explained_Variance_Pct'], color='steelblue')
ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('Scree Plot: PC1 to PC8')
ax.set_ylim(0, explained_df['Explained_Variance_Pct'].max() * 1.15)

plt.tight_layout()
plt.show()

##### Kmean 14
# Use the first 3 principal components (81.75% variance)
X = pca_df[['PC1', 'PC2', 'PC3']].values

k_values = range(2, 9)
inertias = []

for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f'K={k}: Inertia={km.inertia_:.2f}')
elbow_df = pd.DataFrame({'K': list(k_values), 'Inertia': inertias})
print(elbow_df)
plt.plot(elbow_df['K'],elbow_df['Inertia'], marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

##### Step 15 — Choose the number of clusters
# Make sure df and pca_df share the same row order/index before combining
profile_source = df[pca_features].copy()
profile_source.index = pca_df.index  # align just in case
cluster_profiles = {}
for k in range(2, 9):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(X)
        temp = profile_source.copy()
        temp['Cluster'] = labels
        # Count + averages of the 8 original features, per cluster
        summary = temp.groupby('Cluster').agg(
                Num_Customers=('Cluster', 'size'),
                **{f'Avg_{col}': (col, 'mean') for col in pca_features}).round(2)
        cluster_profiles[k] = summary
        
       # print(f'\n=== K = {k} ===')
       # print(summary.to_string())
#print(cluster_profiles.keys())
#print(cluster_profiles[5])

##### Step 16 — Validate K=4 vs K=5 Calculate the silhouette score
silhouette_results = []
for k in range(2, 7):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_results.append({'K': k, 'Silhouette Score': round(score, 4)})

silhouette_df = pd.DataFrame(silhouette_results)
print(silhouette_df.to_string(index=False))

#####Step 17 — Finalize K=5
km5 = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
labels = km5.fit_predict(X)
pca_df['Cluster_K5'] = labels
fig, ax = plt.subplots(figsize=(9, 6))

scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster_K5'],
                                          cmap='tab10', alpha=0.7, s=25, edgecolor='none')
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Customer Segments (K=5) in PCA Space')
# Legend showing cluster numbers
legend1 = ax.legend(*scatter.legend_elements(), title='Cluster')
ax.add_artist(legend1)
plt.tight_layout()
plt.show()


