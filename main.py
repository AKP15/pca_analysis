import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
print(pca_df.head())


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
