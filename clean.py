import pandas as pd
df=pd.read_csv('data/project7_pca_customer_behavior.csv')
c=['Customer_ID', 'Total_Spending', 'Total_Orders', 'Avg_Order_Value', 'Website_Visits', 'Purchase_Frequency', 'Discount_Usage_Pct', 'Returns', 'Engagement_Score']
print(df.shape)
print(df.columns.tolist())
#print(df.dtypes)
#print(df.isnull().sum())
#for column in df.columns:
#    print(df[column].describe())
#print(df['Customer_ID'].value_counts())
#print(df.duplicated().sum())

df['Website_Visits']=df['Website_Visits'].fillna(df['Website_Visits'].median())
df['Discount_Usage_Pct']=df['Discount_Usage_Pct'].fillna(df['Discount_Usage_Pct'].median())
print(df.isnull().sum())

df.to_csv('data/pca_cleaned.csv',index=False)
