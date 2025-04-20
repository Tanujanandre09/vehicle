#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


dataset_path = r"C:\Users\Rasika\Downloads\ABC.csv"  # Ensure the dataset is in CSV format
df = pd.read_csv(dataset_path)


# In[19]:


# Display the first few rows to understand the structure
df.head()


# In[20]:


# Inspect the structure of the dataset
df.info()


# In[21]:


# Check the size of the dataset (rows and columns)
print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# In[22]:


# Identifying missing values
print("Missing values in each column:")
print(df.isnull().sum())


# In[23]:


# Check for duplicate records
print(f"Number of duplicate rows: {df.duplicated().sum()}")


# In[24]:


# Basic statistics of numerical columns
df.describe()


# In[25]:


# Data Cleaning and Preprocessing

# Handling missing values
# Example: Filling missing numerical values with the mean
df.fillna(df.mean(), inplace=True)


# In[ ]:


# Convert data types if necessary (Example: converting 'date' column to datetime)
# df['date_column'] = pd.to_datetime(df['date_column'])

# Dropping duplicate records, if any
df.drop_duplicates(inplace=True)


# In[ ]:


# Handling outliers (you can use IQR method or Z-score for detecting outliers)
# Example: Removing outliers in a 'fuel_consumption' column
Q1 = df['fuel_consumption'].quantile(0.25)
Q3 = df['fuel_consumption'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['fuel_consumption'] >= (Q1 - 1.5 * IQR)) & (df['fuel_consumption'] <= (Q3 + 1.5 * IQR))]


# In[ ]:


# Exploratory Data Analysis (EDA)

# Univariate Analysis
# Numerical Features
df.hist(bins=20, figsize=(15, 10))
plt.show()


# In[ ]:


# Categorical Features# Categorical Features
for col in df.select_dtypes(include='object').columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()
for col in df.select_dtypes(include='object').columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()# Categorical Features
for col in df.select_dtypes(include='object').columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()


# In[ ]:


# Bivariate Analysis
# Example: Relationship between fuel consumption and vehicle age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vehicle_age', y='fuel_consumption', data=df)
plt.title('Fuel Consumption vs Vehicle Age')
plt.show()


# In[ ]:


# Example: Correlation heatmap for numerical variables
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


# Outlier Detection (Boxplot)
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[ ]:


# Feature Engineering

# Creating a new feature: Fuel efficiency (assuming you have columns like distance traveled and fuel consumption)
# df['fuel_efficiency'] = df['distance_traveled'] / df['fuel_consumption']

# Standardizing numerical features if necessary
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# Insights and Recommendations

# Summary of key insights
print("Key Insights:")


# In[ ]:


# Example: Identifying the average fuel consumption by vehicle type
print(df.groupby('vehicle_type')['fuel_consumption'].mean())


# In[ ]:


# Recommendations based on insights
print("Recommendations:")


# In[ ]:


# Example: Recommend actions for improving fuel efficiency based on driver behavior
print("1. Monitor driver behavior more closely to improve fuel efficiency.")
print("2. Regular vehicle maintenance can reduce fuel consumption and increase uptime.")


# In[ ]:





# In[ ]:




