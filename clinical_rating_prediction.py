"""
Predictive Analytics for Healthcare Operations
Objective: Develop a Machine Learning classification model to predict hospital overall ratings based on 94 clinical and operational metrics.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Data Ingestion & Preprocessing
print("Initializing Clinical Data Pipeline...")
df = pd.read_csv('hospital-info.csv')
target = 'Hospital overall rating'

# Drop non-predictive administrative columns
features = df.drop(columns=['Provider ID', 'Hospital Name', 'Address', 'City', 'State', 'ZIP Code', 'County Name', 'Phone Number', target])

# Encode categorical variables for ML processing
for col in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col].astype(str))

# Handle potential missing clinical values with median imputation
features = features.fillna(features.median())

# 2. Predictive Modeling (Random Forest Classifier)
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 3. Model Evaluation
print("\n--- Algorithm Performance Report ---")
predictions = rf_model.predict(X_test)
print(classification_report(y_test, predictions))

# 4. Feature Importance Extraction (Business Intelligence)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15] # Extract Top 15 clinical drivers
top_features = X_train.columns[indices]

# Generate Executive Visual
plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=top_features, palette='mako')
plt.title('Predictive Analytics: Top 15 Drivers of Hospital Overall Rating')
plt.xlabel('Relative Importance (Gini Score)')
plt.ylabel('Clinical & Operational Metrics')
plt.tight_layout()
plt.savefig('hospital_feature_importance.png')

print("Pipeline execution complete. Model trained and insights extracted.")
