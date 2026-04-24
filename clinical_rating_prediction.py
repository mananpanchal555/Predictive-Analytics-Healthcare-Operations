"""
Predictive Analytics for Healthcare Operations
Objective: Develop a Machine Learning classification model to predict hospital
overall ratings based on 94 clinical and operational metrics.
Includes: Model evaluation, feature importance, confusion matrix,
          and real-world inference on unrated hospitals.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from collections import Counter

# ── 1. Data Ingestion & Preprocessing ───────────────────────────────────────
print("=" * 65)
print("  Predictive Analytics for Healthcare Operations")
print("  Initializing Clinical Data Pipeline...")
print("=" * 65)

df = pd.read_csv('hospital-info.csv')
target = 'Hospital overall rating'

# Drop non-predictive administrative columns
admin_cols = [
    'Provider ID', 'Hospital Name', 'Address', 'City',
    'State', 'ZIP Code', 'County Name', 'Phone Number', target
]
features = df.drop(columns=admin_cols)

# Encode categorical variables for ML processing
label_encoders = {}
for col in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col].astype(str))
    label_encoders[col] = le

# Handle potential missing clinical values with median imputation
features = features.fillna(features.median())

print(f"\n  Dataset loaded     : {df.shape[0]:,} hospitals")
print(f"  Features used      : {features.shape[1]} clinical & operational variables")
print(f"  Target variable    : {target}")

# ── 2. Train/Test Split & Model Training ────────────────────────────────────
print("\n  Training Random Forest Classifier...")

X_train, X_test, y_train, y_test = train_test_split(
    features, df[target], test_size=0.2, random_state=42, stratify=df[target]
)

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced'
)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)

print(f"  Training set size  : {X_train.shape[0]:,} hospitals")
print(f"  Test set size      : {X_test.shape[0]:,} hospitals")

# ── 3. Model Evaluation ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RANDOM FOREST — CLASSIFICATION REPORT (Test Set)")
print("=" * 65)

# Filter to only star ratings that appear in y_test
unique_labels = sorted(y_test.unique())
target_names  = [f"{int(s)}-star" for s in unique_labels]

print(classification_report(
    y_test, predictions,
    labels=unique_labels,
    target_names=target_names
))

acc = accuracy_score(y_test, predictions)
print(f"  Overall Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
print("=" * 65)

# ── 4. Confusion Matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, predictions, labels=unique_labels)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[f"{int(s)}★" for s in unique_labels]
)

fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(
    'Random Forest — Confusion Matrix\nHospital CMS Rating Prediction',
    fontsize=13, pad=12
)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Confusion matrix saved → confusion_matrix.png")

# ── 5. Feature Importance Extraction ────────────────────────────────────────
importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1][:15]
top_features = X_train.columns[indices]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=top_features, palette='mako')
plt.title('Predictive Analytics: Top 15 Drivers of Hospital Overall Rating')
plt.xlabel('Relative Importance (Gini Score)')
plt.ylabel('Clinical & Operational Metrics')
plt.tight_layout()
plt.savefig('hospital_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Feature importance chart saved → hospital_feature_importance.png")

# Print top 5 features to console
print("\n" + "=" * 65)
print("  TOP 5 CLINICAL DRIVERS OF HOSPITAL RATING")
print("=" * 65)
for i in range(5):
    feat = X_train.columns[indices[i]]
    score = importances[indices[i]]
    print(f"  #{i+1}  {feat:<45} {score:.4f}")
print("=" * 65)

# ── 6. Real-World Inference on Unrated Hospitals ─────────────────────────────
print("\n" + "=" * 65)
print("  INFERENCE ON UNRATED HOSPITALS (not_yet_rated.csv)")
print("=" * 65)

try:
    unrated_df = pd.read_csv('not_yet_rated.csv')

    # Apply same preprocessing as training data
    unrated_features = unrated_df.drop(
        columns=[c for c in admin_cols if c in unrated_df.columns and c != target],
        errors='ignore'
    )

    # Encode categoricals using same encoders
    for col in unrated_features.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            unrated_features[col] = unrated_features[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            le_new = LabelEncoder()
            unrated_features[col] = le_new.fit_transform(
                unrated_features[col].astype(str)
            )

    # Align columns to training feature set
    unrated_features = unrated_features.reindex(
        columns=features.columns, fill_value=0
    )
    unrated_features = unrated_features.fillna(unrated_features.median())

    # Generate predictions
    predicted_ratings = rf_model.predict(unrated_features)

    # Summary
    rating_counts = Counter(predicted_ratings)
    print(f"  Total unrated facilities scored : {len(predicted_ratings):,}")
    print()
    for star in sorted(rating_counts):
        bar = "█" * int(rating_counts[star] / len(predicted_ratings) * 30)
        print(f"  Predicted {int(star)}-star : {rating_counts[star]:>4}  {bar}")

    # Save predictions to CSV
    output_df = unrated_df.copy()
    output_df['Predicted_Rating'] = predicted_ratings
    output_df.to_csv('unrated_hospitals_predictions.csv', index=False)
    print(f"\n  Predictions saved → unrated_hospitals_predictions.csv")

except FileNotFoundError:
    print("  [SKIP] not_yet_rated.csv not found in working directory.")

# ── 7. Done ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Pipeline execution complete.")
print("  Outputs: confusion_matrix.png | hospital_feature_importance.png")
print("           unrated_hospitals_predictions.csv")
print("=" * 65)
