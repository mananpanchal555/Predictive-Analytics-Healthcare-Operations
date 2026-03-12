# Predictive Analytics for Healthcare Operations

## Executive Summary
This project applies Machine Learning classification algorithms to a dataset of 3,000+ medical facilities to predict the "Hospital Overall Rating" (1-5 stars). By analyzing 94 distinct clinical and operational variables, the model isolates the highest-leverage metrics driving positive patient outcomes and regulatory compliance.

## Business Value & Strategic Outcomes
* **Predictive Quality Assurance:** Engineered a Random Forest classification model to predict hospital ratings, allowing administrators to forecast compliance scores before official regulatory audits.
* **Operational Resource Allocation:** Extracted feature importance (Gini Impurity) to prove that "Patient Experience" and "Readmission Rates" mathematically outweigh standard mortality metrics in driving 5-star ratings, guiding budget allocation for hospital management.
* **Data-Driven Healthcare:** Transformed siloed clinical data (safety, timeliness of care, imaging efficiency) into a unified predictive pipeline for executive decision-making.

## Key Visual Insights
1. **Feature Importance Analysis:** (`hospital_feature_importance.png`) Identifies the specific clinical metrics that have the highest mathematical probability of influencing a hospital's overall rating.
2. **Readmission vs. Patient Experience Correlation:** (`readmission_patient_exp.png`) Maps the intersection of clinical failure (readmission) and front-end service (experience).

## Technical Stack
* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Random Forest, Logistic Regression, Feature Encoding)
* **Data Manipulation & Visualization:** Pandas, NumPy, Seaborn
