# Naive Bayes heart disease classifier (sklearn) — quick & reliable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
data = pd.read_csv("heart.csv")

# Features and label (use whatever features you like)
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = data[features]
y = data['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale continuous features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example query: probability of target=1 for a sample patient
sample = {'age':55,'sex':1,'cp':3,'trestbps':140,'chol':250,'thalach':150,'exang':0,'oldpeak':1.0,'slope':2,'ca':0,'thal':2}
import numpy as np
sample_df = pd.DataFrame([sample])[features]
sample_scaled = scaler.transform(sample_df)
proba = model.predict_proba(sample_scaled)[0]   # [P(not disease), P(disease)]
print(f"\nP(no disease)={proba[0]:.3f}, P(disease)={proba[1]:.3f}")