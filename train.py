import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Heart_disease_cleveland_new.csv")

# Display basic information
print("Dataset Overview:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualizing class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='coolwarm')
plt.title("Class Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Split features and target
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Random Forest model
n_estimators = 200
rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

# Training with manual progress tracking
print("\nTraining Random Forest Model...")
for i in tqdm(range(n_estimators), desc="Training Trees", unit="tree"):
    rf_model.n_estimators = i + 1
    rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Model Evaluation
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"\n✅ Random Forest Accuracy: {rf_accuracy:.4f}")

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the trained model and scaler
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler_rf.pkl")

print("\n✅ Model and scaler saved successfully!")
