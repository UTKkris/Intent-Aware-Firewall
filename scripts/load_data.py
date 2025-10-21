import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load first 100000 rows from each CSV in directory
all_files = glob.glob(r'../data/Processed Traffic Data for ML Algorithms/*.csv')
df_list = [pd.read_csv(f, low_memory=False, nrows=100000) for f in all_files]
df = pd.concat(df_list, axis=0, ignore_index=True)

# Step 2: Robust Timestamp parsing with error coercion
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
print(f"{df['Timestamp'].isnull().sum()} timestamps could not be parsed and are set as NaT.")

# Step 3: Drop rows with missing/unparseable timestamps
df = df[df['Timestamp'].notnull()]
print(f"Remaining rows after dropping bad timestamps: {df.shape[0]}")

# Step 4: Extract time-based features (hour, day of week)
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# Step 5: Drop unused columns like 'Flow ID', 'Src IP', etc. for simplicity
drop_cols = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']
df.drop(columns=drop_cols, inplace=True)

# Step 6: Convert core numeric columns to numeric dtype (add your full list here)
numeric_cols = df.select_dtypes(include=['object']).columns.drop('Label')
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 7: Drop rows with NaN in numeric columns
df.dropna(subset=numeric_cols, inplace=True)

# Step 8: Split features and target
X = df.drop('Label', axis=1)
y = df['Label']

# Step 9: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 10: Replace inf/-inf and NaN with 0 in train/test
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Step 11: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 12: Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Step 13: Predict and evaluate
y_pred = clf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Step 14: Feature importance
importances = clf.feature_importances_
feat_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)
print("Top 10 important features:")
print(feat_importance_df.head(10))

# Step 15: Prepare full dataset for cross-validation
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_scaled = scaler.fit_transform(X)

# Step 16: Cross-validation accuracy
clf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(clf_cv, X_scaled, y, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())