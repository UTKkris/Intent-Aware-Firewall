"""
Main script for training the Intent-Aware Firewall model using Random Forest.
(High-RAM Edition for 64GB PC - No Sampling)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # <-- Changed to Random Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import features as fe
import joblib
import json
import warnings
import sys

# --- Silence Harmless Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Step 1: Load Full *CLEAN* Dataset ---
print("Step 1: Loading full, clean dataset from Parquet file...")
parquet_file = r'../data/cic_ids_2018_full_harmonized_CLEAN.parquet' 

try:
    df = pd.read_parquet(parquet_file)
except FileNotFoundError:
    print(f"Error: Parquet file not found at {parquet_file}")
    print("Please run 'convert_to_parquet.py' first.")
    sys.exit(1)
    
print(f"Loaded {df.shape[0]} total clean rows.")
print(f"Label distribution:\n{df['Label'].value_counts(normalize=True)}")

# --- Step 2: Separate Features (X) and Target (y) ---
print("\nStep 2: Separating features and target...")
X = df.drop('Label', axis=1)
y = df['Label']
print("X and y created.")

# --- Step 3: Train-Test Split ---
print("\nStep 3: Splitting data into train/test sets...")
# This step requires significant RAM (approx 25GB+ for 16M rows)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Step 4: Add STATEFUL Features (AFTER SPLIT) ---
print("\nStep 4: Fitting stateful features on training data...")
size_col = 'Pkt Len Mean' 
if 'Packet Length' in X_train.columns:
    size_col = 'Packet Length'
    
X_train, train_mean, train_std = fe.add_packet_size_flags(X_train.copy(), size_col)
X_test, _, _ = fe.add_packet_size_flags(X_test.copy(), size_col, mean=train_mean, std=train_std)
print("Stateful features added.")

# --- Step 5: Final Prep for ML ---
print("\nStep 5: Final data preparation for modeling...")
drop_cols = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']
X_train = X_train.drop(columns=drop_cols, errors='ignore')
X_test = X_test.drop(columns=drop_cols, errors='ignore')

# Align columns
train_cols = X_train.columns
test_cols = X_test.columns
shared_cols = list(train_cols.intersection(test_cols))
X_train = X_train[shared_cols]
X_test = X_test[shared_cols]
print(f"Aligned columns, using {len(shared_cols)} features.")

# --- Step 6: Scale Features (AFTER SPLIT) ---
print("\nStep 6: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling complete.")

# --- Step 7: Train, Predict, Evaluate ---
print("\nStep 7: Training Random Forest model on ALL 16M+ clean data...")
print("NOTE: Random Forest is slower than LightGBM. This may take 20-40 minutes.")

# Initialize Random Forest Classifier
clf = RandomForestClassifier(
    n_estimators=100,        # Number of trees (100 is standard)
    class_weight='balanced', # Handles imbalance (Benign vs Attack)
    max_depth=30,            # Limit depth to prevent RAM explosion
    n_jobs=-1,               # Use ALL CPU cores (Critical for speed)
    random_state=42,
    verbose=1                # Show progress log
)

# Train the model
clf.fit(X_train_scaled, y_train)
print("Model training complete.")

print("\nEvaluating model...")
y_pred = clf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=['Benign', 'Attack']))

# --- Step 8: Feature Importance ---
print("\nStep 8: Calculating feature importances...")
importances = clf.feature_importances_
feat_importance_df = pd.DataFrame({'Feature': shared_cols, 'Importance': importances})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)
print("Top 10 important features:")
print(feat_importance_df.head(10))

# --- Step 9: Save the Production Model & Assets ---
print("\nStep 9: Saving model and supporting assets...")
# Save as 'random_forest_model.joblib' to avoid overwriting the other one
joblib.dump(clf, 'random_forest_model.joblib')

# Note: We don't strictly need to save the scaler/stats again if they are 
# identical to the LightGBM run, but it's safer to save them to be sure.
joblib.dump(scaler, 'data_scaler_rf.joblib') 

size_stats = {'mean': train_mean, 'std': train_std}
with open('size_stats_rf.json', 'w') as f:
    json.dump(size_stats, f)
with open('shared_features_rf.json', 'w') as f:
    json.dump(shared_cols, f)
    
print("Model and assets saved successfully!")