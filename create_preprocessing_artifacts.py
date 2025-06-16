import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import os

print("Starting preprocessing artifact generation...")

# Create the 'preprocessing' directory if it doesn't exist
os.makedirs('preprocessing', exist_ok=True)

try:
    # This script assumes the CSV file is in the same directory
    df = pd.read_csv("anonymized_sample_fraud_txn.csv")
    print("Successfully loaded anonymized_sample_fraud_txn.csv")
except FileNotFoundError:
    print("Error: 'anonymized_sample_fraud_txn.csv' not found. Please place it in the project's root directory.")
    exit()

# --- Full Preprocessing Pipeline (to create the artifacts) ---
df["TXN_TIMESTAMP"] = pd.to_datetime(df["TXN_TIMESTAMP"], format="%d/%m/%Y %H:%M")
df["hour"] = df["TXN_TIMESTAMP"].dt.hour
df["day_of_week"] = df["TXN_TIMESTAMP"].dt.dayofweek
df["is_night_time"] = df["hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)

df['PAYER_VPA_PROVIDER'] = df['PAYER_VPA'].str.split('@').str[-1].fillna('UNKNOWN')
df['BENEFICIARY_VPA_PROVIDER'] = df['BENEFICIARY_VPA'].str.split('@').str[-1].fillna('UNKNOWN')

# Define all columns to be dropped *after* VPA creation
cols_to_drop = [
    'TXN_TIMESTAMP', 'TRANSACTION_ID', 'RRN', 'CARD_NUMBER', 'UPI_LITE_LRN', 
    'PAYER_ACCOUNT', 'BENEFICIARY_ACCOUNT', 'PAYER_VPA', 'BENEFICIARY_VPA',
    'PAYER_CODE', 'PAYMENT_INSTRUMENT'
]
df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')

# Handle NaNs in remaining object columns by filling with a placeholder
for col in ['BENEFICIARY_IFSC', 'LONGITUDE', 'LATITUDE', 'INITIATION_MODE', 'IP_ADDRESS']:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].fillna('UNKNOWN')

df_cleaned.drop_duplicates(inplace=True)

# Create a copy for the scam-type model before target encoding is applied
df_for_artifacts = df_cleaned.copy()

# --- Create and Save Target Encoding Artifacts ---
# This is based on the IS_FRAUD column
print("Generating Target Encoding map for BENEFICIARY_CODE...")
target_mean_map = df_for_artifacts.groupby('BENEFICIARY_CODE')['IS_FRAUD'].mean().to_dict()
global_mean = df_for_artifacts['IS_FRAUD'].mean()
target_encoding_artifact = {
    'target_mean_map': {str(k): v for k, v in target_mean_map.items()}, # Ensure keys are strings for JSON
    'global_mean': global_mean
}
with open('preprocessing/target_encoding_beneficiary_code.json', 'w') as f:
    json.dump(target_encoding_artifact, f)
print("Saved: Target Encoding map for BENEFICIARY_CODE")

# --- Create and Save Label Encoder Maps ---
print("Generating Label Encoding maps...")
all_label_encoder_maps = {}
label_encode_cols = ['TRN_STATUS', 'RESPONSE_CODE', 'INITIATION_MODE', 'TRANSACTION_TYPE', 'PAYER_VPA_PROVIDER', 'BENEFICIARY_VPA_PROVIDER']
for col in label_encode_cols:
    le = LabelEncoder()
    # Fit on all unique values, including 'UNKNOWN' if present
    le.fit(df_for_artifacts[col].astype(str))
    # Create a dictionary mapping from the string class to its integer encoding
    mapping = dict(zip(le.classes_, le.transform(le.classes_).astype(int).tolist()))
    all_label_encoder_maps[col] = mapping
    print(f"-> Generated map for {col}")

with open('preprocessing/label_encoder_maps.json', 'w') as f:
    json.dump(all_label_encoder_maps, f, indent=4)
print("Saved all label encoder maps to a single file.")

# --- Create and Save Frequency Encoder Maps ---
print("Generating Frequency Encoding maps...")
freq_cols = ['PAYER_IFSC', 'BENEFICIARY_IFSC', 'DEVICE_ID', 'IP_ADDRESS', 'LATITUDE', 'LONGITUDE']
for col in freq_cols:
    freq_map = df_for_artifacts[col].value_counts(normalize=True).to_dict()
    with open(f'preprocessing/freq_map_{col}.json', 'w') as f:
        json.dump(freq_map, f)
    print(f"-> Saved frequency map for {col}")

# --- Create and Save the Scam Type Map ---
print("Generating Scam Type map...")
# This map is manually defined based on your model's training labels
scam_type_map = {
    "0": 'Request Money Scam', 
    "1": 'Fake Customer Care', 
    "2": 'Fake Payment Link', 
    "-1": 'Not Fraud'
}
with open('preprocessing/scam_type_map.json', 'w') as f:
    json.dump(scam_type_map, f, indent=4)
print("Saved: Scam Type map")

print("\nAll preprocessing artifacts have been generated successfully in the 'preprocessing/' directory!")