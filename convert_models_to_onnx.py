import pandas as pd
import numpy as np
import pickle
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
# We need the specific FloatTensorType for onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxToolsFloatTensorType
import os

print("Starting model conversion to ONNX format...")

# --- Define the precise feature columns for each model ---
CLF_MODEL_COLUMNS = [
    'TRN_STATUS', 'RESPONSE_CODE', 'BENEFICIARY_CODE', 'INITIATION_MODE',
    'TRANSACTION_TYPE', 'hour', 'day_of_week', 'is_night_time',
    'PAYER_VPA_PROVIDER', 'BENEFICIARY_VPA_PROVIDER', 'AMOUNT_log',
    'PAYER_IFSC_freq', 'BENEFICIARY_IFSC_freq', 'DEVICE_ID_freq', 'IP_FREQ',
    'LATITUDE_freq_enc', 'LONGITUDE_freq_enc'
]
XGB_MODEL_COLUMNS = CLF_MODEL_COLUMNS + ['BENEFICIARY_CODE_TE']

# --- Load original models ---
try:
    with open('models/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/clf.pkl', 'rb') as f:
        clf_type_model = pickle.load(f)
    print("Original models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Create 'onnx_models' directory if it doesn't exist
os.makedirs('onnx_models', exist_ok=True)


# --- Convert clf_type_model (Random Forest) (This has been working correctly) ---
print("\nConverting Random Forest model (clf.pkl)...")
num_clf_features = len(CLF_MODEL_COLUMNS)
initial_type_clf = [('float_input', FloatTensorType([None, num_clf_features]))]
onnx_clf = convert_sklearn(clf_type_model, initial_types=initial_type_clf, target_opset=12)
onnx_clf_path = 'onnx_models/clf_type_model.onnx'
with open(onnx_clf_path, "wb") as f:
    f.write(onnx_clf.SerializeToString())
print(f"Random Forest model saved to {onnx_clf_path}")


# --- Convert xgb_model (XGBoost) by MANUALLY RENAMING FEATURES ---
print("\nConverting XGBoost model (xgb_model.pkl)...")

# ** THE DEFINITIVE FIX: Manually overwrite the feature names **
# The onnxmltools converter cannot handle string feature names. We will
# replace them with the generic 'f0', 'f1', 'f2'... that it expects.
booster = xgb_model.get_booster()
num_xgb_features = len(XGB_MODEL_COLUMNS)
generic_feature_names = [f'f{i}' for i in range(num_xgb_features)]
booster.feature_names = generic_feature_names
print(f"Manually renamed model features to generic names: {generic_feature_names[:5]}...")

# Now, we define the input type for the converter
initial_type_xgb = [('float_input', OnnxToolsFloatTensorType([None, num_xgb_features]))]

# And convert the booster object, which now has compatible feature names
onnx_xgb = onnxmltools.convert.convert_xgboost(
    booster,
    initial_types=initial_type_xgb,
    target_opset=12
)

onnx_xgb_path = 'onnx_models/xgb_model.onnx'
with open(onnx_xgb_path, "wb") as f:
    f.write(onnx_xgb.SerializeToString())
print(f"XGBoost model saved to {onnx_xgb_path}")

print("\nConversion complete! The 'onnx_models' directory is ready for web deployment.")