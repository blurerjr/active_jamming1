import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Define the list of your 20 features (must match the order used during training!)
selected_features = [
    "tx_total_pkts",
    "tx_total_bytes",
    "tx_ucast_pkts",
    "tx_ucast_bytes",
    "tx_failures",
    "rx_data_pkts",
    "rx_ucast_pkts",
    "rx_data_bytes",
    "tx_data_pkts_retried",
    "tx_total_pkts_sent",
    "tx_pkts_retries",
    "tx_pkts_retry_exhausted",
    "rate_last_tx_pkt_min",
    "rate_last_tx_pkt_max",
    "per_antenna_rssi_last_rx_data_frame_1",
    "per_antenna_rssi_last_rx_data_frame_2",
    "per_antenna_avg_rssi_rx_data_frames_1",
    "per_antenna_avg_rssi_rx_data_frames_2",
    "sinr_per_antenna_1",
    "per_antenna_noise_floor_1"
]

# --- Load the trained model and label encoder ---
# Use st.cache_resource to cache the model and encoder
# This prevents reloading them every time the script reruns
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_jammer_detector_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'random_forest_jammer_detector_model.joblib' is in the repository.")
        return None

@st.cache_resource
def load_encoder():
     try:
         encoder = joblib.load('jammer_label_encoder.joblib')
         return encoder
     except FileNotFoundError:
         st.error("Label encoder file not found. Make sure 'jammer_label_encoder.joblib' is in the repository.")
         return None

model = load_model()
label_encoder = load_encoder()

# Check if model and encoder loaded successfully
if model is not None and label_encoder is not None:

    # --- Streamlit App Title and Description ---
    st.title("Jamming Attack Detector")
    st.write("Enter the feature values below to predict the type of network activity.")

    # --- Create Input Fields for Features ---
    st.header("Input Features")

    # You can create input fields for each of your 20 features
    # For simplicity, let's create number inputs.
    # You might need different input types based on the nature of your features (e.g., text, file uploader)
    input_data = {}
    cols = st.columns(2) # Use columns for better layout

    for i, feature in enumerate(selected_features):
        with cols[i % 2]: # Distribute inputs into two columns
            # You might want to set default values or hints based on your data's range
            input_data[feature] = st.number_input(f"{feature}", key=feature)

    # --- Prediction Button ---
    if st.button("Predict"):
        # --- Prepare Input Data for Prediction ---
        # Convert input data into a pandas DataFrame with the correct feature order
        # The model expects the input to be in the same format (column names and order) as the training data
        input_df = pd.DataFrame([input_data])

        # Ensure the order of columns in input_df matches the selected_features list
        input_df = input_df[selected_features]

        # --- Make Prediction ---
        # The model predicts numerical labels
        prediction_encoded = model.predict(input_df)

        # --- Inverse transform the prediction to get the original label ---
        # Use the loaded label_encoder
        predicted_label = label_encoder.inverse_transform(prediction_encoded)

        # --- Display Prediction ---
        st.header("Prediction")
        st.success(f"Predicted Activity: **{predicted_label[0]}**")

        # Optional: Provide more details about the prediction or the class
        # You could add conditional messages based on the predicted_label

else:
    st.warning("Model or Label Encoder failed to load. Please check the files in the repository.")
