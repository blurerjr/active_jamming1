import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib # Still useful for potential future saving/loading, or if needed by other libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Needed for splitting if you want to evaluate within the app
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer # For handling missing values consistently

# --- Configuration ---
st.set_page_config(page_title="Jamming Attack Detector", layout="wide")

# Define the list of your 20 selected features
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

# --- Data Loading and Preprocessing ---
# Use st.cache_data to cache the data loading and preprocessing steps
# This prevents reloading and reprocessing the data every time the script reruns
@st.cache_data
def load_and_preprocess_data():
    # --- IMPORTANT: Replace with the actual URLs to your raw CSV files in GitHub ---
    # Assuming your files are in a 'data' folder in the root of your repo
    # Example: https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO_NAME/main/data/constant_jammer_gaussian_10db.csv
    # You need to list ALL your CSV file URLs here
    file_urls = [
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/constant_jammer_gaussian_10db.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/constant_jammer_gaussian_20db.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/constant_jammer_gaussian_25db.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/constant_jammer_gaussian_dynamic_gain.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/constant_jammer_pulse_20db.csv', # Add all your files
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/constant_jammer_triangle_20db.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/data_benign_1.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/data_benign_2.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/data_benign_3.csv',
        'hhttps://github.com/panitsasi/JamShield-Dataset/blob/main/data/data_benign_4.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/random_jammer_cos_dynamic_gain.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/random_jammer_gaussian_NLOS.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/random_jammer_pulse_dynamic_gain.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/random_jammer_saw_tooth_dynamic_gain.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/random_jammer_triangle_dynamic_gain.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/reactive_jammer_cos_NLOS.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/reactive_jammer_gaussian_LOS.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/reactive_jammer_gaussian_additional_en.csv', # Check this name
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/data/reactive_jammer_square_NLOS.csv',
        'https://github.com/panitsasi/JamShield-Dataset/blob/main/dataa/reactive_jammer_triangle_NLOS.csv', # Check this name
        # Add any other CSV files from your dataset
    ]

    all_data = []
    for url in file_urls:
        try:
            df = pd.read_csv(url)
            # Extract label from URL (assuming filename is the last part of the URL)
            label = url.split('/')[-1].replace('.csv', '')
            if not df.empty:
                df['label'] = label
                all_data.append(df)
        except Exception as e:
            st.error(f"Error loading data from {url}: {e}")
            return None, None, None # Return None if data loading fails

    if not all_data:
        st.error("No data loaded from the provided URLs.")
        return None, None, None

    combined_df = pd.concat(all_data, ignore_index=True)

    # Separate features and target
    X = combined_df[selected_features] # Use only the selected features
    y = combined_df['label']

    # Handle missing values (Impute with median, fitted on the entire dataset)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=selected_features) # Convert back to DataFrame

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder, imputer # Also return the imputer for consistent preprocessing

# Load and preprocess the data
data_load_state = st.info("Loading data...")
X, y_encoded, label_encoder, imputer = load_and_preprocess_data()
data_load_state.empty() # Clear the loading message

# Check if data loaded successfully
if X is not None and y_encoded is not None and label_encoder is not None and imputer is not None:

    st.title("🤖 Jamming Attack Detector")
    st.write("Upload your network data or use the input fields to predict the type of activity.")

    # --- Optional: Display Raw Data (can be removed for a cleaner app) ---
    with st.expander('Show Raw Data (from GitHub)'):
        st.write("This is the combined raw data loaded from your GitHub repository.")
        st.dataframe(X.head()) # Displaying head of features after imputation
        st.write("Label counts:")
        st.write(pd.Series(label_encoder.inverse_transform(y_encoded)).value_counts())


    # --- Model Training ---
    # Use st.cache_resource to cache the trained model
    # This prevents retraining every time the script reruns (but still trains on initial load)
    @st.cache_resource
    def train_model(features, target):
        st.info("Training the Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(features, target)
        st.success("Model training complete!")
        return model

    # Train the model
    rf_model = train_model(X, y_encoded)


    # --- Input Features Section ---
    st.header("Input Features for Prediction")
    st.write("Enter the values for the 20 selected features:")

    input_data = {}
    # Create a dictionary to hold default values or ranges if you know them
    # For now, using 0.0 as default for number inputs
    default_values = {feature: 0.0 for feature in selected_features}

    cols = st.columns(2) # Use columns for better layout

    for i, feature in enumerate(selected_features):
        with cols[i % 2]: # Distribute inputs into two columns
            # You might want to use st.slider for features with known ranges
            # Example: if 'tx_total_pkts' ranges from 0 to 10000
            # input_data[feature] = st.slider(f"{feature}", 0, 10000, default_values[feature])
            input_data[feature] = st.number_input(f"{feature}", value=default_values[feature], key=f"input_{feature}")


    # --- Prediction Button ---
    if st.button("Predict Activity"):
        # --- Prepare Input Data for Prediction ---
        # Convert input data into a pandas DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure the order of columns in input_df matches the selected_features list
        # This is crucial for the model prediction
        input_df = input_df[selected_features]

        # Apply the same imputation used during training to the input data
        # Use the imputer fitted on the training data
        input_imputed = imputer.transform(input_df)
        input_processed_df = pd.DataFrame(input_imputed, columns=selected_features)


        # --- Make Prediction ---
        # The model predicts numerical labels
        prediction_encoded = rf_model.predict(input_processed_df)

        # --- Inverse transform the prediction to get the original label ---
        # Use the loaded label_encoder
        predicted_label = label_encoder.inverse_transform(prediction_encoded)

        # --- Display Prediction ---
        st.subheader("Prediction Result")
        st.success(f"Predicted Activity: **{predicted_label[0]}**")

        # Optional: Display prediction probabilities
        # prediction_proba = rf_model.predict_proba(input_processed_df)
        # proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
        # st.write("Prediction Probabilities:")
        # st.dataframe(proba_df)


else:
    st.error("App could not load data or train the model. Please check the data URLs and file format.")

