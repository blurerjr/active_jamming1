import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

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
@st.cache_data
def load_and_preprocess_data():
    # --- ACTUAL RAW URLs to your CSV files in GitHub ---
    base_url = 'https://raw.githubusercontent.com/panitsasi/JamShield-Dataset/refs/heads/main/data/'
    file_names = [
        'constant_jammer_gaussian_10db.csv',
        'constant_jammer_gaussian_20db.csv',
        'constant_jammer_gaussian_25db.csv',
        'constant_jammer_gaussian_dynamic_gain.csv',
        'constant_jammer_pulse_20db.csv',
        'constant_jammer_triangle_20db.csv',
        'data_benign_1.csv',
        'data_benign_2.csv',
        'data_benign_3.csv',
        'data_benign_4.csv',
        'random_jammer_cos_dynamic_gain.csv',
        'random_jammer_gaussian_NLOS.csv',
        'random_jammer_pulse_dynamic_gain.csv',
        'random_jammer_saw_tooth_dynamic_gain.csv',
        'random_jammer_triangle_dynamic_gain.csv',
        'reactive_jammer_cos_NLOS.csv',
        'reactive_jammer_gaussian_LOS.csv',
        'reactive_jammer_gaussian_additional_end_devices.csv',
        'reactive_jammer_square_NLOS.csv',
        'reactive_jammer_triangle_NLOS.csv',
    ]

    file_urls = [base_url + name for name in file_names]

    all_data = []
    for url in file_urls:
        try:
            df = pd.read_csv(url)
            label = url.split('/')[-1].replace('.csv', '')
            if not df.empty:
                df['label'] = label
                all_data.append(df)
        except Exception as e:
            st.error(f"Error loading data from {url}: {e}")
            return None, None, None, None

    if not all_data:
        st.error("No data loaded from the provided URLs.")
        return None, None, None, None

    combined_df = pd.concat(all_data, ignore_index=True)

    X = combined_df[selected_features]
    y = combined_df['label']

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=selected_features)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder, imputer

# Load and preprocess the data
data_load_state = st.info("Loading data and preprocessing...")
try:
    X, y_encoded, label_encoder, imputer = load_and_preprocess_data()
    data_load_state.empty()
except Exception as e:
    data_load_state.error(f"An error occurred during data loading and preprocessing: {e}")
    X, y_encoded, label_encoder, imputer = None, None, None, None


# Check if data loaded successfully
if X is not None and y_encoded is not None and label_encoder is not None and imputer is not None:

    st.title("🤖 Jamming Attack Detector")
    st.write("Use the sidebar to enter the feature values and predict the type of activity.")

    # --- Optional: Display Raw Data ---
    with st.expander('Show Raw Data (from GitHub)'):
        st.write("This is the combined raw data loaded from your GitHub repository.")
        st.dataframe(X.head())
        st.write("Label counts:")
        st.write(pd.Series(label_encoder.inverse_transform(y_encoded)).value_counts())


    # --- Model Training ---
    @st.cache_resource
    def train_model(features, target):
        st.info("Training the Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(features, target)
        st.success("Model training complete!")
        return model

    # Train the model
    rf_model = train_model(X, y_encoded)

    # --- Feature Importance ---
    st.header("Feature Importance")
    st.write("Shows which features the model considers most important for classification.")

    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    st.dataframe(feature_importance_df, hide_index=True)

    st.subheader("Feature Importance Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax)
    ax.set_title('Feature Importance from Random Forest')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig)


    # --- Input Features Section (Moved to Sidebar) ---
    with st.sidebar:
        st.header("Input Features")
        st.write("Adjust the values below to get a prediction.")

        input_data = {}
        # Create a dictionary to hold default values.
        # You could potentially calculate min/max/median from your training data (X)
        # to set more meaningful default values or slider ranges.
        default_values = {feature: 0.0 for feature in selected_features} # Using 0.0 as a simple default

        # Using number_input for flexibility, as ranges are not explicitly known
        # If you know the typical range for a feature, st.slider is often better
        for feature in selected_features:
             # Example of using st.slider if you knew the range:
             # if feature == 'tx_total_pkts':
             #     input_data[feature] = st.slider(f"{feature}", 0, 10000, default_values[feature])
             # else:
             #     input_data[feature] = st.number_input(f"{feature}", value=default_values[feature], key=f"sidebar_input_{feature}")

             # Using number_input for all features
             input_data[feature] = st.number_input(f"{feature}", value=default_values[feature], key=f"sidebar_input_{feature}")


    # --- Prediction Button (Can be in main area or sidebar) ---
    # Placing the button in the main area for visibility after inputs are set
    st.header("Get Prediction")
    if st.button("Predict Activity"):
        # --- Prepare Input Data for Prediction ---
        input_df = pd.DataFrame([input_data])
        input_df = input_df[selected_features]

        # Apply the same imputation used during training
        input_imputed = imputer.transform(input_df)
        input_processed_df = pd.DataFrame(input_imputed, columns=selected_features)

        # --- Make Prediction ---
        prediction_encoded = rf_model.predict(input_processed_df)
        predicted_label = label_encoder.inverse_transform(prediction_encoded)

        # --- Display Prediction ---
        st.subheader("Prediction Result")
        st.success(f"Predicted Activity: **{predicted_label[0]}**")


else:
    st.error("App could not load data or train the model. Please check the data URLs and file format.")

