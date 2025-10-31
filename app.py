import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🚗 Vehicle Classification using SVM")
st.write("Upload a dataset to classify vehicles (Bus, Car, or Van).")

# --- Provide sample CSV for users ---
st.info("Download this sample **test.csv** file and upload it below to see the model in action.")
with open("test.csv", "rb") as f:
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )

# --- Initialize session state ---
if "data" not in st.session_state:
    st.session_state.data = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- File uploader with dynamic key ---
uploaded_file = st.file_uploader(
    "📂 Upload CSV file", 
    type=["csv"], 
    key=f"file_upload_{st.session_state.uploader_key}"
)

# --- Load uploaded file ---
if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### Uploaded Data Preview")
    st.write(st.session_state.data.head())

# --- Buttons ---
col1, col2 = st.columns(2)
with col1:
    submit = st.button("Submit for Prediction", key="submit_button")
with col2:
    clear = st.button("🧹 Clear Data", key="clear_button")

# --- Clear button logic ---
if clear:
    st.session_state.data = None
    st.session_state.predictions = None
    st.session_state.uploader_key += 1  # change uploader key to reset widget
    st.rerun()

# Submit button logic
if submit:
    if st.session_state.data is not None:
        data = st.session_state.data.copy()

        # Clean data
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)

        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Predict
        predictions = model.predict(data_scaled)
        st.session_state.predictions = predictions

        # Display results
        st.write("### Model Predictions:")
        st.write(predictions[:5])

        # Combine with original data
        results = data.copy()
        results["Predicted Class"] = predictions
        st.write("### 📈 Predicted Output Preview")
        st.write(results.head())

        # Download predictions
        st.download_button(
            "⬇️ Download Results",
            results.to_csv(index=False),
            "predictions.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Please upload a CSV file before submitting.")
