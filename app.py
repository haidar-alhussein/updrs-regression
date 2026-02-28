import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Layer

# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(1, activation='tanh')
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = self.dense(inputs)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector


# Load Model + Scaler + Features
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "parkinsons_bilstm_attention.h5",
        custom_objects={"Attention": Attention},
        compile=False
    )

    scaler = joblib.load("scaler.pkl")
    selected_features = joblib.load("selected_features.pkl")

    return model, scaler, selected_features


model, scaler, selected_features = load_artifacts()


# App UI
st.title("Parkinson's UPDRS Prediction (BiLSTM + Attention)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    if st.button("Predict UPDRS"):

        if data.shape[0] != 18:
            st.error("CSV must contain exactly 18 rows (one patient sequence).")
            st.stop()

        missing_features = [f for f in selected_features if f not in data.columns]

        if len(missing_features) > 0:
            st.error(f"Missing required columns: {missing_features}")
            st.stop()

        X_input = data[selected_features].values  # shape (18, 13)

        d = len(selected_features)
        X_flat = scaler.transform(X_input.reshape(-1, d))

        X_seq = X_flat.reshape(1, 18, d)

        prediction = model.predict(X_seq)

        st.success(f"Predicted total_UPDRS: {float(prediction[0][0]):.2f}")


st.divider()
st.markdown("""
---
**Prepared by:** Haidar Al-Hussein  
**Model:** BiLSTM + Attention Network  
**Application:** Early Parkinson’s Disease Screening  
""")