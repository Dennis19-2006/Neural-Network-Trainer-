import streamlit as st
import numpy as np
import pandas as pd
import time

# -----------------------------
# Neural network logic (yours)
# -----------------------------
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_network(X, Y, lr, epochs, hidden_neurons):
    np.random.seed(42)

    num_inputs = X.shape[1]
    num_outputs = 1

    W_hidden = np.random.randn(num_inputs, hidden_neurons) * np.sqrt(2.0 / num_inputs)
    W_output = np.random.randn(hidden_neurons, num_outputs) * np.sqrt(2.0 / hidden_neurons)
    B_hidden = np.zeros((1, hidden_neurons))
    B_output = np.zeros((1, num_outputs))

    error_history = []
    activations_snapshots = []

    progress = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        total_error = 0
        indices = np.random.permutation(len(X))

        for idx in indices:
            x = X[idx:idx+1]
            y = Y[idx:idx+1]

            # Forward
            Z_hidden = np.dot(x, W_hidden) + B_hidden
            A_hidden = sigmoid(Z_hidden)
            Z_output = np.dot(A_hidden, W_output) + B_output
            A_output = sigmoid(Z_output)

            error = y - A_output
            total_error += error[0, 0] ** 2

            # Backprop
            dA_output = -2 * error
            dZ_output = dA_output * sigmoid_derivative(A_output)
            dW_output = np.dot(A_hidden.T, dZ_output)
            dB_output = dZ_output

            dA_hidden = np.dot(dZ_output, W_output.T)
            dZ_hidden = dA_hidden * sigmoid_derivative(A_hidden)
            dW_hidden = np.dot(x.T, dZ_hidden)
            dB_hidden = dZ_hidden

            # Update
            W_output -= lr * np.clip(dW_output, -1, 1)
            B_output -= lr * np.clip(dB_output, -1, 1)
            W_hidden -= lr * np.clip(dW_hidden, -1, 1)
            B_hidden -= lr * np.clip(dB_hidden, -1, 1)

        avg_error = total_error / len(X)
        error_history.append(avg_error)

        activations_snapshots.append({
            "hidden": A_hidden[0].tolist(),
            "output": float(A_output[0, 0])
        })

        progress.progress((epoch + 1) / epochs)
        status.text(f"Epoch {epoch+1}/{epochs}  |  Error: {avg_error:.6f}")
        time.sleep(0.01)

    return {
        "W_hidden": W_hidden,
        "W_output": W_output,
        "B_hidden": B_hidden,
        "B_output": B_output,
        "errors": error_history,
        "activations": activations_snapshots
    }

def predict(x, model):
    Z_hidden = np.dot(x, model["W_hidden"]) + model["B_hidden"]
    A_hidden = sigmoid(Z_hidden)
    Z_output = np.dot(A_hidden, model["W_output"]) + model["B_output"]
    return sigmoid(Z_output)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Neural Network Trainer", layout="wide")
st.title("🧠 Neural Network Trainer (Streamlit)")

# Create sample data if no file uploaded
sample_data = None
try:
    sample_data = pd.read_csv("frontend/sample_data.csv")
except:
    # Generate synthetic data if sample_data.csv doesn't exist
    np.random.seed(42)
    X_sample = np.random.randn(100, 2)
    y_sample = (X_sample[:, 0] + X_sample[:, 1] > 0).astype(int)
    sample_data = pd.DataFrame(np.column_stack([X_sample, y_sample]), columns=["Feature1", "Feature2", "Target"])

tab1, tab2 = st.tabs(["📊 Sample Data", "📤 Upload Data"])

with tab1:
    st.subheader("Sample Dataset")
    st.dataframe(sample_data.head(10))
    use_sample = st.button("Use Sample Data", key="sample_btn")
    if use_sample:
        st.session_state["data"] = sample_data

with tab2:
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10))
        use_upload = st.button("Use Uploaded Data", key="upload_btn")
        if use_upload:
            st.session_state["data"] = df

# Process the selected data
if "data" in st.session_state:
    df = st.session_state["data"]
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    X = df.iloc[:, :-1].values.astype(np.float32)
    Y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

    st.sidebar.header("⚙️ Training Settings")
    lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
    epochs = st.sidebar.slider("Epochs", 10, 2000, 500)
    hidden_neurons = st.sidebar.slider("Hidden Neurons", 1, 50, 10)

    if st.button("🚀 Train Model"):
        model = train_network(X, Y, lr, epochs, hidden_neurons)
        st.session_state["model"] = model
        st.session_state["num_features"] = X.shape[1]

        st.success("✨ Training complete!")

        st.subheader("📉 Training Error")
        st.line_chart(model["errors"])

        st.subheader("🔍 Final Neuron Activations (last sample)")
        st.json(model["activations"][-1])

# -----------------------------
# Prediction
# -----------------------------
if "model" in st.session_state:
    st.subheader("🔮 Predict New Input")

    inputs = []
    for i in range(st.session_state["num_features"]):
        val = st.number_input(f"Input {i+1}", value=0.0)
        inputs.append(val)

    if st.button("Predict"):
        x = np.array(inputs).reshape(1, -1)
        pred = predict(x, st.session_state["model"])
        st.metric("Prediction", float(pred[0, 0]))
