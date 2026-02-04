import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from interpreter_bot import InterpreterBot, create_interpreter_bot

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

# Initialize interpreter bot
if "interpreter_bot" not in st.session_state:
    st.session_state["interpreter_bot"] = create_interpreter_bot()

def create_animated_training_neurons(model, X, Y):
    """Create animation showing neuron activations during prediction"""
    
    # Get a few sample predictions to animate
    num_samples = min(5, len(X))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    if num_samples == 1:
        axes = [axes]
    
    for sample_idx in range(num_samples):
        x = X[sample_idx:sample_idx+1]
        
        # Forward pass
        Z_hidden = np.dot(x, model["W_hidden"]) + model["B_hidden"]
        A_hidden = sigmoid(Z_hidden)[0]
        Z_output = np.dot(A_hidden.reshape(1, -1), model["W_output"]) + model["B_output"]
        A_output = sigmoid(Z_output)[0, 0]
        
        ax = axes[sample_idx]
        
        # Create simple visualization
        num_inputs = x.shape[1]
        num_hidden = len(A_hidden)
        
        y_pos_input = np.linspace(0.8, 0.2, num_inputs)
        y_pos_hidden = np.linspace(0.8, 0.2, num_hidden)
        
        # Plot input neurons
        for i, val in enumerate(x[0]):
            color_val = min(abs(val), 1)
            ax.scatter(0.1, y_pos_input[i], s=500, c=[[color_val, 0.5, 1-color_val]], 
                      edgecolors='black', linewidth=2, zorder=3)
            ax.text(0.05, y_pos_input[i], f"{val:.2f}", ha='right', va='center', fontsize=8)
        
        # Plot hidden neurons with glow effect
        max_hidden = np.max(A_hidden) if len(A_hidden) > 0 else 1
        for i, activation in enumerate(A_hidden):
            # Glow effect
            glow_alpha = activation * 0.3
            circle = Circle((0.5, y_pos_hidden[i]), 0.08, 
                           color=(activation, 0.3, 1-activation), alpha=glow_alpha, zorder=1)
            ax.add_patch(circle)
            
            # Main neuron
            ax.scatter(0.5, y_pos_hidden[i], s=400, 
                      c=[[activation, 0.3, 1-activation]], 
                      edgecolors='black', linewidth=2, zorder=3)
            ax.text(0.5, y_pos_hidden[i], f"{activation:.2f}", 
                   ha='center', va='center', fontsize=7, color='white', weight='bold')
        
        # Plot output neuron
        ax.scatter(0.9, 0.5, s=600, c=[[A_output, 0.7, 1-A_output]], 
                  edgecolors='gold', linewidth=3, zorder=3)
        ax.text(0.95, 0.5, f"{A_output:.3f}", ha='left', va='center', fontsize=9, weight='bold')
        
        # Draw connections (optimal path stronger)
        for i in range(num_inputs):
            for j in range(num_hidden):
                weight = model["W_hidden"][i, j]
                alpha = min(abs(weight) * 0.5, 0.5)
                ax.plot([0.15, 0.45], [y_pos_input[i], y_pos_hidden[j]], 
                       'gray', alpha=alpha, linewidth=1, zorder=0)
        
        # Highlight optimal path
        max_input_idx = np.argmax(np.abs(x[0]))
        max_hidden_idx = np.argmax(A_hidden)
        
        ax.plot([0.15, 0.45], [y_pos_input[max_input_idx], y_pos_hidden[max_hidden_idx]], 
               'yellow', linewidth=3, zorder=2)
        ax.plot([0.55, 0.85], [y_pos_hidden[max_hidden_idx], 0.5], 
               'yellow', linewidth=3, zorder=2)
        
        for j in range(num_hidden):
            ax.plot([0.55, 0.85], [y_pos_hidden[j], 0.5], 
                   'lightgray', alpha=0.2, linewidth=1, zorder=0)
        
        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(0.1, 0.9)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Sample {sample_idx+1}\nActual: {Y[sample_idx, 0]:.3f}", 
                    fontsize=10, weight='bold')
    
    plt.tight_layout()
    return fig

# =====================================================================
# HELPER FUNCTION: Generate Full Dataset Assessment
# =====================================================================
def generate_dataset_assessment(df, bot):
    """Generate comprehensive assessment of dataset"""
    st.markdown("---")
    st.markdown("### 📊 Dataset Assessment Report")
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Target Type", "Continuous" if df.iloc[:, -1].dtype in ['float64', 'float32'] else "Categorical")
    with col4:
        st.metric("Data Quality", f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
    
    # Feature analysis
    with st.expander("🔍 Feature Analysis", expanded=True):
        feature_cols = df.columns[:-1]
        for col in feature_cols:
            col_data = df[col].describe()
            st.write(f"**{col}**: μ={col_data['mean']:.4f}, σ={col_data['std']:.4f}, range=[{col_data['min']:.2f}, {col_data['max']:.2f}]")
    
    # Target analysis
    with st.expander("🎯 Target Distribution", expanded=True):
        target_col = df.columns[-1]
        st.write(f"**{target_col} Statistics:**")
        st.write(df[target_col].describe())
        
        if len(df[target_col].unique()) <= 10:
            st.write("**Value Counts:**")
            st.write(df[target_col].value_counts())
    
    st.markdown("---")

# =====================================================================
# HELPER FUNCTION: Generate Training Assessment
# =====================================================================
def generate_training_assessment(model, X, Y, bot):
    """Generate comprehensive assessment after training"""
    st.markdown("---")
    st.markdown("### 📈 Training Assessment Report")
    
    # Make predictions on training data
    predictions = predict(X, model)
    predictions_flat = predictions.flatten()
    Y_flat = Y.flatten()
    
    # Calculate detailed metrics
    mae = np.mean(np.abs(predictions_flat - Y_flat))
    mse = np.mean((predictions_flat - Y_flat) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate true confidence metrics
    mean_pred = np.mean(predictions_flat)
    ss_tot = np.sum((Y_flat - mean_pred) ** 2)
    ss_res = np.sum((Y_flat - predictions_flat) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate prediction spread (how confident/concentrated)
    pred_std = np.std(predictions_flat)
    actual_std = np.std(Y_flat)
    
    # Calculate error distribution
    errors = np.abs(predictions_flat - Y_flat)
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    within_1_std = np.sum(errors <= error_mean + error_std) / len(errors) * 100
    within_2_std = np.sum(errors <= error_mean + 2*error_std) / len(errors) * 100
    
    # Calibration: how well predictions match actual distribution
    calibration_error = abs(np.mean(predictions_flat) - np.mean(Y_flat))
    
    # Real confidence score (inverse of error)
    confidence_score = max(0, 1 - mae)
    confidence_pct = confidence_score * 100
    
    # Display confidence metrics
    st.markdown("### 🎯 Real Confidence Assessment")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confidence Score", f"{confidence_pct:.1f}%", "↑ Higher is better")
    with col2:
        st.metric("R² Score", f"{r_squared:.4f}", "↑ Closer to 1.0 is better")
    with col3:
        st.metric("MAE", f"{mae:.4f}", "↓ Lower is better")
    with col4:
        st.metric("RMSE", f"{rmse:.4f}", "↓ Lower is better")
    
    # Confidence interpretation
    if confidence_pct >= 95:
        conf_text = "🟢 **EXTREMELY HIGH** - Model is very precise and trustworthy"
    elif confidence_pct >= 85:
        conf_text = "🟢 **VERY HIGH** - Model predictions are highly reliable"
    elif confidence_pct >= 75:
        conf_text = "🟡 **HIGH** - Model is reasonably confident, generally trustworthy"
    elif confidence_pct >= 65:
        conf_text = "🟡 **MODERATE** - Model has acceptable precision, use with caution"
    elif confidence_pct >= 50:
        conf_text = "🟠 **LOW** - Model predictions have high variability"
    else:
        conf_text = "🔴 **VERY LOW** - Model is not confident, unreliable predictions"
    
    st.markdown(f"**Confidence Assessment:** {conf_text}")
    
    st.markdown("---")
    
    # Detailed metrics
    with st.expander("📊 Detailed Precision Metrics", expanded=True):
        st.write(f"""
        **Prediction Accuracy Metrics:**
        - **Mean Absolute Error (MAE):** {mae:.6f}
          - On average, predictions differ from actual by {mae*100:.2f}%
        
        - **Root Mean Squared Error (RMSE):** {rmse:.6f}
          - Typical prediction deviation: {rmse*100:.2f}%
        
        - **R² Score:** {r_squared:.6f}
          - Explains {r_squared*100:.2f}% of variance in target
          - 1.0 = perfect fit, 0.0 = no correlation
        
        **Prediction Distribution:**
        - **Prediction Std Dev:** {pred_std:.6f}
          - Model confidence spread: {pred_std*100:.2f}%
        
        - **Actual Std Dev:** {actual_std:.6f}
          - Actual data variability: {actual_std*100:.2f}%
        
        - **Distribution Match:** {abs(pred_std - actual_std):.6f}
          - How well model captures data spread
        
        **Error Distribution:**
        - **Mean Error:** {error_mean:.6f}
        - **Error Std Dev:** {error_std:.6f}
        - **Predictions within 1σ:** {within_1_std:.1f}%
        - **Predictions within 2σ:** {within_2_std:.1f}%
        
        **Calibration:**
        - **Calibration Error:** {calibration_error:.6f}
          - Model mean vs actual mean difference
          - Lower is better (shows how well model centers predictions)
        """)
    
    # Performance insights
    with st.expander("🔍 Performance Insights", expanded=True):
        # Find best and worst predictions
        best_idx = np.argmin(errors)
        worst_idx = np.argmax(errors)
        
        st.write(f"""
        **✅ Best Prediction:**
        - Predicted: {predictions_flat[best_idx]:.4f}
        - Actual: {Y_flat[best_idx]:.4f}
        - Error: {errors[best_idx]:.6f} ({errors[best_idx]*100:.2f}%)
        
        **❌ Worst Prediction:**
        - Predicted: {predictions_flat[worst_idx]:.4f}
        - Actual: {Y_flat[worst_idx]:.4f}
        - Error: {errors[worst_idx]:.6f} ({errors[worst_idx]*100:.2f}%)
        
        **Error Range:**
        - Min Error: {np.min(errors):.6f}
        - Max Error: {np.max(errors):.6f}
        - Error Median: {np.median(errors):.6f}
        """)
        
        # Bias analysis
        mean_error_signed = np.mean(predictions_flat - Y_flat)
        if abs(mean_error_signed) < 0.01:
            bias_text = "✅ **NO BIAS** - Predictions are well-calibrated"
        elif mean_error_signed > 0:
            bias_text = f"📊 **OVERESTIMATION BIAS** - Tends to predict {mean_error_signed*100:.2f}% too high"
        else:
            bias_text = f"📊 **UNDERESTIMATION BIAS** - Tends to predict {abs(mean_error_signed)*100:.2f}% too low"
        
        st.write(f"**Bias Analysis:** {bias_text}")
    
    # Error history visualization
    with st.expander("📉 Training Progress", expanded=True):
        st.line_chart(model["errors"])
        avg_error = np.mean(model["errors"])
        st.write(f"Average training error: {avg_error:.6f}")
    
    # Prediction vs Actual scatter plot
    with st.expander("📈 Prediction Accuracy Visualization", expanded=True):
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Scatter plot: Predicted vs Actual
        ax1.scatter(Y_flat, predictions_flat, alpha=0.6, s=30)
        min_val = min(Y_flat.min(), predictions_flat.min())
        max_val = max(Y_flat.max(), predictions_flat.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predictions vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        ax2.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(error_mean, color='r', linestyle='--', linewidth=2, label=f'Mean Error: {error_mean:.4f}')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
    
    st.markdown("---")

# =====================================================================
# DATA LOADING INTERFACE
# =====================================================================
tab1, tab2 = st.tabs(["📊 Sample Data", "📤 Upload Data"])

with tab1:
    st.subheader("Sample Dataset")
    st.dataframe(sample_data.head(10))
    use_sample = st.button("Use Sample Data", key="sample_btn")
    if use_sample:
        st.session_state["data"] = sample_data
        st.success("Sample data loaded!")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10))
        use_upload = st.button("Use Uploaded Data", key="upload_btn")
        if use_upload:
            st.session_state["data"] = df
            st.success("Data uploaded and loaded!")

# Process the selected data
if "data" in st.session_state:
    df = st.session_state["data"]
    bot = st.session_state["interpreter_bot"]
    
    st.markdown("## 📊 Loaded Dataset")
    st.dataframe(df.head())
    
    # Generate dataset assessment
    generate_dataset_assessment(df, bot)

    X = df.iloc[:, :-1].values.astype(np.float32)
    Y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

    st.sidebar.header("⚙️ Training Settings")
    lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
    epochs = st.sidebar.slider("Epochs", 10, 2000, 500)
    hidden_neurons = st.sidebar.slider("Hidden Neurons", 1, 50, 10)

    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner("Training neural network..."):
            model = train_network(X, Y, lr, epochs, hidden_neurons)
            st.session_state["model"] = model
            st.session_state["num_features"] = X.shape[1]

        st.success("✨ Training complete!")
        
        # Generate training assessment
        generate_training_assessment(model, X, Y, bot)

# Prediction with Interpretation
if "model" in st.session_state:
    st.markdown("---")
    st.markdown("## 🔮 Make New Predictions")

    inputs = []
    for i in range(st.session_state["num_features"]):
        val = st.number_input(f"Feature {i+1}", value=0.0)
        inputs.append(val)

    if st.button("🚀 Get Prediction", key="predict_btn"):
        x = np.array(inputs).reshape(1, -1)
        pred = predict(x, st.session_state["model"])
        pred_value = float(pred[0, 0])
        st.session_state["last_prediction"] = pred_value
        
        bot = st.session_state["interpreter_bot"]
        
        # Show prediction value
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Raw Prediction", f"{pred_value:.4f}")
        with col2:
            st.metric("Percentage", f"{pred_value*100:.1f}%")
        with col3:
            # Prediction confidence based on distance from 0.5
            distance_from_boundary = min(abs(pred_value - 0.5), 0.5)
            pred_confidence = (distance_from_boundary * 2) * 100  # 0-100%
            st.metric("Prediction Confidence", f"{pred_confidence:.1f}%")
        
        # Interpret prediction
        with st.expander("📖 What This Prediction Means", expanded=True):
            insight = bot.interpret_prediction(pred_value, context="neural network output")
            st.markdown(insight.user_friendly_description)
            
            with st.expander("📊 Probability Details"):
                st.write(insight.probability_explanation)
            
            with st.expander("💡 Recommendations"):
                for rec in insight.recommendations:
                    st.write(f"• {rec}")
            
            with st.expander("📈 Trend"):
                st.write(insight.trend_analysis)
        
        # Real confidence explanation
        with st.expander("🎯 Real Confidence Explained", expanded=True):
            st.write(f"""
            **Prediction Confidence: {pred_confidence:.1f}%**
            
            This measures how **far the prediction is from the uncertainty boundary** (0.5).
            
            - **0%** = Completely uncertain (prediction = 0.5)
            - **100%** = Maximum certainty (prediction = 0.0 or 1.0)
            
            **For this prediction:**
            - Value: {pred_value:.4f}
            - Distance from 0.5: {abs(pred_value - 0.5):.4f}
            - Confidence: {pred_confidence:.1f}%
            
            The higher this percentage, the more **decisive** the model is about its prediction.
            """)




