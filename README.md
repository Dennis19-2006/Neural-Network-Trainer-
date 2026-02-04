# Neural-Network-Trainer-

# Neural Network Trainer with Streamlit

A **from-scratch single-layer neural network** with backpropagation, visualized and interactively trained using **Streamlit**.

Perfect for learning how neural networks really work under the hood — no PyTorch, no TensorFlow, just NumPy + Streamlit.

https://github.com/yourusername/neural-network-trainer-streamlit

<p align="center">
  <img src="https://via.placeholder.com/800x450.png?text=App+Screenshot" alt="App Screenshot" width="800"/>
  <!-- Replace with real screenshot later -->
</p>

## Features

- Train a neural network from scratch using only NumPy
- Sigmoid activation + mean squared error
- Interactive Streamlit interface
- Real-time training progress bar & error curve
- Upload your own CSV dataset or use built-in sample data
- Make predictions on new inputs after training
- Visualizes final layer activations (JSON view)
- Simple gradient clipping to prevent explosions

## Demo Look & Feel

https://github.com/yourusername/neural-network-trainer-streamlit/assets/12345678/abcdef12-3456-7890-abcd-ef1234567890  
*(add a short screen recording here later — strongly recommended!)*

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/neural-network-trainer-streamlit.git
cd neural-network-trainer-streamlit

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
