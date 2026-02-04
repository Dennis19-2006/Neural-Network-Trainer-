# Neural-Network-Trainer-

# Neural Network Trainer with Streamlit

A **from-scratch single-layer neural network** with backpropagation, visualized and interactively trained using **Streamlit**.

Perfect for learning how neural networks really work under the hood — no PyTorch, no TensorFlow, just NumPy + Streamlit.

##steamlit app link:
https://neural-network-trainer-123938.streamlit.app/

## Features

- Train a neural network from scratch using only NumPy
- Sigmoid activation + mean squared error
- Interactive Streamlit interface
- Real-time training progress bar & error curve
- Upload your own CSV dataset or use built-in sample data
- Make predictions on new inputs after training
- Visualizes final layer activations (JSON view)
- Simple gradient clipping to prevent explosions 

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Dennis19-2006/neural-network-trainer-streamlit.git
cd neural-network-trainer-streamlit

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
