# Example: How to load and use the saved model
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model (PyTorch 2.7+ compatibility)
model_state = torch.load('model.pt', map_location='cpu', weights_only=False)

# Recreate the model
from your_model_file import PowerMarketLSTM  # Import your model class
config = model_state['model_config']
model = PowerMarketLSTM(**config)
model.load_state_dict(model_state['model_state_dict'])
model.eval()

# Get the scaler and feature columns
scaler = model_state['scaler']
feature_cols = model_state['feature_cols']

# Example prediction
def predict_price(new_features, new_sequence):
    """Make a price prediction with new data"""
    # Scale features
    new_features_scaled = scaler.transform(new_features.reshape(1, -1))
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(new_features_scaled)
    sequence_tensor = torch.FloatTensor(new_sequence).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(features_tensor, sequence_tensor)
    
    return prediction.item()

# Example usage:
# new_features = np.array([...])  # 19 engineered features
# new_sequence = np.array([...])  # 24-hour price sequence
# predicted_price = predict_price(new_features, new_sequence)
# print(f"Predicted next-hour price: ${predicted_price:.2f}")

# Model information
print(f"Model loaded successfully!")
print(f"Features: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")
print(f"Training history: {list(model_state['training_history'].keys())}")
