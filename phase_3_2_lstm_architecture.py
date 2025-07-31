# phase_3_2_lstm_architecture.py - Define LSTM Model Architecture (PyTorch)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

load_dotenv()

def setup_database_connection():
    """Setup database connection"""
    try:
        # Try PostgreSQL first
        pg_user = os.getenv('POSTGRES_USER', 'postgres')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'password')
        pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        pg_port = os.getenv('POSTGRES_PORT', '5432')
        pg_database = os.getenv('POSTGRES_DATABASE', 'smart_dispatch')
        
        pg_connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        engine = create_engine(pg_connection_string)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        print("✅ PostgreSQL connection successful")
        return engine, "postgresql"
        
    except Exception as e:
        print(f"⚠️ PostgreSQL not available, using SQLite")
        sqlite_path = "market_data.db"
        sqlite_connection_string = f"sqlite:///{sqlite_path}"
        engine = create_engine(sqlite_connection_string)
        print(f"✅ SQLite connection successful: {sqlite_path}")
        return engine, "sqlite"

def load_and_prepare_data(engine):
    """Load and prepare data for model training (reusing Phase 3.1 logic)"""
    print(f"🔄 Loading and preparing data...")
    
    # Load feature matrix
    query = "SELECT * FROM features ORDER BY window_id"
    df_features = pd.read_sql(query, engine)
    df_features['target_time'] = pd.to_datetime(df_features['target_time'])
    
    # Define feature columns
    exclude_cols = ['window_id', 'target_time', 'target_price', 'price_sequence_json']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    # Extract features and targets
    X = df_features[feature_cols].values
    y = df_features['target_price'].values
    
    # Extract price sequences
    price_sequences = []
    for seq_json in df_features['price_sequence_json']:
        seq = json.loads(seq_json)
        price_sequences.append(seq)
    price_sequences = np.array(price_sequences)
    
    # Train/test split
    X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
        X, y, price_sequences, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data prepared: {len(X_train_scaled)} train, {len(X_test_scaled)} test samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, seq_train, seq_test, scaler, feature_cols

class PowerMarketDataset(Dataset):
    """PyTorch Dataset for power market data"""
    
    def __init__(self, X, y, sequences):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.sequences[idx], self.y[idx]

class PowerMarketLSTM(nn.Module):
    """
    Phase 3.2: LSTM Model Architecture for Power Market Forecasting
    Input: windowed features; Output: price forecast
    
    Hybrid architecture combining:
    - LSTM for temporal sequence processing (24-hour price sequences)
    - Dense layers for engineered feature processing
    - Combined prediction layer
    """
    
    def __init__(self, n_features, sequence_length=24, lstm_hidden_size=64, 
                 lstm_num_layers=2, dense_hidden_size=32, dropout_rate=0.2):
        super(PowerMarketLSTM, self).__init__()
        
        # Store architecture parameters
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dense_hidden_size = dense_hidden_size
        self.dropout_rate = dropout_rate
        
        # LSTM branch for processing price sequences (temporal patterns)
        self.lstm = nn.LSTM(
            input_size=1,  # Single price value per timestep
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        
        # Dense branch for processing engineered features (statistical patterns)
        self.feature_layers = nn.Sequential(
            nn.Linear(n_features, dense_hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden_size * 2, dense_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined prediction layers
        combined_input_size = lstm_hidden_size + dense_hidden_size
        self.prediction_layers = nn.Sequential(
            nn.Linear(combined_input_size, dense_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden_size, dense_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(dense_hidden_size // 2, 1)  # Single price prediction
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.xavier_uniform_(param)
                else:
                    # Dense layer weights
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, features, sequences):
        """
        Forward pass through the hybrid LSTM architecture
        
        Args:
            features: (batch_size, n_features) - Engineered features
            sequences: (batch_size, sequence_length) - Price sequences
            
        Returns:
            predictions: (batch_size, 1) - Price forecasts
        """
        batch_size = features.size(0)
        
        # LSTM branch: Process price sequences
        # Reshape sequences for LSTM: (batch_size, seq_len, 1)
        lstm_input = sequences.unsqueeze(-1)
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        
        # Use the last hidden state from the final LSTM layer
        lstm_features = hidden[-1]  # (batch_size, lstm_hidden_size)
        
        # Dense branch: Process engineered features
        dense_features = self.feature_layers(features)  # (batch_size, dense_hidden_size)
        
        # Combine LSTM and dense features
        combined_features = torch.cat([lstm_features, dense_features], dim=1)
        
        # Final prediction
        predictions = self.prediction_layers(combined_features)
        
        return predictions.squeeze(-1)  # Remove last dimension: (batch_size,)
    
    def model_summary(self):
        """
        Test Case: model.summary() confirms layers and dimensions
        Display model architecture and parameter counts
        """
        print(f"\n🏗️ PowerMarketLSTM Model Architecture Summary")
        print(f"=" * 60)
        
        # Architecture overview
        print(f"📋 Model Configuration:")
        print(f"   Input Features: {self.n_features}")
        print(f"   Sequence Length: {self.sequence_length}")
        print(f"   LSTM Hidden Size: {self.lstm_hidden_size}")
        print(f"   LSTM Layers: {self.lstm_num_layers}")
        print(f"   Dense Hidden Size: {self.dense_hidden_size}")
        print(f"   Dropout Rate: {self.dropout_rate}")
        
        # Layer details
        print(f"\n🔧 Layer Architecture:")
        print(f"   1. LSTM Branch:")
        print(f"      - Input: (batch_size, {self.sequence_length}, 1)")
        print(f"      - LSTM: {self.lstm_num_layers} layers × {self.lstm_hidden_size} units")
        print(f"      - Output: (batch_size, {self.lstm_hidden_size})")
        
        print(f"   2. Dense Branch:")
        print(f"      - Input: (batch_size, {self.n_features})")
        print(f"      - Dense1: {self.n_features} → {self.dense_hidden_size * 2}")
        print(f"      - Dense2: {self.dense_hidden_size * 2} → {self.dense_hidden_size}")
        print(f"      - Output: (batch_size, {self.dense_hidden_size})")
        
        print(f"   3. Prediction Branch:")
        combined_size = self.lstm_hidden_size + self.dense_hidden_size
        print(f"      - Input: (batch_size, {combined_size}) [LSTM + Dense]")
        print(f"      - Dense3: {combined_size} → {self.dense_hidden_size}")
        print(f"      - Dense4: {self.dense_hidden_size} → {self.dense_hidden_size // 2}")
        print(f"      - Dense5: {self.dense_hidden_size // 2} → 1")
        print(f"      - Output: (batch_size,) [Price prediction]")
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📊 Parameter Statistics:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        
        # Memory estimate
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        print(f"   Estimated Model Size: {model_size_mb:.2f} MB")
        
        # Layer-wise parameter breakdown
        print(f"\n🔍 Layer-wise Parameter Count:")
        for name, module in self.named_modules():
            if len(list(module.parameters())) > 0:
                layer_params = sum(p.numel() for p in module.parameters())
                if layer_params > 0:
                    print(f"   {name}: {layer_params:,} parameters")
        
        print(f"=" * 60)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb
        }

def test_model_architecture(model, sample_features, sample_sequences):
    """Test the model with sample data"""
    print(f"\n🧪 Testing Model Architecture:")
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        predictions = model(sample_features, sample_sequences)
        
        print(f"   Input features shape: {sample_features.shape}")
        print(f"   Input sequences shape: {sample_sequences.shape}")
        print(f"   Output predictions shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[:3].numpy().round(2)}")
    
    return predictions

def main():
    """Execute Phase 3.2 workflow"""
    print("🚀 Phase 3.2: Define LSTM Model Architecture (PyTorch)")
    
    try:
        # Step 1: Setup database and load data
        engine, db_type = setup_database_connection()
        X_train, X_test, y_train, y_test, seq_train, seq_test, scaler, feature_cols = load_and_prepare_data(engine)
        
        # Step 2: Define model architecture (Phase 3.2)
        n_features = len(feature_cols)
        sequence_length = 24
        
        print(f"\n🏗️ Creating LSTM model architecture...")
        print(f"   Features: {n_features}")
        print(f"   Sequence length: {sequence_length}")
        
        # Create model
        model = PowerMarketLSTM(
            n_features=n_features,
            sequence_length=sequence_length,
            lstm_hidden_size=64,
            lstm_num_layers=2,
            dense_hidden_size=32,
            dropout_rate=0.2
        )
        
        # Test Case: model.summary() confirms layers and dimensions
        model_stats = model.model_summary()
        
        # Test model with sample data
        sample_features = torch.FloatTensor(X_train[:3])
        sample_sequences = torch.FloatTensor(seq_train[:3])
        
        predictions = test_model_architecture(model, sample_features, sample_sequences)
        
        print(f"\n✅ Phase 3.2 COMPLETE: LSTM model architecture defined successfully")
        print(f"🏗️ Model specifications:")
        print(f"   Hybrid architecture: LSTM + Dense features")
        print(f"   Total parameters: {model_stats['total_params']:,}")
        print(f"   Model size: {model_stats['model_size_mb']:.2f} MB")
        print(f"   Ready for training on power market data")
        print(f"🔄 Next: Phase 3.3 - Train Model on Sample Data")
        
        # Return model and data for next phase
        training_data = {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'seq_train': seq_train,
            'seq_test': seq_test,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
        
        return training_data
        
    except Exception as e:
        print(f"❌ Phase 3.2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    training_data = main()