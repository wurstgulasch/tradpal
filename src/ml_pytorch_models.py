"""
Advanced PyTorch Models for Time Series Prediction

Implements LSTM, GRU, and Transformer models for trading signal prediction.
Provides more advanced neural network architectures than TensorFlow LSTM.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Advanced neural network features will be disabled.")
    print("   Install with: pip install torch")

from config.settings import (
    SYMBOL, TIMEFRAME, ML_PYTORCH_HIDDEN_SIZE, ML_PYTORCH_NUM_LAYERS,
    ML_PYTORCH_DROPOUT, ML_PYTORCH_LEARNING_RATE, ML_PYTORCH_BATCH_SIZE,
    ML_PYTORCH_EPOCHS, ML_PYTORCH_EARLY_STOPPING_PATIENCE
)


class LSTMModel(nn.Module):
    """
    Advanced LSTM model for time series prediction with PyTorch.
    
    Features:
    - Bidirectional LSTM layers
    - Layer normalization
    - Residual connections
    - Attention mechanism (optional)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, use_attention: bool = False):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out  # Residual connection
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class GRUModel(nn.Module):
    """
    GRU model for time series prediction.
    
    GRUs are similar to LSTMs but with fewer parameters and faster training.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2):
        """Initialize GRU model."""
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # GRU layers
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)
        
        # Use last output
        last_output = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction.
    
    Uses multi-head self-attention instead of recurrence.
    Can be more effective at capturing long-term dependencies.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.2):
        """Initialize Transformer model."""
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network."""
        # Project input to hidden size
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        transformer_out = self.transformer(x)
        
        # Use last output
        last_output = transformer_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PyTorchPredictor:
    """
    PyTorch-based signal predictor with advanced neural network models.
    
    Features:
    - Multiple model architectures (LSTM, GRU, Transformer)
    - GPU acceleration support
    - Early stopping and model checkpointing
    - Learning rate scheduling
    - Advanced training techniques
    """
    
    def __init__(self, model_type: str = 'lstm', model_dir: str = "cache/ml_models",
                 symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                 sequence_length: int = 60, hidden_size: int = ML_PYTORCH_HIDDEN_SIZE,
                 num_layers: int = ML_PYTORCH_NUM_LAYERS, dropout: float = ML_PYTORCH_DROPOUT):
        """
        Initialize PyTorch predictor.
        
        Args:
            model_type: Type of model ('lstm', 'gru', 'transformer')
            model_dir: Directory to store trained models
            symbol: Trading symbol
            timeframe: Timeframe
            sequence_length: Length of input sequences
            hidden_size: Size of hidden layers
            num_layers: Number of layers
            dropout: Dropout rate
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        self.training_history = {}
        
        # Try to load existing model
        self.load_model()
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create a model based on the specified type."""
        if self.model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_attention=True
            )
        elif self.model_type == 'gru':
            model = GRUModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        elif self.model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_heads=4,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features from DataFrame."""
        feature_cols = []
        
        # Price features
        if 'close' in df.columns:
            feature_cols.extend(['close', 'open', 'high', 'low', 'volume'])
        
        # Technical indicators
        indicator_patterns = ['EMA', 'SMA', 'RSI', 'BB', 'ATR', 'MACD', 'ADX', 'Stoch']
        for pattern in indicator_patterns:
            feature_cols.extend([col for col in df.columns if pattern in col])
        
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if not feature_cols:
            raise ValueError("No valid features found in DataFrame")
        
        # Extract features and normalize
        features = df[feature_cols].values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features, feature_cols
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 5) -> np.ndarray:
        """Create binary labels based on future price movement."""
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        close_prices = df['close'].values
        labels = np.zeros(len(close_prices))
        
        for i in range(len(close_prices) - prediction_horizon):
            future_price = close_prices[i + prediction_horizon]
            current_price = close_prices[i]
            
            # Label as 1 if price increases, 0 otherwise
            labels[i] = 1 if future_price > current_price else 0
        
        return labels
    
    def prepare_sequences(self, df: pd.DataFrame, prediction_horizon: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences for training."""
        features, feature_names = self.prepare_features(df)
        self.feature_columns = feature_names
        labels = self.create_labels(df, prediction_horizon)
        
        # Create sequences
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(features)):
            sequences.append(features[i - self.sequence_length:i])
            sequence_labels.append(labels[i])
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(sequence_labels, dtype=np.float32)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        return X_tensor, y_tensor
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2,
                   prediction_horizon: int = 5, epochs: int = ML_PYTORCH_EPOCHS,
                   batch_size: int = ML_PYTORCH_BATCH_SIZE,
                   learning_rate: float = ML_PYTORCH_LEARNING_RATE) -> Dict[str, Any]:
        """
        Train the PyTorch model.
        
        Args:
            df: DataFrame with technical indicators
            test_size: Fraction for test set
            prediction_horizon: Periods ahead to predict
            epochs: Maximum training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training results dictionary
        """
        try:
            print(f"🚀 Training {self.model_type.upper()} model...")
            
            # Prepare data
            X, y = self.prepare_sequences(df, prediction_horizon)
            
            if len(X) < 100:
                raise ValueError(f"Insufficient data: {len(X)} sequences")
            
            # Split data
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"📊 Training: {len(X_train)}, Test: {len(X_test)}")
            
            # Create model
            input_size = X.shape[2]
            self.model = self._create_model(input_size)
            
            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
            
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                
                # Mini-batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i + batch_size]
                    batch_y = y_train[i:i + batch_size]
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= (len(X_train) / batch_size)
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_test)
                    val_loss = criterion(val_outputs, y_test).item()
                    val_predictions = (val_outputs > 0.5).float()
                    val_accuracy = (val_predictions == y_test).float().mean().item()
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_accuracy:.4f}")
                
                if patience_counter >= ML_PYTORCH_EARLY_STOPPING_PATIENCE:
                    print(f"⏸️  Early stopping at epoch {epoch + 1}")
                    break
            
            self.is_trained = True
            self.training_history = history
            
            # Final evaluation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_predictions = (test_outputs > 0.5).float()
                test_accuracy = (test_predictions == y_test).float().mean().item()
            
            results = {
                'model_type': self.model_type,
                'epochs_trained': epoch + 1,
                'best_val_loss': best_loss,
                'final_test_accuracy': test_accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_columns),
                'device': str(self.device)
            }
            
            print(f"✅ Training complete! Test Accuracy: {test_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def predict_signal(self, df: pd.DataFrame, threshold: float = 0.6) -> Dict[str, Any]:
        """Generate prediction for the latest data."""
        if not self.is_trained or self.model is None:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': 'Model not trained'
            }
        
        try:
            # Prepare sequences
            X, _ = self.prepare_sequences(df)
            
            if len(X) == 0:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'error': 'Insufficient data'
                }
            
            # Get prediction for last sequence
            self.model.eval()
            with torch.no_grad():
                last_sequence = X[-1:]
                prediction = self.model(last_sequence)
                confidence = prediction.item()
            
            # Determine signal
            if confidence >= threshold:
                signal = 'BUY'
            elif confidence <= (1 - threshold):
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'model_type': self.model_type,
                'threshold': threshold
            }
            
        except Exception as e:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def save_model(self):
        """Save model to disk."""
        if self.model is None:
            return
        
        model_path = self.model_dir / f"{self.symbol.replace('/', '_')}_{self.timeframe}_{self.model_type}_pytorch.pth"
        metadata_path = self.model_dir / f"{self.symbol.replace('/', '_')}_{self.timeframe}_{self.model_type}_pytorch_meta.pkl"
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'input_size': len(self.feature_columns)
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"💾 Model saved to {model_path}")
    
    def load_model(self):
        """Load model from disk."""
        model_path = self.model_dir / f"{self.symbol.replace('/', '_')}_{self.timeframe}_{self.model_type}_pytorch.pth"
        metadata_path = self.model_dir / f"{self.symbol.replace('/', '_')}_{self.timeframe}_{self.model_type}_pytorch_meta.pkl"
        
        if not model_path.exists() or not metadata_path.exists():
            return
        
        try:
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.is_trained = metadata['is_trained']
            self.training_history = metadata.get('training_history', {})
            
            # Create and load model
            input_size = metadata['input_size']
            self.model = self._create_model(input_size)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            print(f"📂 Loaded {self.model_type.upper()} model from {model_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            self.model = None
            self.is_trained = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns,
            'device': str(self.device),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length
        }


# Global predictor instances
pytorch_predictors = {}


def get_pytorch_predictor(model_type: str = 'lstm', symbol: str = SYMBOL,
                          timeframe: str = TIMEFRAME) -> Optional[PyTorchPredictor]:
    """Get or create PyTorch predictor instance."""
    global pytorch_predictors
    
    key = f"{symbol}_{timeframe}_{model_type}"
    
    if key not in pytorch_predictors and PYTORCH_AVAILABLE:
        try:
            pytorch_predictors[key] = PyTorchPredictor(
                model_type=model_type,
                symbol=symbol,
                timeframe=timeframe
            )
        except Exception as e:
            print(f"❌ Failed to initialize PyTorch predictor: {e}")
            return None
    
    return pytorch_predictors.get(key)


def is_pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return PYTORCH_AVAILABLE
