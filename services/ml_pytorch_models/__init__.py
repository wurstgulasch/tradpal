"""
ML PyTorch Models Service - PyTorch-based models for trading.

Provides PyTorch neural network models for advanced trading predictions.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PyTorch models will be disabled.")


class TradingDataset(Dataset):
    """Dataset for trading data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class TransformerModel(nn.Module):
    """Transformer model for trading prediction."""

    def __init__(self, input_size: int, num_heads: int = 8, num_layers: int = 4, hidden_size: int = 64):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        x = self.sigmoid(x)
        return x


class PyTorchPredictor:
    """PyTorch-based predictor for trading signals."""

    def __init__(self, model_type: str = "lstm"):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_model(self, input_size: int) -> nn.Module:
        """Create PyTorch model."""
        if self.model_type == "lstm":
            return LSTMModel(input_size=input_size)
        elif self.model_type == "transformer":
            return TransformerModel(input_size=input_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the PyTorch model."""
        try:
            # Create model
            input_size = X.shape[1] if len(X.shape) > 1 else 1
            self.model = self.create_model(input_size).to(self.device)

            # Create dataset and dataloader
            dataset = TradingDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            # Training loop
            self.model.train()
            for epoch in range(epochs):
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                    loss.backward()
                    optimizer.step()

            self.is_trained = True
            logger.info(f"PyTorch model trained for {epochs} epochs")

        except Exception as e:
            logger.error(f"PyTorch training failed: {e}")
            self.is_trained = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.is_trained or self.model is None:
            return np.zeros(len(X))

        try:
            self.model.eval()
            with torch.no_grad():
                inputs = torch.FloatTensor(X).to(self.device)
                outputs = self.model(inputs)
                predictions = outputs.cpu().numpy().squeeze()

            return predictions

        except Exception as e:
            logger.error(f"PyTorch prediction failed: {e}")
            return np.zeros(len(X))


def is_pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return PYTORCH_AVAILABLE


def get_pytorch_predictor(model_type: str = "lstm") -> Optional[PyTorchPredictor]:
    """Get PyTorch predictor instance."""
    if not PYTORCH_AVAILABLE:
        return None

    try:
        return PyTorchPredictor(model_type=model_type)
    except Exception as e:
        logger.error(f"Failed to create PyTorch predictor: {e}")
        return None


def create_lstm_predictor(input_size: int) -> Optional[LSTMModel]:
    """Create LSTM predictor model."""
    if not PYTORCH_AVAILABLE:
        return None

    try:
        return LSTMModel(input_size=input_size)
    except Exception as e:
        logger.error(f"Failed to create LSTM model: {e}")
        return None


def create_transformer_predictor(input_size: int) -> Optional[TransformerModel]:
    """Create Transformer predictor model."""
    if not PYTORCH_AVAILABLE:
        return None

    try:
        return TransformerModel(input_size=input_size)
    except Exception as e:
        logger.error(f"Failed to create Transformer model: {e}")
        return None