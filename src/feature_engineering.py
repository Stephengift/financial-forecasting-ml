"""
Feature engineering for financial time series data.
Creates technical indicators and ML-ready sequences.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, sequence_length: int = 60):  # FIX 1: Add __init__
        self.sequence_length = sequence_length
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        """Trying to calculate the change in price when it opens and closes """
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        """Simple Moving Averages"""
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, target_column: str = 'Returns') -> Tuple[np.ndarray, np.ndarray]:  # FIX 2: Add target_column parameter
        df_clean = df.dropna()  # FIX 3: Fix typo def_clean -> df_clean

        feature_columns = ['Returns', 'Volatility', 'SMA_20', 'EMA_20']

        features = df_clean[feature_columns].values
        target = df_clean[target_column].values

        X, y = [], []

        for i in range(self.sequence_length, len(features)):
            X.append(features[i - self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

def main():
    # Load data
    print("Starting main function...")
    data_dir = Path("data")
    with open(data_dir / "market_data.pkl", 'rb') as f:
        data_dict = pickle.load(f)
    
    # Test your methods
    fe = FeatureEngineering()  # This will now work because we added __init__
    aapl_data = data_dict["AAPL"]
    
    print("Before:", list(aapl_data.columns))
    
    # Add price features first
    processed_data = fe.add_price_features(aapl_data)
    print("After price features:", list(processed_data.columns))
    
    # Add technical indicators
    processed_data = fe.add_technical_indicators(processed_data)
    print("After technical indicators:", list(processed_data.columns))
    
    # Show all the new features
    print(processed_data[['Close', 'Returns', 'Volatility', 'SMA_20', 'EMA_20']].tail())
    
    # Test create_sequences - THE MOST IMPORTANT PART
    print("\n🎯 Testing create_sequences (for LSTM):")
    X, y = fe.create_sequences(processed_data)
    print(f"✅ X shape: {X.shape}")  # Should be (samples, 60, 4)
    print(f"✅ y shape: {y.shape}")  # Should be (samples,)
    print("Ready for LSTM! 🚀")


if __name__ == "__main__":
    main()