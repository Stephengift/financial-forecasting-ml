"""
LSTM Model for Stock Price Prediction
Senior Engineer building this step by step for junior developer
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

class StockLSTM:
    def __init__(self):
        """
        Step 1: Initialize our LSTM model
        Think of this as setting up our AI brain
        """
        self.model = None
        self.scaler = MinMaxScaler()  # This makes our data easier for AI to learn
        self.is_trained = False
        
        print("🧠 LSTM Brain initialized!")
    
    def build_model(self, input_shape):
        """
        Step 2: Build the neural network architecture
        Junior: This is like building the brain structure
        """
        print(f"🏗️ Building LSTM model with input shape: {input_shape}")
        
        # Create the model step by step
        self.model = Sequential()
        
        # Layer 1: LSTM layer (the memory part)
        # 50 = number of memory cells
        # return_sequences=True = pass info to next layer
        self.model.add(LSTM(units=50, 
                           return_sequences=True, 
                           input_shape=input_shape))
        
        # Layer 2: Dropout (prevents overfitting - like not memorizing answers)
        self.model.add(Dropout(0.2))
        
        # Layer 3: Another LSTM layer (more memory)
        self.model.add(LSTM(units=50, 
                           return_sequences=False))
        
        # Layer 4: Another dropout
        self.model.add(Dropout(0.2))
        
        # Layer 5: Dense layer (the decision maker)
        self.model.add(Dense(units=25))
        
        # Layer 6: Final output (1 prediction)
        self.model.add(Dense(units=1))
        
        # Compile the model (set up how it learns)
        self.model.compile(optimizer='adam',      # How it learns
                          loss='mean_squared_error',  # How it measures mistakes
                          metrics=['mae'])            # Extra metric to track
        
        print("✅ Model architecture built!")
        self.model.summary()  # Show the brain structure
    
    def prepare_data(self, X, y):
        """
        Step 3: Prepare data for training
        Junior: This scales data so AI can learn better
        """
        print("📊 Preparing data for training...")
        
        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Don't shuffle time series!
        )
        
        # Scale the features (make numbers between 0 and 1)
        X_train_scaled = self.scale_features(X_train)
        X_test_scaled = self.scale_features(X_test)
        
        print(f"✅ Data prepared:")
        print(f"   Training: X{X_train_scaled.shape}, y{y_train.shape}")
        print(f"   Testing:  X{X_test_scaled.shape}, y{y_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def scale_features(self, X):
        """
        Helper function to scale features
        Junior: This makes big numbers small so AI learns better
        """
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Reshape back
        return X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50):
        """
        Step 4: Train the model
        Junior: This is where the AI actually learns!
        """
        print(f"🚀 Training LSTM for {epochs} epochs...")
        print("⏰ This might take a few minutes...")
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,          # How many examples at once
            epochs=epochs,          # How many times through all data
            validation_data=(X_test, y_test),  # Test while training
            verbose=1,              # Show progress
            shuffle=False           # Don't shuffle time series
        )
        
        self.is_trained = True
        print("✅ Training complete!")
        
        return history
    
    def predict(self, X):
        """
        Step 5: Make predictions
        Junior: This is where the AI makes its guesses!
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        predictions = self.model.predict(X)
        return predictions.flatten()  # Make it 1D array
    
    def evaluate(self, X_test, y_test):
        """
        Step 6: See how good our AI is
        Junior: This tells us if our AI is smart or not!
        """
        print("📊 Evaluating model performance...")
        
        # Get predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        
        print(f"📈 Model Performance:")
        print(f"   Mean Squared Error: {mse:.6f}")
        print(f"   Mean Absolute Error: {mae:.6f}")
        
        return predictions, mse, mae

def main():
    """
    Step 7: Put it all together!
    Junior: This runs our entire pipeline - CONNECTED VERSION!
    """
    print("🚀 Starting LSTM Stock Prediction Training!")
    print("=" * 50)
    
    # Step 1: Import and use our feature engineering
    from feature_engineering import FeatureEngineering
    
    # Step 2: Load raw data
    print("📂 Loading raw stock data...")
    data_dir = Path("data")
    with open(data_dir / "market_data.pkl", 'rb') as f:
        data_dict = pickle.load(f)
    
    # Step 3: Get AAPL data and engineer features
    print("🔧 Running feature engineering...")
    fe = FeatureEngineering(sequence_length=60)
    aapl_data = data_dict["AAPL"]
    
    # Add all features
    processed_data = fe.add_price_features(aapl_data)
    processed_data = fe.add_technical_indicators(processed_data)
    
    # Create sequences for LSTM
    X, y = fe.create_sequences(processed_data, target_column='Returns')
    
    print(f"✅ Feature engineering complete!")
    print(f"   X shape: {X.shape}")  # Should be (samples, 60, 4)
    print(f"   y shape: {y.shape}")  # Should be (samples,)
    
    # Step 4: Create and build LSTM model
    print("\n🧠 Building LSTM model...")
    lstm_model = StockLSTM()
    
    # Input shape is (timesteps, features) = (60, 4)
    input_shape = (X.shape[1], X.shape[2])
    lstm_model.build_model(input_shape)
    
    # Step 5: Prepare data for training
    print("\n📊 Preparing training data...")
    X_train, X_test, y_train, y_test = lstm_model.prepare_data(X, y)
    
    # Step 6: Train the model
    print("\n🚀 Training LSTM model...")
    print("⏰ This will take a few minutes...")
    
    history = lstm_model.train(X_train, y_train, X_test, y_test, epochs=25)
    
    # Step 7: Evaluate the model
    print("\n📈 Evaluating model performance...")
    predictions, mse, mae = lstm_model.evaluate(X_test, y_test)
    
    # Step 8: Show some sample predictions
    print("\n🎯 Sample Predictions vs Actual:")
    for i in range(5):
        print(f"   Predicted: {predictions[i]:+.4f}, Actual: {y_test[i]:+.4f}")
    
    print("\n🎉 LSTM Training Complete!")
    print("✅ Your AI model is now trained and ready!")
    
    return lstm_model, history, predictions

if __name__ == "__main__":
    main()