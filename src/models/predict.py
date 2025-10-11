"""
Prediction utilities for the trading signal model.
"""

import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradingSignalPredictor:
    """Wrapper class for making predictions with the trained model."""
    
    def __init__(self, model_path: str = "models/saved_models/trading_signal_model.keras",
                 scaler_path: str = "data/features/scaler.pkl"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler
        """
        self.config = load_config()
        
        # Load signal classes from config
        # Convert string keys to integers for proper indexing
        signal_classes_raw = self.config['signals']['classes']
        self.signal_classes = {int(k): v for k, v in signal_classes_raw.items()}
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scaler
        logger.info(f"Loading scaler from {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        feature_names_path = Path("data/features/feature_names.txt")
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Predictor initialized with {len(self.feature_names)} features")
        logger.info(f"Signal classes: {self.signal_classes}")
    
    def predict(self, features: np.ndarray) -> dict:
        """
        Make a prediction for given features.
        
        Args:
            features: Feature array (can be single sample or batch)
        
        Returns:
            Dictionary with prediction results
        """
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Validate feature count
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
            )
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        probabilities = self.model.predict(features_scaled, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            # Get signal name from prediction index
            pred_idx = int(pred)
            signal = self.signal_classes[pred_idx]
            confidence = float(probabilities[i][pred_idx])
            
            # Create probability dictionary for all signals
            prob_dict = {}
            for class_idx, class_name in self.signal_classes.items():
                prob_dict[class_name] = float(probabilities[i][class_idx])
            
            results.append({
                'signal': signal,
                'confidence': confidence,
                'probabilities': prob_dict
            })
        
        # Return single result if single input, otherwise list
        return results[0] if len(results) == 1 else results
    
    def predict_from_dict(self, feature_dict: dict) -> dict:
        """
        Make prediction from a dictionary of feature values.
        
        Args:
            feature_dict: Dictionary with feature names as keys
        
        Returns:
            Prediction result dictionary
        
        Raises:
            ValueError: If required features are missing
        """
        # Check for missing features
        missing_features = set(self.feature_names) - set(feature_dict.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Convert dict to array in correct order
        features = np.array([feature_dict[name] for name in self.feature_names])
        
        return self.predict(features)
    
    def get_feature_names(self) -> list:
        """Get list of required feature names."""
        return self.feature_names.copy()
    
    def get_signal_classes(self) -> dict:
        """Get signal class mapping."""
        return self.signal_classes.copy()


def load_predictor() -> TradingSignalPredictor:
    """
    Convenience function to load the predictor.
    
    Returns:
        Initialized TradingSignalPredictor instance
    """
    return TradingSignalPredictor()


if __name__ == "__main__":
    # Test the predictor
    print("\n" + "=" * 60)
    print("TRADING SIGNAL PREDICTOR TEST")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = TradingSignalPredictor()
        
        print("\n[OK] Predictor loaded successfully!")
        print(f"Features required: {len(predictor.feature_names)}")
        print(f"Signal classes: {list(predictor.signal_classes.values())}")
        
        # Test 1: Predict with random features
        print("\n" + "-" * 60)
        print("Test 1: Random feature prediction")
        print("-" * 60)
        
        dummy_features = np.random.randn(14)  # 14 features
        result = predictor.predict(dummy_features)
        
        print(f"\nPredicted Signal: {result['signal'].upper().replace('_', ' ')}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        print("\nProbability Distribution:")
        for signal, prob in result['probabilities'].items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {signal:15s} {bar} {prob:6.2%}")
        
        # Test 2: Predict from dictionary
        print("\n" + "-" * 60)
        print("Test 2: Dictionary-based prediction")
        print("-" * 60)
        
        # Create example feature dict
        feature_dict = {
            'SMA_10': 150.5,
            'SMA_20': 148.2,
            'SMA_50': 145.8,
            'EMA_12': 151.0,
            'EMA_26': 149.5,
            'RSI_14': 65.3,
            'MACD': 1.5,
            'MACD_signal': 1.2,
            'MACD_hist': 0.3,
            'BB_upper': 155.0,
            'BB_middle': 150.0,
            'BB_lower': 145.0,
            'volume_sma_20': 1000000.0,
            'price_change': 0.02
        }
        
        result2 = predictor.predict_from_dict(feature_dict)
        
        print(f"\nPredicted Signal: {result2['signal'].upper().replace('_', ' ')}")
        print(f"Confidence: {result2['confidence']:.2%}")
        
        # Test 3: Batch prediction
        print("\n" + "-" * 60)
        print("Test 3: Batch prediction (3 samples)")
        print("-" * 60)
        
        batch_features = np.random.randn(3, 14)  # 3 samples, 14 features each
        batch_results = predictor.predict(batch_features)
        
        for idx, result in enumerate(batch_results, 1):
            print(f"\nSample {idx}: {result['signal'].upper().replace('_', ' ')} "
                  f"(confidence: {result['confidence']:.2%})")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All tests passed!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("[ERROR] Test failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print()