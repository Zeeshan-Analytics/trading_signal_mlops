"""
Prepare train/validation/test datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def select_features(df: pd.DataFrame, config: dict) -> tuple:
    """
    Select features and target for modeling.
    
    Args:
        df: DataFrame with all data
        config: Configuration dictionary
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Define feature columns (exclude non-feature columns)
    exclude_columns = ['Datetime', 'ticker', 'future_return', 'signal', 
                      'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    logger.info(f"Selected {len(feature_columns)} features")
    logger.info(f"Features: {feature_columns}")
    
    X = df[feature_columns].values
    y = df['signal'].values
    
    return X, y, feature_columns


def split_data(X: np.ndarray, y: np.ndarray, config: dict) -> tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        config: Configuration dictionary
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    
    logger.info(f"Splitting data: train={train_split}, val={val_split}, test={test_split}")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    
    # Second split: separate train and validation
    val_ratio = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
    
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    logger.info("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Features scaled successfully")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test, 
                  feature_names, scaler, output_dir: str):
    """
    Save all datasets and scaler.
    
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Target vectors
        feature_names: List of feature names
        scaler: Fitted scaler object
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets as numpy arrays
    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'X_val.npy', X_val)
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_val.npy', y_val)
    np.save(output_path / 'y_test.npy', y_test)
    
    # Save feature names
    with open(output_path / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    # Save scaler
    joblib.dump(scaler, output_path / 'scaler.pkl')
    
    logger.info(f"All datasets saved to {output_path}")


def main():
    """Main function to prepare datasets."""
    try:
        # Load configuration
        config = load_config()
        
        logger.info("=" * 50)
        logger.info("STARTING DATASET PREPARATION")
        logger.info("=" * 50)
        
        # Load processed features
        input_path = "data/processed/features.csv"
        logger.info(f"Loading features from {input_path}...")
        df = pd.read_csv(input_path)
        
        # Select features and target
        X, y, feature_names = select_features(df, config)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test
        )
        
        # Save datasets
        output_dir = "data/features"
        save_datasets(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            feature_names, scaler, output_dir
        )
        
        logger.info("=" * 50)
        logger.info("DATASET PREPARATION COMPLETED")
        logger.info("=" * 50)
        
        # Print summary
        print("\n" + "=" * 50)
        print("DATASET PREPARATION SUMMARY")
        print("=" * 50)
        print(f"Number of features: {len(feature_names)}")
        print(f"Train samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Test samples: {len(y_test)}")
        print(f"Output directory: {output_dir}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        raise


if __name__ == "__main__":
    main()