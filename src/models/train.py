"""
Train the trading signal model with Wandb tracking.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os
import json
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model import create_model, get_callbacks
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
import wandb
#from wandb.integration.keras import WandbCallback

logger = get_logger(__name__)


def load_datasets(data_dir: str = "data/features"):
    """
    Load train, validation, and test datasets.
    
    Args:
        data_dir: Directory containing the datasets
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    logger.info(f"Loading datasets from {data_dir}...")
    
    data_path = Path(data_dir)
    
    X_train = np.load(data_path / 'X_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    X_test = np.load(data_path / 'X_test.npy')
    y_train = np.load(data_path / 'y_train.npy')
    y_val = np.load(data_path / 'y_val.npy')
    y_test = np.load(data_path / 'y_test.npy')
    
    # Load feature names
    with open(data_path / 'feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Features: {len(feature_names)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def convert_labels_to_categorical(y_train, y_val, y_test, num_classes: int = 5):
    """
    Convert integer labels to one-hot encoded format.
    
    Args:
        y_train, y_val, y_test: Label arrays
        num_classes: Number of classes
    
    Returns:
        One-hot encoded labels
    """
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return y_train_cat, y_val_cat, y_test_cat


def initialize_wandb(config: dict, feature_names: list):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        config: Project configuration
        feature_names: List of feature names
    """
    # Check if WANDB_API_KEY is set
    wandb_key = os.getenv('WANDB_API_KEY')
    if not wandb_key:
        logger.warning("WANDB_API_KEY not found in environment. Make sure .env is loaded.")
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity'),
        config={
            'architecture': config['model']['architecture'],
            'training': config['model']['training'],
            'data': {
                'tickers': config['data']['tickers'],
                'interval': config['data']['interval'],
                'features': feature_names,
                'num_features': len(feature_names)
            }
        },
        name=f"training_{wandb.util.generate_id()}"
    )
    
    logger.info(f"Wandb initialized: {wandb.run.name}")


def train_model(model, X_train, y_train, X_val, y_val, config: dict):
    """
    Train the model.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Configuration dictionary
    
    Returns:
        Training history
    """
    logger.info("=" * 50)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 50)
    
    epochs = config['model']['training']['epochs']
    batch_size = config['model']['training']['batch_size']
    
    # Get callbacks (Early Stopping and ReduceLR)
    callbacks = get_callbacks(config)
    
    # DON'T use WandbCallback - it's deprecated and buggy
    # We'll log metrics manually after training
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Log training history to wandb manually
    for epoch in range(len(history.history['loss'])):
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': history.history['loss'][epoch],
            'train_accuracy': history.history['accuracy'][epoch],
            'train_precision': history.history['precision'][epoch],
            'train_recall': history.history['recall'][epoch],
            'val_loss': history.history['val_loss'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch],
            'val_precision': history.history['val_precision'][epoch],
            'val_recall': history.history['val_recall'][epoch],
        })
    
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 50)
    
    return history


def evaluate_model(model, X_test, y_test, config: dict):
    """
    Evaluate the model on test set.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        config: Configuration dictionary
    
    Returns:
        Test metrics dictionary
    """
    logger.info("Evaluating model on test set...")
    
    # Evaluate
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    # Create metrics dictionary
    metrics = {
        'test_loss': test_results[0],
        'test_accuracy': test_results[1],
        'test_precision': test_results[2],
        'test_recall': test_results[3]
    }
    
    # Calculate F1 score
    precision = metrics['test_precision']
    recall = metrics['test_recall']
    if precision + recall > 0:
        metrics['test_f1'] = 2 * (precision * recall) / (precision + recall)
    else:
        metrics['test_f1'] = 0.0
    
    # Log to wandb
    wandb.log(metrics)
    
    # Print results
    logger.info("Test Set Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics


def save_model(model, output_dir: str = "models/saved_models"):
    """
    Save the trained model.
    
    Args:
        model: Trained Keras model
        output_dir: Directory to save the model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model in Keras format
    model_path = output_path / "trading_signal_model.keras"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Also save as SavedModel format for deployment
    savedmodel_path = output_path / "trading_signal_model"
    model.export(savedmodel_path)
    logger.info(f"Model exported to {savedmodel_path}")
    
    # Log model artifact to wandb (instead of symlink)
    try:
        artifact = wandb.Artifact('trading-signal-model', type='model')
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        logger.info("Model logged to Wandb as artifact")
    except Exception as e:
        logger.warning(f"Could not log model to Wandb: {e}")
        logger.info("Model is still saved locally - this is fine!")


def main():
    """Main training function."""
    try:
        # Load configuration
        config = load_config()
        
        # Load datasets
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_datasets()
        
        # Convert labels to categorical
        y_train_cat, y_val_cat, y_test_cat = convert_labels_to_categorical(
            y_train, y_val, y_test
        )
        
        # Initialize Wandb
        initialize_wandb(config, feature_names)
        
        # Create model
        input_size = X_train.shape[1]
        model = create_model(input_size, config)
        
        # Train model
        history = train_model(model, X_train, y_train_cat, X_val, y_val_cat, config)
        
        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test_cat, config)
        
        # Save model
        save_model(model)
        
        # Finish wandb run
        wandb.finish()
        
        # Print final summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Final test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Final test F1-score: {metrics['test_f1']:.4f}")
        print(f"Model saved to: models/saved_models/")
        print("=" * 50 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()