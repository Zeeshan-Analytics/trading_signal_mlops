"""
Neural Network model for trading signal classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_model(input_size: int, config: dict) -> keras.Model:
    """
    Create a feedforward neural network for signal classification.
    
    Args:
        input_size: Number of input features
        config: Model configuration dictionary
    
    Returns:
        Compiled Keras model
    """
    architecture = config['model']['architecture']
    training = config['model']['training']
    
    hidden_layers = architecture['hidden_layers']
    dropout_rate = architecture['dropout_rate']
    output_size = architecture['output_size']
    
    logger.info(f"Creating model with input_size={input_size}")
    logger.info(f"Hidden layers: {hidden_layers}")
    logger.info(f"Dropout rate: {dropout_rate}")
    
    # Build model
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(hidden_layers[0], activation='relu', name='hidden_1'),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_layers[1], activation='relu', name='hidden_2'),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_layers[2], activation='relu', name='hidden_3'),
        layers.Dropout(dropout_rate),
        layers.Dense(output_size, activation='softmax', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training['learning_rate']),
        loss=training['loss'],
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    logger.info("Model created and compiled successfully")
    logger.info(f"\nModel Summary:")
    model.summary(print_fn=logger.info)
    
    return model


def get_callbacks(config: dict) -> list:
    """
    Create training callbacks.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    
    # Early stopping
    if config['model']['training']['early_stopping']['enabled']:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=config['model']['training']['early_stopping']['monitor'],
            patience=config['model']['training']['early_stopping']['patience'],
            restore_best_weights=config['model']['training']['early_stopping']['restore_best_weights'],
            verbose=1
        )
        callbacks.append(early_stopping)
        logger.info("Early stopping enabled")
    
    # Reduce learning rate on plateau
    if config['model']['training']['reduce_lr']['enabled']:
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['model']['training']['reduce_lr']['factor'],
            patience=config['model']['training']['reduce_lr']['patience'],
            min_lr=config['model']['training']['reduce_lr']['min_lr'],
            verbose=1
        )
        callbacks.append(reduce_lr)
        logger.info("Learning rate reduction enabled")
    
    return callbacks


if __name__ == "__main__":
    # Test model creation
    from src.utils.config_loader import load_config
    
    config = load_config()
    model = create_model(input_size=20, config=config)
    
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE TEST")
    print("="*50)
    print(f"Total parameters: {model.count_params():,}")
    print("="*50)