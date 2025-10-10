"""
Build features from raw stock data.
Calculate technical indicators and prepare features for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_sma(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return df[column].rolling(window=window).mean()


def calculate_ema(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return df[column].ewm(span=window, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, column: str, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df: pd.DataFrame, column: str) -> tuple:
    """Calculate MACD, Signal line, and Histogram."""
    ema_12 = calculate_ema(df, column, 12)
    ema_26 = calculate_ema(df, column, 26)
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger_bands(df: pd.DataFrame, column: str, window: int = 20) -> tuple:
    """Calculate Bollinger Bands."""
    sma = calculate_sma(df, column, window)
    std = df[column].rolling(window=window).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, sma, lower


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    logger.info("Calculating technical indicators...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Sort by ticker and datetime
    df = df.sort_values(['ticker', 'Datetime']).reset_index(drop=True)
    
    # Process each ticker separately
    processed_dfs = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        
        # Moving Averages
        ticker_df['SMA_10'] = calculate_sma(ticker_df, 'Close', 10)
        ticker_df['SMA_20'] = calculate_sma(ticker_df, 'Close', 20)
        ticker_df['SMA_50'] = calculate_sma(ticker_df, 'Close', 50)
        ticker_df['EMA_12'] = calculate_ema(ticker_df, 'Close', 12)
        ticker_df['EMA_26'] = calculate_ema(ticker_df, 'Close', 26)
        
        # RSI
        ticker_df['RSI_14'] = calculate_rsi(ticker_df, 'Close', 14)
        
        # MACD
        macd, signal, hist = calculate_macd(ticker_df, 'Close')
        ticker_df['MACD'] = macd
        ticker_df['MACD_signal'] = signal
        ticker_df['MACD_hist'] = hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(ticker_df, 'Close', 20)
        ticker_df['BB_upper'] = bb_upper
        ticker_df['BB_middle'] = bb_middle
        ticker_df['BB_lower'] = bb_lower
        
        # Volume indicators
        ticker_df['volume_sma_20'] = calculate_sma(ticker_df, 'Volume', 20)
        
        # Price changes (for labels)
        ticker_df['price_change'] = ticker_df['Close'].pct_change()
        ticker_df['future_return'] = ticker_df['Close'].pct_change().shift(-1)  # Next period return
        
        processed_dfs.append(ticker_df)
    
    # Combine all tickers
    result_df = pd.concat(processed_dfs, ignore_index=True)
    
    logger.info(f"Technical indicators calculated. Shape: {result_df.shape}")
    
    return result_df


def create_labels(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Create trading signal labels based on future returns.
    
    Args:
        df: DataFrame with features
        thresholds: Dictionary with signal thresholds
    
    Returns:
        DataFrame with labels added
    """
    logger.info("Creating labels...")
    
    df = df.copy()
    
    def classify_signal(future_return):
        if pd.isna(future_return):
            return 2  # hold (for missing values)
        elif future_return >= thresholds['strong_buy']:
            return 4  # strong_buy
        elif future_return >= thresholds['buy']:
            return 3  # buy
        elif future_return <= thresholds['strong_sell']:
            return 0  # strong_sell
        elif future_return <= thresholds['sell']:
            return 1  # sell
        else:
            return 2  # hold
    
    df['signal'] = df['future_return'].apply(classify_signal)
    
    # Log label distribution
    label_counts = df['signal'].value_counts().sort_index()
    logger.info("Label distribution:")
    signal_names = ['strong_sell', 'sell', 'hold', 'buy', 'strong_buy']
    for label, count in label_counts.items():
        logger.info(f"  {signal_names[label]}: {count} ({count/len(df)*100:.2f}%)")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing NaN values and outliers.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Data shape before cleaning: {df.shape}")
    
    # Remove rows with NaN in critical columns
    critical_columns = ['Close', 'Volume', 'SMA_20', 'RSI_14', 'MACD', 'signal']
    df = df.dropna(subset=critical_columns)
    
    logger.info(f"Data shape after cleaning: {df.shape}")
    
    return df


def save_features(df: pd.DataFrame, output_path: str):
    """Save processed features to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    logger.info(f"Features saved to {output_file}")


def main():
    """Main function to build features."""
    try:
        # Load configuration
        config = load_config()
        
        logger.info("=" * 50)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("=" * 50)
        
        # Load raw data
        input_path = "data/raw/stock_data.csv"
        logger.info(f"Loading raw data from {input_path}...")
        df = pd.read_csv(input_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Create labels
        thresholds = config['signals']['thresholds']
        df = create_labels(df, thresholds)
        
        # Clean data
        df = clean_data(df)
        
        # Save processed features
        output_path = "data/processed/features.csv"
        save_features(df, output_path)
        
        logger.info("=" * 50)
        logger.info("FEATURE ENGINEERING COMPLETED")
        logger.info("=" * 50)
        
        # Print summary
        print("\n" + "=" * 50)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 50)
        print(f"Total records: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        print(f"Feature columns: {df.columns.tolist()}")
        print(f"Output file: {output_path}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to build features: {e}")
        raise


if __name__ == "__main__":
    main()