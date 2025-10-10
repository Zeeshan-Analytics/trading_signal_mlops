"""
Fetch historical stock data from yfinance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_ticker_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetch data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        interval: Data interval (e.g., '1h')
        period: Historical period (e.g., '730d')
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        logger.info(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return None
        
        # Add ticker column
        df['ticker'] = ticker
        
        # Reset index to make datetime a column
        df.reset_index(inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} records for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_all_data(config: dict) -> pd.DataFrame:
    """
    Fetch data for all tickers in configuration.
    
    Args:
        config: Project configuration dictionary
    
    Returns:
        Combined DataFrame with all tickers
    """
    tickers = config['data']['tickers']
    interval = config['data']['interval']
    period = config['data']['period']
    
    logger.info(f"Starting data fetch for {len(tickers)} tickers...")
    logger.info(f"Parameters: interval={interval}, period={period}")
    
    all_data = []
    
    for ticker in tickers:
        df = fetch_ticker_data(ticker, interval, period)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No data fetched for any ticker!")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Total records fetched: {len(combined_df)}")
    logger.info(f"Date range: {combined_df['Datetime'].min()} to {combined_df['Datetime'].max()}")
    
    return combined_df


def save_raw_data(df: pd.DataFrame, output_path: str):
    """
    Save raw data to CSV.
    
    Args:
        df: DataFrame to save
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    logger.info(f"Raw data saved to {output_file}")


def main():
    """Main function to fetch and save data."""
    try:
        # Load configuration
        config = load_config()
        
        # Fetch data
        logger.info("=" * 50)
        logger.info("STARTING DATA FETCH")
        logger.info("=" * 50)
        
        df = fetch_all_data(config)
        
        # Save raw data
        output_path = "data/raw/stock_data.csv"
        save_raw_data(df, output_path)
        
        logger.info("=" * 50)
        logger.info("DATA FETCH COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        # Print summary
        print("\n" + "=" * 50)
        print("DATA FETCH SUMMARY")
        print("=" * 50)
        print(f"Total records: {len(df)}")
        print(f"Tickers: {df['ticker'].unique().tolist()}")
        print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Output file: {output_path}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise


if __name__ == "__main__":
    main()