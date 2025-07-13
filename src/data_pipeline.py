"""
Data pipeline for financial time series data from Yahoo Finance.
"""

import datetime
from typing import Dict, List, Optional
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration settings for the pipeline"""
    
    @staticmethod
    def get_primary_tickers() -> List[str]:
        """Get list of primary tickers for analysis"""
        return ["AAPL", "TSLA", "SPY", "MSFT", "GOOGL"]

class FinancialDataPipeline:
    """
    Production-ready financial data pipeline with error handling,
    caching and data validation.
    """
    
    def __init__(self, data_dir: str = "data"):  # Fixed: double underscores
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def download_ticker_data(
            self,
            ticker: str,
            start_date: str,
            end_date: str,
            use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a single ticker with error handling and caching.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.csv"

        # Check cache first
        if use_cache and cache_file.exists():
            try:
                logger.info(f"Loading {ticker} from cache")
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return data
            except Exception as e:
                logger.warning(f"Cache read failed for {ticker}: {e}")

        # Download fresh data
        try:
            logger.info(f"Downloading {ticker} data from {start_date} to {end_date}")
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date)

            if data.empty:
                logger.error(f"No data received for {ticker}")
                return None
            
            # Validate data quality
            if not self._validate_data(data, ticker):
                return None
            
            # Save to cache
            data.to_csv(cache_file)
            logger.info(f"Data for {ticker} saved to cache")
            return data
        
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {e}")
            return None
        
    def download_multiple_tickers(
            self,
            tickers: List[str],
            start_date: str,
            end_date: str,
            use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Download data for multiple tickers"""
        
        results = {}
        failed_tickers = []

        logger.info(f"Downloading data for {len(tickers)} tickers")

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{len(tickers)})")

            data = self.download_ticker_data(ticker, start_date, end_date, use_cache)

            if data is not None:
                results[ticker] = data
            else:
                failed_tickers.append(ticker)

        # Summary
        logger.info(f"Successfully downloaded data for {len(results)} tickers")
        if failed_tickers:
            logger.warning("Failed to download data for: " + ", ".join(failed_tickers))

        return results

    def _validate_data(self, data: pd.DataFrame, ticker: str) -> bool:
        """Validate downloaded data quality"""
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Check required columns
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns in {ticker} data")
            return False
        
        # Check for reasonable data ranges
        if (data['Close'] <= 0).any():
            logger.error(f"Invalid price data for {ticker} (non-positive prices)")
            return False
        
        # Check for excessive missing data
        missing_pct = data['Close'].isna().sum() / len(data)
        if missing_pct > 0.1:  # More than 10% missing
            logger.error(f"Too much missing data for {ticker}: {missing_pct:.1%}")
            return False
        
        # Check for reasonable volume (unless it's crypto)
        if not ticker.endswith('-USD') and (data['Volume'] < 0).any():
            logger.error(f"Invalid volume data for {ticker}")
            return False
        
        logger.info(f"Data validation passed for {ticker}")
        return True
    
    def get_market_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics for multiple tickers.
        
        Args:
            data_dict: Dictionary of ticker symbols to DataFrames
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for ticker, data in data_dict.items():
            if data.empty:
                continue
            
            stats = {
                'Ticker': ticker,
                'Start Date': data.index.min().strftime('%Y-%m-%d'),
                'End Date': data.index.max().strftime('%Y-%m-%d'),
                'Total Days': len(data),
                'Mean Close': data['Close'].mean(),
                'Max Close': data['Close'].max(),
                'Min Close': data['Close'].min(),
                'Mean Volume': data['Volume'].mean()
            }
            summary_data.append(stats)
        
        return pd.DataFrame(summary_data)
    
    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame], filename: str = "market_data.pkl"):
        """
        Save processed data to a file.
        
        Args:
            data_dict: Dictionary of ticker DataFrames
            filename: Name of the output file
        """
        output_path = self.data_dir / filename
        
        try:
            pd.to_pickle(data_dict, output_path)
            logger.info(f"Processed data saved to {output_path}")

            # Also save summary as CSV for easy inspection
            summary = self.get_market_data_summary(data_dict)
            summary_path = self.data_dir / "market_data_summary.csv"
            summary.to_csv(summary_path, index=False)
            logger.info(f"Data summary saved to {summary_path}")

        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")

def main():
    """Test the data pipeline with sample tickers"""
    
    # Initialize pipeline
    pipeline = FinancialDataPipeline()

    # Get primary tickers for testing
    tickers = Config.get_primary_tickers()
    start_date = "2022-01-01"  # Extended date range for more data
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    print(" Downloading market data...")
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
    print("-" * 50)

    # Download data
    data_dict = pipeline.download_multiple_tickers(
        tickers=tickers, 
        start_date=start_date, 
        end_date=end_date
    )
    
    # Generate summary
    if data_dict:
        summary = pipeline.get_market_data_summary(data_dict)
        print("\n Data Download Summary:")
        print(summary.to_string(index=False))

        # Save processed data
        pipeline.save_processed_data(data_dict)

        print("\n Data processing complete!")
        print(f"Processed data saved to {pipeline.data_dir}")
    else:
        print(" No data downloaded. Please check the tickers and date range.")

if __name__ == "__main__":
    main()