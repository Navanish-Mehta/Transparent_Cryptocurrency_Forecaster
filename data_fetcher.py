"""
Data fetching and preprocessing utilities for Transparent Cryptocurrency Price Forecaster.

- Fetches OHLCV from Yahoo Finance via yfinance as a robust free source
- Cleans data, fills missing values
- Generates engineered features (returns, rolling stats, lag features)
- Splits into train/test sets reproducibly

Author: Your Name
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

try:
	import yfinance as yf
except Exception:  # pragma: no cover
	raise


SUPPORTED_SYMBOLS = {
	"BTC-USD": "Bitcoin",
	"ETH-USD": "Ethereum",
	"BNB-USD": "Binance Coin",
	"ADA-USD": "Cardano",
	"SOL-USD": "Solana",
}


@dataclass
class Dataset:
	"""Container for features and target split."""
	X_train: pd.DataFrame
	X_test: pd.DataFrame
	y_train: pd.Series
	y_test: pd.Series
	feature_names: list
	meta: Dict


def fetch_ohlcv(symbol: str, start: str = "2018-01-01", end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
	"""Download OHLCV data using yfinance.

	Parameters
	----------
	symbol : str
		Ticker like "BTC-USD".
	start : str
		Start date YYYY-MM-DD
	end : Optional[str]
		End date YYYY-MM-DD. Defaults to today.
	interval : str
		Supported by yfinance (e.g., "1d", "1h").

	Returns
	-------
	pd.DataFrame
		Indexed by Datetime, columns: Open, High, Low, Close, Adj Close, Volume
	"""
	if end is None:
		end = dt.date.today().isoformat()

	data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
	if data is None or data.empty:
		raise ValueError(f"No data returned for {symbol} with interval={interval}")

	# Handle MultiIndex columns if present
	if isinstance(data.columns, pd.MultiIndex):
		# Flatten MultiIndex columns to single level
		data.columns = data.columns.get_level_values(0)
	
	# Ensure datetime index and consistent column names
	data = data.rename(columns=lambda c: c.strip().title()).sort_index()
	data.index = pd.to_datetime(data.index)
	
	# Remove any duplicate or unwanted columns
	expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
	available_columns = [col for col in expected_columns if col in data.columns]
	data = data[available_columns]
	
	return data


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
	"""Basic cleaning: forward/backward fill, drop duplicates, enforce dtypes."""
	df = df.copy()
	df = df[~df.index.duplicated(keep="first")]
	# Fill missing values sensibly
	numeric_cols = df.columns
	df[numeric_cols] = df[numeric_cols].ffill().bfill()
	return df


def add_features(df: pd.DataFrame, target_col: str = "Close", lags: int = 10) -> pd.DataFrame:
	"""Engineer time-series features for ML models.

	Features include:
	- Daily return and log-return
	- Rolling mean/volatility over multiple windows
	- Lagged target values up to `lags`
	"""
	df = df.copy()
	# Returns
	df["return"] = df[target_col].pct_change()
	df["log_return"] = np.log(df[target_col]).diff()

	# Rolling features
	for window in [3, 7, 14, 30]:
		df[f"roll_mean_{window}"] = df[target_col].rolling(window).mean()
		df[f"roll_std_{window}"] = df[target_col].rolling(window).std()

	# Lag features for target leakage-safe ML
	for lag in range(1, lags + 1):
		df[f"lag_{lag}"] = df[target_col].shift(lag)

	# Volume related signal (scaled)
	if "Volume" in df.columns:
		df["vol_change"] = df["Volume"].pct_change()

	# Drop initial NaNs
	df = df.replace([np.inf, -np.inf], np.nan).dropna()
	return df


def train_test_split_ts(df: pd.DataFrame, target_col: str = "Close", test_size: float = 0.2) -> Dataset:
	"""Time-series split preserving order.

	Returns Dataset object with features/target split.
	"""
	if not 0.0 < test_size < 1.0:
		raise ValueError("test_size must be in (0,1)")

	feature_cols = [c for c in df.columns if c != target_col]
	X = df[feature_cols]
	y = df[target_col].squeeze()  # Ensure y is a Series, not DataFrame

	n = len(df)
	split_idx = int(n * (1 - test_size))
	X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
	y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

	return Dataset(
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		feature_names=feature_cols,
		meta={"split_index": split_idx},
	)


def prepare_dataset(symbol: str = "BTC-USD", start: str = "2018-01-01", end: Optional[str] = None, interval: str = "1d", lags: int = 10, test_size: float = 0.2) -> Tuple[pd.DataFrame, Dataset]:
	"""High-level convenience: fetch, clean, feature-engineer, split.

	Returns (full_df_with_features, Dataset)
	"""
	df = fetch_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
	df = clean_ohlcv(df)
	df_features = add_features(df, target_col="Close", lags=lags)
	dataset = train_test_split_ts(df_features, target_col="Close", test_size=test_size)
	return df, dataset