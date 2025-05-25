"""
Adaptive Indicator Service

Provides machine learning enhanced indicators that adapt parameters based on market conditions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from hmmlearn import hmm
import logging
import pandas_ta as ta # Import pandas_ta
import os
import joblib

logger = logging.getLogger(__name__)

# Define a cache directory for adaptive models
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "cache", "adaptive_models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class AdaptiveIndicatorService:
    """
    Service for adaptive technical indicators using machine learning and statistical methods.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {} # To store trained models (in-memory for current session)
        self.model_cache_dir = MODEL_CACHE_DIR

    def _get_model_path(self, model_name: str) -> str:
        """Generates a filepath for a given model name in the cache directory."""
        return os.path.join(self.model_cache_dir, f"{model_name}.joblib")

    def _save_model(self, model: Any, model_name: str):
        """Saves a model to disk."""
        try:
            model_path = self._get_model_path(model_name)
            joblib.dump(model, model_path)
            logger.info(f"Saved model '{model_name}' to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model '{model_name}': {e}", exc_info=True)

    def _load_model(self, model_name: str) -> Optional[Any]:
        """Loads a model from disk if it exists."""
        model_path = self._get_model_path(model_name)
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info(f"Loaded model '{model_name}' from {model_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading model '{model_name}': {e}. Will retrain.", exc_info=True)
                return None
        return None

    def adaptive_moving_average(self, df: pd.DataFrame, price_col: str = 'close', base_window: int = 20, 
                                volatility_window: int = 20, volatility_factor: float = 1.0, 
                                min_alpha: float = 0.01, max_alpha: float = 0.5) -> pd.Series:
        """
        Calculate a volatility-adaptive Exponential Moving Average (EMA).
        The smoothing factor (alpha) of the EMA adapts based on recent price volatility.
        Higher volatility leads to a more responsive EMA (higher alpha), 
        lower volatility to a smoother EMA (lower alpha).

        Args:
            df: DataFrame with price data.
            price_col: Column to use for price (default 'close').
            base_window: Base window for initial EMA smoothing, also used for ATR calculation if not provided.
            volatility_window: Window for calculating volatility (e.g., standard deviation of returns or ATR).
            volatility_factor: Multiplier to scale normalized volatility for alpha adjustment.
            min_alpha: Minimum allowed alpha for the EMA.
            max_alpha: Maximum allowed alpha for the EMA.

        Returns:
            pd.Series of adaptive moving average values.
        """
        if price_col not in df.columns:
            logger.error(f"Price column '{price_col}' not found in DataFrame.")
            return pd.Series(dtype=float)
        
        if len(df) < max(base_window, volatility_window):
            logger.warning("Data length too short for adaptive moving average calculation.")
            return pd.Series(dtype=float, index=df.index)

        close_prices = df[price_col].astype(float)
        
        # 1. Calculate Volatility (e.g., using standard deviation of log returns)
        # Using log returns for volatility calculation
        log_returns = np.log(close_prices / close_prices.shift(1))
        # Alternatively, use ATR (Average True Range) from pandas_ta or implement here
        # For this example, std dev of log returns is used.
        volatility = log_returns.rolling(window=volatility_window, min_periods=volatility_window // 2).std()
        volatility.fillna(method='bfill', inplace=True) # Fill initial NaNs
        volatility.fillna(method='ffill', inplace=True) # Fill remaining NaNs

        # 2. Normalize Volatility (e.g., to a 0-1 range or Z-score)
        # Simple min-max normalization for this example over the available volatility series
        min_vol = volatility.min()
        max_vol = volatility.max()
        
        if max_vol == min_vol: # Avoid division by zero if volatility is constant
            normalized_volatility = pd.Series(0.5, index=volatility.index) # Default to a mid-point alpha
        else:
            normalized_volatility = (volatility - min_vol) / (max_vol - min_vol)
        
        # 3. Calculate Adaptive Alpha
        # Higher normalized volatility -> higher alpha (more responsive)
        # Alpha = min_alpha + (max_alpha - min_alpha) * (normalized_volatility ^ scaling_power)
        # A simpler linear scaling for now:
        adaptive_alpha = min_alpha + (max_alpha - min_alpha) * (normalized_volatility * volatility_factor)
        adaptive_alpha = adaptive_alpha.clip(lower=min_alpha, upper=max_alpha)
        
        # 4. Calculate Adaptive EMA
        # EMA_t = alpha_t * Price_t + (1 - alpha_t) * EMA_t-1
        adaptive_ema_values = pd.Series(index=df.index, dtype=float)
        
        if adaptive_ema_values.empty:
             return adaptive_ema_values

        # Initialize the first EMA value (e.g., with SMA or first price)
        # Using SMA for initial values can sometimes be problematic if adaptive_alpha starts very low.
        # Simple initialization: first price point.
        first_valid_index = close_prices.first_valid_index()
        if first_valid_index is None:
            logger.warning("No valid prices to initialize AMA.")
            return pd.Series(dtype=float, index=df.index)
            
        adaptive_ema_values[first_valid_index] = close_prices[first_valid_index]

        for i in range(df.index.get_loc(first_valid_index) + 1, len(df)):
            current_idx = df.index[i]
            prev_idx = df.index[i-1]
            
            alpha = adaptive_alpha.get(current_idx, min_alpha) # Get current alpha, default if NaN
            current_price = close_prices.get(current_idx)
            prev_ema = adaptive_ema_values.get(prev_idx)

            if pd.isna(current_price) or pd.isna(prev_ema):
                adaptive_ema_values[current_idx] = prev_ema # Carry forward if current price is NaN
                continue
            
            adaptive_ema_values[current_idx] = alpha * current_price + (1 - alpha) * prev_ema
            
        logger.info(f"Calculated adaptive moving average for {price_col}.")
        return adaptive_ema_values

    def dynamic_rsi_periods(self, df: pd.DataFrame, price_col: str = 'close', 
                              base_period: int = 14, 
                              regime_series: Optional[pd.Series] = None,
                              regime_map: Optional[Dict[int, int]] = None,
                              min_rsi_period: int = 7, max_rsi_period: int = 28,
                              n_hmm_states: int = 2, hmm_random_seed: Optional[int] = None) -> pd.Series:
        """
        Calculate RSI with a period that adapts to detected market regime.
        If regime_series is not provided, it will attempt to calculate it using market_regime_detector.

        Args:
            df: DataFrame with price data.
            price_col: Column to use for price (default 'close').
            base_period: Default RSI period if no regime adaptation occurs.
            regime_series: Optional pre-calculated series of market regimes (integers).
            regime_map: Optional dictionary mapping regime label (int) to RSI period (int).
                        Example: {0: 21, 1: 7} (regime 0 -> 21 period, regime 1 -> 7 period).
                        If None, a default heuristic might be applied based on n_hmm_states.
            min_rsi_period: Minimum allowable RSI period.
            max_rsi_period: Maximum allowable RSI period.
            n_hmm_states: Number of HMM states if regimes need to be calculated internally.
            hmm_random_seed: Random seed for internal HMM calculation for reproducibility.

        Returns:
            pd.Series of adaptive RSI values.
        """
        if price_col not in df.columns:
            logger.error(f"Price column '{price_col}' not found in DataFrame for dynamic RSI.")
            return pd.Series(dtype=float)

        if regime_series is None:
            logger.info("Regime series not provided for dynamic RSI, attempting to calculate internally.")
            regime_series = self.market_regime_detector(df, price_col=price_col, n_states=n_hmm_states, random_state_seed=hmm_random_seed)
        
        if regime_series is None or regime_series.empty or regime_series.isnull().all():
            logger.warning("Could not obtain or calculate market regimes. Falling back to base RSI period.")
            return df.ta.rsi(close=df[price_col], length=base_period) # Use pandas_ta for RSI

        # Ensure regime_series is aligned with df.index and forward fill NaNs
        regime_series = regime_series.reindex(df.index).ffill().bfill()

        # Define regime_map if not provided (example heuristic)
        # This mapping is crucial and should ideally be based on research or optimization.
        # Example: state 0 = longer period (low vol/ranging), state 1 = shorter period (high vol/trending)
        # This interpretation depends on how HMM states correlate with market behavior (e.g. mean/variance of returns in state)
        if regime_map is None:
            if n_hmm_states == 2:
                # Query the HMM model to interpret states (e.g., based on volatility)
                # This requires the model to be trained and available in self.models
                model_key = f"hmm_regime_{price_col}_{n_hmm_states}_log_returns"
                hmm_model = self.models.get(model_key)
                if hmm_model:
                    # Assuming states are ordered by volatility (e.g. HMM sorts them or we sort by covars/means)
                    # This is a simplification; proper state interpretation is complex.
                    # Let's say state with lower variance of returns is low-vol regime.
                    state_variances = [hmm_model.covars_[i][0,0] for i in range(hmm_model.n_components)]
                    low_vol_state = np.argmin(state_variances)
                    high_vol_state = np.argmax(state_variances)
                    if low_vol_state == high_vol_state: # only one state effectively or equal variance
                         regime_map = {state: base_period for state in range(n_hmm_states)}
                    else:
                        regime_map = {low_vol_state: max_rsi_period, high_vol_state: min_rsi_period}
                        # Fill any other states if n_hmm_states > 2 with base_period
                        for s in range(n_hmm_states):
                            if s not in regime_map: regime_map[s] = base_period
                    logger.info(f"Auto-generated regime_map based on HMM state variances: {regime_map}")
                else:
                    logger.warning("HMM model not found for regime interpretation. Using default equal periods for regime_map.")
                    regime_map = {state: base_period for state in range(n_hmm_states)}
            else: # For >2 states, default all to base_period if no map given.
                regime_map = {state: base_period for state in range(n_hmm_states)}
            logger.info(f"Using default/derived regime_map for RSI periods: {regime_map}")

        # Calculate RSI with adaptive periods
        # This requires iterating or grouping by period, then applying RSI.
        # A fully vectorized solution is complex if periods change frequently.
        # Iterative approach for clarity:
        adaptive_rsi_values = pd.Series(index=df.index, dtype=float)
        
        # Pre-calculate RSI for all relevant periods to speed up the loop
        # This is only efficient if the number of unique periods in regime_map is small.
        unique_periods = sorted(list(set(regime_map.values()))) 
        rsi_precalculated = {}
        for period in unique_periods:
            clamped_period = max(2, min(period, len(df)-1)) # Ensure period is valid for ta.rsi length
            if clamped_period < 2 : continue # RSI needs at least 2 periods
            try:
                rsi_precalculated[period] = df.ta.rsi(close=df[price_col], length=clamped_period)
            except Exception as e:
                logger.error(f"Error pre-calculating RSI for period {clamped_period}: {e}")
                rsi_precalculated[period] = pd.Series(np.nan, index=df.index) # Fill with NaNs on error

        current_period = base_period
        for idx, row_data in df.iterrows():
            regime = regime_series.get(idx)
            if pd.isna(regime):
                # If regime is NaN, use the last known period or base_period
                pass # current_period remains unchanged
            else:
                current_period = regime_map.get(int(regime), base_period)
            
            # Clamp period again just in case regime_map has extreme values
            clamped_current_period = max(min_rsi_period, min(max_rsi_period, current_period))
            clamped_current_period = max(2, clamped_current_period) # RSI min length for pandas-ta is usually 2

            if clamped_current_period in rsi_precalculated:
                adaptive_rsi_values[idx] = rsi_precalculated[clamped_current_period].get(idx)
            else:
                # This fallback is less efficient, should ideally be covered by precalculation
                logger.debug(f"Calculating RSI on-the-fly for period {clamped_current_period} at index {idx}")
                temp_rsi = df.ta.rsi(close=df[price_col], length=clamped_current_period)
                adaptive_rsi_values[idx] = temp_rsi.get(idx)
        
        logger.info(f"Calculated dynamic period RSI for {price_col}.")
        return adaptive_rsi_values

    def market_regime_detector(self, df: pd.DataFrame, price_col: str = 'close', n_states: int = 2, 
                               feature_col: Optional[str] = None, random_state_seed: Optional[int]=None,
                               use_cached_model: bool = True, retrain_model: bool = False) -> pd.Series:
        """
        Detect market regimes (e.g., trending, mean-reverting) using a Gaussian Hidden Markov Model (HMM).
        
        Args:
            df: DataFrame with price data. Must have a DatetimeIndex.
            price_col: Column to use for price (default 'close').
            n_states: Number of hidden states (regimes) to detect (default 2).
            feature_col: Optional pre-calculated feature column (e.g., 'returns') to use for HMM. 
                         If None, daily log returns of 'price_col' will be calculated and used.
            random_state_seed: Optional seed for HMM's random_state for reproducibility.
            use_cached_model: If True, attempts to load a pre-trained HMM model from cache.
            retrain_model: If True, forces retraining even if a cached model exists.

        Returns:
            pd.Series of detected regime labels (integers) indexed same as input df.
            Returns an empty Series if detection fails or data is insufficient.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be a DatetimeIndex for market_regime_detector.")
            return pd.Series(dtype=int)

        if price_col not in df.columns and feature_col not in df.columns and feature_col is None:
            logger.error(f"Price column '{price_col}' not found in DataFrame.")
            return pd.Series(dtype=int)
        
        if feature_col and feature_col not in df.columns:
            logger.error(f"Specified feature column '{feature_col}' not found in DataFrame.")
            return pd.Series(dtype=int)

        data_for_hmm = None
        if feature_col:
            data_for_hmm = df[[feature_col]].copy()
            logger.info(f"Using pre-calculated feature column '{feature_col}' for HMM.")
        else:
            # Calculate daily log returns as features for the HMM
            if df[price_col].isnull().any():
                logger.warning(f"Price column '{price_col}' contains NaNs. Dropping them for return calculation.")
            
            # Ensure prices are numeric and positive before log
            prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
            prices = prices[prices > 0]

            if len(prices) < n_states * 5: # Heuristic: need enough data
                 logger.error(f"Insufficient data points (after cleaning: {len(prices)}) to fit HMM with {n_states} states.")
                 return pd.Series(dtype=int)
            
            log_returns = np.log(prices / prices.shift(1)).dropna()
            data_for_hmm = log_returns.to_frame(name='returns')
            logger.info(f"Calculated log returns from '{price_col}' for HMM. Shape: {data_for_hmm.shape}")


        if data_for_hmm is None or data_for_hmm.empty or data_for_hmm.isnull().values.any():
            logger.error("No valid data available to train HMM after processing features.")
            return pd.Series(dtype=int)
        
        if len(data_for_hmm) < n_states * 5: # Heuristic: need enough data points per state
            logger.error(f"Insufficient data points ({len(data_for_hmm)}) to robustly fit HMM with {n_states} states.")
            return pd.Series(dtype=int)

        model_name = f"hmm_regime_{self._generate_hmm_model_suffix(price_col, n_states, feature_col, df.index.name or 'generic_index')}"

        model: Optional[hmm.GaussianHMM] = None

        if use_cached_model and not retrain_model:
            model = self._load_model(model_name)
            if model and model.n_components != n_states: # Check if loaded model matches requested states
                logger.warning(f"Cached HMM model '{model_name}' has {model.n_components} states, but {n_states} were requested. Retraining.")
                model = None # Force retrain
            elif model:
                 logger.info(f"Using cached HMM model: {model_name}")

        if model is None:
            logger.info(f"Training new HMM model: {model_name}")
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states, 
                    covariance_type="diag", 
                    n_iter=1000, 
                    random_state=np.random.RandomState(random_state_seed) if random_state_seed is not None else None,
                    init_params="cm", 
                    params="cmt"
                )
                model.fit(data_for_hmm.values)
                self._save_model(model, model_name) # Save after successful training
            except Exception as e:
                logger.error(f"Error training HMM model '{model_name}': {e}", exc_info=True)
                return pd.Series(dtype=int)
        
        try:
            # Predict the hidden states (regimes)
            hidden_states = model.predict(data_for_hmm.values)
            
            # Store the trained model in memory if needed for later inspection or re-use in current session
            self.models[model_name] = model 
            logger.info(f"HMM prediction complete. Detected {len(np.unique(hidden_states))} unique states for model {model_name}.")
            logger.debug(f"HMM means ({model_name}):\n{model.means_}\nHMM covariances ({model_name}):\n{model.covars_}")

            # Create a Pandas Series for the regimes, aligned with the original DataFrame's index
            # where data_for_hmm was derived from.
            regime_series = pd.Series(hidden_states, index=data_for_hmm.index, name=f'regime_{n_states}_states')
            
            # Reindex to match the original DataFrame's full index, filling gaps if any
            # This ensures the output series has the same length and index as df
            # Gaps created by initial .dropna() or .shift(1) will be NaN, can be ffilled or handled by caller
            return regime_series.reindex(df.index)

        except Exception as e:
            logger.error(f"Error during HMM market regime detection: {e}", exc_info=True)
            return pd.Series(dtype=int)

    def _generate_hmm_model_suffix(self, price_col: str, n_states: int, feature_col: Optional[str], index_name_hint: str) -> str:
        """Generates a consistent suffix for HMM model filenames/keys."""
        feat_part = feature_col if feature_col else f"{price_col}_logret"
        # Adding a hint from the index (e.g. symbol_timeframe) can help differentiate models if service is used for multiple series
        # This is a simplification. A more robust approach might involve hashing the input data characteristics or a UUID.
        return f"{index_name_hint}_{feat_part}_{n_states}states"

    def _get_adaptive_indicator_educational_content(self) -> Dict[str, Any]:
        """
        Provides educational content about adaptive indicators and market regimes.
        """
        return {
            "title": "Understanding Adaptive Indicators & Market Regimes",
            "introduction":
                "Adaptive indicators automatically adjust their parameters (like lookback periods or smoothing factors) "
                "in response to changing market conditions, such as volatility or detected market regimes. "
                "This helps them remain relevant and effective across different market dynamics.",
            "market_regimes": {
                "title": "Market Regimes (via Hidden Markov Models - HMM)",
                "concept":
                    "Markets don't behave the same way all the time. They can switch between different 'regimes' or states, "
                    "such as trending (high volatility, directional moves) or mean-reverting/ranging (lower volatility, sideways movement). "
                    "Hidden Markov Models (HMMs) are statistical models that can be used to identify these underlying, unobserved regimes based on price behavior (e.g., returns).",
                "interpretation":
                    "An HMM will assign each period to a regime (e.g., State 0, State 1). The characteristics of these states (like average return, volatility within the state) "
                    "must be analyzed to understand what each state represents (e.g., 'State 0 is a low-volatility, ranging market'). This interpretation is key.",
                "application": "Knowing the current regime can help in selecting appropriate trading strategies or indicator parameters."
            },
            "adaptive_moving_average": {
                "title": "Volatility-Adaptive Moving Average (AMA)",
                "concept": 
                    "A Volatility-Adaptive MA adjusts its responsiveness based on market volatility. "
                    "In highly volatile periods, it becomes more sensitive to recent prices (shorter effective period). "
                    "In calm periods, it becomes smoother and less sensitive (longer effective period).",
                "example_logic": "One common method is Kaufman's Adaptive Moving Average (KAMA). Another is to adjust an EMA's alpha based on a volatility measure like the Average True Range (ATR) or standard deviation of returns."
            },
            "dynamic_rsi": {
                "title": "Dynamic Period RSI",
                "concept":
                    "A Dynamic Period RSI changes its lookback period based on market conditions. "
                    "For example, in a strong trending regime, a shorter RSI period might be more effective for identifying overbought/oversold conditions relative to the trend. "
                    "In a ranging market, a longer period might be preferred to filter out noise.",
                "application": "The RSI period can be linked to the output of a market regime model or scaled by volatility."
            },
            "benefits_of_adaptive_indicators": [
                "Improved responsiveness in fast-moving markets.",
                "Reduced noise and false signals in choppy or ranging markets.",
                "Potentially more robust performance across different market conditions compared to fixed-parameter indicators."
            ],
            "challenges": [
                "Increased complexity in design and interpretation.",
                "Risk of overfitting if adaptation rules are too complex or poorly chosen.",
                "HMM state interpretation requires careful analysis and may not always be clear-cut."
            ],
            "disclaimer":
                "Adaptive indicators are tools, not guarantees of profit. Always use them in conjunction with a sound trading plan and risk management."
        } 