"""
Risk Management Service

This module provides a service for performing various risk management calculations,
including Value at Risk (VaR), portfolio correlation analysis, position sizing,
and volatility analysis.
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from scipy.stats import norm

from .cache_service import (
    get_cached_json_data,
    store_json_data,
    generate_indicator_cache_key 
)
from .data_fetcher import get_historical_data

logger = logging.getLogger(__name__)

class RiskManagementService:
    """
    Provides methods for risk management calculations.
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the RiskManagementService.

        Args:
            cache_enabled (bool): Whether to use caching for calculations.
        """
        self.cache_enabled = cache_enabled

    def _get_returns(self, symbol: str, timeframe: str, days: int) -> Optional[pd.Series]:
        """Helper to fetch and calculate returns."""
        df = get_historical_data(symbol, timeframe, days=days, use_cache=self.cache_enabled)
        if df is None or df.empty or 'close' not in df:
            logger.warning(f"Could not fetch or process data for {symbol} ({timeframe}) for returns calculation.")
            return None
        return df['close'].pct_change().dropna()

    def calculate_var(
        self,
        symbol: str,
        timeframe: str,
        days: int = 252,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate Value at Risk (VaR).

        Args:
            symbol (str): The trading symbol (e.g., 'BTCUSDT').
            timeframe (str): The timeframe for data (e.g., '1d', '4h').
            days (int): Number of historical days to use for calculation.
            confidence_level (float): The confidence level for VaR (e.g., 0.95 for 95%).
            method (str): 'historical' or 'parametric'.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing VaR result or None if calculation fails.
                Includes 'var_value', 'confidence_level', 'method', 'period_days', 'educational_note'.
        """
        cache_key_params = {
            "symbol": symbol, 
            "timeframe": timeframe, 
            "days": days, 
            "confidence": confidence_level,
            "method": method
        }
        cache_key = generate_indicator_cache_key("var", cache_key_params, symbol, timeframe)

        if self.cache_enabled:
            cached_result = get_cached_json_data(cache_key)
            if cached_result:
                logger.debug(f"Using cached VaR for {symbol} {timeframe}")
                return cached_result

        returns = self._get_returns(symbol, timeframe, days)
        if returns is None or returns.empty:
            return None

        var_value = None
        if method == "historical":
            if not returns.empty:
                var_value = -np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            if not returns.empty:
                mean_return = np.mean(returns)
                std_dev = np.std(returns)
                # Z-score for the specified confidence level
                z_score = norm.ppf(confidence_level)
                var_value = -(mean_return - z_score * std_dev) # Negative because VaR is a loss
        else:
            logger.error(f"Invalid VaR method: {method}. Choose 'historical' or 'parametric'.")
            return None

        if var_value is None:
            return None

        result = {
            "var_value": round(var_value * 100, 4),  # As percentage
            "confidence_level": confidence_level,
            "method": method,
            "period_days": days,
            "symbol": symbol,
            "timeframe": timeframe,
            "educational_note": (
                f"Value at Risk ({method.capitalize()}) at {confidence_level*100:.0f}% confidence level "
                f"suggests that the maximum expected loss for {symbol} over the next period "
                f"(based on {days} days of historical data for the {timeframe} timeframe) "
                f"is {var_value*100:.2f}%. This means there is a {(1-confidence_level)*100:.0f}% chance "
                f"of experiencing a loss greater than this amount."
            )
        }

        if self.cache_enabled:
            store_json_data(cache_key, result, timeframe=timeframe)
            logger.debug(f"Stored VaR for {symbol} {timeframe} in cache.")
            
        return result

    def correlation_analysis(
        self,
        symbols: List[str],
        timeframe: str,
        days: int = 90
    ) -> Optional[Dict[str, Any]]:
        """
        Perform correlation analysis between multiple symbols.

        Args:
            symbols (List[str]): List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT']).
            timeframe (str): The timeframe for data (e.g., '1d', '4h').
            days (int): Number of historical days to use for calculation.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with correlation matrix and educational note, or None.
        """
        if len(symbols) < 2:
            logger.error("Correlation analysis requires at least two symbols.")
            return None

        cache_key_params = {"symbols": sorted(symbols), "timeframe": timeframe, "days": days}
        # Use the first symbol and timeframe for generic cache key generation parts
        cache_key = generate_indicator_cache_key("correlation", cache_key_params, symbols[0], timeframe)

        if self.cache_enabled:
            cached_result = get_cached_json_data(cache_key)
            if cached_result:
                logger.debug(f"Using cached correlation for {', '.join(symbols)} {timeframe}")
                return cached_result
        
        all_returns = {}
        for symbol in symbols:
            returns = self._get_returns(symbol, timeframe, days)
            if returns is not None and not returns.empty:
                all_returns[symbol] = returns
            else:
                logger.warning(f"Could not fetch returns for {symbol} for correlation analysis.")
        
        if len(all_returns) < 2:
            logger.error("Not enough data to perform correlation analysis for the given symbols.")
            return None

        returns_df = pd.DataFrame(all_returns).dropna()
        
        if returns_df.shape[0] < 2 or returns_df.shape[1] < 2 : # Need at least 2 data points and 2 assets
            logger.error("Insufficient data points or assets after aligning timestamps for correlation.")
            return None

        correlation_matrix = returns_df.corr()

        result = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "period_days": days,
            "timeframe": timeframe,
            "symbols": symbols,
            "educational_note": (
                f"Correlation analysis for {', '.join(symbols)} over {days} days on the {timeframe} timeframe. "
                "Values range from -1 to 1. A value close to 1 indicates a strong positive correlation "
                "(assets tend to move in the same direction). A value close to -1 indicates a strong "
                "negative correlation (assets tend to move in opposite directions). A value near 0 "
                "suggests little to no linear relationship."
            )
        }

        if self.cache_enabled:
            store_json_data(cache_key, result, timeframe=timeframe)
            logger.debug(f"Stored correlation for {', '.join(symbols)} {timeframe} in cache.")

        return result

    def position_sizing(
        self,
        account_balance: float,
        risk_per_trade_percent: float,
        stop_loss_percent: float,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        asset_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate position size based on account risk and stop-loss.
        Either (entry_price and stop_loss_price) or (asset_price and stop_loss_percent) must be provided.

        Args:
            account_balance (float): Total trading account balance.
            risk_per_trade_percent (float): Percentage of account balance to risk per trade (e.g., 1 for 1%).
            stop_loss_percent (float): Percentage distance of stop-loss from entry price (e.g., 5 for 5%).
                                     Used if entry_price and stop_loss_price are not given.
            entry_price (Optional[float]): The entry price of the asset.
            stop_loss_price (Optional[float]): The stop-loss price for the asset.
            asset_price (Optional[float]): Current price of the asset (used with stop_loss_percent).


        Returns:
            Optional[Dict[str, Any]]: Dictionary with position sizing details or None if inputs are invalid.
        """
        if account_balance <= 0:
            logger.error("Account balance must be positive.")
            return None
        if not (0 < risk_per_trade_percent <= 100):
            logger.error("Risk per trade percent must be between 0 and 100 (exclusive of 0).")
            return None

        amount_to_risk = account_balance * (risk_per_trade_percent / 100.0)
        position_size_units = None
        position_size_value = None
        actual_stop_loss_percent = None

        if entry_price is not None and stop_loss_price is not None:
            if entry_price <= 0 or stop_loss_price <= 0:
                logger.error("Entry price and stop loss price must be positive.")
                return None
            if entry_price == stop_loss_price:
                logger.error("Entry price and stop loss price cannot be the same.")
                return None
            
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit == 0: # Should be caught by previous check, but as safeguard
                 logger.error("Risk per unit is zero, cannot calculate position size.")
                 return None
            position_size_units = amount_to_risk / risk_per_unit
            position_size_value = position_size_units * entry_price
            actual_stop_loss_percent = (risk_per_unit / entry_price) * 100

        elif asset_price is not None and stop_loss_percent is not None:
            if asset_price <= 0:
                logger.error("Asset price must be positive.")
                return None
            if not (0 < stop_loss_percent <= 100):
                logger.error("Stop loss percent must be between 0 and 100 (exclusive of 0).")
                return None
            
            risk_per_unit = asset_price * (stop_loss_percent / 100.0)
            if risk_per_unit == 0:
                 logger.error("Risk per unit is zero based on stop_loss_percent, cannot calculate position size.")
                 return None
            position_size_units = amount_to_risk / risk_per_unit
            position_size_value = position_size_units * asset_price
            actual_stop_loss_percent = stop_loss_percent
        else:
            logger.error("Either (entry_price and stop_loss_price) or (asset_price and stop_loss_percent) must be provided.")
            return None

        if position_size_units is None or position_size_value is None:
            return None

        result = {
            "account_balance": round(account_balance, 2),
            "risk_per_trade_percent": risk_per_trade_percent,
            "amount_to_risk": round(amount_to_risk, 2),
            "stop_loss_percent_input": stop_loss_percent if entry_price is None else None, # The user input SL %
            "entry_price_input": entry_price,
            "stop_loss_price_input": stop_loss_price,
            "asset_price_input": asset_price,
            "calculated_stop_loss_percent": round(actual_stop_loss_percent, 2) if actual_stop_loss_percent is not None else None,
            "position_size_units": round(position_size_units, 8), # typically for crypto
            "position_size_value": round(position_size_value, 2),
            "educational_note": (
                f"For an account of ${account_balance:,.2f}, risking {risk_per_trade_percent:.2f}% (${amount_to_risk:,.2f}) per trade, "
                f"with a calculated stop-loss of {actual_stop_loss_percent:.2f}%, the suggested position size is "
                f"{position_size_units:.8f} units, valued at approximately ${position_size_value:,.2f}. "
                "This helps manage risk by ensuring a single trade's loss doesn't exceed the predefined risk tolerance."
            )
        }
        return result

    def volatility_analysis(
        self,
        symbol: str,
        timeframe: str,
        days: int = 30,
        window: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Perform volatility analysis (e.g., rolling standard deviation of returns).

        Args:
            symbol (str): The trading symbol.
            timeframe (str): The timeframe for data.
            days (int): Number of historical days to use.
            window (int): Rolling window for volatility calculation.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with volatility data or None.
        """
        cache_key_params = {"symbol": symbol, "timeframe": timeframe, "days": days, "window": window}
        cache_key = generate_indicator_cache_key("volatility", cache_key_params, symbol, timeframe)

        if self.cache_enabled:
            cached_result = get_cached_json_data(cache_key)
            if cached_result:
                logger.debug(f"Using cached volatility for {symbol} {timeframe}")
                return cached_result

        returns = self._get_returns(symbol, timeframe, days)
        if returns is None or returns.empty or len(returns) < window:
            logger.warning(f"Not enough data for volatility analysis of {symbol} ({timeframe}) with window {window}. Requires at least {window} data points, got {len(returns) if returns is not None else 0}.")
            return None

        rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252) # Annualized if daily returns
        # Adjust sqrt factor based on timeframe if more precision needed, e.g. sqrt(252*6) for 4h if 252 is annual trading days
        # For simplicity, using 252 as a general annualization factor for returns.
        # If timeframe is not daily, this annualization might be misleading.
        # Consider if raw period volatility is more appropriate or a more complex annualization.

        current_volatility = rolling_volatility.iloc[-1] if not rolling_volatility.empty else None
        
        if current_volatility is None:
            return None

        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "period_days": days,
            "calculation_window": window,
            "current_annualized_volatility_percent": round(current_volatility * 100, 4) if pd.notna(current_volatility) else None,
            # "volatility_series": {str(idx): round(val * 100, 4) if pd.notna(val) else None for idx, val in rolling_volatility.items()}, # Optional: include full series
            "educational_note": (
                f"The {window}-period rolling annualized volatility for {symbol} on the {timeframe} timeframe is currently "
                f"{current_volatility*100:.2f}%. Higher volatility indicates greater price fluctuations and potential risk, "
                f"while lower volatility suggests more stability. This is based on {days} days of data."
            )
        }
        
        if self.cache_enabled:
            store_json_data(cache_key, result, timeframe=timeframe)
            logger.debug(f"Stored volatility for {symbol} {timeframe} in cache.")

        return result

    def get_risk_assessment_summary(
        self,
        symbol: str,
        timeframe: str,
        account_balance: float,
        risk_per_trade_percent: float,
        stop_loss_percent: float, # Default stop loss percent for asset if specific not given
        var_days: int = 252,
        var_confidence: float = 0.95,
        vol_days: int = 30,
        vol_window: int = 20,
        asset_price: Optional[float] = None # Current asset price, needed for position sizing based on SL %
    ) -> Optional[Dict[str, Any]]:
        """
        Provides a consolidated risk assessment summary for a symbol.
        
        Args:
            symbol (str): Trading symbol.
            timeframe (str): Data timeframe.
            account_balance (float): User's account balance.
            risk_per_trade_percent (float): Desired risk per trade.
            stop_loss_percent (float): General stop loss assumption for position sizing.
            var_days (int): Days for VaR calculation.
            var_confidence (float): Confidence for VaR.
            vol_days (int): Days for volatility calculation.
            vol_window (int): Window for volatility.
            asset_price (Optional[float]): Current price of the asset. Fetched if None.

        Returns:
            Optional[Dict[str, Any]]: A summary dictionary or None.
        """
        
        if asset_price is None:
            price_data = get_historical_data(symbol, timeframe, days=2, use_cache=self.cache_enabled) # get latest price
            if price_data is not None and not price_data.empty and 'close' in price_data:
                asset_price = price_data['close'].iloc[-1]
            else:
                logger.warning(f"Could not fetch current asset price for {symbol} for risk summary.")
                # We can proceed without position sizing if asset price is unavailable
        
        var_data = self.calculate_var(symbol, timeframe, days=var_days, confidence_level=var_confidence)
        vol_data = self.volatility_analysis(symbol, timeframe, days=vol_days, window=vol_window)
        
        pos_size_data = None
        if asset_price is not None: # Only calculate if we have asset_price
            pos_size_data = self.position_sizing(
                account_balance, 
                risk_per_trade_percent, 
                stop_loss_percent,
                asset_price=asset_price
            )
        else:
            logger.warning(f"Asset price for {symbol} is unavailable, skipping position sizing in summary.")


        if not var_data and not vol_data and not pos_size_data:
            logger.error(f"Failed to compute any risk metrics for {symbol} on {timeframe}.")
            return None

        summary = {
            "symbol": symbol,
            "timeframe": timeframe,
            "value_at_risk": var_data,
            "volatility": vol_data,
            "position_sizing_example": pos_size_data,
            "educational_summary": (
                f"Risk Assessment for {symbol} ({timeframe}):\\n"
                f"VaR ({var_confidence*100}%): Expect to not lose more than {var_data['var_value']:.2f}% in the next period (based on {var_days} days).\\n"
                f"Volatility: Current annualized volatility is {vol_data['current_annualized_volatility_percent']:.2f}% (based on {vol_days} days, {vol_window}-period window).\\n"
            )
        }
        if pos_size_data:
             summary["educational_summary"] += (
                f"Position Sizing: For a ${account_balance:,.2f} account, risking {risk_per_trade_percent}%, "
                f"with a {stop_loss_percent}% stop-loss from current price ({asset_price:,.2f}), you could trade approx. "
                f"{pos_size_data['position_size_units']:.4f} units (value ${pos_size_data['position_size_value']:,.2f}).\\n"
                f"Always adapt risk parameters to your strategy and market conditions."
             )
        else:
            summary["educational_summary"] += f"Position sizing example could not be generated due to missing asset price.\\n"
            
        summary["educational_summary"] += "This information is for educational purposes and not financial advice."


        return summary

# Example Usage (for testing or direct script execution):
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG) # Enable debug logging for example
    
    # Setup console logging for example
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.DEBUG)


    risk_service = RiskManagementService(cache_enabled=True)
    
    # Test VaR
    var_result_hist = risk_service.calculate_var("BTCUSDT", "1d", days=252, confidence_level=0.99, method="historical")
    if var_result_hist:
        print(f"\nHistorical VaR: {var_result_hist['var_value']}%")
        print(var_result_hist['educational_note'])
    else:
        print("\nFailed to calculate Historical VaR.")

    var_result_param = risk_service.calculate_var("BTCUSDT", "1d", days=252, confidence_level=0.99, method="parametric")
    if var_result_param:
        print(f"\nParametric VaR: {var_result_param['var_value']}%")
        print(var_result_param['educational_note'])
    else:
        print("\nFailed to calculate Parametric VaR.")

    # Test Correlation
    correlation_result = risk_service.correlation_analysis(["BTCUSDT", "ETHUSDT", "SOLUSDT"], "1d", days=90)
    if correlation_result:
        print(f"\nCorrelation Matrix:\n{pd.DataFrame(correlation_result['correlation_matrix'])}")
        print(correlation_result['educational_note'])
    else:
        print("\nFailed to calculate Correlation Matrix.")

    # Test Position Sizing (using asset_price and stop_loss_percent)
    btc_price_for_sizing = get_historical_data("BTCUSDT", "1d", days=2)['close'].iloc[-1] # approx current price
    
    position_size_result_sl_percent = risk_service.position_sizing(
        account_balance=10000, 
        risk_per_trade_percent=1, 
        stop_loss_percent=5,
        asset_price=btc_price_for_sizing
    )
    if position_size_result_sl_percent:
        print(f"\nPosition Sizing (SL %): Max units {position_size_result_sl_percent['position_size_units']}, value ${position_size_result_sl_percent['position_size_value']:.2f}")
        print(position_size_result_sl_percent['educational_note'])
    else:
        print("\nFailed to calculate Position Size with SL %.")

    # Test Position Sizing (using entry_price and stop_loss_price)
    entry = 60000
    stop_loss = 57000 # 5% stop loss from 60000
    position_size_result_prices = risk_service.position_sizing(
        account_balance=10000, 
        risk_per_trade_percent=1, 
        stop_loss_percent=None, # Explicitly None
        entry_price=entry,
        stop_loss_price=stop_loss 
    )
    if position_size_result_prices:
        print(f"\nPosition Sizing (Prices): Max units {position_size_result_prices['position_size_units']}, value ${position_size_result_prices['position_size_value']:.2f}")
        print(position_size_result_prices['educational_note'])
    else:
        print("\nFailed to calculate Position Size with prices.")


    # Test Volatility Analysis
    volatility_result = risk_service.volatility_analysis("BTCUSDT", "1d", days=90, window=20)
    if volatility_result:
        print(f"\nAnnualized Volatility: {volatility_result['current_annualized_volatility_percent']}%")
        print(volatility_result['educational_note'])
    else:
        print("\nFailed to calculate Volatility.")

    # Test Risk Assessment Summary
    summary = risk_service.get_risk_assessment_summary(
        symbol="ETHUSDT",
        timeframe="4h",
        account_balance=25000,
        risk_per_trade_percent=2,
        stop_loss_percent=3,
        asset_price=3000 # Example price
    )
    if summary:
        print("\n--- Risk Assessment Summary for ETHUSDT (4h) ---")
        print(f"Value at Risk ({summary['value_at_risk']['confidence_level']*100}%): {summary['value_at_risk']['var_value']:.2f}% (Method: {summary['value_at_risk']['method']})")
        print(f"Annualized Volatility: {summary['volatility']['current_annualized_volatility_percent']:.2f}% (Window: {summary['volatility']['calculation_window']})")
        if summary['position_sizing_example']:
            print(f"Example Position Size (Units): {summary['position_sizing_example']['position_size_units']:.4f}")
            print(f"Example Position Size (Value): ${summary['position_sizing_example']['position_size_value']:,.2f}")
        print(f"Educational Summary:\n{summary['educational_summary']}")
    else:
        print("\nFailed to generate Risk Assessment Summary for ETHUSDT.") 