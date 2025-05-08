"""
Market analyzer module for financial data analysis.

This module provides a MarketAnalyzer class that can be used to fetch, analyze, and visualize
financial market data using different trading timeframes.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.jupyter.kernel import trading_styles
from src.jupyter.kernel import market_data

# Set up logging
logger = logging.getLogger(__name__)

# Helper function to convert confidence levels to numeric values for comparison
def confidence_level(confidence: str) -> int:
    """Convert confidence string to numeric value for comparison."""
    levels = {"high": 3, "medium": 2, "low": 1}
    return levels.get(confidence.lower(), 0)

class MarketAnalyzer:
    """
    Market analyzer for financial data analysis.
    
    This class provides methods to fetch, analyze, and visualize financial market data
    using different trading timeframes (short, medium, long).
    
    Args:
        symbol: Stock ticker symbol to analyze
        timeframe: Trading timeframe ('short', 'medium', 'long')
        use_test_data: If True, uses test data instead of fetching from API
    """
    
    def __init__(self, symbol: str, timeframe: str = "medium", use_test_data: bool = False):
        """
        Initialize the market analyzer.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Trading timeframe ('short', 'medium', 'long')
            use_test_data: If True, uses test data instead of fetching from API
        
        Raises:
            ValueError: If the timeframe is invalid
        """
        self.symbol = symbol
        self.timeframe = timeframe.lower()
        self.use_test_data = use_test_data
        self.trading_style = self._get_trading_style(self.timeframe)
        self.data = None
        self.performance = None
        self.analysis_results = {}
        self.visualizations = {}
        
        logger.info(f"Initialized MarketAnalyzer for {symbol} with {timeframe} timeframe")
    
    def _get_trading_style(self, timeframe: str) -> Dict[str, Any]:
        """
        Get the trading style settings for the specified timeframe.
        
        Args:
            timeframe: Trading timeframe ('short', 'medium', 'long')
            
        Returns:
            Dict containing the trading style settings
            
        Raises:
            ValueError: If the timeframe is invalid
        """
        if timeframe == "short":
            return trading_styles.SHORT_SETTINGS
        elif timeframe == "medium":
            return trading_styles.MEDIUM_SETTINGS
        elif timeframe == "long":
            return trading_styles.LONG_SETTINGS
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch market data for the symbol using the configured timeframe.
        
        Returns:
            DataFrame containing the market data
            
        Raises:
            ValueError: If no data is found for the symbol
        """
        from src.jupyter.kernel.market_data import fetch_market_data
        
        # Use test data for testing purposes
        if self.use_test_data:
            # Create mock data with 100 rows
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min')
            
            # Generate random price data with a slight uptrend
            base_price = 50000  # Base price for BTC
            trend = np.linspace(0, 0.05, 100)  # 5% uptrend
            noise = np.random.normal(0, 0.01, 100)  # 1% daily volatility
            
            # Calculate OHLCV data
            close_prices = base_price * (1 + trend + noise)
            open_prices = close_prices * (1 + np.random.normal(0, 0.002, 100))
            high_prices = np.maximum(close_prices, open_prices) * (1 + abs(np.random.normal(0, 0.003, 100)))
            low_prices = np.minimum(close_prices, open_prices) * (1 - abs(np.random.normal(0, 0.003, 100)))
            volumes = np.random.normal(1000, 200, 100) * (1 + abs(noise) * 5)
            
            # Create DataFrame
            self.data = pd.DataFrame({
                'date': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            # Set index
            self.data.set_index('date', inplace=True)
            logger.info(f"Using test data for {self.symbol}")
            
            return self.data
        
        # Otherwise fetch real data
        interval = self.trading_style['intervals'][1]  # Middle option as default
        period = self.trading_style['periods'][1]      # Middle option as default
        
        logger.info(f"Fetching data for {self.symbol} with interval={interval}, period={period}")
        
        try:
            # Fetch the data
            self.data = fetch_market_data(
                symbol=self.symbol,
                interval=interval,
                period=period
            )
            
            return self.data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run market analysis on the data.
        
        This will:
        1. Fetch data if not already fetched
        2. Add technical indicators
        3. Calculate performance metrics
        
        Returns:
            Dict containing the analysis results:
            - 'data': DataFrame with technical indicators
            - 'performance': Dict with performance metrics
            - 'trading_style': Dict with trading style settings
        """
        # Fetch data if not already fetched
        if self.data is None:
            self.fetch_data()
        
        logger.info(f"Running analysis for {self.symbol}")
        
        # Use the appropriate window size for the current timeframe
        window_size = self.trading_style['window_sizes'][0]  # Use the first window size
        
        # Add technical indicators with timeframe-specific window size
        self.data = market_data.add_technical_indicators(self.data, window_size=window_size)
        
        # Calculate performance metrics
        self.performance = market_data.get_performance_summary(self.data)
        
        return {
            'data': self.data,
            'performance': self.performance,
            'trading_style': self.trading_style
        }
    
    def generate_visualizations(self) -> Dict[str, go.Figure]:
        """
        Generate visualizations for the data.
        
        This will:
        1. Run analysis if not already run
        2. Generate price history, technical analysis, and candlestick charts
        
        Returns:
            Dict containing the visualizations:
            - 'price': Price history chart
            - 'technical': Technical analysis chart
            - 'candlestick': Candlestick chart
        """
        # Run analysis if not already run
        if self.data is None:
            self.run_analysis()
        
        logger.info(f"Generating visualizations for {self.symbol}")
        
        # Generate charts
        price_fig = market_data.plot_price_history(self.data)
        technical_fig = market_data.plot_technical_analysis(self.data)
        candlestick_fig = market_data.plot_candlestick(self.data)
        
        return {
            'price': price_fig,
            'technical': technical_fig,
            'candlestick': candlestick_fig
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis.
        
        This will:
        1. Run analysis if not already run
        2. Determine the trend
        3. Summarize technical indicators
        
        Returns:
            Dict containing the summary:
            - 'symbol': Stock symbol
            - 'timeframe': Trading timeframe
            - 'current_price': Latest closing price
            - 'period_return': Total return over the period
            - 'volatility': Price volatility
            - 'trend': Overall price trend (string or detailed dictionary)
            - 'indicators': Summary of technical indicators (interpretations)
            - 'indicator_data': Detailed indicator data with values
        """
        # Run analysis if not already run
        if self.data is None or self.performance is None:
            self.run_analysis()
        
        logger.info(f"Creating summary for {self.symbol}")
        
        # Initialize default values
        current_price = 0.0
        period_return = 0.0
        volatility = 0.0
        
        # Get current price with improved handling
        if self.data is not None and not self.data.empty:
            if 'close' in self.data.columns:
                # Try the last close price
                last_close = self.data['close'].iloc[-1]
                
                if not pd.isna(last_close):
                    current_price = last_close
                    logger.info(f"Current price for {self.symbol}: {current_price}")
                else:
                    logger.warning(f"Current price for {self.symbol} is NaN, attempting to find last valid price")
                    # Try to get the last valid close price with better error handling
                    try:
                        valid_prices = self.data['close'].dropna()
                        if not valid_prices.empty:
                            current_price = valid_prices.iloc[-1]
                            logger.info(f"Using last valid price for {self.symbol}: {current_price}")
                        else:
                            # If no valid prices, try to get the mean of open, high, low
                            alt_price_cols = ['open', 'high', 'low']
                            alt_prices = []
                            for col in alt_price_cols:
                                if col in self.data.columns:
                                    val = self.data[col].iloc[-1]
                                    if not pd.isna(val):
                                        alt_prices.append(val)
                            
                            if alt_prices:
                                current_price = sum(alt_prices) / len(alt_prices)
                                logger.warning(f"No valid close price found, using average of other price columns: {current_price}")
                            else:
                                logger.error(f"Could not find any valid price for {self.symbol}, defaulting to 0.0")
                    except Exception as e:
                        logger.error(f"Error finding valid price for {self.symbol}: {e}, defaulting to 0.0")
            else:
                logger.warning(f"Close column missing for {self.symbol}, defaulting price to 0.0")
        else:
            logger.warning(f"Data is None or empty for {self.symbol}, defaulting price to 0.0")
        
        # Handle performance metrics with better error checking
        if self.performance is not None:
            # Handle 24H change safely
            if 'total_return_pct' in self.performance:
                period_return_val = self.performance['total_return_pct']
                if not pd.isna(period_return_val):
                    period_return = period_return_val
                else:
                    logger.warning(f"24H change for {self.symbol} is NaN, defaulting to 0.0")
            else:
                logger.warning(f"Total return not found in performance data for {self.symbol}, defaulting to 0.0")
            
            # Handle volatility with fallbacks
            volatility_val = None
            
            # Try annualized_volatility_pct first
            if 'annualized_volatility_pct' in self.performance:
                volatility_val = self.performance['annualized_volatility_pct']
            
            # If that's not available or is NaN, try volatility
            if volatility_val is None or pd.isna(volatility_val):
                if 'volatility' in self.performance:
                    volatility_val = self.performance['volatility']
            
            # If we have a valid volatility value, use it
            if volatility_val is not None and not pd.isna(volatility_val):
                volatility = volatility_val
            else:
                logger.warning(f"Volatility for {self.symbol} is NaN or missing, defaulting to 0.0")
        else:
            logger.warning(f"Performance data is None for {self.symbol}, using default values for return and volatility")
        
        # Determine trend
        trend_data = self._determine_trend()
        if not isinstance(trend_data, dict):
            # Wrap string in a dict with sensible defaults
            trend_data = {
                'direction': trend_data,
                'strength': "Unknown",
                'confidence': "Unknown",
                'signals': {
                    'short_term': "Unknown",
                    'medium_term': "Unknown",
                    'long_term': "Unknown",
                    'action': "Hold"
                },
                'explanation': "Trend returned as string, not dict. Upstream logic should be improved."
            }
        trend = trend_data['direction']
        
        # Summarize indicators
        indicator_data = self._summarize_indicators()
        
        # For backward compatibility, extract just the interpretations
        indicators = indicator_data.get('interpretations', {})
        
        # Build the summary with validated values
        summary = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_price': current_price,
            'period_return': period_return,
            'volatility': volatility,
            'trend': trend_data,  # Use the full trend data dictionary
            'trend_direction': trend,  # For backward compatibility
            'indicators': indicators,
            'indicator_data': indicator_data
        }
        
        # Log summary price information with safer formatting
        try:
            logger.info(f"Summary for {self.symbol}: price={summary['current_price']}, return={summary['period_return']:.2f}%, volatility={summary['volatility']:.2f}%")
        except Exception as e:
            logger.error(f"Error logging summary for {self.symbol}: {e}")
        
        return summary
    
    def _determine_trend(self) -> Union[str, Dict[str, Any]]:
        """
        Determine the overall price trend with comprehensive analysis.
        
        Returns:
            Dictionary with detailed trend analysis:
            - 'direction': String describing the trend ('Uptrend', 'Downtrend', or 'Sideways')
            - 'strength': Trend strength ('Strong', 'Moderate', 'Weak')
            - 'confidence': Confidence level ('High', 'Medium', 'Low')
            - 'signals': Dictionary with timeframe signals and action recommendation
            - 'explanation': Detailed explanation of trend determination
            - 'advanced_recommendation': Advanced trading recommendations (if available)
        """
        # Check if data is available
        if self.data is None or self.data.empty:
            return {
                'direction': "Insufficient data",
                'strength': "Unknown",
                'confidence': "Low",
                'signals': {
                    'short_term': "Unknown",
                    'medium_term': "Unknown",
                    'long_term': "Unknown",
                    'action': "Hold"
                },
                'explanation': "No data available for trend determination."
            }
        
        # Get close values with NaN handling
        close_values = self.data['close'].dropna().values
        
        # If we have no valid close values, return default
        if len(close_values) < 2:
            return {
                'direction': "Insufficient data",
                'strength': "Unknown",
                'confidence': "Low",
                'signals': {
                    'short_term': "Unknown",
                    'medium_term': "Unknown",
                    'long_term': "Unknown",
                    'action': "Hold"
                },
                'explanation': "Insufficient price data for trend determination."
            }
        
        # Calculate the linear regression slope of the closing prices
        x = np.arange(len(close_values))
        
        # Simple linear regression to determine trend
        try:
            slope, intercept = np.polyfit(x, close_values, 1)
        except Exception as e:
            logger.warning(f"Error calculating trend slope: {e}, using default values")
            slope, intercept = 0, close_values[0] if len(close_values) > 0 else 0
        
        # Get the first and last prices
        first_price = close_values[0]
        last_price = close_values[-1]
        
        # Calculate percent change with safety checks
        try:
            if first_price != 0:
                percent_change = (last_price - first_price) / first_price * 100
            else:
                percent_change = 0
                logger.warning("First price is zero, using percent_change=0")
        except Exception as e:
            logger.warning(f"Error calculating percent change: {e}, using default value")
            percent_change = 0
        
        # Set default direction
        direction = "Sideways"
        
        # Determine basic trend direction based on slope and percent change
        # Handle NaN values using the _ensure_float helper
        safe_slope = self._ensure_float(slope, 0.0)
        safe_percent_change = self._ensure_float(percent_change, 0.0)
        
        if safe_slope > 0 and safe_percent_change > 1:
            direction = "Uptrend"
        elif safe_slope < 0 and safe_percent_change < -1:
            direction = "Downtrend"
        
        # Get the latest technical indicator values
        try:
            latest = self.data.iloc[-1]
        except Exception as e:
            logger.error(f"Error accessing latest data: {e}")
            latest = pd.Series()
        
        # Determine trend strength using ADX with better NaN handling
        adx = None
        if not latest.empty:
            adx_val = latest.get('ADX_14', None)
            adx = self._ensure_float(adx_val, None)
        
        strength = "Unknown"
        
        if adx is not None:
            if adx > 25:
                strength = "Strong"
            elif adx > 15:
                strength = "Moderate"
            else:
                strength = "Weak"
        elif safe_slope != 0:
            # Fall back to slope if ADX is not available
            normalized_slope = abs(safe_slope) / max(first_price, 0.1)  # Normalize by first price with safety
            if normalized_slope > 0.001:  # Arbitrary threshold, adjust based on testing
                strength = "Strong"
            elif normalized_slope > 0.0005:
                strength = "Moderate"
            else:
                strength = "Weak"
        else:
            # Default to Weak if we can't determine strength
            strength = "Weak"
        
        # Collect indicator signals for determining confidence and timeframe signals
        signals = []
        
        # SMA signal (comparing price to SMA) with safer handling
        if not latest.empty:
            window_size = self.trading_style['window_sizes'][0]
            sma_key = f'sma_{window_size}'
            sma = self._ensure_float(latest.get(sma_key, None), None)
            
            if sma is not None and last_price > 0:
                signals.append(1 if last_price > sma else -1)
            
            # RSI signal with safer handling
            rsi_val = latest.get('rsi_14', None)
            rsi = self._ensure_float(rsi_val, None)
            
            if rsi is not None:
                if rsi > 70:
                    signals.append(-1)  # Overbought, potentially bearish
                elif rsi < 30:
                    signals.append(1)   # Oversold, potentially bullish
                else:
                    # Neutral but leaning
                    signals.append(0.5 if rsi > 50 else -0.5)
            
            # MACD signal with safer handling
            macd_val = latest.get('MACD_12_26_9', None)
            macd_signal_val = latest.get('MACDs_12_26_9', None)
            
            macd = self._ensure_float(macd_val, None)
            macd_signal = self._ensure_float(macd_signal_val, None)
            
            if macd is not None and macd_signal is not None:
                signals.append(1 if macd > macd_signal else -1)
            
            # Bollinger Bands signal with safer handling
            bb_upper_key = f'BBU_{window_size}_2.0'
            bb_lower_key = f'BBL_{window_size}_2.0'
            
            bb_upper_val = latest.get(bb_upper_key, None)
            bb_lower_val = latest.get(bb_lower_key, None)
            
            bb_upper = self._ensure_float(bb_upper_val, None)
            bb_lower = self._ensure_float(bb_lower_val, None)
            
            if bb_upper is not None and bb_lower is not None:
                if last_price > bb_upper:
                    signals.append(-1)  # Overbought
                elif last_price < bb_lower:
                    signals.append(1)   # Oversold
                else:
                    # Inside bands, neutral
                    signals.append(0)
            
            # Stochastic signal with safer handling
            stoch_k_val = latest.get('STOCHk_14_3_3', None)
            stoch_d_val = latest.get('STOCHd_14_3_3', None)
            
            stoch_k = self._ensure_float(stoch_k_val, None)
            stoch_d = self._ensure_float(stoch_d_val, None)
            
            if stoch_k is not None and stoch_d is not None:
                if stoch_k > 80 and stoch_d > 80:
                    signals.append(-1)  # Overbought
                elif stoch_k < 20 and stoch_d < 20:
                    signals.append(1)   # Oversold
                elif stoch_k > stoch_d:
                    signals.append(0.5)  # Bullish crossover
                elif stoch_k < stoch_d:
                    signals.append(-0.5)  # Bearish crossover
                else:
                    signals.append(0)   # Neutral
        
        # Calculate confidence based on signal agreement with safer handling
        confidence = "Low"
        
        if signals:
            try:
                # Average the signals
                avg_signal = sum(signals) / len(signals)
                signal_std = np.std(signals) if len(signals) > 1 else 0
                
                # Higher agreement (lower std) means higher confidence
                if signal_std < 0.5 and abs(avg_signal) > 0.3:
                    confidence = "High"
                elif signal_std < 0.8:
                    confidence = "Medium"
            except Exception as e:
                logger.warning(f"Error calculating confidence: {e}, defaulting to 'Low'")
        
        # Determine timeframe signals with safer handling
        short_signal = "Neutral"
        medium_signal = "Neutral"
        long_signal = "Neutral"
        
        # Focus on short-term signals with safer handling
        if rsi is not None:
            if rsi > 70:
                short_signal = "Bearish"
            elif rsi < 30:
                short_signal = "Bullish"
        
        # Medium-term signals with safer handling 
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                medium_signal = "Bullish"
            elif macd < macd_signal:
                medium_signal = "Bearish"
        
        # Long-term signals: trend direction, moving averages
        long_signal = direction
        if direction == "Sideways":
            long_signal = "Neutral"
            
        # Determine recommended action based on signals
        action = "Hold"
        if short_signal == "Bullish" and medium_signal == "Bullish":
            action = "Buy"
        elif short_signal == "Bearish" and medium_signal == "Bearish":
            action = "Sell"
        
        # Format values for explanation with safer handling
        slope_str = "N/A"
        percent_change_str = "N/A"
        adx_str = "N/A"
        
        try:
            if not pd.isna(slope):
                slope_str = f"{slope:.6f}"
        except:
            pass
            
        try:
            if not pd.isna(percent_change):
                percent_change_str = f"{percent_change:.2f}"
        except:
            pass
            
        try:
            if adx is not None and not pd.isna(adx):
                adx_str = f"{adx:.2f}"
        except:
            pass

        # Generate explanation text with handled NaN values
        explanation = f"Trend direction determined by linear regression slope "
        explanation += f"({slope_str}) and percent change ({percent_change_str}%). "
        
        if adx is not None and not pd.isna(adx):
            explanation += f"Trend strength based on ADX value of {adx_str}. "
        else:
            explanation += f"Trend strength estimated from price movement. "
            
        explanation += f"Signals derived from {len(signals)} technical indicators. "
        
        # Add signal summary
        if len(signals) > 0:
            bullish_count = sum(1 for s in signals if s > 0)
            bearish_count = sum(1 for s in signals if s < 0)
            neutral_count = len(signals) - bullish_count - bearish_count
            
            explanation += f"Signal breakdown: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral. "
        
        # Generate advanced trading recommendations if available
        advanced_recommendation = None
        try:
            # Only import when needed to avoid circular imports
            from src.services.trading_strategies import generate_trade_recommendation
            advanced_recommendation = generate_trade_recommendation(self.data)
            
            # Use advanced recommendation for the action if confidence is higher
            if (advanced_recommendation and 
                advanced_recommendation.get('confidence') and 
                confidence_level(advanced_recommendation['confidence']) > confidence_level(confidence)):
                action = advanced_recommendation['action'].capitalize()
                explanation += f"\nAdvanced analysis suggests {advanced_recommendation['strategy']} "
                explanation += f"strategy with {advanced_recommendation['confidence']} confidence. "
                
                # Add market condition information
                if 'market_condition' in advanced_recommendation:
                    market_condition = advanced_recommendation['market_condition']
                    if 'condition' in market_condition and 'sub_condition' in market_condition:
                        explanation += f"Market condition: {market_condition['condition']} "
                        explanation += f"({market_condition['sub_condition']}). "
                
                # Add entry/exit information if available
                if 'entry_points' in advanced_recommendation and advanced_recommendation['entry_points']:
                    entry_point = advanced_recommendation['entry_points'][0]
                    if 'price' in entry_point and entry_point['price'] is not None:
                        explanation += f"Suggested entry near {entry_point['price']:.2f}. "
                
                if 'exit_points' in advanced_recommendation:
                    exit_points = advanced_recommendation['exit_points']
                    if 'take_profit' in exit_points and exit_points['take_profit']:
                        explanation += f"Target: {exit_points['take_profit'][0]:.2f}. "
                    if 'stop_loss' in exit_points and exit_points['stop_loss'] is not None:
                        explanation += f"Stop loss: {exit_points['stop_loss']:.2f}. "
                
                # Add risk information if available
                if 'risk_assessment' in advanced_recommendation:
                    risk = advanced_recommendation['risk_assessment']
                    if 'risk_reward_ratio' in risk and risk['risk_reward_ratio'] is not None:
                        explanation += f"Risk/Reward: {risk['risk_reward_ratio']:.2f}. "
        except Exception as e:
            logger.warning(f"Error generating advanced recommendations: {e}")
            advanced_recommendation = None
        
        # Construct the full trend analysis dictionary
        trend_analysis = {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'signals': {
                'short_term': short_signal,
                'medium_term': medium_signal,
                'long_term': long_signal,
                'action': action
            },
            'explanation': explanation,
            'values': {
                'slope': self._ensure_float(slope, 0.0),
                'percent_change': self._ensure_float(percent_change, 0.0),
                'adx': self._ensure_float(adx, 0.0),
                'signal_count': len(signals)
            }
        }
        
        # Add advanced recommendation if available
        if advanced_recommendation:
            trend_analysis['advanced_recommendation'] = advanced_recommendation
        
        return trend_analysis

    def _summarize_indicators(self) -> Dict[str, Any]:
        """
        Summarize the technical indicators.
        
        Returns:
            Dict with interpretations and values of technical indicators:
            - 'rsi': RSI data (interpretation and value)
            - 'macd': MACD data (interpretation and values)
            - 'bollinger': Bollinger Bands data (interpretation and values)
            - 'adx': ADX data (interpretation and value)
            - 'stochastic': Stochastic Oscillator data (interpretation and values)
            - 'cci': CCI data (interpretation and value)
            - 'atr': ATR data (interpretation and value)
            - 'obv': OBV data (interpretation and value)
        """
        # Initialize the result dictionary to store all indicator data
        result = {}
        
        # Ensure data is available
        if self.data is None or self.data.empty:
            logger.warning(f"No data available for {self.symbol}, using default indicator values")
            return self._get_default_indicators()
        
        # Get the latest values
        try:
            latest = self.data.iloc[-1]
        except Exception as e:
            logger.error(f"Error accessing latest data for {self.symbol}: {e}")
            return self._get_default_indicators()
        
        # Get the window size used for this timeframe
        window_size = self.trading_style['window_sizes'][0]  # Use the first window size
        
        # Initialize close variable explicitly with better error handling
        close = None
        try:
            close = latest.get('close', None)
            if pd.isna(close):
                # Try to find a valid close price
                close_values = self.data['close'].dropna()
                if not close_values.empty:
                    close = close_values.iloc[-1]
                    logger.warning(f"Latest close is NaN for {self.symbol}, using last valid close: {close}")
                else:
                    # If no valid close prices, try other price columns
                    for col in ['open', 'high', 'low']:
                        if col in latest and not pd.isna(latest[col]):
                            close = latest[col]
                            logger.warning(f"No valid close prices for {self.symbol}, using {col}: {close}")
                            break
        except Exception as e:
            logger.error(f"Error getting close price for {self.symbol}: {e}")
        
        if close is None or pd.isna(close):
            logger.warning(f"Could not determine a valid price for {self.symbol}, using 0.0")
            close = 0.0
        
        # RSI interpretation and value with better fallback handling
        rsi = latest.get('rsi_14', None)
        if rsi is not None and not pd.isna(rsi):
            # Valid RSI value
            if rsi > 70:
                rsi_interp = "Overbought"
            elif rsi < 30:
                rsi_interp = "Oversold"
            else:
                rsi_interp = "Neutral"
            
            # Ensure value is a number, not an array or other structure
            if isinstance(rsi, (np.ndarray, list, pd.Series, pd.DataFrame)):
                try:
                    rsi = float(rsi.iloc[0] if hasattr(rsi, 'iloc') else rsi[0])
                except (IndexError, TypeError, ValueError):
                    logger.warning(f"RSI value for {self.symbol} couldn't be converted to float, defaulting to 50")
                    rsi = 50.0
        else:
            # Use fallback value (neutral RSI = 50)
            logger.warning(f"RSI value is None or NaN for {self.symbol}, defaulting to 'Neutral' (50)")
            rsi_interp = "Neutral"
            rsi = 50.0
        
        # MACD interpretation and values with better fallback handling
        macd = latest.get('MACD_12_26_9', None)
        macd_signal = latest.get('MACDs_12_26_9', None)
        macd_hist = latest.get('MACDh_12_26_9', None)
        
        # Initialize with default values
        macd_values = {'line': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        if (macd is not None and macd_signal is not None and 
            not pd.isna(macd) and not pd.isna(macd_signal)):
            # Valid MACD values
            if macd > macd_signal and macd > 0:
                macd_interp = "Bullish"
            elif macd < macd_signal and macd < 0:
                macd_interp = "Bearish"
            else:
                macd_interp = "Neutral"
            
            # Update the values dictionary with actual values
            macd_values['line'] = self._ensure_float(macd, 0.0)
            macd_values['signal'] = self._ensure_float(macd_signal, 0.0)
            if macd_hist is not None and not pd.isna(macd_hist):
                macd_values['histogram'] = self._ensure_float(macd_hist, 0.0)
        else:
            # Use fallback values
            logger.warning(f"MACD values are None or NaN for {self.symbol}, defaulting to 'Neutral' (0)")
            macd_interp = "Neutral"
        
        # Bollinger Bands interpretation with enhanced error handling
        logger.debug(f"Using window_size={window_size} for Bollinger Bands calculation")
        bb_lower_key = f'BBL_{window_size}_2.0'
        bb_middle_key = f'BBM_{window_size}_2.0'
        bb_upper_key = f'BBU_{window_size}_2.0'
        
        bb_lower = latest.get(bb_lower_key, None)
        bb_middle = latest.get(bb_middle_key, None)
        bb_upper = latest.get(bb_upper_key, None)
        
        # Initialize with default values
        bb_values = {
            'upper': close * 1.02 if close else 0.0,  # Default: price + 2%
            'middle': close if close else 0.0,        # Default: price
            'lower': close * 0.98 if close else 0.0,  # Default: price - 2%
            'close': close if close else 0.0,
            'percent': 0.0                            # Default: neutral (0%)
        }
        
        if (bb_lower is not None and bb_upper is not None and close is not None and 
            not pd.isna(bb_lower) and not pd.isna(bb_upper) and not pd.isna(close)):
            # Valid Bollinger Bands values
            if close > bb_upper:
                bb_interp = "Overbought"
            elif close < bb_lower:
                bb_interp = "Oversold"
            else:
                bb_interp = "Neutral"
            
            # Update the values dictionary with actual values
            bb_values['upper'] = self._ensure_float(bb_upper, bb_values['upper'])
            bb_values['lower'] = self._ensure_float(bb_lower, bb_values['lower'])
            
            # Calculate percentage distance from the middle band
            if bb_middle is not None and not pd.isna(bb_middle) and float(bb_middle) != 0:
                bb_percent = ((close - float(bb_middle)) / float(bb_middle)) * 100
                bb_values['middle'] = self._ensure_float(bb_middle, close)
                bb_values['percent'] = self._ensure_float(bb_percent, 0.0)
        else:
            # Use fallback values
            logger.warning(f"Bollinger Bands values are None or NaN for {self.symbol}, defaulting to 'Neutral'")
            bb_interp = "Neutral"
        
        # ADX interpretation and value with better fallback handling
        adx = latest.get('ADX_14', None)
        dmp = latest.get('DMP_14', None)
        dmn = latest.get('DMN_14', None)
        
        if adx is not None and not pd.isna(adx):
            # Valid ADX value
            if adx > 25:
                adx_interp = "Strong Trend"
            elif adx > 15:
                adx_interp = "Moderate Trend"
            else:
                adx_interp = "Weak Trend"
            
            adx = self._ensure_float(adx, 15.0)
        else:
            # Use fallback value (weak trend ADX = 15)
            logger.warning(f"ADX value is None or NaN for {self.symbol}, defaulting to 'Weak Trend' (15)")
            adx_interp = "Weak Trend"
            adx = 15.0
        
        # Stochastic Oscillator with better fallback handling
        stoch_k = latest.get('STOCHk_14_3_3', None)
        stoch_d = latest.get('STOCHd_14_3_3', None)
        
        # Initialize with default values
        stoch_values = {'k': 50.0, 'd': 50.0}  # Neutral values
        
        if (stoch_k is not None and stoch_d is not None and 
            not pd.isna(stoch_k) and not pd.isna(stoch_d)):
            # Valid Stochastic values
            if stoch_k > 80 and stoch_d > 80:
                stoch_interp = "Overbought"
            elif stoch_k < 20 and stoch_d < 20:
                stoch_interp = "Oversold"
            elif stoch_k > stoch_d:
                stoch_interp = "Bullish Crossover"
            elif stoch_k < stoch_d:
                stoch_interp = "Bearish Crossover"
            else:
                stoch_interp = "Neutral"
            
            # Update the values dictionary with actual values
            stoch_values['k'] = self._ensure_float(stoch_k, 50.0)
            stoch_values['d'] = self._ensure_float(stoch_d, 50.0)
        else:
            # Use fallback values
            logger.warning(f"Stochastic values are None or NaN for {self.symbol}, defaulting to 'Neutral' (50)")
            stoch_interp = "Neutral"
        
        # CCI interpretation and value with better fallback handling
        cci = latest.get('CCI_20', None)
        
        if cci is not None and not pd.isna(cci):
            # Valid CCI value
            if cci > 100:
                cci_interp = "Overbought"
            elif cci < -100:
                cci_interp = "Oversold"
            else:
                cci_interp = "Neutral"
            
            cci = self._ensure_float(cci, 0.0)
        else:
            # Use fallback value (neutral CCI = 0)
            logger.warning(f"CCI value is None or NaN for {self.symbol}, defaulting to 'Neutral' (0)")
            cci_interp = "Neutral"
            cci = 0.0
        
        # ATR interpretation and value with better fallback handling
        atr = latest.get('ATR_14', None)
        
        # Default ATR interpretation and value
        atr_interp = "Unavailable"
        atr_value = close * 0.01 if close else 0.0  # Default to 1% of price
        
        if (atr is not None and close is not None and close > 0 and 
            not pd.isna(atr) and not pd.isna(close)):
            # Valid ATR value
            atr_value = self._ensure_float(atr, atr_value)
            
            # Calculate ATR as percentage of price
            atr_pct = (atr_value / close) * 100
            
            if atr_pct > 3:
                atr_interp = "High Volatility"
            elif atr_pct > 1.5:
                atr_interp = "Moderate Volatility"
            else:
                atr_interp = "Low Volatility"
        else:
            # Use fallback value
            logger.warning(f"ATR value or close price is None or NaN for {self.symbol}, defaulting to 'Low Volatility'")
            atr_interp = "Low Volatility"
        
        # OBV interpretation and value with better fallback handling
        obv = latest.get('OBV', None)
        
        # Default OBV interpretation and value
        obv_interp = "Neutral"
        obv_value = 0.0
        
        if obv is not None and not pd.isna(obv) and len(self.data) > 1:
            # Valid OBV value
            obv_value = self._ensure_float(obv, 0.0)
            
            # Compare with previous OBV if possible
            try:
                if 'OBV' in self.data.columns and 'close' in self.data.columns:
                    prev_obv = self.data['OBV'].iloc[-2]
                    prev_close = self.data['close'].iloc[-2]
                    
                    if not pd.isna(prev_obv) and not pd.isna(prev_close):
                        if obv_value > prev_obv and close > prev_close:
                            obv_interp = "Confirming Uptrend"
                        elif obv_value < prev_obv and close < prev_close:
                            obv_interp = "Confirming Downtrend"
                        elif obv_value < prev_obv and close > prev_close:
                            obv_interp = "Divergence (Bearish)"
                        elif obv_value > prev_obv and close < prev_close:
                            obv_interp = "Divergence (Bullish)"
                        else:
                            obv_interp = "Neutral"
            except Exception as e:
                logger.warning(f"Error comparing OBV values for {self.symbol}: {e}, defaulting to 'Neutral'")
        else:
            # Use fallback value
            logger.warning(f"OBV value is None or NaN for {self.symbol}, defaulting to 'Neutral' (0)")
        
        # SMA interpretation and values with better fallback handling
        sma_key = f'sma_{window_size}'
        
        # Default SMA interpretation and values
        sma_interp = "Unavailable"
        sma_values = {
            'sma': close if close else 0.0,  # Default to current price
            'close': close if close else 0.0,
            'position': 'Unknown'
        }
        
        # Get SMA value with error handling
        try:
            sma = None
            if sma_key in latest:
                sma = latest.get(sma_key, None)
                if sma is not None and not pd.isna(sma):
                    sma = self._ensure_float(sma, close)
                
            if sma is not None and not pd.isna(sma) and close is not None and close > 0:
                # Valid SMA value
                sma_values['sma'] = sma
                
                # Check if we have enough data for trend calculation
                try:
                    if len(self.data) > 10 and sma_key in self.data.columns:
                        # Calculate SMA trend by comparing with SMA from 10 periods ago
                        historical_index = -11 if len(self.data) > 10 else 0
                        historical_sma = self.data[sma_key].iloc[historical_index]
                        
                        # Calculate percentage difference between price and SMA
                        price_sma_diff_pct = ((close - sma) / sma) * 100
                        
                        # Calculate SMA trend as percentage change
                        sma_trend_pct = 0
                        if historical_sma is not None and historical_sma > 0 and not pd.isna(historical_sma):
                            sma_trend_pct = ((sma - historical_sma) / historical_sma) * 100
                        
                        # Determine price position relative to SMA
                        if close > sma:
                            sma_values['position'] = 'Above'
                            if sma_trend_pct > 1 or price_sma_diff_pct > 3:
                                sma_interp = "Bullish"
                            elif -1 <= sma_trend_pct <= 1 and abs(price_sma_diff_pct) <= 3:
                                sma_interp = "Neutral"
                            else:
                                sma_interp = "Mildly Bullish"
                        else:
                            sma_values['position'] = 'Below'
                            if sma_trend_pct < -1 or price_sma_diff_pct < -3:
                                sma_interp = "Bearish"
                            elif -1 <= sma_trend_pct <= 1 and abs(price_sma_diff_pct) <= 3:
                                sma_interp = "Neutral"
                            else:
                                sma_interp = "Mildly Bearish"
                    else:
                        # Not enough historical data, do basic position check
                        if close > sma:
                            sma_values['position'] = 'Above'
                            sma_interp = "Above (Potentially Bullish)"
                        else:
                            sma_values['position'] = 'Below'
                            sma_interp = "Below (Potentially Bearish)"
                except Exception as e:
                    logger.warning(f"Error calculating SMA trend for {self.symbol}: {e}")
                    if close > sma:
                        sma_values['position'] = 'Above'
                        sma_interp = "Above SMA"
                    else:
                        sma_values['position'] = 'Below'
                        sma_interp = "Below SMA"
            else:
                # Use fallback value
                logger.warning(f"SMA value or close price is None or NaN for {self.symbol}, defaulting to 'Neutral'")
                sma_interp = "Neutral"
        except Exception as e:
            logger.error(f"Error interpreting SMA for {self.symbol}: {e}")
        
        # Create structured results with both interpretations and values
        result = {
            'rsi': {
                'interpretation': rsi_interp,
                'value': round(rsi, 2)
            },
            'macd': {
                'interpretation': macd_interp,
                'values': {
                    'line': round(macd_values['line'], 4),
                    'signal': round(macd_values['signal'], 4),
                    'histogram': round(macd_values['histogram'], 4)
                }
            },
            'bollinger': {
                'interpretation': bb_interp,
                'values': {
                    'upper': round(bb_values['upper'], 2),
                    'middle': round(bb_values['middle'], 2),
                    'lower': round(bb_values['lower'], 2),
                    'close': round(bb_values['close'], 2),
                    'percent': round(bb_values['percent'], 2)
                }
            },
            'sma': {
                'interpretation': sma_interp,
                'values': {
                    'sma': round(sma_values['sma'], 2),
                    'close': round(sma_values['close'], 2),
                    'position': sma_values['position']
                }
            },
            'adx': {
                'interpretation': adx_interp,
                'value': round(adx, 2)
            },
            'stochastic': {
                'interpretation': stoch_interp,
                'values': {
                    'k': round(stoch_values['k'], 2),
                    'd': round(stoch_values['d'], 2)
                }
            },
            'cci': {
                'interpretation': cci_interp,
                'value': round(cci, 2)
            },
            'atr': {
                'interpretation': atr_interp,
                'value': round(atr_value, 2)
            },
            'obv': {
                'interpretation': obv_interp,
                'value': obv_value
            }
        }
        
        # Add Ichimoku Cloud interpretation
        try:
            # Check if necessary Ichimoku components are available
            tenkan_key = 'ITS_9'  # Tenkan-sen (Conversion Line)
            kijun_key = 'IKS_26'  # Kijun-sen (Base Line)
            senkou_a_key = 'ISA_9'  # Senkou Span A (Leading Span A)
            senkou_b_key = 'ISB_26'  # Senkou Span B (Leading Span B)
            chikou_key = 'ICS_26'  # Chikou Span (Lagging Span)
            
            # Get values from the dataframe with safer access
            tenkan_val = self._ensure_float(latest.get(tenkan_key, None), close)
            kijun_val = self._ensure_float(latest.get(kijun_key, None), close)
            senkou_a_val = self._ensure_float(latest.get(senkou_a_key, None), close)
            senkou_b_val = self._ensure_float(latest.get(senkou_b_key, None), close * 0.98)
            chikou_val = self._ensure_float(latest.get(chikou_key, None), close)
            
            # Check for NaN values
            if (pd.isna(tenkan_val) or pd.isna(kijun_val) or 
                pd.isna(senkou_a_val) or pd.isna(senkou_b_val)):
                logger.debug("Some Ichimoku values are NaN, using fallback interpretation")
                ichimoku_interp = "Neutral"
                result['ichimoku'] = {
                    'interpretation': ichimoku_interp,
                    'values': {
                        'tenkan_sen': tenkan_val,
                        'kijun_sen': kijun_val,
                        'senkou_span_a': senkou_a_val,
                        'senkou_span_b': senkou_b_val,
                        'chikou_span': chikou_val,
                        'cloud_bullish': senkou_a_val > senkou_b_val,
                        'position': "Neutral",
                        'tk_cross': "Neutral"
                    }
                }
            else:
                # Price position relative to cloud
                cloud_bullish = latest.get('ichimoku_cloud_bullish', senkou_a_val > senkou_b_val)
                
                # Is price above or below the cloud?
                price_above_cloud = close > max(senkou_a_val, senkou_b_val) if not pd.isna(close) else False
                price_below_cloud = close < min(senkou_a_val, senkou_b_val) if not pd.isna(close) else False
                
                # Tenkan-Kijun Cross
                tk_cross = "Neutral"
                if tenkan_val > kijun_val:
                    tk_cross = "Bullish"
                elif tenkan_val < kijun_val:
                    tk_cross = "Bearish"
                
                # Overall position
                position = "Neutral"
                if price_above_cloud:
                    position = "Bullish"
                elif price_below_cloud:
                    position = "Bearish"
                
                # Determine the overall Ichimoku interpretation
                if price_above_cloud and tenkan_val > kijun_val and cloud_bullish:
                    ichimoku_interp = "Strong Bullish"
                elif price_below_cloud and tenkan_val < kijun_val and not cloud_bullish:
                    ichimoku_interp = "Strong Bearish"
                elif price_above_cloud or (tenkan_val > kijun_val and cloud_bullish):
                    ichimoku_interp = "Bullish"
                elif price_below_cloud or (tenkan_val < kijun_val and not cloud_bullish):
                    ichimoku_interp = "Bearish"
                else:
                    ichimoku_interp = "Neutral"
                
                # Add Ichimoku to the result
                result['ichimoku'] = {
                    'interpretation': ichimoku_interp,
                    'values': {
                        'tenkan_sen': tenkan_val,
                        'kijun_sen': kijun_val,
                        'senkou_span_a': senkou_a_val,
                        'senkou_span_b': senkou_b_val,
                        'chikou_span': chikou_val,
                        'cloud_bullish': cloud_bullish,
                        'position': position,
                        'tk_cross': tk_cross
                    }
                }
        except Exception as e:
            logger.debug(f"Error interpreting Ichimoku Cloud: {e}")
            # Default Ichimoku interpretation when there's an error
            ichimoku_interp = "Neutral"
            result['ichimoku'] = {
                'interpretation': ichimoku_interp,
                'values': {
                    'tenkan_sen': close,
                    'kijun_sen': close,
                    'senkou_span_a': close,
                    'senkou_span_b': close * 0.98,
                    'chikou_span': close,
                    'cloud_bullish': False,
                    'position': "Neutral",
                    'tk_cross': "Neutral"
                }
            }
        
        # Backward compatibility - add string interpretations as top-level keys
        indicator_interpretations = {
            'rsi': rsi_interp,
            'macd': macd_interp,
            'bollinger': bb_interp,
            'sma': sma_interp,
            'adx': adx_interp,
            'stochastic': stoch_interp,
            'cci': cci_interp,
            'atr': atr_interp,
            'obv': obv_interp
        }
        
        # Add Ichimoku to interpretations if available
        if 'ichimoku' in result:
            indicator_interpretations['ichimoku'] = result['ichimoku']['interpretation']
        
        # Always return the structured format with interpretations included
        result.update({
            # Include a flat dictionary of interpretations for backward compatibility
            'interpretations': indicator_interpretations
        })
        
        return result
    
    def _ensure_float(self, value, default=0.0):
        """
        Helper method to ensure a value is a float, handling various data types.
        
        Args:
            value: The value to convert to float
            default: Default value to use if conversion fails
            
        Returns:
            Float value or default if conversion fails
        """
        if value is None or pd.isna(value):
            return default
            
        try:
            # Handle arrays, series, etc.
            if isinstance(value, (np.ndarray, list, pd.Series, pd.DataFrame)):
                # For array-like objects, try to get the first element
                if hasattr(value, 'iloc'):
                    return float(value.iloc[0])
                elif hasattr(value, 'item') and value.size == 1:
                    return float(value.item())
                elif len(value) > 0:
                    return float(value[0])
                else:
                    return default
            else:
                # For scalar values, convert directly
                return float(value)
        except (TypeError, ValueError, IndexError, AttributeError):
            return default
    
    def _get_default_indicators(self):
        """
        Provide default indicator values and interpretations when no data is available.
        
        Returns:
            Dict with default indicator interpretations and values
        """
        # Create a default result dictionary
        result = {
            'rsi': {
                'interpretation': "Neutral",
                'value': 50.0
            },
            'macd': {
                'interpretation': "Neutral",
                'values': {
                    'line': 0.0,
                    'signal': 0.0,
                    'histogram': 0.0
                }
            },
            'bollinger': {
                'interpretation': "Neutral",
                'values': {
                    'upper': 110.0,
                    'middle': 100.0,
                    'lower': 90.0,
                    'close': 100.0,
                    'percent': 0.0
                }
            },
            'sma': {
                'interpretation': "Neutral",
                'values': {
                    'sma': 100.0,
                    'close': 100.0,
                    'position': 'Unknown'
                }
            },
            'adx': {
                'interpretation': "Weak Trend",
                'value': 15.0
            },
            'stochastic': {
                'interpretation': "Neutral",
                'values': {
                    'k': 50.0,
                    'd': 50.0
                }
            },
            'cci': {
                'interpretation': "Neutral",
                'value': 0.0
            },
            'atr': {
                'interpretation': "Low Volatility",
                'value': 1.0
            },
            'obv': {
                'interpretation': "Neutral",
                'value': 0.0
            },
            'ichimoku': {
                'interpretation': "Neutral",
                'values': {
                    'tenkan_sen': 100.0,
                    'kijun_sen': 100.0,
                    'senkou_span_a': 100.0,
                    'senkou_span_b': 98.0,
                    'chikou_span': 100.0,
                    'cloud_bullish': False,
                    'position': "Neutral",
                    'tk_cross': "Neutral"
                }
            }
        }
        
        # Add flat interpretations dictionary for backward compatibility
        result['interpretations'] = {
            'rsi': "Neutral",
            'macd': "Neutral",
            'bollinger': "Neutral",
            'sma': "Neutral",
            'adx': "Weak Trend",
            'stochastic': "Neutral",
            'cci': "Neutral",
            'atr': "Low Volatility",
            'obv': "Neutral",
            'ichimoku': "Neutral"
        }
        
        return result

    def present_cases(self) -> Dict[str, Any]:
        """
        Present bullish, bearish, and neutral cases based on all indicators.
        
        Returns:
            Dict containing analysis of bearish, neutral, and bullish cases 
            with supporting indicators and confidence levels
        """
        # First ensure we have run analysis
        if self.data is None:
            self.run_analysis()
            
        logger = logging.getLogger(__name__)
        
        # Get indicator data including interpretations
        indicators = self._summarize_indicators()
        
        # Group indicators by their signal
        bullish_indicators = []
        bearish_indicators = []
        neutral_indicators = []
        
        # Process interpretations from the flat dictionary
        for indicator, data in indicators.items():
            if indicator == 'interpretations':
                continue
                
            interpretation = None
            # Handle both the new and legacy formats
            if isinstance(data, dict) and 'interpretation' in data:
                interpretation = data['interpretation']
            elif indicator in indicators.get('interpretations', {}):
                interpretation = indicators['interpretations'][indicator]
                
            if interpretation:
                logger.debug(f"Processing indicator: {indicator} with interpretation: {interpretation}")
                interp_lower = interpretation.lower()
                
                # Skip unavailable indicators
                if interp_lower in ['unavailable', 'nan']:
                    continue
                
                # Categorize by signal words in the interpretation
                if any(term in interp_lower for term in ['bullish', 'oversold', 'uptrend', 'confirming uptrend', 'strong bullish']):
                    bullish_indicators.append((indicator, interpretation))
                elif any(term in interp_lower for term in ['bearish', 'overbought', 'downtrend', 'confirming downtrend', 'strong bearish']):
                    bearish_indicators.append((indicator, interpretation))
                elif 'neutral' in interp_lower:
                    neutral_indicators.append((indicator, interpretation))
                    
        logger.debug(f"Bullish indicators: {bullish_indicators}")
        logger.debug(f"Bearish indicators: {bearish_indicators}")
        logger.debug(f"Neutral indicators: {neutral_indicators}")
        
        # Calculate confidence levels based on weighted indicators
        # Some indicators are more reliable than others
        indicator_weights = {
            'macd': 1.5,
            'rsi': 1.2,
            'bollinger': 1.2,
            'ichimoku': 1.5,
            'adx': 1.0,
            'stochastic': 1.0,
            'cci': 0.8,
            'obv': 1.0,
            'sma': 1.0,
            'atr': 0.5  # Not directly a direction indicator
        }
        
        # Calculate weighted counts
        bullish_weight = sum(indicator_weights.get(ind[0], 1.0) for ind in bullish_indicators)
        bearish_weight = sum(indicator_weights.get(ind[0], 1.0) for ind in bearish_indicators)
        neutral_weight = sum(indicator_weights.get(ind[0], 1.0) for ind in neutral_indicators)
        
        total_weight = bullish_weight + bearish_weight + neutral_weight
        
        # Handle the case when there are no valid indicators
        if total_weight == 0:
            logger.warning("No valid indicators found for case analysis")
            overall_sentiment = "Neutral"
            bullish_confidence = 0
            bearish_confidence = 0
            neutral_confidence = 1.0  # Default to neutral when no indicators are available
            
            bullish_confidence_pct = "0.0%"
            bearish_confidence_pct = "0.0%"
            neutral_confidence_pct = "100.0%"
        else:
            # Calculate confidence percentages
            bullish_confidence = bullish_weight / total_weight
            bearish_confidence = bearish_weight / total_weight
            neutral_confidence = neutral_weight / total_weight
            
            # Format as percentages
            bullish_confidence_pct = f"{bullish_confidence * 100:.1f}%"
            bearish_confidence_pct = f"{bearish_confidence * 100:.1f}%"
            neutral_confidence_pct = f"{neutral_confidence * 100:.1f}%"
            
            # Determine overall sentiment
            overall_sentiment = "Neutral"
            if bullish_confidence > max(bearish_confidence, neutral_confidence):
                overall_sentiment = "Bullish"
            elif bearish_confidence > max(bullish_confidence, neutral_confidence):
                overall_sentiment = "Bearish"
        
        # Build the result with supporting evidence
        result = {
            'overall_sentiment': overall_sentiment,
            'cases': {
                'bullish': {
                    'confidence': bullish_confidence_pct,
                    'confidence_raw': bullish_confidence,
                    'supporting_indicators': bullish_indicators
                },
                'bearish': {
                    'confidence': bearish_confidence_pct,
                    'confidence_raw': bearish_confidence,
                    'supporting_indicators': bearish_indicators
                },
                'neutral': {
                    'confidence': neutral_confidence_pct,
                    'confidence_raw': neutral_confidence,
                    'supporting_indicators': neutral_indicators
                }
            }
        }
        
        # Add overall explanation
        supporting_case = overall_sentiment.lower()
        num_supporters = len(result['cases'][supporting_case]['supporting_indicators'])
        
        if total_weight == 0:
            result['explanation'] = "No valid indicators available for analysis. Defaulting to Neutral."
        else:
            result['explanation'] = (
                f"The {supporting_case} case has the highest confidence at {result['cases'][supporting_case]['confidence']} "
                f"with {num_supporters} supporting indicators."
            )
        
        return result

    def get_advanced_analytics(self) -> Dict[str, Any]:
        """
        Compute advanced analytics: volatility forecast, regime detection, strategy suggestion, and open interest analysis.
        Returns:
            Dict with keys:
                - 'volatility_forecast': dict of forecasts for 24h, 4h, 1h
                - 'regime': output of detect_regime
                - 'strategy_suggestion': output of suggest_strategy_for_regime
                - 'watch_for_signals': list of 'what to watch for' signals (optional)
                - 'open_interest_analysis': dict with OI regime and summary (educational, actionable)
        """
        from src.services.indicators import forecast_volatility
        from src.services.trading_strategies import detect_regime, suggest_strategy_for_regime, generate_watch_for_signals
        from src.services.open_interest import fetch_open_interest

        if self.data is None or self.data.empty:
            return {
                'volatility_forecast': {},
                'regime': {'trend_regime': 'ambiguous', 'volatility_regime': 'ambiguous', 'confidence': 'low', 'metrics': {}},
                'strategy_suggestion': {'strategy': 'insufficient_data', 'educational_rationale': 'Not enough data to determine a safe or effective strategy.', 'actionable_advice': 'Do not open new positions.'},
                'watch_for_signals': ["Watch for more price action and indicator signals to develop before trading."],
                'open_interest_analysis': {'regime': 'insufficient_data', 'confidence': 'low', 'summary': 'No open interest data available.'}
            }
        # Volatility forecasts
        volatility_forecast = {
            horizon: forecast_volatility(self.data, horizon=horizon)
            for horizon in ["24h", "4h", "1h"]
        }
        # Regime detection
        regime = detect_regime(self.data)
        # Strategy suggestion
        strategy_suggestion = suggest_strategy_for_regime(regime)
        # Watch for signals (only if strategy is 'insufficient_data' or 'reduce_exposure')
        watch_for_signals = []
        if strategy_suggestion.get('strategy') in ['insufficient_data', 'reduce_exposure']:
            watch_for_signals = generate_watch_for_signals(regime, regime.get('metrics', {}))
        
        # Open Interest Analytics
        try:
            # Fetch open interest data that already includes enhanced analysis
            oi_data = fetch_open_interest(self.symbol)
            
            # Get the latest price from our data
            latest_price = float(self.data['close'].iloc[-1]) if not self.data.empty else 0
            
            # If we have successful data but it's missing the enhanced fields, add them
            if not oi_data.get('error') and 'metrics' not in oi_data and 'trading_signals' not in oi_data:
                # Create a simulated time series for analyze_open_interest
                from src.services.open_interest import analyze_open_interest
                
                # Create historical data points for analysis
                historical_data = []
                
                # Current point
                current_oi = oi_data.get('open_interest_value', 0)
                current_timestamp = int(datetime.now().timestamp() * 1000)
                
                # Calculate previous point based on 24h change
                oi_change_24h = oi_data.get('open_interest_change_24h', 0)
                previous_oi = current_oi / (1 + (oi_change_24h / 100)) if oi_change_24h != -100 else current_oi
                
                # Generate simple data points for analysis
                historical_data = [
                    {
                        'timestamp': current_timestamp - (24 * 3600 * 1000),  # 24 hours ago
                        'open_interest': previous_oi,
                        'price': latest_price * 0.98,  # Approximate previous price
                        'volume': 1000  # Placeholder volume
                    },
                    {
                        'timestamp': current_timestamp,
                        'open_interest': current_oi,
                        'price': latest_price,
                        'volume': 1000  # Placeholder volume
                    }
                ]
                
                # Get enhanced analysis
                enhanced_analysis = analyze_open_interest(historical_data)
                
                # Merge the enhanced analysis with existing data
                oi_data.update(enhanced_analysis)
            
            # Use the obtained OI data (either with existing or added enhanced fields)
            # This preserves all available information
            oi_analysis = oi_data
            
        except Exception as e:
            logging.error(f"Error fetching open interest data: {e}")
            oi_analysis = {
                'regime': 'error', 
                'confidence': 'low',
                'summary': f'Error fetching or analyzing open interest: {e}',
                'value': 0,
                'change_24h': 0,
                'trading_signals': {'signal': 'neutral', 'action': 'wait', 'entry': None, 'stop_loss': None, 'take_profit': None},
                'metrics': {},
                'divergence': {'detected': False, 'type': None, 'strength': 0}
            }
        
        return {
            'volatility_forecast': volatility_forecast,
            'regime': regime,
            'strategy_suggestion': strategy_suggestion,
            'watch_for_signals': watch_for_signals,
            'open_interest_analysis': oi_analysis
        } 