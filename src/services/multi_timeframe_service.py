"""
Multi-Timeframe Analysis Service

This module provides services for fetching, caching, and analyzing market data
across multiple timeframes to identify signal confluence and generate insights.
"""

import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from src.services.data_fetcher import DataFetcher, CCXT_TIMEFRAMES
from src.services.cache_service import store_dataframe, get_cached_dataframe
from src.services.indicators.get_indicator import get_indicator # Assuming this can be used for multi-timeframe

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES = ['1h', '4h', '1d', '1w']

class TimeframeAnalyzer:
    """
    Analyzes market data across multiple timeframes.
    """

    def __init__(self, symbol: str, timeframes: Optional[List[str]] = None, exchange_name: str = "okx"):
        """
        Initialize the TimeframeAnalyzer.

        Args:
            symbol: The trading symbol (e.g., BTC/USDT).
            timeframes: A list of timeframes to analyze (e.g., ['1h', '4h', '1d']).
                        Defaults to ['1h', '4h', '1d', '1w'].
            exchange_name: The name of the exchange to use.
        """
        self.symbol = symbol
        self.timeframes = timeframes if timeframes else DEFAULT_TIMEFRAMES
        self.data_fetcher = DataFetcher(exchange_name=exchange_name)
        self.exchange_name = exchange_name
        self._validate_timeframes()
        logger.info(f"TimeframeAnalyzer initialized for {symbol} with timeframes: {self.timeframes}")

    def _validate_timeframes(self):
        """Validate if the provided timeframes are supported."""
        valid_timeframes = []
        for tf in self.timeframes:
            if tf in CCXT_TIMEFRAMES:
                valid_timeframes.append(tf)
            else:
                logger.warning(f"Unsupported timeframe '{tf}' provided. It will be ignored. "
                               f"Supported timeframes: {list(CCXT_TIMEFRAMES.keys())}")
        self.timeframes = valid_timeframes
        if not self.timeframes:
            logger.error("No valid timeframes provided for analysis. Defaulting to ['1h', '4h', '1d', '1w'].")
            self.timeframes = DEFAULT_TIMEFRAMES


    async def fetch_multi_timeframe_data(self, limit: int = 100, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetches OHLCV data for the symbol across all specified timeframes.
        Uses asyncio for concurrent fetching.

        Args:
            limit: Number of candles to fetch for each timeframe.
            use_cache: Whether to use cached data.

        Returns:
            A dictionary where keys are timeframes and values are pandas DataFrames.
        """
        logger.info(f"Fetching multi-timeframe data for {self.symbol}, timeframes: {self.timeframes}, limit: {limit}")
        tasks = {}
        for tf in self.timeframes:
            # Pass exchange_name to generate_cache_key_for_df if it's a required param
            # Based on current cache_service.py, it might not be needed directly if cache_key is self-contained
            cache_key_parts = [
                "ohlcv", self.exchange_name, self.symbol.replace('/', '_'), 
                tf, str(limit)
            ]
            # Add 'since' if it were a parameter, similar to DataFetcher. For now, assume not.
            cache_key = "_".join(cache_key_parts)
            
            if use_cache:
                cached_df = get_cached_dataframe(cache_key)
                if cached_df is not None:
                    logger.debug(f"Cache hit for {self.symbol} - {tf} (key: {cache_key})")
                    tasks[tf] = cached_df # Store DataFrame directly
                    continue # Skip fetching if cached
                logger.debug(f"Cache miss for {self.symbol} - {tf} (key: {cache_key})")

            # If not cached, create a task to fetch it
            # Note: DataFetcher.fetch_historical_ohlcv is not async, so true concurrency is limited here
            # For true async, fetch_historical_ohlcv would need to be async or run in a thread pool
            tasks[tf] = asyncio.to_thread(
                self.data_fetcher.fetch_historical_ohlcv,
                self.symbol,
                tf,
                limit,
                use_cache=False # Already handled cache check above, direct fetch now
            )

        results = {}
        fetched_data_for_caching = {}

        # Await async tasks and process results
        for tf, task_or_df in tasks.items():
            if isinstance(task_or_df, pd.DataFrame): # Already a DataFrame (from cache)
                results[tf] = task_or_df
            else: # An awaitable task
                try:
                    df = await task_or_df
                    if df is not None and not df.empty:
                        results[tf] = df
                        fetched_data_for_caching[tf] = df
                        logger.debug(f"Successfully fetched data for {self.symbol} - {tf}")
                    else:
                        results[tf] = pd.DataFrame()
                        logger.warning(f"No data fetched for {self.symbol} - {tf}")
                except Exception as e:
                    logger.error(f"Error fetching data for {self.symbol} - {tf}: {e}", exc_info=True)
                    results[tf] = pd.DataFrame()
        
        # Store newly fetched data to cache
        if use_cache:
            for tf, df in fetched_data_for_caching.items():
                # Regenerate cache key for storing
                cache_key_store_parts = [
                     "ohlcv", self.exchange_name, self.symbol.replace('/', '_'), 
                     tf, str(limit)
                ]
                cache_key_store = "_".join(cache_key_store_parts)

                metadata = {
                    'symbol': self.symbol, 'timeframe': tf, 'limit': limit, 
                    'source': self.exchange_name, 'service': 'multi_timeframe_service'
                }
                store_dataframe(cache_key_store, df, metadata=metadata, timeframe=tf)
                logger.debug(f"Stored fetched data for {self.symbol} - {tf} to cache (key: {cache_key_store})")

        return results

    def calculate_indicators_for_all_timeframes(
        self, 
        multi_timeframe_data: Dict[str, pd.DataFrame],
        indicator_configs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates specified indicators for each timeframe's data.

        Args:
            multi_timeframe_data: Dict of DataFrames, keys are timeframes.
            indicator_configs: List of indicator configurations. Each config is a dict:
                               {'name': 'sma', 'params': {'length': 20}}

        Returns:
            Dict where keys are timeframes, and values are dicts of indicator results.
            Example: {'1h': {'sma_20': {...values...}}, '4h': {'sma_20': {...values...}}}
        """
        all_indicator_results = {}
        for tf, df in multi_timeframe_data.items():
            if df.empty:
                logger.warning(f"Skipping indicator calculation for {tf} due to empty data for {self.symbol}.")
                all_indicator_results[tf] = {}
                continue

            tf_results = {}
            logger.info(f"Calculating indicators for {self.symbol} on timeframe {tf}...")
            for config in indicator_configs:
                indicator_name = config.get('name')
                params = config.get('params')
                try:
                    # get_indicator should handle caching internally if use_cache=True
                    # For MTA, we might want to force re-calc or ensure cache keys are distinct enough
                    # if underlying data for a timeframe changes.
                    # Assuming get_indicator's caching is robust for this.
                    indicator_output = get_indicator(
                        df.copy(), # Pass a copy to avoid modification issues
                        indicator_name, 
                        params=params, 
                        symbol=self.symbol, # Pass symbol for context if get_indicator uses it
                        timeframe=tf,      # Pass timeframe for context
                        use_cache=True     # Leverage indicator-level caching
                    )
                    
                    # Construct a unique key for the result, e.g., 'sma_20'
                    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items())) if params else ""
                    result_key = f"{indicator_name}{'_' + param_str if param_str else ''}"
                    
                    tf_results[result_key] = indicator_output # Store the full output (values, signals, etc.)
                    logger.debug(f"Calculated {result_key} for {self.symbol} on {tf}")
                except Exception as e:
                    logger.error(f"Error calculating indicator {indicator_name} for {self.symbol} on {tf}: {e}", exc_info=True)
                    tf_results[indicator_name] = {'error': str(e), 'values': None, 'signals': None}
            all_indicator_results[tf] = tf_results
            logger.info(f"Finished calculating indicators for {self.symbol} on timeframe {tf}.")
        return all_indicator_results

    def analyze_confluence(
        self, 
        multi_timeframe_indicator_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes confluence of signals across different timeframes.
        This is a placeholder for a more sophisticated confluence logic.

        Args:
            multi_timeframe_indicator_results: Indicator results per timeframe.

        Returns:
            A dictionary summarizing confluence findings.
        """
        logger.info(f"Analyzing signal confluence for {self.symbol}...")
        confluence_summary = {
            "summary": "Confluence analysis placeholder. Logic to be implemented.",
            "details_per_indicator": {}, # E.g., {'sma_20': {'bullish_timeframes': ['1h', '4h'], ...}}
            "overall_sentiment_score": 0 # Example: -1 (bearish) to 1 (bullish)
        }
        
        # Example: Iterate through indicators and timeframes to find alignment
        # This needs to be adapted based on how 'signals' are structured in get_indicator output
        
        # Identify all unique indicator keys (e.g., 'sma_20', 'rsi_14')
        all_indicator_keys = set()
        for tf_results in multi_timeframe_indicator_results.values():
            all_indicator_keys.update(tf_results.keys())

        for indicator_key in all_indicator_keys:
            confluence_summary["details_per_indicator"][indicator_key] = {
                "signals_by_timeframe": {},
                "aligned_signal": "neutral", # neutral, bullish, bearish, mixed
                "aligned_timeframes": []
            }
            
            signals_across_timeframes = []
            for tf in self.timeframes:
                indicator_result = multi_timeframe_indicator_results.get(tf, {}).get(indicator_key)
                if indicator_result and not indicator_result.get('error'):
                    # Assuming 'signals' might be a list of signal dicts or a summary signal string/dict
                    # This part is highly dependent on the structure of `get_indicator`'s 'signals' output.
                    # For now, let's assume `indicator_result['signals']` gives a primary signal like 'buy', 'sell', 'hold'
                    # or a more complex dict that can be interpreted.
                    # Let's simplify and assume a hypothetical 'primary_signal' field
                    
                    primary_signal = None
                    if isinstance(indicator_result.get('signals'), list) and indicator_result['signals']:
                        # Example: take the latest signal if it's a list of dicts with 'signal_type'
                        # This needs to be robust. What if signals is a dict itself?
                        # Or just a string?
                        # For now, let's assume it's a dict and has a 'general_signal' field.
                        # This part needs to be adapted based on the actual structure of `get_indicator` output.
                        sig_data = indicator_result['signals']
                        if isinstance(sig_data, dict) and 'general_signal' in sig_data:
                            primary_signal = sig_data['general_signal'] # e.g., 'bullish', 'bearish', 'neutral'
                        elif isinstance(sig_data, str): # if it's just a string like 'buy'
                             primary_signal = sig_data
                        # Add more sophisticated parsing if needed
                            
                    elif isinstance(indicator_result.get('signals'), dict):
                         primary_signal = indicator_result['signals'].get('general_signal', 'unknown')


                    if primary_signal:
                         confluence_summary["details_per_indicator"][indicator_key]["signals_by_timeframe"][tf] = primary_signal
                         signals_across_timeframes.append(primary_signal)
                    else:
                        confluence_summary["details_per_indicator"][indicator_key]["signals_by_timeframe"][tf] = "no_signal"


            # Basic confluence logic: if all signals are the same (and not 'neutral' or 'no_signal')
            if signals_across_timeframes:
                first_valid_signal = next((s for s in signals_across_timeframes if s not in ['neutral', 'no_signal', 'unknown', None]), None)
                if first_valid_signal and all(s == first_valid_signal or s in ['neutral', 'no_signal', 'unknown', None] for s in signals_across_timeframes):
                    confluence_summary["details_per_indicator"][indicator_key]["aligned_signal"] = first_valid_signal
                    confluence_summary["details_per_indicator"][indicator_key]["aligned_timeframes"] = [
                        tf for tf, sig in confluence_summary["details_per_indicator"][indicator_key]["signals_by_timeframe"].items() if sig == first_valid_signal
                    ]
                elif len(set(s for s in signals_across_timeframes if s not in ['neutral', 'no_signal', 'unknown', None])) > 1:
                     confluence_summary["details_per_indicator"][indicator_key]["aligned_signal"] = "mixed"
                else: # All neutral or no signals
                    confluence_summary["details_per_indicator"][indicator_key]["aligned_signal"] = "neutral"


        # Placeholder for overall sentiment score calculation
        # Could be based on the number of bullish vs bearish aligned signals, weighted by timeframe, etc.
        bullish_confluences = 0
        bearish_confluences = 0
        for ind_data in confluence_summary["details_per_indicator"].values():
            aligned_sig = ind_data.get("aligned_signal", "neutral").lower()
            if "bull" in aligned_sig or "buy" in aligned_sig : # crude check
                bullish_confluences +=1
            elif "bear" in aligned_sig or "sell" in aligned_sig: # crude check
                bearish_confluences +=1
        
        if bullish_confluences > bearish_confluences:
            confluence_summary["overall_sentiment_score"] = 0.5 + 0.5 * (bullish_confluences / (len(all_indicator_keys) or 1))
        elif bearish_confluences > bullish_confluences:
             confluence_summary["overall_sentiment_score"] = -0.5 - 0.5 * (bearish_confluences / (len(all_indicator_keys) or 1))
        else: # equal or no strong confluences
            confluence_summary["overall_sentiment_score"] = 0.0
        
        # Clamp score between -1 and 1
        confluence_summary["overall_sentiment_score"] = max(-1, min(1, confluence_summary["overall_sentiment_score"]))


        logger.info(f"Confluence analysis for {self.symbol} complete. Overall score: {confluence_summary['overall_sentiment_score']:.2f}")
        return confluence_summary
        
    def generate_timeframe_summary(
        self,
        multi_timeframe_data: Dict[str, pd.DataFrame],
        multi_timeframe_indicator_results: Dict[str, Dict[str, Any]],
        confluence_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive summary of the multi-timeframe analysis.
        Includes educational content about timeframe relationships.

        Args:
            multi_timeframe_data: Original OHLCV data per timeframe.
            multi_timeframe_indicator_results: Calculated indicators per timeframe.
            confluence_analysis: Results of the confluence analysis.

        Returns:
            A dictionary containing the full analysis report.
        """
        logger.info(f"Generating multi-timeframe summary for {self.symbol}...")
        
        report = {
            "symbol": self.symbol,
            "analyzed_timeframes": self.timeframes,
            "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
            "data_summary": {},
            "indicator_summary": multi_timeframe_indicator_results, # Already structured
            "confluence_analysis": confluence_analysis,
            "educational_notes": self._get_educational_content()
        }

        for tf, df in multi_timeframe_data.items():
            if not df.empty:
                report["data_summary"][tf] = {
                    "start_date": df.index.min().isoformat() if not df.empty else None,
                    "end_date": df.index.max().isoformat() if not df.empty else None,
                    "candle_count": len(df),
                    "latest_close": df['close'].iloc[-1] if not df.empty and 'close' in df.columns else None,
                }
            else:
                report["data_summary"][tf] = {
                     "start_date": None, "end_date": None, "candle_count": 0, "latest_close": None, "status": "No data"
                }
        
        logger.info(f"Multi-timeframe summary for {self.symbol} generated successfully.")
        return report

    def _get_educational_content(self) -> Dict[str, str]:
        """
        Provides educational content about multi-timeframe analysis.
        This can be expanded with more detailed explanations.
        """
        return {
            "title": "Understanding Multi-Timeframe Analysis (MTA)",
            "introduction": 
                "Multi-timeframe analysis involves monitoring the same asset across different chart periodicities. "
                "Shorter timeframes (e.g., 1h, 4h) show detailed price action and are useful for entry/exit timing. "
                "Longer timeframes (e.g., 1d, 1w) establish the dominant trend and key support/resistance levels.",
            "why_mta":
                "Using MTA helps traders: 
"
                "1. Identify the primary trend on higher timeframes.
"
                "2. Find lower-risk entries in the direction of that trend on shorter timeframes.
"
                "3. Avoid 'noise' and false signals common on very short timeframes.
"
                "4. Gain a broader market perspective.",
            "confluence":
                "Confluence occurs when multiple technical indicators or signals align across different timeframes, "
                "pointing to the same market direction. Strong confluence can increase the probability of a successful trade. "
                "For example, if the 1-day chart shows an uptrend and a buy signal, and the 4-hour chart also confirms with a buy signal, "
                "this is a point of bullish confluence.",
            "common_approach":
                "A common approach is the 'Top-Down Analysis':
"
                "1. Weekly Chart (Primary Trend): Determine the overall market direction and major levels.
"
                "2. Daily Chart (Intermediate Trend): Refine the trend, identify chart patterns and secondary levels.
"
                "3. 4-Hour Chart (Short-Term Trend/Setup): Look for specific trade setups in line with higher timeframe trends.
"
                "4. 1-Hour Chart (Entry/Exit): Fine-tune entry and exit points, manage risk.",
            "disclaimer":
                "Multi-timeframe analysis, like all trading strategies, is not foolproof. "
                "Always use risk management and consider other factors like market news and volume."
        }

    async def run_full_analysis(
        self, 
        indicator_configs: List[Dict[str, Any]], 
        data_limit: int = 100, 
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Runs the complete multi-timeframe analysis pipeline.

        Args:
            indicator_configs: List of indicator configurations to calculate.
            data_limit: Number of candles to fetch for each timeframe.
            use_cache: Whether to use caching for data and indicators.

        Returns:
            A comprehensive analysis report.
        """
        logger.info(f"Starting full multi-timeframe analysis for {self.symbol}...")
        
        # 1. Fetch data for all timeframes
        # The use_cache here applies to the data fetching step
        multi_timeframe_data = await self.fetch_multi_timeframe_data(limit=data_limit, use_cache=use_cache)
        
        # Check if any data was fetched
        if not any(not df.empty for df in multi_timeframe_data.values()):
            logger.error(f"No data could be fetched for any timeframe for {self.symbol}. Aborting analysis.")
            return {
                "error": "Failed to fetch data for any specified timeframe.",
                "symbol": self.symbol,
                "analyzed_timeframes": self.timeframes,
                "indicator_configs": indicator_configs
            }

        # 2. Calculate indicators for all timeframes
        # The use_cache within get_indicator will handle indicator caching
        multi_timeframe_indicator_results = self.calculate_indicators_for_all_timeframes(
            multi_timeframe_data, 
            indicator_configs
        )

        # 3. Analyze confluence of signals
        confluence_analysis_results = self.analyze_confluence(multi_timeframe_indicator_results)

        # 4. Generate final report
        full_report = self.generate_timeframe_summary(
            multi_timeframe_data,
            multi_timeframe_indicator_results,
            confluence_analysis_results
        )
        
        logger.info(f"Full multi-timeframe analysis for {self.symbol} completed.")
        return full_report

# Example Usage (for testing purposes, can be removed or put in a test file)
async def example_mta_run():
    symbol_to_analyze = "BTC/USDT"
    # Define indicators to calculate. Ensure 'get_indicator' supports these.
    indicator_configurations = [
        {'name': 'sma', 'params': {'length': 20}},
        {'name': 'ema', 'params': {'length': 50}},
        {'name': 'rsi', 'params': {'length': 14}},
        # {'name': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}, # If MACD is supported by get_indicator
    ]

    analyzer = TimeframeAnalyzer(symbol=symbol_to_analyze, timeframes=['1h', '4h', '1d'])
    
    print(f"Running full MTA for {symbol_to_analyze}...")
    try:
        analysis_report = await analyzer.run_full_analysis(
            indicator_configs=indicator_configurations,
            data_limit=100, # Fetch 100 candles per timeframe
            use_cache=True
        )
        
        # Pretty print the report (or parts of it)
        print("\n--- MTA Report ---")
        print(f"Symbol: {analysis_report.get('symbol')}")
        print(f"Timeframes: {analysis_report.get('analyzed_timeframes')}")
        print(f"Timestamp: {analysis_report.get('timestamp')}")
        
        print("\n--- Confluence ---")
        if 'confluence_analysis' in analysis_report:
            print(f"  Overall Sentiment Score: {analysis_report['confluence_analysis'].get('overall_sentiment_score')}")
            # For brevity, print only overall score. Full details are in analysis_report['confluence_analysis']['details_per_indicator']
        
        print("\n--- Educational Notes ---")
        if 'educational_notes' in analysis_report:
            print(f"  Title: {analysis_report['educational_notes'].get('title')}")

        # To see the full structure, you might dump to JSON
        # import json
        # print(json.dumps(analysis_report, indent=2, default=str)) # default=str for pandas Timestamps etc.

    except Exception as e:
        print(f"An error occurred during MTA example run: {e}")
        logger.error("Error in example_mta_run", exc_info=True)

if __name__ == '__main__':
    # Setup basic logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # To run the example:
    asyncio.run(example_mta_run()) 