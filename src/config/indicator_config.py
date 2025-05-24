"""
Indicator configuration and types for technical analysis.
"""
from typing import Dict, List, Optional, TypedDict

class IndicatorParamTyped(TypedDict, total=False):
    length: int
    column: str
    high: str  # For MFI, KC etc.
    low: str
    close: str
    volume: str  # For MFI, CMF, VWAP
    fast: int  # For MACD
    slow: int  # For MACD
    signal: int  # For MACD
    std: float  # For BBands
    atr_length: int  # For KC
    mamode: str  # For KC
    scalar: float  # For KC
    k: int  # For Stoch
    d: int  # For Stoch
    smooth_k: int  # For Stoch
    constant: float  # For CCI
    anchor: str  # For VWAP
    tenkan: int  # For Ichimoku
    kijun: int  # For Ichimoku
    senkou: int  # For Ichimoku

class IndicatorConfigTyped(TypedDict):
    name: str
    params: IndicatorParamTyped
    col_name: Optional[str]  # Made optional, present for single output
    multi_cols: Optional[Dict[str, str]]  # Made optional, present for multi output

def get_indicator_configurations(window_size: int, price_col: str) -> List[IndicatorConfigTyped]:
    """
    Return the indicator configuration list, using window_size and price_col for dynamic fields.
    """
    return [
        {
            'name': 'sma',
            'params': {'length': window_size, 'column': price_col},
            'col_name': f'sma_{window_size}',
            'multi_cols': None
        },
        {
            'name': 'sma',
            'params': {'length': 50, 'column': price_col},
            'col_name': 'sma_50',
            'multi_cols': None
        },
        {
            'name': 'ema',
            'params': {'length': window_size, 'column': price_col},
            'col_name': f'ema_{window_size}',
            'multi_cols': None
        },
        {
            'name': 'rsi',
            'params': {'length': 14, 'column': price_col},
            'col_name': 'rsi_14',
            'multi_cols': None
        },
        {
            'name': 'mfi',
            'params': {
                'length': 14, 'high': 'high', 'low': 'low', 'close': 'close',
                'volume': 'volume'
            },
            'col_name': 'mfi_14',
            'multi_cols': None
        },
        {
            'name': 'macd',
            'params': {'fast': 12, 'slow': 26, 'signal': 9, 'column': price_col},
            'multi_cols': {
                'MACD_12_26_9': 'MACD_12_26_9',
                'MACDh_12_26_9': 'MACDh_12_26_9',
                'MACDs_12_26_9': 'MACDs_12_26_9'
            },
            'col_name': None
        },
        {
            'name': 'bbands',
            'params': {'length': window_size, 'std': 2.0, 'column': price_col},
            'multi_cols': {
                f'BBL_{window_size}_2.0': f'BBL_{window_size}_2.0',
                f'BBM_{window_size}_2.0': f'BBM_{window_size}_2.0',
                f'BBU_{window_size}_2.0': f'BBU_{window_size}_2.0'
            },
            'col_name': None
        },
        {
            'name': 'kc',
            'params': {
                'length': 20, 'atr_length': 10, 'mamode': 'ema', 'scalar': 2.0,
                'high': 'high', 'low': 'low', 'close': 'close'
            },
            'multi_cols': {
                'KCLe_20_2.0': 'KCLe_20_2.0',
                'KCBe_20_2.0': 'KCMe_20_2.0',
                'KCUe_20_2.0': 'KCUe_20_2.0'
            },
            'col_name': None
        },
        {
            'name': 'adx',
            'params': {'length': 14},
            'multi_cols': {
                'ADX_14': 'ADX_14',
                'DMP_14': 'DMP_14',
                'DMN_14': 'DMN_14'
            },
            'col_name': None
        },
        {
            'name': 'stoch',
            'params': {'k': 14, 'd': 3, 'smooth_k': 3},
            'multi_cols': {
                'STOCHk_14_3_3': 'STOCHk_14_3_3',
                'STOCHd_14_3_3': 'STOCHd_14_3_3'
            },
            'col_name': None
        },
        {
            'name': 'cci',
            'params': {'length': 20, 'constant': 0.015},
            'col_name': 'CCI_20_0.015',
            'multi_cols': None
        },
        {
            'name': 'atr',
            'params': {'length': 14},
            'col_name': 'ATR_14',
            'multi_cols': None
        },
        {
            'name': 'obv',
            'params': {'column': 'volume'},
            'col_name': 'obv',
            'multi_cols': None
        },
        {
            'name': 'cmf',
            'params': {
                'length': 20, 'high': 'high', 'low': 'low', 'close': 'close',
                'volume': 'volume'
            },
            'col_name': 'cmf_20',
            'multi_cols': None
        },
        {
            'name': 'vwap',
            'params': {
                'anchor': 'D', 'high': 'high', 'low': 'low', 'close': 'close',
                'volume': 'volume'
            },
            'col_name': 'vwap_D',
            'multi_cols': None
        },
        {
            'name': 'ichimoku',
            'params': {'tenkan': 9, 'kijun': 26, 'senkou': 52},
            'multi_cols': {
                'ITS_9': 'ITS_9',
                'IKS_26': 'IKS_26',
                'ISA_9': 'ISA_9',
                'ISB_26': 'ISB_26',
                'ICS_26': 'ICS_26'
            },
            'col_name': None
        }
    ] 

# Mapping from interval string to recommended window size for indicators
INTERVAL_WINDOW_SIZE_MAP = {
    '1m': 10,
    '5m': 14,
    '15m': 20,
    '30m': 20,
    '1h': 20,
    '4h': 50,
    '1d': 100,
    '1wk': 200,
    '1mo': 200
}

def get_window_size_for_interval(interval: str) -> int:
    """
    Return the recommended window size for a given interval string.
    Defaults to 20 if interval is not recognized.
    """
    return INTERVAL_WINDOW_SIZE_MAP.get(interval.lower(), 20) 