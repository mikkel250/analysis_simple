import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Any

# Visualization functions (formerly in src.analysis.market_data)

def plot_price_history(
    data: pd.DataFrame, 
    title: str = 'Stock Price History',
    price_col: str = 'close'
    # figsize: Tuple[int, int] = DEFAULT_FIGSIZE, # Removed, not used by plotly
    # style: str = DEFAULT_STYLE # Removed, not used by plotly
) -> go.Figure:
    """
    Plot stock price history using plotly.
    
    Args:
        data: DataFrame containing price data
        title: Plot title
        price_col: Column name containing price data
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Check if we have multiple symbols
    if 'symbol' in data.columns and len(data['symbol'].unique()) > 1:
        for symbol, group in data.groupby('symbol'):
            fig.add_trace(
                go.Scatter(
                    x=group.index, 
                    y=group[price_col], 
                    name=symbol,
                    mode='lines'
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data[price_col],
                name=price_col,
                mode='lines'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    
    return fig

def plot_candlestick(
    data: pd.DataFrame, 
    title: str = 'Candlestick Chart',
    include_volume: bool = True
) -> go.Figure:
    """
    Create a candlestick chart using plotly.
    
    Args:
        data: DataFrame containing OHLC data
        title: Chart title
        include_volume: Whether to include volume subplot
    
    Returns:
        Plotly figure
    """
    if include_volume:
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('', 'Volume'),
            row_heights=[0.8, 0.2]
        )
    else:
        fig = go.Figure()
    
    # Add candlestick trace
    trace_kwargs = {}
    if include_volume:
        trace_kwargs['row'] = 1
        trace_kwargs['col'] = 1
    
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='OHLC'
        ),
        **trace_kwargs
    )
    
    # Add volume trace if requested
    if include_volume and 'volume' in data.columns:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig

def plot_technical_analysis(
    data: pd.DataFrame, 
    title: str = 'Technical Analysis',
    include_volume: bool = True
) -> go.Figure:
    """
    Create a technical analysis chart with indicators using a data-driven configuration.
    Assumes 'data' DataFrame already contains all necessary indicator columns.
    
    Args:
        data: DataFrame containing price data and indicators
        title: Chart title
        include_volume: Whether to include volume subplot
    
    Returns:
        Plotly figure
    """

    PLOT_CONFIG = [
        {
            "name": "Price/Candlestick",
            "type": "candlestick",
            "overlays": [
                {"type": "bband", "id": "bband_lower", "prefix": "BBL", "name": "Lower Band", "line_style": {"width": 1}, "legendgroup": "Bollinger Bands", "showlegend_master": True},
                {"type": "bband", "id": "bband_middle", "prefix": "BBM", "name": "Middle Band", "line_style": {"width": 1}, "legendgroup": "Bollinger Bands"},
                {"type": "bband", "id": "bband_upper", "prefix": "BBU", "name": "Upper Band", "line_style": {"width": 1}, "legendgroup": "Bollinger Bands"},
                {"type": "line", "id": "ichimoku_tenkan", "col": "ITS_9", "name": "Tenkan-sen", "line_style": {"color": "cyan", "width": 1}, "legendgroup": "Ichimoku"},
                {"type": "line", "id": "ichimoku_kijun", "col": "IKS_26", "name": "Kijun-sen", "line_style": {"color": "magenta", "width": 1}, "legendgroup": "Ichimoku"},
                {"type": "line", "id": "ichimoku_chikou", "col": "ICS_26", "name": "Chikou Span", "line_style": {"color": "green", "width": 1, "dash": "dot"}, "legendgroup": "Ichimoku"},
                {"type": "fill", "id": "ichimoku_cloud", "col_a": "ISA_9", "col_b": "ISB_26", "name_a": "Senkou A", "name_b": "Senkou B", 
                 "line_a_style": {"color": "rgba(119, 210, 131, 0.3)"}, 
                 "line_b_style": {"color": "rgba(210, 131, 119, 0.3)"},
                 "fillcolor": "rgba(180, 180, 180, 0.2)", "legendgroup": "Ichimoku", "showlegend_master": False}
            ]
        },
        {
            "name": "Volume",
            "type": "bar",
            "col": "volume",
            "is_visible": lambda d, iv: 'volume' in d.columns and iv,
            "subplot_title": "Volume",
            "row_height": 0.15,
            "trace_params": {"name": "Volume", "marker_color": "rgba(0, 0, 255, 0.5)"}
        },
        {
            "name": "RSI",
            "type": "line",
            "col": "rsi_14",
            "is_visible": lambda d, iv: 'rsi_14' in d.columns,
            "subplot_title": "RSI (14)",
            "row_height": 0.15,
            "trace_params": {"name": "RSI (14)", "line": {"color": "purple", "width": 1}},
            "hlines": [
                {"y": 70, "line_dash": "dash", "line_color": "red"},
                {"y": 30, "line_dash": "dash", "line_color": "green"}
            ]
        },
        {
            "name": "MACD",
            "type": "custom_macd",
            "cols": {"macd": "MACD_12_26_9", "signal": "MACDs_12_26_9", "hist": "MACDh_12_26_9"},
            "is_visible": lambda d, iv: all(c in d.columns for c in ["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"]),
            "subplot_title": "MACD",
            "row_height": 0.20,
            "trace_params": {
                "macd_line": {"name": "MACD Line", "line": {"color": "blue", "width": 1}},
                "signal_line": {"name": "MACD Signal", "line": {"color": "red", "width": 1}},
                "hist_bar": {"name": "MACD Hist"} # Colors are dynamic
            }
        },
        {
            "name": "Stochastic",
            "type": "custom_stochastic",
            "cols": {"k": "STOCHk_14_3_3", "d": "STOCHd_14_3_3"},
            "is_visible": lambda d, iv: all(c in d.columns for c in ["STOCHk_14_3_3", "STOCHd_14_3_3"]),
            "subplot_title": "Stochastic",
            "row_height": 0.15,
            "trace_params": {
                "k_line": {"name": "%K", "line": {"color": "blue", "width": 1}},
                "d_line": {"name": "%D", "line": {"color": "red", "width": 1}}
            },
            "hlines": [
                {"y": 80, "line_dash": "dash", "line_color": "red"},
                {"y": 20, "line_dash": "dash", "line_color": "green"}
            ]
        }
    ]

    active_subplots = []
    subplot_titles_list = []
    row_heights_list = []

    # Main price plot is always first
    price_plot_config = next(p for p in PLOT_CONFIG if p["type"] == "candlestick")
    active_subplots.append(price_plot_config)
    subplot_titles_list.append("Price") # Main plot title can be less verbose
    row_heights_list.append(0.6) # Default height for price plot

    # Determine active indicator subplots
    for config_item in PLOT_CONFIG:
        if config_item["type"] != "candlestick": # Already handled
            if config_item.get("is_visible", lambda d, iv: True)(data, include_volume):
                active_subplots.append(config_item)
                subplot_titles_list.append(config_item["subplot_title"])
                row_heights_list.append(config_item.get("row_height", 0.15))
    
    num_rows = len(active_subplots)

    if num_rows == 0: # Should not happen if price plot is always there
        return go.Figure() # Return empty figure

    # Normalize heights if custom subplots exist
    if num_rows > 1:
        total_other_heights = sum(row_heights_list[1:])
        if total_other_heights > 0.8: # Cap other subplots total height relative to price
            scale_factor = 0.8 / total_other_heights
            for i in range(1, len(row_heights_list)):
                row_heights_list[i] *= scale_factor
        row_heights_list[0] = 1.0 - sum(row_heights_list[1:]) # Price gets the rest
    elif num_rows == 1: # Only price plot
        row_heights_list = [1.0]

    fig = make_subplots(
        rows=num_rows, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles_list,
        row_heights=row_heights_list
    )

    current_row_idx = 1 # Plotly rows are 1-indexed

    for plot_item in active_subplots:
        plot_type = plot_item["type"]

        if plot_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='OHLC'
                ),
                row=current_row_idx, col=1
            )
            # Process overlays for candlestick
            for overlay in plot_item.get("overlays", []):
                overlay_type = overlay["type"]
                if overlay_type == "bband":
                    bband_col = next((col for col in data.columns if col.startswith(f'{overlay["prefix"]}_')), None)
                    if bband_col:
                        fig.add_trace(go.Scatter(x=data.index, y=data[bband_col], name=overlay["name"],
                                                line=overlay.get("line_style", {}),
                                                legendgroup=overlay.get("legendgroup"),
                                                showlegend=overlay.get("showlegend_master", False)),
                                      row=current_row_idx, col=1)
                elif overlay_type == "line" and overlay["col"] in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data[overlay["col"]], name=overlay["name"],
                                            line=overlay.get("line_style", {}),
                                            legendgroup=overlay.get("legendgroup")),
                                  row=current_row_idx, col=1)
                elif overlay_type == "fill" and overlay["col_a"] in data.columns and overlay["col_b"] in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data[overlay["col_a"]], name=overlay["name_a"],
                                            line=overlay.get("line_a_style", {}),
                                            legendgroup=overlay.get("legendgroup")),
                                  row=current_row_idx, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=data[overlay["col_b"]], name=overlay["name_b"],
                                            line=overlay.get("line_b_style", {}),
                                            fill='tonexty', fillcolor=overlay.get("fillcolor"),
                                            legendgroup=overlay.get("legendgroup"),
                                            showlegend=overlay.get("showlegend_master", True)),
                                  row=current_row_idx, col=1)
        
        elif plot_type == "bar" and plot_item["col"] in data.columns:
            fig.add_trace(
                go.Bar(x=data.index, y=data[plot_item["col"]], **plot_item.get("trace_params", {})),
                row=current_row_idx, col=1
            )

        elif plot_type == "line" and plot_item["col"] in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[plot_item["col"]], **plot_item.get("trace_params", {})),
                row=current_row_idx, col=1
            )
            if "hlines" in plot_item:
                for hline in plot_item["hlines"]:
                    fig.add_hline(row=current_row_idx, col=1, **hline)
        
        elif plot_type == "custom_macd":
            macd_cols = plot_item["cols"]
            params = plot_item.get("trace_params", {})
            fig.add_trace(go.Scatter(x=data.index, y=data[macd_cols["macd"]], **params.get("macd_line", {})), row=current_row_idx, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data[macd_cols["signal"]], **params.get("signal_line", {})), row=current_row_idx, col=1)
            hist_colors = np.where(data[macd_cols["hist"]] >= 0, 'rgba(0,128,0,0.7)', 'rgba(255,0,0,0.7)')
            fig.add_trace(go.Bar(x=data.index, y=data[macd_cols["hist"]], marker_color=hist_colors, **params.get("hist_bar", {})), row=current_row_idx, col=1)
            # Hlines for MACD? Typically not, but could be added to config.

        elif plot_type == "custom_stochastic":
            stoch_cols = plot_item["cols"]
            params = plot_item.get("trace_params", {})
            fig.add_trace(go.Scatter(x=data.index, y=data[stoch_cols["k"]], **params.get("k_line", {})), row=current_row_idx, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data[stoch_cols["d"]], **params.get("d_line", {})), row=current_row_idx, col=1)
            if "hlines" in plot_item:
                for hline in plot_item["hlines"]:
                    fig.add_hline(row=current_row_idx, col=1, **hline)
        
        current_row_idx +=1 # Increment for the next subplot (if any)

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=max(600, 200 * num_rows)  # Adjust height based on number of subplots, min 600
    )
    
    return fig 