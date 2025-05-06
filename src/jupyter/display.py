"""
Visualization Functions for Jupyter Notebooks

This module provides visualization functions for cryptocurrency analysis,
including price charts, technical indicators, and market summaries.
"""

from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_price_chart(df: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
    """
    Create an interactive price chart with OHLC and volume data.
    
    Args:
        df: DataFrame with price data (must contain open, high, low, close, volume columns)
        title: Chart title
        
    Returns:
        Plotly figure object with price chart
    """
    # Create figure with secondary y-axis (for volume)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                      vertical_spacing=0.03, row_heights=[0.7, 0.3],
                      subplot_titles=(title, "Volume"))
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1000,
        autosize=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        plot_bgcolor="rgba(30, 30, 30, 1)",
        paper_bgcolor="rgba(30, 30, 30, 1)",
        margin=dict(l=50, r=50, t=80, b=30)
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def create_indicator_chart(
    df: pd.DataFrame, 
    indicator_data: Dict[str, Any], 
    title: str = "Indicator Chart",
    include_price: bool = True
) -> go.Figure:
    """
    Create a chart showing a technical indicator and optionally price.
    
    Args:
        df: DataFrame with price data
        indicator_data: Dictionary containing indicator values and metadata
        title: Chart title
        include_price: Whether to include price chart above indicator
        
    Returns:
        Plotly figure object with indicator chart
    """
    # Extract basic info
    indicator_type = indicator_data.get("type", "unknown").lower()
    
    # Set up subplots
    n_rows = 2 if include_price else 1
    row_heights = [0.7, 0.3] if include_price else [1.0]
    subplot_titles = (title, f"{indicator_type.upper()} Indicator") if include_price else (f"{indicator_type.upper()} Indicator",)
    
    # Create figure with appropriate subplots
    fig = make_subplots(rows=n_rows, cols=1, 
                      shared_xaxes=True,
                      vertical_spacing=0.03,
                      row_heights=row_heights,
                      subplot_titles=subplot_titles)
    
    # Add price chart if requested
    if include_price:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        indicator_row = 2
    else:
        indicator_row = 1
    
    # Add the indicator
    # We need to handle different types of indicators differently
    
    # Check if indicator has single-value data
    if "value" in indicator_data:
        # Get the value and signal
        value = indicator_data.get("value", 0)
        signal = indicator_data.get("signal", "neutral")
        
        # Different styling based on signal
        color = "green" if signal == "bullish" else "red" if signal == "bearish" else "yellow"
        
        # Add a point for current value
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[value],
                mode="markers+text",
                marker=dict(size=12, color=color),
                text=[f"{value:.2f}"],
                textposition="middle right",
                name=indicator_type.upper()
            ),
            row=indicator_row, col=1
        )
        
        # If it's a full series
        if "series" in indicator_data:
            series = indicator_data["series"]
            timestamps = [pd.to_datetime(ts) for ts in series.keys()]
            indicator_values = list(series.values())
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=indicator_values,
                    mode="lines",
                    line=dict(color=color, width=1),
                    name=f"{indicator_type.upper()} Line"
                ),
                row=indicator_row, col=1
            )
            
            # Add reference lines for certain indicators
            if indicator_type.lower() == 'rsi':
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=indicator_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=indicator_row, col=1)
                fig.update_yaxes(range=[0, 100], row=indicator_row, col=1)
        
        # Handle multi-line indicators (like MACD, Bollinger Bands)
        if "components" in indicator_data:
            components = indicator_data["components"]
            
            for key, series in components.items():
                # For MACD components
                if key.startswith("MACD"):
                    color = 'blue' if 'MACD_' in key else 'orange' if 'MACDs_' in key else 'green'
                    mode = 'lines' if 'MACD_' in key or 'MACDs_' in key else 'markers'
                    
                    timestamps = [pd.to_datetime(ts) for ts in series.keys()]
                    indicator_values = list(series.values())
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=indicator_values,
                            mode=mode,
                            line=dict(color=color, width=1),
                            marker=dict(color=color, size=3),
                            name=key
                        ),
                        row=indicator_row, col=1
                    )
                
                # For Bollinger Bands
                elif key.startswith("BBU") or key.startswith("BBM") or key.startswith("BBL"):
                    color = 'red' if 'BBU_' in key else 'blue' if 'BBM_' in key else 'green'
                    
                    timestamps = [pd.to_datetime(ts) for ts in series.keys()]
                    indicator_values = list(series.values())
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=indicator_values,
                            mode='lines',
                            line=dict(color=color, width=1),
                            name=key
                        ),
                        row=indicator_row, col=1
                    )
                
                # For Stochastic
                elif key.startswith("STOCH"):
                    color = 'blue' if 'STOCHK_' in key else 'orange'
                    
                    timestamps = [pd.to_datetime(ts) for ts in series.keys()]
                    indicator_values = list(series.values())
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=indicator_values,
                            mode='lines',
                            line=dict(color=color, width=1),
                            name=key
                        ),
                        row=indicator_row, col=1
                    )
                    
                    # Add reference lines for stochastic
                    fig.add_hline(y=80, line_dash="dash", line_color="red", row=indicator_row, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color="green", row=indicator_row, col=1)
                    fig.update_yaxes(range=[0, 100], row=indicator_row, col=1)
    
    # Update layout
    fig.update_layout(
        height=700 if include_price else 400,
        width=1000,
        autosize=True,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=30),
        template="plotly_dark",
        plot_bgcolor="rgba(30, 30, 30, 1)",
        paper_bgcolor="rgba(30, 30, 30, 1)"
    )
    
    return fig


def create_summary_dashboard(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple panels for market analysis.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        Plotly figure object with multiple panels
    """
    # Extract data from analysis results
    metadata = analysis_results.get("metadata", {})
    symbol = metadata.get("symbol", "BTC")
    vs_currency = metadata.get("vs_currency", "usd").upper()
    timeframe = metadata.get("timeframe", "1d")
    
    price_data = analysis_results.get("price_data", {})
    current_price = price_data.get("current_price", 0)
    price_change_24h = price_data.get("price_change_24h", 0)
    price_change_pct_24h = price_data.get("price_change_percentage_24h", 0)
    
    # Get summary data
    summary = analysis_results.get("summary", {})
    trend = summary.get("trend", {})
    signals = summary.get("signals", {})
    
    # Create dashboard with multiple sections
    fig = make_subplots(
        rows=3, 
        cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "domain"}, {"type": "domain"}],
            [{"colspan": 2, "type": "table"}, None]
        ],
        subplot_titles=(
            f"{symbol}/{vs_currency} Price", 
            "24h Change",
            "Trend Analysis", 
            "Market Signals",
            "Analysis Summary"
        )
    )
    
    # Add current price indicator
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=current_price,
            number={"prefix": "$" if vs_currency == "USD" else "€" if vs_currency == "EUR" else "",
                   "font": {"size": 40}},
            title={"text": f"{symbol}/{vs_currency}",
                  "font": {"size": 20}},
            domain={"row": 0, "column": 0}
        ),
        row=1, col=1
    )
    
    # Add price change indicator
    color = "green" if price_change_pct_24h >= 0 else "red"
    symbol_prefix = "+" if price_change_pct_24h >= 0 else ""
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=current_price,
            number={"prefix": "$" if vs_currency == "USD" else "€" if vs_currency == "EUR" else "",
                   "font": {"size": 30}},
            delta={"reference": current_price - price_change_24h,
                  "relative": True,
                  "valueformat": ".2%",
                  "font": {"color": color}},
            title={"text": f"24h Change<br>{symbol_prefix}{price_change_pct_24h:.2f}%",
                  "font": {"size": 20}},
            domain={"row": 0, "column": 1}
        ),
        row=1, col=2
    )
    
    # Add trend analysis gauge
    trend_direction = trend.get("direction", "neutral")
    trend_strength = trend.get("strength", "neutral")
    
    # Map trend to numeric value for the gauge
    trend_value_map = {
        "bearish": 0 if trend_strength == "strong" else 0.25 if trend_strength == "moderate" else 0.35,
        "neutral": 0.5,
        "bullish": 1 if trend_strength == "strong" else 0.75 if trend_strength == "moderate" else 0.65
    }
    trend_value = trend_value_map.get(trend_direction, 0.5)
    
    fig.add_trace(
        go.Indicator(
            mode="gauge",
            value=trend_value,
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "rgba(0,0,0,0)"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 0.33], "color": "red"},
                    {"range": [0.33, 0.67], "color": "yellow"},
                    {"range": [0.67, 1], "color": "green"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": trend_value
                }
            },
            title={"text": f"{trend_direction.title()} {trend_strength.title()} Trend",
                  "font": {"size": 20}},
            domain={"row": 1, "column": 0}
        ),
        row=2, col=1
    )
    
    # Add market signals gauge
    action = signals.get("action", "hold")
    short_term = signals.get("short_term", "neutral")
    medium_term = signals.get("medium_term", "neutral")
    long_term = signals.get("long_term", "neutral")
    
    # Map action to numeric value for the gauge
    action_value_map = {
        "sell": 0.15,
        "hold": 0.5,
        "buy": 0.85
    }
    action_value = action_value_map.get(action, 0.5)
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=action_value,
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "darkblue", "visible": False},
                "bar": {"color": action_value < 0.33 and "red" or action_value > 0.67 and "green" or "yellow"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray"
            },
            title={"text": f"Action: {action.upper()}",
                  "font": {"size": 20}},
            domain={"row": 1, "column": 1}
        ),
        row=2, col=2
    )
    
    # Add analysis summary text
    analysis_text = trend.get("analysis", "No analysis available.")
    
    # Format market trend analysis with symbols for direction
    short_term_symbol = "▲" if short_term == "bullish" else "▼" if short_term == "bearish" else "◆"
    medium_term_symbol = "▲" if medium_term == "bullish" else "▼" if medium_term == "bearish" else "◆"
    long_term_symbol = "▲" if long_term == "bullish" else "▼" if long_term == "bearish" else "◆"
    
    # Define colors - we'll use these in our text formatting
    bullish_color = "chartreuse"
    bearish_color = "red"
    neutral_color = "orange"
    
    # Format the market status with colors
    short_term_color = bullish_color if short_term == "bullish" else bearish_color if short_term == "bearish" else neutral_color
    medium_term_color = bullish_color if medium_term == "bullish" else bearish_color if medium_term == "bearish" else neutral_color
    long_term_color = bullish_color if long_term == "bullish" else bearish_color if long_term == "bearish" else neutral_color
    
    # Create HTML-formatted table content - using only simple HTML tags that Plotly supports
    html_content = f"""<b>MARKET TREND ANALYSIS</b><br /><br />

<b>Short-term:</b> <span style='color:{short_term_color}'>{short_term_symbol} {short_term.upper()}</span> Short-term analysis considers immediate price movements, typically 1-7 days. Best viewed on 15min, 1hr, or 4hr charts.<br />

<b>Medium-term:</b> <span style='color:{medium_term_color}'>{medium_term_symbol} {medium_term.upper()}</span> Medium-term analysis looks at 1-4 week trends. Best viewed on daily or 4hr charts.<br />

<b>Long-term:</b> <span style='color:{long_term_color}'>{long_term_symbol} {long_term.upper()}</span> Long-term analysis examines multi-month trends and overall market direction. Best viewed on daily, weekly, or monthly charts.<br />

<b>Analysis:</b><br />
{analysis_text}
"""
    
    # Create a table with proper HTML rendering
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Market Analysis</b>"],
                font=dict(size=20, color="white"),
                height=40,
                align="center",
                fill_color="rgba(50, 50, 50, 1)"
            ),
            cells=dict(
                values=[[html_content]],
                align="left",
                font=dict(size=14, family="Arial", color="white"),
                height=400,
                line_color="darkslategray", 
                fill_color="rgba(50, 50, 50, 1)"
            ),
            columnwidth=[100]  # Use full width
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        autosize=True,
        showlegend=False,
        margin=dict(l=5, r=5, t=100, b=50, pad=0),
        template="plotly_dark",
        title=f"{symbol}/{vs_currency} Market Analysis ({timeframe})",
        plot_bgcolor="rgba(30, 30, 30, 1)",
        paper_bgcolor="rgba(30, 30, 30, 1)"
    )
    
    return fig


def create_multi_indicator_chart(
    df: pd.DataFrame, 
    analysis_results: Dict[str, Any],
    title: str = "Technical Analysis"
) -> go.Figure:
    """
    Create a comprehensive chart with price and multiple indicators.
    
    Args:
        df: DataFrame with price data
        analysis_results: Dictionary containing analysis results
        title: Chart title
        
    Returns:
        Plotly figure object with multiple indicators
    """
    # Extract metadata
    metadata = analysis_results.get("metadata", {})
    symbol = metadata.get("symbol", "BTC")
    vs_currency = metadata.get("vs_currency", "usd").upper()
    timeframe = metadata.get("timeframe", "1d")
    
    # Create figure with multiple subplots
    fig = make_subplots(
        rows=4, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(
            f"{symbol}/{vs_currency} Price ({timeframe})",
            "Volume",
            "RSI",
            "MACD"
        )
    )
    
    # Add price chart with any overlay indicators
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add overlay indicators (SMA, EMA, Bollinger Bands)
    trend_indicators = analysis_results.get("trend_indicators", {})
    volatility_indicators = analysis_results.get("volatility_indicators", {})
    
    # Add SMA if available
    sma_data = trend_indicators.get("sma", {})
    if sma_data and "value" in sma_data:
        # For illustration - in a real implementation, we'd extract the full SMA series
        # This is just showing the last value as a horizontal line
        sma_value = sma_data.get("value", 0)
        sma_length = sma_data.get("params", {}).get("length", 20)
        fig.add_hline(
            y=sma_value, 
            line_dash="solid", 
            line_color="blue",
            annotation_text=f"SMA({sma_length})",
            annotation_position="bottom right",
            row=1, col=1
        )
    
    # Add EMA if available
    ema_data = trend_indicators.get("ema", {})
    if ema_data and "value" in ema_data:
        ema_value = ema_data.get("value", 0)
        ema_length = ema_data.get("params", {}).get("length", 20)
        fig.add_hline(
            y=ema_value, 
            line_dash="solid", 
            line_color="orange",
            annotation_text=f"EMA({ema_length})",
            annotation_position="bottom left",
            row=1, col=1
        )
    
    # Add Bollinger Bands if available
    bbands_data = volatility_indicators.get("bbands", {})
    if bbands_data:
        upper = bbands_data.get("upper", 0)
        middle = bbands_data.get("middle", 0)
        lower = bbands_data.get("lower", 0)
        
        fig.add_hline(
            y=upper, 
            line_dash="dot", 
            line_color="red",
            annotation_text="Upper BB",
            annotation_position="top left",
            row=1, col=1
        )
        
        fig.add_hline(
            y=middle, 
            line_dash="dot", 
            line_color="purple",
            annotation_text="Middle BB",
            annotation_position="middle left",
            row=1, col=1
        )
        
        fig.add_hline(
            y=lower, 
            line_dash="dot", 
            line_color="green",
            annotation_text="Lower BB",
            annotation_position="bottom left",
            row=1, col=1
        )
    
    # Add volume bar chart
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Add RSI
    momentum_indicators = analysis_results.get("momentum_indicators", {})
    rsi_data = momentum_indicators.get("rsi", {})
    if rsi_data and "value" in rsi_data:
        rsi_value = rsi_data.get("value", 50)
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[rsi_value],
                mode="markers+text",
                marker=dict(size=10, color="yellow"),
                text=[f"{rsi_value:.1f}"],
                textposition="middle right",
                name="RSI"
            ),
            row=3, col=1
        )
        
        # Add reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    # Add MACD
    macd_data = trend_indicators.get("macd", {})
    if macd_data:
        macd_line = macd_data.get("macd_line", 0)
        signal_line = macd_data.get("signal_line", 0)
        histogram = macd_data.get("histogram", 0)
        
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[macd_line],
                mode="markers+text",
                marker=dict(size=10, color="blue"),
                text=[f"MACD: {macd_line:.2f}"],
                textposition="middle right",
                name="MACD"
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[signal_line],
                mode="markers+text",
                marker=dict(size=10, color="orange"),
                text=[f"Signal: {signal_line:.2f}"],
                textposition="middle left",
                name="Signal"
            ),
            row=4, col=1
        )
        
        # Add histogram
        # This is just a visual representation at the last point
        color = "green" if histogram > 0 else "red"
        fig.add_trace(
            go.Bar(
                x=[df.index[-1]],
                y=[histogram],
                marker_color=color,
                name="Histogram"
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        autosize=True,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=30),
        template="plotly_dark",
        plot_bgcolor="rgba(30, 30, 30, 1)",
        paper_bgcolor="rgba(30, 30, 30, 1)",
        title=title
    )
    
    return fig


def plot_indicator_comparison(df: pd.DataFrame, indicators: List[Dict], title: str = "Indicator Comparison") -> plt.Figure:
    """
    Create a matplotlib figure comparing multiple indicators.
    
    Args:
        df: DataFrame with price data
        indicators: List of indicator data dictionaries
        title: Chart title
        
    Returns:
        Matplotlib figure object
    """
    # Set Seaborn style
    sns.set(style="darkgrid")
    
    # Create figure
    fig, axes = plt.subplots(len(indicators) + 1, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # Plot price in the first subplot
    axes[0].plot(df.index, df['close'], color='blue', linewidth=1.5)
    axes[0].set_title('Price')
    axes[0].set_ylabel('Price')
    
    # Plot each indicator
    for i, indicator_data in enumerate(indicators):
        indicator_type = indicator_data.get("indicator", "unknown")
        values = indicator_data.get("values", {})
        
        ax = axes[i + 1]
        ax.set_title(indicator_type.upper())
        
        # Handle different indicator types
        if isinstance(values, dict):
            # Single-value indicators
            if all(isinstance(k, str) and k.isdigit() for k in values.keys()):
                timestamps = [pd.to_datetime(ts) for ts in values.keys()]
                indicator_values = list(values.values())
                ax.plot(timestamps, indicator_values, color='orange', linewidth=1.5)
            
            # Multi-line indicators
            else:
                for key, series in values.items():
                    timestamps = [pd.to_datetime(ts) for ts in series.keys()]
                    indicator_values = list(series.values())
                    
                    # Choose color based on indicator component
                    if 'MACD_' in key:
                        color = 'blue'
                    elif 'MACDs_' in key:
                        color = 'red'
                    elif 'MACDh_' in key:
                        color = 'green'
                    elif 'BBU_' in key:
                        color = 'red'
                    elif 'BBM_' in key:
                        color = 'blue'
                    elif 'BBL_' in key:
                        color = 'green'
                    elif 'STOCHk_' in key:
                        color = 'blue'
                    elif 'STOCHd_' in key:
                        color = 'orange'
                    else:
                        color = 'purple'
                    
                    ax.plot(timestamps, indicator_values, color=color, linewidth=1.5, label=key)
                
                ax.legend()
        
        # Add reference lines for certain indicators
        if indicator_type.lower() == 'rsi':
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax.set_ylim([0, 100])
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig


def create_correlation_heatmap(df_dict: Dict[str, pd.DataFrame], title: str = "Cryptocurrency Correlation Matrix") -> plt.Figure:
    """
    Create a correlation heatmap between multiple cryptocurrency price series.
    
    Args:
        df_dict: Dictionary mapping cryptocurrency symbols to their price DataFrames
        title: Chart title
        
    Returns:
        Matplotlib figure object with correlation heatmap
    """
    # Extract close prices from each DataFrame
    close_prices = {}
    for symbol, df in df_dict.items():
        if 'close' in df.columns:
            close_prices[symbol] = df['close']
    
    # Create a DataFrame with all close prices
    correlation_df = pd.DataFrame(close_prices)
    
    # Calculate correlation matrix
    correlation_matrix = correlation_df.corr()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1,
        center=0,
        fmt=".2f"
    )
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()


def plot_returns_distribution(df: pd.DataFrame, title: str = "Returns Distribution") -> plt.Figure:
    """
    Create a plot showing the distribution of daily returns.
    
    Args:
        df: DataFrame with price data
        title: Chart title
        
    Returns:
        Matplotlib figure object
    """
    # Calculate daily returns
    if 'close' in df.columns:
        returns = df['close'].pct_change().dropna() * 100  # as percentage
    else:
        return plt.figure()  # return empty figure if no close prices
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create distribution plot
    sns.histplot(returns, kde=True, color='skyblue', ax=ax)
    
    # Add a vertical line at 0
    plt.axvline(0, color='red', linestyle='--')
    
    # Add mean and median lines
    mean_return = returns.mean()
    median_return = returns.median()
    plt.axvline(mean_return, color='green', linestyle='-', label=f'Mean: {mean_return:.2f}%')
    plt.axvline(median_return, color='orange', linestyle='-', label=f'Median: {median_return:.2f}%')
    
    # Add labels and title
    plt.xlabel('Daily Returns (%)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    
    # Add basic statistics
    stats_text = (
        f"Min: {returns.min():.2f}%\n"
        f"Max: {returns.max():.2f}%\n"
        f"Mean: {mean_return:.2f}%\n"
        f"Median: {median_return:.2f}%\n"
        f"Std Dev: {returns.std():.2f}%\n"
        f"Skewness: {returns.skew():.2f}\n"
        f"Kurtosis: {returns.kurtosis():.2f}"
    )
    
    # Position the text box in figure coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    return fig 