"""
Interactive Widgets for Jupyter Notebooks

This module provides interactive widgets for parameter selection and controls
in Jupyter notebooks, allowing users to adjust settings and see results in real-time.
"""

from typing import Dict, Optional, Callable, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import traceback

# Import the core analysis functions
from src.jupyter.analysis import run_analysis, get_data, clear_cache
from src.jupyter.display import (
    create_price_chart, create_indicator_chart,
    create_summary_dashboard, create_multi_indicator_chart
)


def create_symbol_selector(default: str = "BTC", 
                         on_change: Optional[Callable] = None) -> widgets.Dropdown:
    """
    Create a dropdown widget for cryptocurrency symbol selection.
    
    Args:
        default: Default symbol to select
        on_change: Callback function for value change
        
    Returns:
        Dropdown widget for symbol selection
    """
    # Common cryptocurrency symbols
    symbols = [
        "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "XLM",
        "DOGE", "SOL", "UNI", "AAVE", "MATIC", "AVAX", "SHIB", "ATOM", "ALGO", "FIL"
    ]
    
    # Create widget
    symbol_widget = widgets.Dropdown(
        options=symbols,
        value=default,
        description='Symbol:',
        disabled=False,
        layout=widgets.Layout(width='200px')
    )
    
    # Add callback if provided
    if on_change:
        symbol_widget.observe(on_change, names='value')
    
    return symbol_widget


def create_timeframe_selector(default: str = "1d", 
                            on_change: Optional[Callable] = None) -> widgets.Dropdown:
    """
    Create a dropdown widget for timeframe selection.
    
    Args:
        default: Default timeframe to select
        on_change: Callback function for value change
        
    Returns:
        Dropdown widget for timeframe selection
    """
    # Common timeframes
    timeframes = ["1d", "4h", "1h", "15m", "5m"]
    
    # Create widget
    timeframe_widget = widgets.Dropdown(
        options=timeframes,
        value=default,
        description='Timeframe:',
        disabled=False,
        layout=widgets.Layout(width='200px')
    )
    
    # Add callback if provided
    if on_change:
        timeframe_widget.observe(on_change, names='value')
    
    return timeframe_widget


def create_days_slider(default: int = 100, 
                     min_val: int = 7, 
                     max_val: int = 365, 
                     on_change: Optional[Callable] = None) -> widgets.IntSlider:
    """
    Create a slider widget for selecting number of days of data.
    
    Args:
        default: Default number of days
        min_val: Minimum number of days
        max_val: Maximum number of days
        on_change: Callback function for value change
        
    Returns:
        IntSlider widget for days selection
    """
    # Create widget
    days_widget = widgets.IntSlider(
        value=default,
        min=min_val,
        max=max_val,
        step=1,
        description='Days:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='300px')
    )
    
    # Add callback if provided
    if on_change:
        days_widget.observe(on_change, names='value')
    
    return days_widget


def create_vs_currency_selector(default: str = "usd",
                              on_change: Optional[Callable] = None) -> widgets.Dropdown:
    """
    Create a dropdown widget for vs_currency selection.
    
    Args:
        default: Default currency (usd, eur, etc.)
        on_change: Callback function for value change
        
    Returns:
        Dropdown widget for currency selection
    """
    # Common currencies
    currencies = ["usd", "eur", "gbp", "jpy", "aud", "cad", "chf"]
    
    # Create widget
    currency_widget = widgets.Dropdown(
        options=currencies,
        value=default,
        description='Currency:',
        disabled=False,
        layout=widgets.Layout(width='200px')
    )
    
    # Add callback if provided
    if on_change:
        currency_widget.observe(on_change, names='value')
    
    return currency_widget


def create_parameter_controls() -> Dict[str, widgets.Widget]:
    """
    Create a set of parameter control widgets.
    
    Returns:
        Dictionary of widgets for parameter selection
    """
    controls = {
        "symbol": create_symbol_selector(),
        "timeframe": create_timeframe_selector(),
        "days": create_days_slider(),
        "vs_currency": create_vs_currency_selector(),
        "refresh": widgets.Checkbox(value=False, description='Force refresh', layout=widgets.Layout(width='150px')),
        "forecast": widgets.Checkbox(value=False, description='Include forecast', layout=widgets.Layout(width='150px'))
    }
    
    return controls


def create_indicator_params_widgets() -> Dict[str, Dict[str, widgets.Widget]]:
    """
    Create widgets for adjusting indicator parameters.
    
    Returns:
        Dictionary of widgets grouped by indicator type
    """
    # Trend indicator parameters
    trend_widgets = {
        "sma": {
            "length": widgets.IntSlider(value=20, min=5, max=200, step=1, description='Length:',
                                     continuous_update=False)
        },
        "ema": {
            "length": widgets.IntSlider(value=20, min=5, max=200, step=1, description='Length:',
                                     continuous_update=False)
        },
        "macd": {
            "fast": widgets.IntSlider(value=12, min=3, max=50, step=1, description='Fast:',
                                   continuous_update=False),
            "slow": widgets.IntSlider(value=26, min=10, max=100, step=1, description='Slow:',
                                   continuous_update=False),
            "signal": widgets.IntSlider(value=9, min=3, max=30, step=1, description='Signal:',
                                     continuous_update=False)
        },
        "adx": {
            "length": widgets.IntSlider(value=14, min=5, max=50, step=1, description='Length:',
                                     continuous_update=False)
        }
    }
    
    # Momentum indicator parameters
    momentum_widgets = {
        "rsi": {
            "length": widgets.IntSlider(value=14, min=5, max=50, step=1, description='Length:',
                                     continuous_update=False)
        },
        "stoch": {
            "k": widgets.IntSlider(value=14, min=5, max=50, step=1, description='%K Length:',
                                continuous_update=False),
            "d": widgets.IntSlider(value=3, min=1, max=20, step=1, description='%D Length:',
                                continuous_update=False),
            "smooth_k": widgets.IntSlider(value=3, min=1, max=20, step=1, description='Smooth K:',
                                       continuous_update=False)
        },
        "cci": {
            "length": widgets.IntSlider(value=20, min=5, max=100, step=1, description='Length:',
                                     continuous_update=False),
            "constant": widgets.FloatSlider(value=0.015, min=0.005, max=0.05, step=0.005, 
                                         description='Constant:', continuous_update=False)
        }
    }
    
    # Volatility indicator parameters
    volatility_widgets = {
        "bbands": {
            "length": widgets.IntSlider(value=20, min=5, max=100, step=1, description='Length:',
                                     continuous_update=False),
            "std": widgets.FloatSlider(value=2.0, min=0.5, max=4.0, step=0.1, description='StdDev:',
                                    continuous_update=False)
        },
        "atr": {
            "length": widgets.IntSlider(value=14, min=5, max=50, step=1, description='Length:',
                                     continuous_update=False)
        }
    }
    
    # Volume indicator parameters
    volume_widgets = {
        "obv": {
            # OBV has no specific parameters
        }
    }
    
    return {
        "trend": trend_widgets,
        "momentum": momentum_widgets,
        "volatility": volatility_widgets,
        "volume": volume_widgets
    }


def create_indicator_selector(on_change: Optional[Callable] = None) -> widgets.Dropdown:
    """
    Create a dropdown widget for indicator selection.
    
    Args:
        on_change: Callback function for value change
        
    Returns:
        Dropdown widget for indicator selection
    """
    # All available indicators grouped by type
    indicators = [
        # Trend indicators
        ("SMA - Simple Moving Average", "sma"),
        ("EMA - Exponential Moving Average", "ema"),
        ("MACD - Moving Average Convergence Divergence", "macd"),
        ("ADX - Average Directional Index", "adx"),
        
        # Momentum indicators
        ("RSI - Relative Strength Index", "rsi"),
        ("Stochastic Oscillator", "stoch"),
        ("CCI - Commodity Channel Index", "cci"),
        
        # Volatility indicators
        ("Bollinger Bands", "bbands"),
        ("ATR - Average True Range", "atr"),
        
        # Volume indicators
        ("OBV - On Balance Volume", "obv")
    ]
    
    # Create widget with display text and value pairs
    indicator_widget = widgets.Dropdown(
        options=indicators,
        value="rsi",  # Default to RSI
        description='Indicator:',
        disabled=False,
        layout=widgets.Layout(width='350px')
    )
    
    # Add callback if provided
    if on_change:
        indicator_widget.observe(on_change, names='value')
    
    return indicator_widget


def create_chart_type_selector(on_change: Optional[Callable] = None) -> widgets.RadioButtons:
    """
    Create a widget for selecting chart type.
    
    Args:
        on_change: Callback function for value change
        
    Returns:
        RadioButtons widget for chart type selection
    """
    # Chart type options
    chart_types = [
        ("Price Chart", "price"),
        ("Indicator Chart", "indicator"),
        ("Multi-Indicator Chart", "multi"),
        ("Dashboard", "dashboard")
    ]
    
    # Create widget
    chart_type_widget = widgets.RadioButtons(
        options=chart_types,
        value="price",  # Default to price chart
        description='Chart Type:',
        disabled=False,
        layout=widgets.Layout(width='200px')
    )
    
    # Add callback if provided
    if on_change:
        chart_type_widget.observe(on_change, names='value')
    
    return chart_type_widget


def create_display_options() -> Dict[str, widgets.Widget]:
    """
    Create widgets for display options.
    
    Returns:
        Dictionary of display option widgets
    """
    display_options = {
        "chart_height": widgets.IntSlider(value=600, min=300, max=1000, step=50, 
                                       description='Chart Height:', continuous_update=False),
        "theme": widgets.Dropdown(options=["plotly_dark", "plotly", "plotly_white", "ggplot2"], 
                               value="plotly_dark", description='Theme:'),
        "include_price": widgets.Checkbox(value=True, description='Include Price Chart'),
        "show_volume": widgets.Checkbox(value=True, description='Show Volume')
    }
    
    return display_options


def create_analysis_dashboard(output_widget: Optional[widgets.Output] = None) -> Tuple[Dict[str, widgets.Widget], widgets.VBox]:
    """
    Create a complete analysis dashboard with controls and output area.
    
    Args:
        output_widget: Optional output widget to display results
        
    Returns:
        Tuple containing the controls dictionary and the dashboard widget
    """
    # Create an output widget if not provided
    if output_widget is None:
        output_widget = widgets.Output()
    
    # Create parameter controls
    controls = create_parameter_controls()
    
    # Create chart type selector
    chart_type = create_chart_type_selector()
    
    # Create indicator selector
    indicator_selector = create_indicator_selector()
    
    # Create indicator parameter widgets (but don't display them yet)
    indicator_params = create_indicator_params_widgets()
    
    # Create display options
    display_options = create_display_options()
    
    # Create a tab for each group of settings
    basic_tab = widgets.VBox([
        widgets.HBox([controls["symbol"], controls["timeframe"], controls["vs_currency"]]),
        widgets.HBox([controls["days"], controls["refresh"], controls["forecast"]])
    ])
    
    indicator_widgets = {}
    
    # Function to update indicator parameters based on selection
    def update_indicator_params(change):
        indicator = change.new
        
        # Clear previous widgets
        indicator_widgets.clear()
        indicators_box.children = ()
        
        # Find the category and parameters for the selected indicator
        for category, indicators in indicator_params.items():
            if indicator in indicators:
                # Add the widgets for this indicator
                indicator_widgets.update(indicators[indicator])
                indicators_box.children = tuple(indicator_widgets.values())
                break
    
    # Connect the update function to the indicator selector
    indicator_selector.observe(update_indicator_params, names='value')
    
    # Container for indicator parameters
    indicators_box = widgets.VBox([])
    
    # Initialize with the default selected indicator
    update_indicator_params(type('obj', (object,), {'new': indicator_selector.value}))
    
    indicator_tab = widgets.VBox([
        indicator_selector,
        indicators_box
    ])
    
    display_tab = widgets.VBox([
        widgets.HBox([display_options["theme"], display_options["chart_height"]]),
        widgets.HBox([display_options["include_price"], display_options["show_volume"]])
    ])
    
    # Create tabs for the settings
    tabs = widgets.Tab()
    tabs.children = [basic_tab, indicator_tab, display_tab]
    tabs.set_title(0, "Basic Settings")
    tabs.set_title(1, "Indicator Parameters")
    tabs.set_title(2, "Display Options")
    
    # Create the run button
    run_button = widgets.Button(
        description='Run Analysis',
        button_style='success',
        tooltip='Run analysis with the selected parameters',
        icon='play'
    )
    
    # Create the clear cache button
    clear_cache_button = widgets.Button(
        description='Clear Cache',
        button_style='warning',
        tooltip='Clear the data and analysis cache',
        icon='trash'
    )
    
    def run_analysis_callback(b):
        with output_widget:
            clear_output(wait=True)
            try:
                # Get parameter values
                symbol = controls["symbol"].value
                timeframe = controls["timeframe"].value
                days = controls["days"].value
                vs_currency = controls["vs_currency"].value
                force_refresh = controls["refresh"].value
                forecast = controls["forecast"].value
                
                # Get selected chart type
                chart_type_value = chart_type.value
                
                # Get display options
                theme = display_options["theme"].value
                chart_height = display_options["chart_height"].value
                include_price = display_options["include_price"].value
                
                # Fetch data and run analysis
                df, current_price_data = get_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    force_refresh=force_refresh,
                    vs_currency=vs_currency
                )
                
                if df.empty:
                    print("No data available. Try changing parameters or refresh.")
                    return
                
                # Run the analysis
                analysis_results = run_analysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    force_refresh=force_refresh,
                    forecast=forecast,
                    vs_currency=vs_currency
                )
                
                # Create the appropriate chart based on selection
                if chart_type_value == "price":
                    fig = create_price_chart(df, title=f"{symbol}/{vs_currency.upper()} Price Chart")
                    fig.update_layout(height=chart_height, template=theme)
                    display(fig)
                
                elif chart_type_value == "indicator":
                    # Get the selected indicator
                    indicator = indicator_selector.value
                    
                    # Prepare indicator parameters
                    ind_params = {}
                    for name, widget in indicator_widgets.items():
                        ind_params[name] = widget.value
                    
                    # Find the indicator category
                    for category, indicators in indicator_params.items():
                        if indicator in indicators:
                            if category == "trend":
                                indicator_data = analysis_results.get("trend_indicators", {}).get(indicator, {})
                            elif category == "momentum":
                                indicator_data = analysis_results.get("momentum_indicators", {}).get(indicator, {})
                            elif category == "volatility":
                                indicator_data = analysis_results.get("volatility_indicators", {}).get(indicator, {})
                            elif category == "volume":
                                indicator_data = analysis_results.get("volume_indicators", {}).get(indicator, {})
                            
                            # Create the indicator chart
                            fig = create_indicator_chart(
                                df=df, 
                                indicator_data=indicator_data, 
                                title=f"{symbol}/{vs_currency.upper()} {indicator.upper()}",
                                include_price=include_price
                            )
                            fig.update_layout(height=chart_height, template=theme)
                            display(fig)
                            break
                
                elif chart_type_value == "multi":
                    # Create multi-indicator chart
                    fig = create_multi_indicator_chart(
                        df=df,
                        analysis_results=analysis_results,
                        title=f"{symbol}/{vs_currency.upper()} Technical Analysis"
                    )
                    fig.update_layout(height=chart_height, template=theme)
                    display(fig)
                
                elif chart_type_value == "dashboard":
                    # Create dashboard
                    fig = create_summary_dashboard(analysis_results)
                    fig.update_layout(height=chart_height, template=theme)
                    display(fig)
                
                # Display forecast information if requested
                if forecast and "forecast" in analysis_results:
                    forecast_data = analysis_results["forecast"]
                    
                    # Create a nice HTML table for forecast data
                    if "data" in forecast_data and not "error" in forecast_data:
                        html = """
                        <div style="background-color:#f0f8ff;padding:10px;border-radius:5px;margin:10px 0;">
                            <h3>Price Forecast</h3>
                            <table style="width:100%;border-collapse:collapse;">
                                <tr style="background-color:#4682b4;color:white;">
                                    <th style="padding:8px;text-align:left;">Period</th>
                                    <th style="padding:8px;text-align:left;">Predicted Price</th>
                                    <th style="padding:8px;text-align:left;">Change</th>
                                </tr>
                        """
                        
                        for i, item in enumerate(forecast_data["data"]):
                            bg_color = "#f5f5f5" if i % 2 == 0 else "white"
                            color = "green" if item["change"] >= 0 else "red"
                            html += f"""
                                <tr style="background-color:{bg_color};">
                                    <td style="padding:8px;">{item["period_label"]}</td>
                                    <td style="padding:8px;">${item["value"]:.2f}</td>
                                    <td style="padding:8px;color:{color};">{item["change_pct"]}</td>
                                </tr>
                            """
                        
                        html += """
                            </table>
                            <p style="color:#666;font-style:italic;margin-top:10px;">
                                Forecasts are based on historical patterns and should not be used as financial advice.
                            </p>
                        </div>
                        """
                        
                        display(HTML(html))
                    elif "error" in forecast_data:
                        print(f"Forecast error: {forecast_data['error']}")
            
            except Exception as e:
                print(f"Error running analysis: {str(e)}")
                print(traceback.format_exc())
    
    def clear_cache_callback(b):
        with output_widget:
            clear_output(wait=True)
            clear_cache()
            display(HTML("<div style='background-color:#e8f5e9;padding:10px;border-radius:5px;'>Cache cleared successfully</div>"))
    
    # Connect the callbacks to the buttons
    run_button.on_click(run_analysis_callback)
    clear_cache_button.on_click(clear_cache_callback)
    
    # Create the main UI layout
    ui = widgets.VBox([
        widgets.HBox([chart_type]),
        tabs,
        widgets.HBox([run_button, clear_cache_button]),
        output_widget
    ])
    
    # Collect all controls for reference
    all_controls = {
        **controls,
        "chart_type": chart_type,
        "indicator_selector": indicator_selector,
        "indicator_params": indicator_params,
        "display_options": display_options,
        "output": output_widget,
        "run_button": run_button,
        "clear_cache_button": clear_cache_button
    }
    
    return all_controls, ui


def set_notebook_width(width='100%'):
    """
    Set the width of the Jupyter notebook to the specified value.
    This ensures plots and widgets can display at full viewport width.
    
    Args:
        width: Width of the notebook (default: '100%')
    """
    from IPython.display import display, HTML
    # Create a comprehensive CSS that works for both classic Jupyter Notebook and JupyterLab
    css = f"""
    <style>
      /* Classic Jupyter Notebook */
      .container, .container-lg, .container-fluid, 
      .notebook_app .grid-container,
      #notebook-container {{
        width: {width} !important;
        max-width: {width} !important;
        padding-left: 2% !important;
        padding-right: 2% !important;
      }}
      
      /* JupyterLab */
      .jp-RenderedHTMLCommon, 
      .jp-OutputArea-output, 
      .jp-Cell-outputArea,
      .jp-Cell {{
        width: {width} !important;
        max-width: {width} !important;
      }}
      
      /* Output containers */
      .output_area, 
      .output_wrapper,
      .output_subarea {{
        width: {width} !important;
        max-width: {width} !important;
        flex: 1 1 auto;
      }}
      
      /* For plotly figures */
      .plotly.plot {{
        width: 100% !important;
      }}
    </style>
    """
    display(HTML(css))


def create_quick_analysis_widget(output_widget: Optional[widgets.Output] = None) -> widgets.VBox:
    """
    Create a simplified widget for quick analysis.
    
    Args:
        output_widget: Optional output widget to display results
        
    Returns:
        Widget containing the quick analysis controls
    """
    # Set notebook to full width for better display
    set_notebook_width('100%')
    
    # Create an output widget if not provided
    if output_widget is None:
        output_widget = widgets.Output()
    
    # Create simplified controls
    symbol = create_symbol_selector()
    timeframe = create_timeframe_selector()
    refresh = widgets.Checkbox(value=False, description='Force refresh')
    forecast = widgets.Checkbox(value=False, description='Include forecast')
    
    # Create the run button
    run_button = widgets.Button(
        description='Run Quick Analysis',
        button_style='success',
        tooltip='Run a quick analysis with the selected parameters',
        icon='play'
    )
    
    def run_quick_analysis(b):
        with output_widget:
            clear_output(wait=True)
            try:
                # Get data and run analysis
                df, current_price_data = get_data(symbol.value, timeframe.value, 100, refresh.value)
                analysis_results = run_analysis(symbol.value, timeframe.value, 100, refresh.value, forecast.value)
                
                # Display a multi-indicator chart
                fig = create_multi_indicator_chart(
                    df=df,
                    analysis_results=analysis_results,
                    title=f"{symbol.value}/USD Technical Analysis"
                )
                display(fig)
                
                # Display the dashboard
                fig = create_summary_dashboard(analysis_results)
                display(fig)
            
            except Exception as e:
                print(f"Error running quick analysis: {str(e)}")
    
    # Connect the callback to the button
    run_button.on_click(run_quick_analysis)
    
    # Create the UI
    ui = widgets.VBox([
        widgets.HBox([symbol, timeframe, refresh]),
        widgets.HBox([forecast]),
        run_button,
        output_widget
    ])
    
    return ui


def create_comparison_widget(output_widget: Optional[widgets.Output] = None) -> widgets.VBox:
    """
    Create a widget for comparing multiple cryptocurrencies.
    
    Args:
        output_widget: Optional output widget to display results
        
    Returns:
        Widget containing the comparison controls
    """
    # Create an output widget if not provided
    if output_widget is None:
        output_widget = widgets.Output()
    
    # Common cryptocurrency symbols
    all_symbols = [
        "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "XLM",
        "DOGE", "SOL", "UNI", "AAVE", "MATIC", "AVAX", "SHIB", "ATOM", "ALGO", "FIL"
    ]
    
    # Create multi-select widget for symbols
    symbols = widgets.SelectMultiple(
        options=all_symbols,
        value=["BTC", "ETH", "SOL", "ADA"],
        description='Symbols:',
        disabled=False,
        layout=widgets.Layout(width='300px', height='150px')
    )
    
    # Create timeframe selector
    timeframe = create_timeframe_selector()
    
    # Create vs_currency selector
    vs_currency = create_vs_currency_selector()
    
    # Create the run button
    run_button = widgets.Button(
        description='Compare',
        button_style='success',
        tooltip='Run comparison analysis',
        icon='play'
    )
    
    def run_comparison(b):
        with output_widget:
            clear_output(wait=True)
            try:
                selected_symbols = list(symbols.value)
                
                if not selected_symbols:
                    print("Please select at least one symbol to compare")
                    return
                
                # Display a loading message
                display(HTML("<div>Running comparison analysis... This may take a few moments.</div>"))
                
                # Run batch analysis for all selected symbols
                from src.jupyter.analysis import batch_analyze, get_comparison_data
                
                analysis_results = batch_analyze(
                    symbols=selected_symbols,
                    timeframe=timeframe.value,
                    days=100,
                    vs_currency=vs_currency.value
                )
                
                # Convert to DataFrame for easy comparison
                comparison_df = get_comparison_data(analysis_results)
                
                # Clear the loading message
                clear_output(wait=True)
                
                # Style the DataFrame for display
                styled_df = comparison_df.style.format({
                    'Price': '${:.2f}',
                    '24h Change': '{:.2f}%'
                }).background_gradient(
                    subset=['24h Change'], 
                    cmap='RdYlGn',
                    vmin=-5,
                    vmax=5
                ).applymap(
                    lambda x: 'background-color: #e8f5e9' if x == 'bullish' else 
                             ('background-color: #ffebee' if x == 'bearish' else ''),
                    subset=['Trend', 'Short Term', 'Medium Term', 'Long Term']
                ).applymap(
                    lambda x: 'background-color: #e8f5e9' if x == 'buy' else 
                             ('background-color: #ffebee' if x == 'sell' else 
                              'background-color: #fff9c4'),
                    subset=['Action']
                )
                
                # Display the styled DataFrame
                display(styled_df)
                
                # Create charts for each symbol
                for symbol in selected_symbols[:4]:  # Limit to first 4 to avoid overload
                    if symbol in analysis_results:
                        result = analysis_results[symbol]
                        
                        if "price_data" not in result:
                            continue
                            
                        # Create a mini dashboard for this symbol
                        price = result["price_data"].get("current_price", 0)
                        change = result["price_data"].get("price_change_percentage_24h", 0)
                        trend = result["summary"]["trend"]["direction"]
                        action = result["summary"]["signals"]["action"]
                        
                        color = "green" if change >= 0 else "red"
                        trend_color = "green" if trend == "bullish" else ("red" if trend == "bearish" else "orange")
                        action_color = "green" if action == "buy" else ("red" if action == "sell" else "orange")
                        
                        html = f"""
                        <div style="display:inline-block;margin:10px;padding:15px;border-radius:8px;border:1px solid #ddd;width:220px;">
                            <h3 style="margin-top:0;color:#333;">{symbol}/{vs_currency.value.upper()}</h3>
                            <div style="font-size:1.4em;margin:8px 0;">${price:.2f}</div>
                            <div style="color:{color};margin:8px 0;">{'+'if change>=0 else ''}{change:.2f}%</div>
                            <div style="margin:8px 0;">Trend: <span style="color:{trend_color};">{trend.upper()}</span></div>
                            <div style="margin:8px 0;">Action: <span style="color:{action_color};">{action.upper()}</span></div>
                        </div>
                        """
                        display(HTML(html))
            
            except Exception as e:
                print(f"Error running comparison: {str(e)}")
                print(traceback.format_exc())
    
    # Connect the callback to the button
    run_button.on_click(run_comparison)
    
    # Create the UI
    ui = widgets.VBox([
        widgets.HBox([symbols, widgets.VBox([timeframe, vs_currency, run_button])]),
        output_widget
    ])
    
    return ui


def create_dashboard_widget(default_symbol="bitcoin", default_vs_currency="usd", default_days=30):
    """
    Create an interactive dashboard widget with multiple analysis components.
    
    Args:
        default_symbol: Default cryptocurrency symbol
        default_vs_currency: Default reference currency
        default_days: Default number of days for analysis
        
    Returns:
        Interactive widget object
    """
    # Set notebook to full width for better display
    set_notebook_width('100%')
    
    # Create an output widget if not provided
    output_widget = widgets.Output()
    
    # Create simplified controls
    symbol = create_symbol_selector(default_symbol)
    timeframe = create_timeframe_selector()
    refresh = widgets.Checkbox(value=False, description='Force refresh')
    forecast = widgets.Checkbox(value=False, description='Include forecast')
    
    # Create the run button
    run_button = widgets.Button(
        description='Run Quick Analysis',
        button_style='success',
        tooltip='Run a quick analysis with the selected parameters',
        icon='play'
    )
    
    def run_quick_analysis(b):
        with output_widget:
            clear_output(wait=True)
            try:
                # Get data and run analysis
                df, current_price_data = get_data(symbol.value, timeframe.value, default_days, refresh.value)
                analysis_results = run_analysis(symbol.value, timeframe.value, default_days, refresh.value, forecast.value)
                
                # Display a multi-indicator chart
                fig = create_multi_indicator_chart(
                    df=df,
                    analysis_results=analysis_results,
                    title=f"{symbol.value}/{default_vs_currency.upper()} Technical Analysis"
                )
                display(fig)
                
                # Display the dashboard
                fig = create_summary_dashboard(analysis_results)
                display(fig)
            
            except Exception as e:
                print(f"Error running quick analysis: {str(e)}")
    
    # Connect the callback to the button
    run_button.on_click(run_quick_analysis)
    
    # Create the UI
    ui = widgets.VBox([
        widgets.HBox([symbol, timeframe, refresh]),
        widgets.HBox([forecast]),
        run_button,
        output_widget
    ])
    
    return ui 