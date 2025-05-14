import numbers

def calculate_atr_trade_parameters(
    entry_price: float,
    atr: float,
    position_type: str,
    risk_amount: float = 100.0,
    multiplier: float = 1.0,
) -> dict:
    """
    Calculates position size and stop-loss percentage based on ATR.

    Args:
        entry_price: The entry price of the asset.
        atr: The Average True Range value.
        position_type: Type of position, either 'long' or 'short'.
        risk_amount: The amount of capital to risk on the trade (default: 100.0).
        multiplier: The ATR multiplier for stop-loss (default: 1.0).

    Returns:
        A dictionary containing:
            'position_size': The calculated position size.
            'stop_loss_percentage': The stop-loss as a percentage of entry price.

    Raises:
        ValueError: If any input parameters are invalid.
    """
    # Validate input types
    if not all(isinstance(val, numbers.Number) for val in [entry_price, atr, risk_amount, multiplier]):
        raise ValueError("entry_price, atr, risk_amount, and multiplier must be numeric.")

    if not isinstance(position_type, str):
        raise ValueError("position_type must be a string.")

    # Validate input values
    if entry_price <= 0:
        raise ValueError("entry_price must be positive.")
    if atr <= 0:
        raise ValueError("atr must be positive.")
    if risk_amount <= 0:
        raise ValueError("risk_amount must be positive.")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")
    if position_type.lower() not in ['long', 'short']:
        raise ValueError("position_type must be either 'long' or 'short'.")

    stop_loss_distance = atr * multiplier
    
    if stop_loss_distance == 0:
        # This case should be caught by atr <= 0 or multiplier <= 0,
        # but as a safeguard to prevent division by zero if inputs were, for example, very small.
        raise ValueError("Calculated stop_loss_distance is zero, cannot calculate position size.")

    position_size = risk_amount / stop_loss_distance
    stop_loss_percentage = (stop_loss_distance / entry_price) * 100

    return {
        "position_size": position_size,
        "stop_loss_percentage": stop_loss_percentage,
    } 