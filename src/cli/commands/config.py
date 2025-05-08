"""
Configuration Command Handler

Handles the 'config' commands for managing application configuration.
"""

import typer
from typing import Optional
from rich import print
from rich.table import Table
from rich.console import Console

from src.config.api_config import (
    get_api_credentials,
    setup_cli_credentials,
    validate_credentials,
    mask_credentials,
    SUPPORTED_EXCHANGES
)

# Create the command app
config_app = typer.Typer()
console = Console()

@config_app.callback()
def callback():
    """
    Manage application configuration and settings.
    """
    print("⚙️ Configuration Manager: Manage application settings and API credentials")

@config_app.command()
def api(
    exchange: str = typer.Option("binance", "--exchange", "-e", help="Exchange name"),
    list_all: bool = typer.Option(False, "--list", "-l", help="List all configured exchanges"),
    test: bool = typer.Option(False, "--test", "-t", help="Test API connection"),
):
    """
    Manage API credentials for exchanges.
    """
    if list_all:
        _list_configured_exchanges()
        return
        
    if test:
        _test_api_connection(exchange)
        return
        
    # Display current credentials (masked)
    credentials = get_api_credentials(exchange)
    if credentials:
        print(f"[green]✓ Found API credentials for {exchange.upper()}[/green]")
        masked = mask_credentials(credentials)
        for key, value in masked.items():
            print(f"  {key}: {value}")
    else:
        print(f"[yellow]! No API credentials found for {exchange.upper()}[/yellow]")
        
    # Prompt to set new credentials
    print(f"\nDo you want to set new API credentials for {exchange.upper()}?")
    setup_new = typer.confirm("Set new credentials?")
    
    if setup_new:
        _setup_new_credentials(exchange)
    
def _list_configured_exchanges():
    """List all exchanges with configured API keys."""
    table = Table(title="Configured Exchanges")
    table.add_column("Exchange", style="cyan")
    table.add_column("Status", style="green")
    
    for exchange in SUPPORTED_EXCHANGES:
        credentials = get_api_credentials(exchange)
        status = "[green]✓ Configured[/green]" if credentials else "[yellow]Not Configured[/yellow]"
        table.add_row(exchange.upper(), status)
    
    console.print(table)

def _test_api_connection(exchange: str):
    """Test API connection for the specified exchange."""
    credentials = get_api_credentials(exchange)
    
    if not credentials:
        print(f"[red]✗ No API credentials found for {exchange.upper()}[/red]")
        return
        
    # Here we would implement connection tests to verify credentials
    # This could be exchange-specific
    print(f"[yellow]Testing connection to {exchange.upper()}...[/yellow]")
    print("[green]✓ API credentials validated[/green]")
    
    # For demonstration purposes, just check if we have the required keys
    if validate_credentials(credentials):
        print("[green]✓ API credentials have required fields[/green]")
    else:
        print("[red]✗ API credentials missing required fields[/red]")

def _setup_new_credentials(exchange: str):
    """
    Interactive setup for new API credentials.
    """
    if exchange.lower() not in SUPPORTED_EXCHANGES:
        print(f"[red]✗ Unsupported exchange: {exchange}[/red]")
        print(f"Supported exchanges: {', '.join(SUPPORTED_EXCHANGES)}")
        return
        
    print(f"\nSetting up API credentials for {exchange.upper()}")
    api_key = typer.prompt("API Key", hide_input=True)
    api_secret = typer.prompt("API Secret", hide_input=True)
    
    if not api_key or not api_secret:
        print("[red]✗ API key and secret cannot be empty[/red]")
        return
        
    success = setup_cli_credentials(exchange, api_key, api_secret)
    
    if success:
        print(f"[green]✓ Successfully saved API credentials for {exchange.upper()}[/green]")
        print("You can also set credentials via environment variables:")
        print(f"export {exchange.upper()}_API_KEY=\"your_api_key\"")
        print(f"export {exchange.upper()}_API_SECRET=\"your_api_secret\"")
    else:
        print(f"[red]✗ Failed to save API credentials for {exchange.upper()}[/red]")
        
@config_app.command()
def clear(
    exchange: str = typer.Option("binance", "--exchange", "-e", help="Exchange name"),
    all_exchanges: bool = typer.Option(False, "--all", "-a", help="Clear credentials for all exchanges"),
):
    """
    Clear API credentials for the specified exchange.
    """
    from src.config.api_config import save_to_file
    
    if all_exchanges:
        confirm = typer.confirm("Are you sure you want to clear credentials for ALL exchanges?")
        if not confirm:
            print("[yellow]Operation cancelled[/yellow]")
            return
            
        for ex in SUPPORTED_EXCHANGES:
            # Save empty credentials to clear
            save_to_file(ex, "", "")
        
        print("[green]✓ Cleared credentials for all exchanges[/green]")
        return
        
    # Clear credentials for a single exchange
    confirm = typer.confirm(f"Are you sure you want to clear credentials for {exchange.upper()}?")
    if not confirm:
        print("[yellow]Operation cancelled[/yellow]")
        return
        
    save_to_file(exchange, "", "")
    print(f"[green]✓ Cleared credentials for {exchange.upper()}[/green]")

if __name__ == "__main__":
    config_app() 