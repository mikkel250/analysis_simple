#!/usr/bin/env python
"""
Installation script for the Financial Analysis Jupyter kernel.

This script installs and registers the custom kernel with Jupyter,
allowing it to be selected when creating new notebooks.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.jupyter.kernel import __version__

KERNEL_NAME = "financial_analysis"
DISPLAY_NAME = "Financial Analysis"
KERNELSPEC_DIR = "kernelspec"


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "ipykernel", "pandas", "numpy", "matplotlib", "plotly", 
        "yfinance", "pandas_ta", "jupyterlab"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        install = input("Would you like to install them now? [y/N] ")
        if install.lower() == 'y':
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_packages
            ])
            print("Dependencies installed successfully.")
        else:
            print("Please install the missing dependencies before proceeding.")
            sys.exit(1)
    else:
        print("All dependencies are installed.")


def get_kernel_dir():
    """Get the kernel module directory."""
    return os.path.dirname(os.path.abspath(__file__))


def prepare_kernelspec_directory(temp_dir):
    """
    Prepare the kernelspec directory for installation.
    
    Args:
        temp_dir: Temporary directory to store files for installation
        
    Returns:
        Path to the prepared kernelspec directory
    """
    # Create temporary kernelspec directory
    temp_kernelspec_dir = os.path.join(temp_dir, KERNEL_NAME)
    os.makedirs(temp_kernelspec_dir, exist_ok=True)
    
    # Get source directories
    kernel_dir = get_kernel_dir()
    kernelspec_source_dir = os.path.join(kernel_dir, KERNELSPEC_DIR)
    
    # Copy the kernelspec files
    kernel_json_file = os.path.join(kernelspec_source_dir, "kernel.json")
    if os.path.exists(kernel_json_file):
        with open(kernel_json_file, "r") as f:
            kernel_json = json.load(f)
    else:
        # Create kernel.json if it doesn't exist
        kernel_json = {
            "argv": [
                sys.executable,
                "-m",
                "src.jupyter.kernel.kernel",
                "-f",
                "{connection_file}"
            ],
            "display_name": DISPLAY_NAME,
            "language": "python",
            "metadata": {
                "debugger": True
            },
            "env": {
                "PROJECT_PATH": "${PWD}"
            }
        }
    
    # Write kernel.json to the temporary directory
    with open(os.path.join(temp_kernelspec_dir, "kernel.json"), "w") as f:
        json.dump(kernel_json, f, indent=2)
    
    # Copy any other resources (like logos) if they exist
    for res_file in os.listdir(kernelspec_source_dir):
        if res_file != "kernel.json" and not res_file.startswith("."):
            src_path = os.path.join(kernelspec_source_dir, res_file)
            dst_path = os.path.join(temp_kernelspec_dir, res_file)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
    
    return temp_kernelspec_dir


def install_kernel(user=True, prefix=None, verbose=False):
    """
    Install the kernel to Jupyter.
    
    Args:
        user: Whether to install for the current user only
        prefix: Installation prefix path
        verbose: Whether to print verbose output
    
    Returns:
        True if installation was successful, False otherwise
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        kernelspec_dir = prepare_kernelspec_directory(temp_dir)
        
        # Build the command to install the kernelspec
        cmd = [sys.executable, "-m", "jupyter", "kernelspec", "install", kernelspec_dir]
        
        if user:
            cmd.append("--user")
        
        if prefix:
            cmd.extend(["--prefix", prefix])
        
        if verbose:
            cmd.append("--verbose")
            print(f"Running command: {' '.join(cmd)}")
        
        try:
            # Run the installation command
            subprocess.check_call(cmd)
            print(f"Financial Analysis kernel v{__version__} installed successfully!")
            
            # Print information about how to use the kernel
            print("\nTo use the kernel:")
            print("1. Start Jupyter Lab or Notebook")
            print("2. Create a new notebook and select 'Financial Analysis' as the kernel")
            print("3. Try using the %short, %medium, or %long magic commands to set trading timeframes")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing kernel: {e}")
            return False


def uninstall_kernel(verbose=False):
    """
    Uninstall the kernel from Jupyter.
    
    Args:
        verbose: Whether to print verbose output
    
    Returns:
        True if uninstallation was successful, False otherwise
    """
    cmd = [sys.executable, "-m", "jupyter", "kernelspec", "uninstall", KERNEL_NAME]
    
    if verbose:
        cmd.append("--verbose")
        print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Add -y to force removal without confirmation
        subprocess.check_call(cmd + ["-y"])
        print(f"Financial Analysis kernel v{__version__} uninstalled successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error uninstalling kernel: {e}")
        return False


def main():
    """Main entry point for the installation script."""
    parser = argparse.ArgumentParser(description="Install the Financial Analysis Jupyter kernel")
    parser.add_argument(
        "--user", action="store_true", default=True,
        help="Install for the current user only (default)"
    )
    parser.add_argument(
        "--sys", action="store_true", dest="system",
        help="Install system-wide (requires admin privileges)"
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Installation prefix path"
    )
    parser.add_argument(
        "--uninstall", action="store_true",
        help="Uninstall the kernel"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Financial Analysis Kernel Installer v{__version__}")
    
    # Check if Jupyter is installed
    try:
        import jupyter
    except ImportError:
        print("Jupyter is not installed. Please install it first.")
        sys.exit(1)
    
    # Check if --sys and --prefix are specified together
    if args.system and args.prefix:
        print("Error: Cannot specify both --sys and --prefix.")
        sys.exit(1)
    
    # Determine user installation based on args
    user_install = not args.system if args.prefix is None else False
    
    if args.uninstall:
        success = uninstall_kernel(verbose=args.verbose)
    else:
        check_dependencies()
        success = install_kernel(
            user=user_install,
            prefix=args.prefix,
            verbose=args.verbose
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
