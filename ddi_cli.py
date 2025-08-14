#!/usr/bin/env python3
"""
Drug-Drug Interaction Prediction CLI - Entry Point

Simple entry point for the CLI application that can be run from project root.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from cli.cli_app import cli

if __name__ == '__main__':
    cli()