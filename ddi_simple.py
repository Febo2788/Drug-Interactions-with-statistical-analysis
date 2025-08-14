#!/usr/bin/env python3
"""
Drug-Drug Interaction Prediction CLI - Simple Entry Point
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from cli.simple_cli import cli

if __name__ == '__main__':
    cli()