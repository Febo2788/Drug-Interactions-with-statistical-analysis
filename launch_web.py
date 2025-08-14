#!/usr/bin/env python3
"""
Simple launcher for the Drug-Drug Interaction Web Interface.
This script can be run with 'python launch_web.py' from anywhere.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit web application."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    web_app_path = script_dir / "web_app.py"
    
    print("🌐 Launching Drug-Drug Interaction Web Interface...")
    print(f"📁 Project directory: {script_dir}")
    print(f"🚀 Starting Streamlit server...")
    print("")
    print("Once started, open: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(web_app_path),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped by user")
    except Exception as e:
        print(f"❌ Error launching web interface: {e}")
        print("\nTry running manually:")
        print(f"streamlit run {web_app_path}")

if __name__ == "__main__":
    main()