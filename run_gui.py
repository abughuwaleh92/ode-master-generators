#!/usr/bin/env python
"""
GUI Launcher for ODE Master Generator System
"""

import sys
import subprocess
import os

def check_services():
    """Check if required services are running"""
    print("Checking required services...")
    
    # Check API server
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        print("✓ API Server is running")
    except:
        print("✗ API Server is not running")
        print("  Start it with: python scripts/production_server.py")
        return False
    
    return True

def main():
    """Launch the GUI"""
    print("ODE Master Generator - GUI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("gui/integrated_interface.py"):
        print("Error: Cannot find gui/integrated_interface.py")
        print("Make sure you're running this from the project root directory")
        sys.exit(1)
    
    # Check services
    if not check_services():
        print("\nWould you like to start the API server? (y/n)")
        if input().lower() == 'y':
            print("Starting API server in background...")
            subprocess.Popen(["python", "scripts/production_server.py"])
            import time
            time.sleep(5)  # Wait for server to start
    
    # Launch GUI
    print("\nStarting GUI...")
    print("The interface will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop")
    
    try:
        subprocess.run(["streamlit", "run", "gui/integrated_interface.py"])
    except KeyboardInterrupt:
        print("\nGUI stopped")

if name == "__main__":  # <- This line was wrong (had single underscores)
    main()