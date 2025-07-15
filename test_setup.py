#!/usr/bin/env python3
"""
Test script to verify the setup for the Image Hashtag Generator application.
This script checks:
1. Required Python packages
2. ExifTool installation
3. Connection to ollama API
"""

import importlib.util
import os
import platform
import subprocess
import sys

import requests


def check_package(package_name):
    """Check if a Python package is installed."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ {package_name} is not installed. Install it with: pip install {package_name}")
        return False
    print(f"✅ {package_name} is installed")
    return True

def find_exiftool():
    """Find exiftool in the system."""
    system = platform.system()
    exiftool_path = None
    
    if system == "Darwin":  # macOS
        default_path = "/usr/local/bin/exiftool"
        if os.path.exists(default_path):
            exiftool_path = default_path
        else:
            # Try to find in other common locations
            for path in ["/opt/homebrew/bin/exiftool", "/usr/bin/exiftool"]:
                if os.path.exists(path):
                    exiftool_path = path
                    break
    
    elif system == "Windows":
        # Check if exiftool is in PATH
        try:
            result = subprocess.run(["where", "exiftool"], capture_output=True, text=True)
            if result.returncode == 0:
                exiftool_path = result.stdout.strip().split("\n")[0]
        except:
            pass
        
        # Check common installation locations
        if not exiftool_path:
            for path in [
                r"C:\Program Files\ExifTool\exiftool.exe",
                r"C:\Program Files (x86)\ExifTool\exiftool.exe"
            ]:
                if os.path.exists(path):
                    exiftool_path = path
                    break
    
    else:  # Linux and others
        try:
            result = subprocess.run(["which", "exiftool"], capture_output=True, text=True)
            if result.returncode == 0:
                exiftool_path = result.stdout.strip()
        except:
            pass
    
    return exiftool_path

def test_exiftool(exiftool_path):
    """Test if exiftool is working."""
    try:
        result = subprocess.run([exiftool_path, "-ver"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ ExifTool is installed (version {version})")
            return True
        else:
            print(f"❌ ExifTool test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ ExifTool test failed: {str(e)}")
        return False

def test_ollama_connection():
    """Test connection to ollama API."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print(f"✅ Connected to ollama API. Available models: {', '.join([m.get('name', '') for m in models])}")
            else:
                print("✅ Connected to ollama API, but no models found. Pull a model with: docker exec -it ollama ollama pull llava")
            return True
        else:
            print(f"❌ ollama API connection failed: {response.status_code} {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to ollama API. Make sure ollama is running in Docker on port 11434.")
        return False
    except Exception as e:
        print(f"❌ ollama API connection failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Testing setup for Image Hashtag Generator...\n")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    major, minor, _ = python_version.split(".")
    if int(major) < 3 or (int(major) == 3 and int(minor) < 7):
        print("❌ Python 3.7+ is required")
    else:
        print("✅ Python version is compatible")
    
    print("\nChecking required packages:")
    packages_ok = all([
        check_package("PyQt5"),
        check_package("requests"),
        check_package("PIL"),
    ])
    
    print("\nChecking ExifTool:")
    exiftool_path = find_exiftool()
    if exiftool_path:
        print(f"✅ ExifTool found at: {exiftool_path}")
        exiftool_ok = test_exiftool(exiftool_path)
    else:
        print("❌ ExifTool not found. Please install ExifTool:")
        print("   - macOS: brew install exiftool")
        print("   - Windows: Download from https://exiftool.org/ and add to PATH")
        print("   - Linux: sudo apt install libimage-exiftool-perl (or equivalent)")
        exiftool_ok = False
    
    print("\nChecking ollama API connection:")
    ollama_ok = test_ollama_connection()
    
    print("\nSummary:")
    if packages_ok and exiftool_ok and ollama_ok:
        print("✅ All checks passed! You're ready to run the Image Hashtag Generator.")
        print("   Run the application with: python main.py")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above before running the application.")
        return 1

if __name__ == "__main__":
    sys.exit(main())