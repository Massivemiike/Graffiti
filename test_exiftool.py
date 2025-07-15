import os
import platform
import subprocess
import sys

def find_exiftool():
    """Find exiftool based on the operating system."""
    if platform.system() == "Darwin":  # macOS
        default_path = "/usr/local/bin/exiftool"
        if os.path.exists(default_path):
            return default_path
        # Try homebrew location
        homebrew_path = "/opt/homebrew/bin/exiftool"
        if os.path.exists(homebrew_path):
            return homebrew_path
    elif platform.system() == "Windows":
        # Check if exiftool is in PATH
        try:
            result = subprocess.run(["where", "exiftool"], 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)
            if result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except subprocess.CalledProcessError:
            pass
    
    # Try to find in current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exiftool_path = os.path.join(current_dir, "exiftool")
    if os.path.exists(exiftool_path):
        return exiftool_path
        
    return None

def test_exiftool():
    """Test if ExifTool is installed and working."""
    print("Testing ExifTool installation...")
    
    exiftool_path = find_exiftool()
    
    if exiftool_path:
        print(f"✅ ExifTool found at: {exiftool_path}")
        
        # Test ExifTool version
        try:
            result = subprocess.run([exiftool_path, "-ver"], 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)
            version = result.stdout.strip()
            print(f"✅ ExifTool version: {version}")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running ExifTool: {str(e)}")
            return False
    else:
        print("❌ ExifTool not found.")
        print("\nInstallation instructions:")
        
        if platform.system() == "Darwin":  # macOS
            print("On macOS, install with Homebrew:")
            print("  brew install exiftool")
        elif platform.system() == "Windows":
            print("On Windows:")
            print("1. Download ExifTool from https://exiftool.org/")
            print("2. Extract the contents and rename exiftool(-k).exe to exiftool.exe")
            print("3. Move exiftool.exe to a directory in your PATH or add its location to your PATH")
        else:
            print("On Linux:")
            print("  sudo apt-get install libimage-exiftool-perl")
            print("  or")
            print("  sudo yum install perl-Image-ExifTool")
        
        return False

if __name__ == "__main__":
    success = test_exiftool()
    sys.exit(0 if success else 1)