#!/usr/bin/env python3
"""
Test script to verify the Gemini Text to Knowledge setup
"""

import os
import sys
from pathlib import Path

def test_environment():
    """Test if the environment is properly configured."""
    print("Testing Gemini Text to Knowledge Setup")
    print("=" * 40)
    
    # Test Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("Python 3.7 or higher is required")
        return False
    else:
        print("Python version is compatible")
    
    # Test required directories
    required_dirs = ['input', 'output']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"Directory '{dir_name}' exists")
        else:
            print(f"Directory '{dir_name}' not found")
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"Created directory '{dir_name}'")
            except Exception as e:
                print(f"Failed to create directory '{dir_name}': {e}")
                return False
    
    # Test required files
    required_files = ['gemini_prompt.txt', 'gemini_text_to_knowledge.py']
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"File '{file_name}' exists")
        else:
            print(f"File '{file_name}' not found")
            return False
    
    # Test .env file
    env_file = Path('.env')
    if env_file.exists():
        print(".env file exists")
        # Check if it contains the API key
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if 'GOOGLE_API_KEY' in content:
                    print("GOOGLE_API_KEY found in .env")
                else:
                    print("GOOGLE_API_KEY not found in .env")
                    return False
        except Exception as e:
            print(f"Error reading .env file: {e}")
            return False
    else:
        print(".env file not found")
        print("Please create a .env file with your GOOGLE_API_KEY")
        return False
    
    # Test dependencies
    try:
        import google.generativeai
        print("google-generativeai library available")
    except ImportError:
        print("google-generativeai library not available")
        print("Run: pip install -r requirements.txt")
        return False
    
    try:
        import dotenv
        print("python-dotenv library available")
    except ImportError:
        print("python-dotenv library not available")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n" + "=" * 40)
    print("Setup verification completed successfully!")
    print("You can now run: python gemini_text_to_knowledge.py")
    return True

if __name__ == "__main__":
    success = test_environment()
    if not success:
        print("\nSetup verification failed. Please fix the issues above.")
        sys.exit(1)
