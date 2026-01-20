#!/usr/bin/env python3
"""
Setup script for DSPy 0-to-1 Guide

This script helps set up the development environment and test the installation.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False, e.stderr

def check_python_version():
    """Check if Python version is 3.9+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported. Requires Python 3.9+")
        return False

def check_ollama():
    """Check if Ollama is available and has models."""
    print("🔍 Checking Ollama availability...")

    # Check if ollama command exists
    success, _ = run_command("which ollama", "Finding Ollama")
    if not success:
        print("⚠️  Ollama not found. Install from https://ollama.ai")
        return False

    # Check if Ollama API is responding
    success, output = run_command("curl -s http://localhost:11434/api/tags", "Testing Ollama API")
    if not success:
        print("⚠️  Ollama API not responding. Run 'ollama serve' to start it")
        return False

    # Check available models
    success, output = run_command("ollama list", "Listing Ollama models")
    if success and "llama3" in output:
        print("✅ Ollama is ready with models available")
        return True
    else:
        print("⚠️  No suitable models found. Run 'ollama pull llama3' to download a model")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")

    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment. Consider using: python -m venv .venv && source .venv/bin/activate")

    # Install dependencies
    success, _ = run_command("pip install -e .", "Installing project dependencies")
    return success

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file already exists")
        return True

    print("📝 Creating .env file from template...")
    try:
        with open(".env.example", "r") as source:
            content = source.read()

        with open(".env", "w") as target:
            target.write(content)

        print("✅ .env file created")
        print("⚠️  Please edit .env to add your API keys if you want to use cloud models")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_basic_functionality():
    """Test basic DSPy functionality with Ollama."""
    print("🧪 Testing basic DSPy functionality...")

    test_script = '''
import os
import dspy

try:
    # Configure DSPy with Ollama
    dspy.configure(lm=dspy.LM('ollama_chat/llama3'))

    # Create a simple test
    predictor = dspy.Predict("question -> answer")
    result = predictor(question="What is 2+2?")

    print(f"Test result: {result.answer}")
    print("DSPy with Ollama is working!")

except Exception as e:
    print(f"Test failed: {e}")
    exit(1)
'''

    with open("test_setup.py", "w") as f:
        f.write(test_script)

    success, output = run_command("python test_setup.py", "Testing DSPy with Ollama")

    # Clean up test file
    try:
        os.remove("test_setup.py")
    except:
        pass

    if success:
        print("✅ DSPy is working correctly with Ollama!")

    return success

def main():
    """Main setup function."""
    print("🚀 DSPy 0-to-1 Guide Setup")
    print("=" * 50)

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Check Ollama
    ollama_available = check_ollama()
    if not ollama_available:
        print("⚠️  Ollama issues detected. Some examples may not work.")

    # Step 3: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Step 4: Create .env file
    create_env_file()

    # Step 5: Test functionality
    if ollama_available:
        if not test_basic_functionality():
            print("❌ Basic functionality test failed")
            sys.exit(1)

    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("   1. Run examples: make run-basic")
    print("   2. Try personas: make run-personas")
    print("   3. Run tests: make test")
    print("   4. Start with: python examples/basic/hello_world.py")

if __name__ == "__main__":
    main()