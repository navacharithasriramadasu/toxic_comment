#!/usr/bin/env python3
"""
Setup Script for Toxic Comment Detection System
Summer Internship Project | Summer of AI
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr.strip()}")
        return False

def main():
    """Setup the project"""
    print("🚀 Setting up Toxic Comment Detection System")
    print("=" * 50)
    
    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    else:
        print("\n✅ Virtual environment already exists")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip install -r requirements.txt"
    else:  # Unix/Linux/Mac
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip install -r requirements.txt"
    
    if not run_command(pip_cmd, "Installing dependencies"):
        print("\n⚠️  If installation failed, try running manually:")
        print(f"   {activate_cmd}")
        print("   pip install -r requirements.txt")
        return False
    
    # Create sample dataset if it doesn't exist
    if not os.path.exists("data/train.csv"):
        if not run_command("python scripts/create_sample_dataset.py", "Creating sample dataset"):
            return False
    else:
        print("\n✅ Sample dataset already exists")
    
    # Create test negative comments if they don't exist
    if not os.path.exists("data/test_negative_comments.csv"):
        if not run_command("python scripts/generate_negative_comments.py", "Creating test negative comments"):
            return False
    else:
        print("\n✅ Test negative comments already exist")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run the server:")
    print("   python backend/main.py")
    print("3. Open browser:")
    print("   http://localhost:8000")
    print("\n📚 For training:")
    print("   python scripts/train.py --data data/train.csv --output models/bert-toxic-finetuned")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
