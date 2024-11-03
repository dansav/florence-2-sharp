#!/bin/bash

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run setup script
python setup.py

echo "Setup completed successfully! You can now run the main script with 'python main.py'."