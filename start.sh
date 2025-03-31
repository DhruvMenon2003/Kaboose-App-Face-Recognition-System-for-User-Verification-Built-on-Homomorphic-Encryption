#!/bin/bash

# Create virtual environment if it doesn't exist
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=verify.py
export FLASK_ENV=production

# Start the application
gunicorn verify:app --bind 0.0.0.0:$PORT