#!/bin/bash
echo "=================================================="
echo " 🎙️ Starting Elena - AI Spoken English Coach"
echo "=================================================="

# Ensure cv venv is used
source cv/bin/activate

# Launch server
python server.py
