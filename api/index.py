"""
FiberGuard - Vercel Serverless Entrypoint
Exposes the FastAPI application for Vercel Serverless Python runtime.
"""

import sys
import os

# Add root directory to sys.path so backend modules can be imported
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.main import app

# Vercel looks for 'app' in api/index.py
