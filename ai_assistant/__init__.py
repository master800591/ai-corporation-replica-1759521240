"""
AI Personal Assistant Toolkit

A comprehensive toolkit of utilities for:
- AI model management and operations
- Blockchain and P2P networking tools  
- Development platform utilities
- Autonomous development assistance

No demos or interactive code - just pure utility functions.
"""

__version__ = "2.0.0"
__author__ = "Steve Cornell"
__email__ = "your.email@example.com"

# Import utility modules
from . import utils

# Version and metadata
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "utils",
]

def get_version():
    """Get package version"""
    return __version__

def get_info():
    """Get package information"""
    return {
        "name": "ai-personal-assistant",
        "version": __version__,
        "author": __author__,
        "description": "Pure utility functions for AI development"
    }