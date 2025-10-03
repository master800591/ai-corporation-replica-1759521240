#!/usr/bin/env python3
"""
Ollama Utility Functions

Pure utility functions for Ollama operations.
No demos, no interactive code, just tools.
"""

from typing import List, Dict, Any, Optional

try:
    from ..core.ollama_toolkit import OllamaToolkit
    from ..core.model_manager import OllamaModelManager
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def create_ollama_toolkit():
    """Create Ollama toolkit if available"""
    if not OLLAMA_AVAILABLE:
        return None
    return OllamaToolkit()


def create_model_manager():
    """Create model manager if available"""
    if not OLLAMA_AVAILABLE:
        return None
    return OllamaModelManager()


def check_ollama_availability() -> bool:
    """Check if Ollama functionality is available"""
    return OLLAMA_AVAILABLE


def get_ollama_status() -> Dict[str, Any]:
    """Get Ollama system status"""
    return {
        'available': OLLAMA_AVAILABLE,
        'dependencies': {
            'ollama': _check_dependency('ollama'),
            'requests': _check_dependency('requests'),
        }
    }


def list_local_models() -> List[str]:
    """List locally available models"""
    if not OLLAMA_AVAILABLE:
        return []
    
    try:
        manager = OllamaModelManager()
        return manager.get_local_models()
    except Exception:
        return []


def get_popular_models(limit: int = 10) -> List[Dict[str, Any]]:
    """Get popular models from registry"""
    if not OLLAMA_AVAILABLE:
        return []
    
    try:
        manager = OllamaModelManager()
        return manager.get_popular_models(limit)
    except Exception:
        return []


def pull_model(model_name: str) -> bool:
    """Pull a specific model"""
    if not OLLAMA_AVAILABLE:
        return False
    
    try:
        manager = OllamaModelManager()
        result = manager.pull_model(model_name)
        return result.get('success', False)
    except Exception:
        return False


def _check_dependency(module_name: str) -> bool:
    """Check if a dependency is available"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False