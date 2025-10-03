"""
AI Personal Assistant Toolkit

A comprehensive toolkit for AI model management, autonomous development, 
and peer-to-peer AI collaboration.
"""

__version__ = "2.0.0"
__author__ = "Steve Cornell"
__email__ = "your.email@example.com"

# Core imports for easy access
from .core.ollama_toolkit import OllamaToolkit
from .core.model_manager import OllamaModelManager
from .core.ai_platform import AIPlatformEnhanced

# Version and metadata
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "OllamaToolkit",
    "OllamaModelManager", 
    "AIPlatformEnhanced",
]

# Optional P2P imports with graceful degradation
try:
    from .p2p.node import P2PNode
    from .p2p.integration import DistributedOllamaNode
    __all__.extend(["P2PNode", "DistributedOllamaNode"])
except ImportError:
    # P2P dependencies not available
    pass

# Check component availability
def check_components():
    """Check availability of optional components"""
    components = {
        'ollama': False,
        'p2p': False,
        'crewai': False
    }
    
    try:
        import ollama
        components['ollama'] = True
    except ImportError:
        pass
    
    try:
        from .p2p.node import P2PNode
        components['p2p'] = True
    except ImportError:
        pass
    
    try:
        import crewai
        components['crewai'] = True
    except ImportError:
        pass
    
    return components

# Convenience function for quick access
def get_version():
    """Get package version"""
    return __version__

def get_info():
    """Get package information"""
    return {
        "name": "ai-personal-assistant",
        "version": __version__,
        "author": __author__,
        "components": check_components()
    }