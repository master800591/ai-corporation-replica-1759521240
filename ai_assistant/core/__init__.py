"""
Core AI Assistant functionality

This package contains the main AI assistant components:
- AI Corporation system with democratic governance
- Autonomous learning and development
- Global operations management
"""

# Core AI Corporation components
from .ai_corporation import AICorporation, create_ai_corporation
from .autonomous_learning import AutonomousDevelopmentSystem, create_autonomous_development_system
from .global_operations import GlobalOperationsManager, create_global_operations_manager

# Ollama components with graceful degradation
try:
    from .ollama_toolkit import OllamaToolkit
    from .model_manager import OllamaModelManager
    ollama_available = True
except ImportError:
    ollama_available = False

# AI Platform import with graceful degradation
try:
    from .ai_platform import AIPlatformEnhanced
    ai_platform_available = True
except ImportError:
    ai_platform_available = False

# Build __all__ based on available components
__all__ = ["AICorporation", "create_ai_corporation", 
           "AutonomousDevelopmentSystem", "create_autonomous_development_system",
           "GlobalOperationsManager", "create_global_operations_manager"]

if ollama_available:
    __all__.extend(["OllamaToolkit", "OllamaModelManager"])

if ai_platform_available:
    __all__.append("AIPlatformEnhanced")