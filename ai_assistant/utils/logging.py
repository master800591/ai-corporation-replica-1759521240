"""
Logging utilities for AI Assistant

Provides consistent logging setup across all components with UTF-8 safe handling.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration for AI Assistant
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console: Whether to log to console
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create handlers
    handlers = []
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(console_handler)
    
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_file, 
            encoding='utf-8', 
            errors='replace'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format=format_string,
        force=True  # Override any existing configuration
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with AI Assistant formatting
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers exist, set up basic logging
    if not logger.handlers and not logging.getLogger().handlers:
        setup_logging()
    
    return logger

def log_component_status(component_name: str, available: bool, logger: logging.Logger) -> None:
    """
    Log component availability with consistent formatting
    
    Args:
        component_name: Name of the component
        available: Whether component is available
        logger: Logger instance to use
    """
    if available:
        logger.info(f"[OK] {component_name} available")
    else:
        logger.warning(f"[MISSING] {component_name} not available")

def log_operation_result(operation: str, success: bool, logger: logging.Logger, details: str = "") -> None:
    """
    Log operation results with consistent formatting
    
    Args:
        operation: Description of the operation
        success: Whether operation was successful
        logger: Logger instance to use
        details: Additional details
    """
    if success:
        logger.info(f"[OK] {operation}{' - ' + details if details else ''}")
    else:
        logger.error(f"[ERROR] {operation}{' - ' + details if details else ''}")

# Global logger for utilities
logger = get_logger(__name__)