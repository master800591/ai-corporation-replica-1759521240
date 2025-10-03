"""
Utility functions and helpers for AI Assistant

This package provides common utilities used across the toolkit:
- Logging configuration
- Authentication and user management  
- Cryptographic operations
- Security utilities
- File handling utilities
- Environment detection
"""

from .logging import setup_logging, get_logger

# Authentication utilities
from .auth_utils import (
    User, Session, UserManager, create_user_manager,
    validate_password_strength, generate_user_id, generate_session_id
)

# Cryptographic utilities  
from .simple_crypto_utils import (
    KeyPair, Signature, SimpleCryptoManager, create_crypto_manager,
    hash_message, generate_random_string, check_crypto_availability
)

# Combined security utilities
from .security_utils_clean import (
    SecurityManager, SecureUser, create_security_manager,
    quick_user_setup, authenticate_user, sign_message_for_user,
    verify_message_signature, hash_data_secure, generate_secure_token
)

__all__ = [
    # Logging
    "setup_logging", "get_logger",
    
    # Authentication
    "User", "Session", "UserManager", "create_user_manager",
    "validate_password_strength", "generate_user_id", "generate_session_id",
    
    # Cryptography
    "KeyPair", "Signature", "SimpleCryptoManager", "create_crypto_manager", 
    "hash_message", "generate_random_string", "check_crypto_availability",
    
    # Security (combined)
    "SecurityManager", "SecureUser", "create_security_manager",
    "quick_user_setup", "authenticate_user", "sign_message_for_user",
    "verify_message_signature", "hash_data_secure", "generate_secure_token"
]