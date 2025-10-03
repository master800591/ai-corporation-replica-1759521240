#!/usr/bin/env python3
"""
Security Utilities

Combined utilities for authentication and cryptography:
- User authentication and session management
- Key generation and digital signatures
- Secure password handling
- Message signing and verification
"""

import hashlib
import secrets
import base64
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import the separate utility modules
from .auth_utils import (
    User, Session, UserManager,
    create_user_manager, generate_salt, hash_password, verify_password,
    validate_password_strength, generate_user_id, generate_session_id
)

from .crypto_utils import (
    KeyPair, Signature, CryptoManager,
    create_crypto_manager, hash_message, generate_random_bytes,
    generate_random_string, encode_base64, decode_base64,
    check_crypto_availability
)


@dataclass
class SecureUser:
    """Enhanced user with cryptographic keys"""
    user: User
    key_pair: Optional[KeyPair] = None
    key_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'user': self.user.to_dict(),
            'key_id': self.key_id
        }
        if self.key_pair:
            data['public_key'] = self.key_pair.public_key
            data['key_algorithm'] = self.key_pair.algorithm
        return data


class SecurityManager:
    """Combined security manager for authentication and cryptography"""
    
    def __init__(self):
        """Initialize security manager"""
        self.user_manager = create_user_manager()
        self.crypto_manager = create_crypto_manager()
        self.secure_users: Dict[str, SecureUser] = {}
    
    def create_secure_user(self, username: str, email: str, password: str,
                          generate_keys: bool = True, key_algorithm: str = 'rsa',
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a user with cryptographic keys"""
        try:
            # Validate password strength
            password_check = validate_password_strength(password)
            if not password_check['is_strong']:
                return {
                    'success': False, 
                    'error': 'Weak password',
                    'password_issues': password_check['issues']
                }
            
            # Create user
            user_result = self.user_manager.create_user(username, email, password, metadata)
            if not user_result['success']:
                return user_result
            
            user = User.from_dict(user_result['user'])
            
            # Generate keys if requested
            key_pair = None
            key_id = None
            
            if generate_keys:
                key_result = self.crypto_manager.generate_key_pair(key_algorithm)
                if key_result['success']:
                    key_pair = KeyPair.from_dict(key_result['key_pair'])
                    key_id = key_result['key_id']
            
            # Create secure user
            secure_user = SecureUser(
                user=user,
                key_pair=key_pair,
                key_id=key_id
            )
            
            self.secure_users[user.user_id] = secure_user
            
            return {
                'success': True,
                'user_id': user.user_id,
                'key_id': key_id,
                'secure_user': secure_user.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def authenticate_and_sign(self, identifier: str, password: str, 
                             message: str = None) -> Dict[str, Any]:
        """Authenticate user and optionally sign a message"""
        try:
            # Authenticate user
            auth_result = self.user_manager.authenticate_user(identifier, password)
            if not auth_result['success']:
                return auth_result
            
            user_id = auth_result['user_id']
            secure_user = self.secure_users.get(user_id)
            
            if not secure_user:
                return {'success': False, 'error': 'Secure user not found'}
            
            result = {
                'success': True,
                'user_id': user_id,
                'user': auth_result['user']
            }
            
            # Sign message if provided and keys available
            if message and secure_user.key_pair:
                sign_result = self.crypto_manager.sign_message(
                    message, 
                    secure_user.key_pair.private_key,
                    secure_user.key_pair.algorithm
                )
                
                if sign_result['success']:
                    result['signature'] = sign_result['signature']
                    result['message_signed'] = True
                else:
                    result['signature_error'] = sign_result['error']
                    result['message_signed'] = False
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_secure_session(self, user_id: str, duration_hours: int = 24,
                             sign_session: bool = True) -> Dict[str, Any]:
        """Create a session with optional cryptographic signature"""
        try:
            # Create regular session
            session_result = self.user_manager.create_session(user_id, duration_hours)
            if not session_result['success']:
                return session_result
            
            result = session_result.copy()
            
            # Sign session if requested and keys available
            if sign_session:
                secure_user = self.secure_users.get(user_id)
                if secure_user and secure_user.key_pair:
                    session_data = f"{session_result['session_id']}:{user_id}:{session_result['expires_at']}"
                    
                    sign_result = self.crypto_manager.sign_message(
                        session_data,
                        secure_user.key_pair.private_key,
                        secure_user.key_pair.algorithm
                    )
                    
                    if sign_result['success']:
                        result['session_signature'] = sign_result['signature']
                        result['session_signed'] = True
                    else:
                        result['signature_error'] = sign_result['error']
                        result['session_signed'] = False
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def verify_signed_message(self, message: str, signature_data: Dict[str, Any],
                             user_id: str = None) -> Dict[str, Any]:
        """Verify a signed message"""
        try:
            signature_obj = Signature.from_dict(signature_data)
            
            # If user_id provided, use their public key
            if user_id:
                secure_user = self.secure_users.get(user_id)
                if not secure_user or not secure_user.key_pair:
                    return {'success': False, 'error': 'User keys not found'}
                
                public_key = secure_user.key_pair.public_key
                algorithm = secure_user.key_pair.algorithm
            else:
                # Use public key from signature
                public_key = signature_obj.public_key
                algorithm = signature_obj.algorithm
            
            # Verify signature
            verify_result = self.crypto_manager.verify_signature(
                message,
                signature_obj.signature,
                public_key,
                algorithm
            )
            
            return verify_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_public_key(self, user_id: str) -> Optional[str]:
        """Get user's public key"""
        secure_user = self.secure_users.get(user_id)
        if secure_user and secure_user.key_pair:
            return secure_user.key_pair.public_key
        return None
    
    def regenerate_user_keys(self, user_id: str, algorithm: str = 'rsa') -> Dict[str, Any]:
        """Regenerate cryptographic keys for a user"""
        try:
            secure_user = self.secure_users.get(user_id)
            if not secure_user:
                return {'success': False, 'error': 'User not found'}
            
            # Generate new keys
            key_result = self.crypto_manager.generate_key_pair(algorithm)
            if not key_result['success']:
                return key_result
            
            # Update user's keys
            secure_user.key_pair = KeyPair.from_dict(key_result['key_pair'])
            secure_user.key_id = key_result['key_id']
            
            return {
                'success': True,
                'key_id': key_result['key_id'],
                'public_key': secure_user.key_pair.public_key
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def export_user_keys(self, user_id: str, include_private: bool = False) -> Dict[str, Any]:
        """Export user's cryptographic keys"""
        try:
            secure_user = self.secure_users.get(user_id)
            if not secure_user or not secure_user.key_pair:
                return {'success': False, 'error': 'User keys not found'}
            
            result = {
                'success': True,
                'user_id': user_id,
                'key_id': secure_user.key_id,
                'public_key': secure_user.key_pair.public_key,
                'algorithm': secure_user.key_pair.algorithm,
                'key_format': secure_user.key_pair.key_format,
                'created_at': secure_user.key_pair.created_at
            }
            
            if include_private:
                result['private_key'] = secure_user.key_pair.private_key
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def list_secure_users(self) -> List[Dict[str, Any]]:
        """List all secure users (without sensitive data)"""
        users = []
        for user_id, secure_user in self.secure_users.items():
            user_data = secure_user.to_dict()
            # Remove sensitive fields
            if 'user' in user_data:
                user_data['user'].pop('password_hash', None)
                user_data['user'].pop('salt', None)
            users.append(user_data)
        return users
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security system status"""
        return {
            'users_total': len(self.user_manager.users),
            'users_with_keys': len(self.secure_users),
            'active_sessions': len([s for s in self.user_manager.sessions.values() if s.is_active]),
            'crypto_available': self.crypto_manager.backend_available,
            'crypto_algorithms': check_crypto_availability()
        }


# Main utility functions
def create_security_manager() -> SecurityManager:
    """Create a new security manager instance"""
    return SecurityManager()


def quick_user_setup(username: str, email: str, password: str, 
                    generate_keys: bool = True) -> Dict[str, Any]:
    """Quick setup for a secure user with keys"""
    manager = create_security_manager()
    return manager.create_secure_user(username, email, password, generate_keys)


def authenticate_user(manager: SecurityManager, identifier: str, 
                     password: str) -> Dict[str, Any]:
    """Quick user authentication"""
    return manager.authenticate_and_sign(identifier, password)


def sign_message_for_user(manager: SecurityManager, user_id: str, 
                         message: str) -> Dict[str, Any]:
    """Sign a message for a specific user"""
    secure_user = manager.secure_users.get(user_id)
    if not secure_user or not secure_user.key_pair:
        return {'success': False, 'error': 'User or keys not found'}
    
    return manager.crypto_manager.sign_message(
        message,
        secure_user.key_pair.private_key,
        secure_user.key_pair.algorithm
    )


def verify_message_signature(manager: SecurityManager, message: str,
                           signature_data: Dict[str, Any], 
                           user_id: str = None) -> Dict[str, Any]:
    """Verify a message signature"""
    return manager.verify_signed_message(message, signature_data, user_id)


def hash_data_secure(data: str, algorithm: str = 'sha256') -> str:
    """Securely hash data"""
    return hash_message(data, algorithm)


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure token"""
    return generate_random_string(length)