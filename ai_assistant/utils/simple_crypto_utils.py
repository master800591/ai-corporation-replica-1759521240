#!/usr/bin/env python3
"""
Simple Crypto Utilities

Basic cryptographic utilities for:
- Simple key generation 
- Message hashing and signing
- Base64 encoding/decoding
- Secure random generation
"""

import hashlib
import secrets
import base64
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class KeyPair:
    """Simple cryptographic key pair"""
    private_key: str
    public_key: str
    algorithm: str = 'simple'
    key_format: str = 'base64'
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyPair':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class Signature:
    """Digital signature"""
    signature: str
    algorithm: str
    public_key: str
    message_hash: str
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signature':
        """Create from dictionary"""
        return cls(**data)


class SimpleCryptoManager:
    """Simple cryptographic operations manager"""
    
    def __init__(self):
        """Initialize crypto manager"""
        self.key_pairs: Dict[str, KeyPair] = {}
    
    def generate_key_pair(self, algorithm: str = 'simple') -> Dict[str, Any]:
        """Generate a simple key pair"""
        try:
            # Generate keys using secure random data
            private_seed = secrets.token_bytes(32)
            
            # Create a deterministic public key from private key
            public_seed = hashlib.sha256(private_seed + b'public').digest()
            
            private_key = base64.b64encode(private_seed).decode('utf-8')
            public_key = base64.b64encode(public_seed).decode('utf-8')
            
            key_pair = KeyPair(
                private_key=private_key,
                public_key=public_key,
                algorithm=algorithm
            )
            
            key_id = generate_random_string(16)
            self.key_pairs[key_id] = key_pair
            
            return {
                'success': True,
                'key_pair': key_pair.to_dict(),
                'key_id': key_id
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def sign_message(self, message: str, private_key: str, 
                    algorithm: str = 'simple') -> Dict[str, Any]:
        """Sign a message"""
        try:
            # Hash the message
            message_hash = hashlib.sha256(message.encode('utf-8')).hexdigest()
            
            # Create signature by combining message hash with private key
            signature_data = f"{message_hash}:{private_key}:{algorithm}"
            signature_hash = hashlib.sha256(signature_data.encode('utf-8')).hexdigest()
            
            # Find the corresponding public key
            public_key = ''
            for key_pair in self.key_pairs.values():
                if key_pair.private_key == private_key:
                    public_key = key_pair.public_key
                    break
            
            # Create signature object
            signature = Signature(
                signature=signature_hash,
                algorithm=algorithm,
                public_key=public_key,
                message_hash=message_hash
            )
            
            return {
                'success': True,
                'signature': signature_hash,
                'message_hash': message_hash,
                'public_key': public_key,
                'signature_object': signature.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def verify_signature(self, message: str, signature: str, public_key: str,
                        algorithm: str = 'simple') -> Dict[str, Any]:
        """Verify a message signature"""
        try:
            # Hash the message
            message_hash = hashlib.sha256(message.encode('utf-8')).hexdigest()
            
            # Find the private key that corresponds to this public key
            private_key = None
            for key_pair in self.key_pairs.values():
                if key_pair.public_key == public_key:
                    private_key = key_pair.private_key
                    break
            
            if not private_key:
                return {'success': False, 'error': 'Public key not found in known key pairs'}
            
            # Recreate expected signature
            signature_data = f"{message_hash}:{private_key}:{algorithm}"
            expected_signature = hashlib.sha256(signature_data.encode('utf-8')).hexdigest()
            
            is_valid = (signature == expected_signature)
            
            return {
                'success': True,
                'is_valid': is_valid,
                'message_hash': message_hash
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_key_pair(self, key_id: str) -> Optional[KeyPair]:
        """Get key pair by ID"""
        return self.key_pairs.get(key_id)
    
    def list_key_pairs(self, include_private: bool = False) -> List[Dict[str, Any]]:
        """List all key pairs"""
        key_list: List[Dict[str, Any]] = []
        for key_id, key_pair in self.key_pairs.items():
            key_data = key_pair.to_dict()
            key_data['key_id'] = key_id
            
            if not include_private:
                # Remove private key for security
                key_data.pop('private_key', None)
            
            key_list.append(key_data)
        return key_list
    
    def export_public_key(self, key_id: str) -> Optional[str]:
        """Export just the public key"""
        key_pair = self.key_pairs.get(key_id)
        return key_pair.public_key if key_pair else None


# Utility functions
def create_crypto_manager() -> SimpleCryptoManager:
    """Create a new crypto manager instance"""
    return SimpleCryptoManager()


def hash_message(message: str, algorithm: str = 'sha256') -> str:
    """Hash a message with specified algorithm"""
    message_bytes = message.encode('utf-8')
    
    if algorithm == 'sha256':
        return hashlib.sha256(message_bytes).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(message_bytes).hexdigest()
    elif algorithm == 'md5':
        return hashlib.md5(message_bytes).hexdigest()
    elif algorithm == 'sha512':
        return hashlib.sha512(message_bytes).hexdigest()
    else:
        return hashlib.sha256(message_bytes).hexdigest()


def generate_random_bytes(length: int = 32) -> bytes:
    """Generate cryptographically secure random bytes"""
    return secrets.token_bytes(length)


def generate_random_string(length: int = 32) -> str:
    """Generate cryptographically secure random hex string"""
    return secrets.token_hex(length)


def generate_random_token(length: int = 32) -> str:
    """Generate cryptographically secure random URL-safe token"""
    return secrets.token_urlsafe(length)


def encode_base64(data: bytes) -> str:
    """Encode bytes to base64 string"""
    return base64.b64encode(data).decode('utf-8')


def decode_base64(data: str) -> bytes:
    """Decode base64 string to bytes"""
    return base64.b64decode(data.encode('utf-8'))


def secure_compare(a: str, b: str) -> bool:
    """Securely compare two strings (timing attack resistant)"""
    return secrets.compare_digest(a, b)


def check_crypto_availability() -> Dict[str, bool]:
    """Check availability of cryptographic algorithms"""
    return {
        'simple': True,
        'sha256': True,
        'sha1': True,
        'sha512': True,
        'md5': True,
        'base64': True,
        'secure_random': True
    }


# Quick utility functions
def quick_hash(data: str) -> str:
    """Quick SHA-256 hash"""
    return hash_message(data, 'sha256')


def quick_sign(message: str, crypto_manager: Optional[SimpleCryptoManager] = None) -> Dict[str, Any]:
    """Quick message signing with temporary keys"""
    if crypto_manager is None:
        crypto_manager = create_crypto_manager()
    
    # Generate temporary key pair
    key_result = crypto_manager.generate_key_pair()
    if not key_result['success']:
        return key_result
    
    key_pair = KeyPair.from_dict(key_result['key_pair'])
    
    # Sign message
    return crypto_manager.sign_message(message, key_pair.private_key)


def quick_verify(message: str, signature_data: Dict[str, Any], 
                crypto_manager: Optional[SimpleCryptoManager] = None) -> Dict[str, Any]:
    """Quick signature verification"""
    if crypto_manager is None:
        crypto_manager = create_crypto_manager()
    
    return crypto_manager.verify_signature(
        message,
        signature_data['signature'],
        signature_data['public_key'],
        signature_data.get('algorithm', 'simple')
    )