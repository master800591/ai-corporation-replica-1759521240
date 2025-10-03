#!/usr/bin/env python3
"""
Crypto Utilities

Simple cryptographic utilities for:
- Key generation 
- Digital signatures
- Message hashing
- Base64 encoding/decoding
"""

import hashlib
import secrets
import base64
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Try to import cryptography library for RSA support
try:
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


@dataclass
class KeyPair:
    """Cryptographic key pair"""
    private_key: str
    public_key: str
    algorithm: str = 'simple'
    key_format: str = 'pem'
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


class CryptoManager:
    """Cryptographic operations manager"""
    
    def __init__(self):
        """Initialize crypto manager"""
        self.backend_available = CRYPTO_AVAILABLE
        self.key_pairs: Dict[str, KeyPair] = {}
    
    def generate_simple_key_pair(self) -> Dict[str, Any]:
        """Generate a simple key pair (for demonstration)"""
        try:
            # Generate simple keys using random data
            private_seed = secrets.token_bytes(32)
            public_seed = hashlib.sha256(private_seed).digest()
            
            private_key = base64.b64encode(private_seed).decode('utf-8')
            public_key = base64.b64encode(public_seed).decode('utf-8')
            
            key_pair = KeyPair(
                private_key=private_key,
                public_key=public_key,
                algorithm='simple',
                key_format='base64'
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
    
    def generate_rsa_key_pair(self) -> Dict[str, Any]:
        """Generate RSA key pair using cryptography library"""
        if not self.backend_available:
            return {'success': False, 'error': 'cryptography library not available'}
        
        try:
            # Generate RSA private key
            private_key_obj = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Serialize private key
            private_pem = private_key_obj.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            
            # Serialize public key
            public_pem = private_key_obj.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
            
            key_pair = KeyPair(
                private_key=private_pem,
                public_key=public_pem,
                algorithm='rsa',
                key_format='pem'
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
    
    def generate_key_pair(self, algorithm: str = 'simple') -> Dict[str, Any]:
        """Generate a key pair with specified algorithm"""
        if algorithm == 'rsa' and self.backend_available:
            return self.generate_rsa_key_pair()
        else:
            return self.generate_simple_key_pair()
    
    def sign_message_simple(self, message: str, private_key: str) -> Dict[str, Any]:
        """Sign message using simple algorithm"""
        try:
            # Hash the message
            message_hash = hashlib.sha256(message.encode('utf-8')).hexdigest()
            
            # Create signature by combining message hash with private key
            signature_data = f"{message_hash}:{private_key}"
            signature_hash = hashlib.sha256(signature_data.encode('utf-8')).hexdigest()
            
            return {
                'success': True,
                'signature': signature_hash,
                'message_hash': message_hash
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def sign_message_rsa(self, message: str, private_key_pem: str) -> Dict[str, Any]:
        """Sign message using RSA algorithm"""
        if not self.backend_available:
            return {'success': False, 'error': 'cryptography library not available'}
        
        try:
            # Load private key
            private_key_obj = serialization.load_pem_private_key(
                private_key_pem.encode('utf-8'),
                password=None,
                backend=default_backend()
            )
            
            # Sign the message
            message_bytes = message.encode('utf-8')
            signature_bytes = private_key_obj.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            signature_b64 = base64.b64encode(signature_bytes).decode('utf-8')
            message_hash = hashlib.sha256(message_bytes).hexdigest()
            
            # Get public key for verification
            public_key_pem = private_key_obj.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
            
            return {
                'success': True,
                'signature': signature_b64,
                'message_hash': message_hash,
                'public_key': public_key_pem
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def sign_message(self, message: str, private_key: str, 
                    algorithm: str = 'simple') -> Dict[str, Any]:
        """Sign a message with specified algorithm"""
        if algorithm == 'rsa' and self.backend_available:
            result = self.sign_message_rsa(message, private_key)
        else:
            result = self.sign_message_simple(message, private_key)
        
        if result['success']:
            # Find the corresponding key pair for public key
            public_key = None
            for key_pair in self.key_pairs.values():
                if key_pair.private_key == private_key:
                    public_key = key_pair.public_key
                    break
            
            # Create signature object
            signature = Signature(
                signature=result['signature'],
                algorithm=algorithm,
                public_key=public_key or result.get('public_key', ''),
                message_hash=result['message_hash']
            )
            
            result['signature_object'] = signature.to_dict()
        
        return result
    
    def verify_signature_simple(self, message: str, signature: str, 
                               public_key: str) -> Dict[str, Any]:
        """Verify signature using simple algorithm"""
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
                # Try to derive private key from public key (simple algorithm only)
                try:
                    public_bytes = base64.b64decode(public_key.encode('utf-8'))
                    # This is a simplified reverse operation - not secure for real use
                    private_key = public_key  # Placeholder
                except:
                    return {'success': False, 'error': 'Cannot verify signature'}
            
            # Recreate expected signature
            signature_data = f"{message_hash}:{private_key}"
            expected_signature = hashlib.sha256(signature_data.encode('utf-8')).hexdigest()
            
            is_valid = (signature == expected_signature)
            
            return {
                'success': True,
                'is_valid': is_valid,
                'message_hash': message_hash
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def verify_signature_rsa(self, message: str, signature: str, 
                           public_key_pem: str) -> Dict[str, Any]:
        """Verify signature using RSA algorithm"""
        if not self.backend_available:
            return {'success': False, 'error': 'cryptography library not available'}
        
        try:
            # Load public key
            public_key_obj = serialization.load_pem_public_key(
                public_key_pem.encode('utf-8'),
                backend=default_backend()
            )
            
            # Decode signature
            signature_bytes = base64.b64decode(signature.encode('utf-8'))
            message_bytes = message.encode('utf-8')
            
            # Verify signature
            try:
                public_key_obj.verify(
                    signature_bytes,
                    message_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                is_valid = True
            except:
                is_valid = False
            
            message_hash = hashlib.sha256(message_bytes).hexdigest()
            
            return {
                'success': True,
                'is_valid': is_valid,
                'message_hash': message_hash
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def verify_signature(self, message: str, signature: str, public_key: str,
                        algorithm: str = 'simple') -> Dict[str, Any]:
        """Verify a message signature"""
        if algorithm == 'rsa' and self.backend_available:
            return self.verify_signature_rsa(message, signature, public_key)
        else:
            return self.verify_signature_simple(message, signature, public_key)
    
    def get_key_pair(self, key_id: str) -> Optional[KeyPair]:
        """Get key pair by ID"""
        return self.key_pairs.get(key_id)
    
    def list_key_pairs(self) -> List[Dict[str, Any]]:
        """List all key pairs (without private keys)"""
        key_list: List[Dict[str, Any]] = []
        for key_id, key_pair in self.key_pairs.items():
            key_data = key_pair.to_dict()
            key_data['key_id'] = key_id
            # Remove private key for security
            key_data.pop('private_key', None)
            key_list.append(key_data)
        return key_list


# Utility functions
def create_crypto_manager() -> CryptoManager:
    """Create a new crypto manager instance"""
    return CryptoManager()


def hash_message(message: str, algorithm: str = 'sha256') -> str:
    """Hash a message with specified algorithm"""
    if algorithm == 'sha256':
        return hashlib.sha256(message.encode('utf-8')).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(message.encode('utf-8')).hexdigest()
    elif algorithm == 'md5':
        return hashlib.md5(message.encode('utf-8')).hexdigest()
    else:
        return hashlib.sha256(message.encode('utf-8')).hexdigest()


def generate_random_bytes(length: int = 32) -> bytes:
    """Generate cryptographically secure random bytes"""
    return secrets.token_bytes(length)


def generate_random_string(length: int = 32) -> str:
    """Generate cryptographically secure random string"""
    return secrets.token_hex(length)


def encode_base64(data: bytes) -> str:
    """Encode bytes to base64 string"""
    return base64.b64encode(data).decode('utf-8')


def decode_base64(data: str) -> bytes:
    """Decode base64 string to bytes"""
    return base64.b64decode(data.encode('utf-8'))


def check_crypto_availability() -> Dict[str, bool]:
    """Check availability of cryptographic algorithms"""
    return {
        'simple': True,
        'rsa': CRYPTO_AVAILABLE,
        'cryptography_lib': CRYPTO_AVAILABLE
    }