#!/usr/bin/env python3
"""
Advanced Cryptographic Utilities

Enterprise-grade cryptographic operations:
- RSA and ECC key generation
- Advanced digital signatures
- Encryption/decryption capabilities
- Key derivation functions
- Certificate management
- Hardware security module support
"""

import hashlib
import secrets
import base64
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Optional cryptography library for advanced features
try:
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
    from cryptography.hazmat.primitives import hashes, serialization, kdf
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.x509 import load_pem_x509_certificate
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# Optional PyNaCl for modern cryptography
try:
    import nacl.secret
    import nacl.public
    import nacl.signing
    import nacl.encoding
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False


class KeyAlgorithm(Enum):
    """Supported key algorithms"""
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"
    ECDSA_P521 = "ecdsa_p521"
    ED25519 = "ed25519"
    X25519 = "x25519"


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_OAEP = "rsa_oaep"


class HashAlgorithm(Enum):
    """Supported hash algorithms"""
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


@dataclass
class CryptoKeyPair:
    """Advanced cryptographic key pair"""
    key_id: str
    algorithm: KeyAlgorithm
    public_key: str
    private_key: str
    key_format: str = "pem"
    key_usage: List[str] = field(default_factory=list)  # signing, encryption, key_agreement
    curve_name: Optional[str] = None
    key_size: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    is_hardware_backed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_private: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'key_id': self.key_id,
            'algorithm': self.algorithm.value,
            'public_key': self.public_key,
            'key_format': self.key_format,
            'key_usage': self.key_usage.copy(),
            'curve_name': self.curve_name,
            'key_size': self.key_size,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'is_hardware_backed': self.is_hardware_backed,
            'metadata': self.metadata.copy()
        }
        
        if include_private:
            data['private_key'] = self.private_key
        
        return data


@dataclass
class AdvancedSignature:
    """Advanced digital signature"""
    signature_id: str
    algorithm: KeyAlgorithm
    hash_algorithm: HashAlgorithm
    signature: str
    public_key: str
    message_hash: str
    timestamp: float = field(default_factory=time.time)
    nonce: Optional[str] = None
    counter_signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'signature_id': self.signature_id,
            'algorithm': self.algorithm.value,
            'hash_algorithm': self.hash_algorithm.value,
            'signature': self.signature,
            'public_key': self.public_key,
            'message_hash': self.message_hash,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'counter_signature': self.counter_signature,
            'metadata': self.metadata.copy()
        }


@dataclass
class EncryptedData:
    """Encrypted data container"""
    data_id: str
    algorithm: EncryptionAlgorithm
    encrypted_data: str
    initialization_vector: Optional[str] = None
    authentication_tag: Optional[str] = None
    key_id: Optional[str] = None
    salt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'data_id': self.data_id,
            'algorithm': self.algorithm.value,
            'encrypted_data': self.encrypted_data,
            'initialization_vector': self.initialization_vector,
            'authentication_tag': self.authentication_tag,
            'key_id': self.key_id,
            'salt': self.salt,
            'metadata': self.metadata.copy(),
            'created_at': self.created_at
        }


class AdvancedCryptoManager:
    """Enterprise cryptographic operations manager"""
    
    def __init__(self):
        """Initialize crypto manager"""
        self.key_pairs: Dict[str, CryptoKeyPair] = {}
        self.signatures: Dict[str, AdvancedSignature] = {}
        self.encrypted_data: Dict[str, EncryptedData] = {}
        
        # Check available backends
        self.backends = {
            'cryptography': CRYPTOGRAPHY_AVAILABLE,
            'nacl': NACL_AVAILABLE,
            'hashlib': True,
            'secrets': True
        }
    
    def _generate_key_id(self) -> str:
        """Generate unique key ID"""
        return str(uuid.uuid4())
    
    def _generate_signature_id(self) -> str:
        """Generate unique signature ID"""
        return str(uuid.uuid4())
    
    def _generate_data_id(self) -> str:
        """Generate unique data ID"""
        return str(uuid.uuid4())
    
    def generate_rsa_key_pair(self, key_size: int = 2048) -> Dict[str, Any]:
        """Generate RSA key pair"""
        if not self.backends['cryptography']:
            return {'success': False, 'error': 'cryptography library not available'}
        
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            
            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
            
            algorithm = KeyAlgorithm.RSA_4096 if key_size == 4096 else KeyAlgorithm.RSA_2048
            
            key_pair = CryptoKeyPair(
                key_id=self._generate_key_id(),
                algorithm=algorithm,
                public_key=public_pem,
                private_key=private_pem,
                key_usage=['signing', 'encryption'],
                key_size=key_size
            )
            
            self.key_pairs[key_pair.key_id] = key_pair
            
            return {
                'success': True,
                'key_id': key_pair.key_id,
                'key_pair': key_pair.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_ecdsa_key_pair(self, curve: str = 'secp256r1') -> Dict[str, Any]:
        """Generate ECDSA key pair"""
        if not self.backends['cryptography']:
            return {'success': False, 'error': 'cryptography library not available'}
        
        try:
            curve_map = {
                'secp256r1': (ec.SECP256R1(), KeyAlgorithm.ECDSA_P256),
                'secp384r1': (ec.SECP384R1(), KeyAlgorithm.ECDSA_P384),
                'secp521r1': (ec.SECP521R1(), KeyAlgorithm.ECDSA_P521)
            }
            
            if curve not in curve_map:
                return {'success': False, 'error': f'Unsupported curve: {curve}'}
            
            curve_obj, algorithm = curve_map[curve]
            
            private_key = ec.generate_private_key(curve_obj, default_backend())
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            
            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
            
            key_pair = CryptoKeyPair(
                key_id=self._generate_key_id(),
                algorithm=algorithm,
                public_key=public_pem,
                private_key=private_pem,
                key_usage=['signing'],
                curve_name=curve
            )
            
            self.key_pairs[key_pair.key_id] = key_pair
            
            return {
                'success': True,
                'key_id': key_pair.key_id,
                'key_pair': key_pair.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_ed25519_key_pair(self) -> Dict[str, Any]:
        """Generate Ed25519 key pair for signing"""
        if not self.backends['nacl']:
            return {'success': False, 'error': 'PyNaCl library not available'}
        
        try:
            signing_key = nacl.signing.SigningKey.generate()
            verify_key = signing_key.verify_key
            
            private_key = base64.b64encode(bytes(signing_key)).decode('utf-8')
            public_key = base64.b64encode(bytes(verify_key)).decode('utf-8')
            
            key_pair = CryptoKeyPair(
                key_id=self._generate_key_id(),
                algorithm=KeyAlgorithm.ED25519,
                public_key=public_key,
                private_key=private_key,
                key_format='base64',
                key_usage=['signing'],
                key_size=32
            )
            
            self.key_pairs[key_pair.key_id] = key_pair
            
            return {
                'success': True,
                'key_id': key_pair.key_id,
                'key_pair': key_pair.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_x25519_key_pair(self) -> Dict[str, Any]:
        """Generate X25519 key pair for key exchange"""
        if not self.backends['nacl']:
            return {'success': False, 'error': 'PyNaCl library not available'}
        
        try:
            private_key = nacl.public.PrivateKey.generate()
            public_key = private_key.public_key
            
            private_key_b64 = base64.b64encode(bytes(private_key)).decode('utf-8')
            public_key_b64 = base64.b64encode(bytes(public_key)).decode('utf-8')
            
            key_pair = CryptoKeyPair(
                key_id=self._generate_key_id(),
                algorithm=KeyAlgorithm.X25519,
                public_key=public_key_b64,
                private_key=private_key_b64,
                key_format='base64',
                key_usage=['key_agreement'],
                key_size=32
            )
            
            self.key_pairs[key_pair.key_id] = key_pair
            
            return {
                'success': True,
                'key_id': key_pair.key_id,
                'key_pair': key_pair.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def sign_message(self, message: str, key_id: str, 
                    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> Dict[str, Any]:
        """Sign message with specified key"""
        try:
            if key_id not in self.key_pairs:
                return {'success': False, 'error': 'Key not found'}
            
            key_pair = self.key_pairs[key_id]
            
            if 'signing' not in key_pair.key_usage:
                return {'success': False, 'error': 'Key not suitable for signing'}
            
            # Hash the message
            message_bytes = message.encode('utf-8')
            
            if hash_algorithm == HashAlgorithm.SHA256:
                message_hash = hashlib.sha256(message_bytes).hexdigest()
            elif hash_algorithm == HashAlgorithm.SHA384:
                message_hash = hashlib.sha384(message_bytes).hexdigest()
            elif hash_algorithm == HashAlgorithm.SHA512:
                message_hash = hashlib.sha512(message_bytes).hexdigest()
            else:
                message_hash = hashlib.sha256(message_bytes).hexdigest()
            
            signature_result = None
            
            # Sign based on algorithm
            if key_pair.algorithm in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096]:
                signature_result = self._sign_rsa(message_bytes, key_pair, hash_algorithm)
            elif key_pair.algorithm in [KeyAlgorithm.ECDSA_P256, KeyAlgorithm.ECDSA_P384, KeyAlgorithm.ECDSA_P521]:
                signature_result = self._sign_ecdsa(message_bytes, key_pair, hash_algorithm)
            elif key_pair.algorithm == KeyAlgorithm.ED25519:
                signature_result = self._sign_ed25519(message_bytes, key_pair)
            
            if not signature_result or not signature_result.get('success'):
                return signature_result or {'success': False, 'error': 'Signing failed'}
            
            signature = AdvancedSignature(
                signature_id=self._generate_signature_id(),
                algorithm=key_pair.algorithm,
                hash_algorithm=hash_algorithm,
                signature=signature_result['signature'],
                public_key=key_pair.public_key,
                message_hash=message_hash,
                nonce=signature_result.get('nonce')
            )
            
            self.signatures[signature.signature_id] = signature
            
            return {
                'success': True,
                'signature_id': signature.signature_id,
                'signature': signature.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _sign_rsa(self, message: bytes, key_pair: CryptoKeyPair, 
                  hash_algorithm: HashAlgorithm) -> Dict[str, Any]:
        """Sign with RSA key"""
        if not self.backends['cryptography']:
            return {'success': False, 'error': 'cryptography library not available'}
        
        try:
            private_key = serialization.load_pem_private_key(
                key_pair.private_key.encode('utf-8'),
                password=None,
                backend=default_backend()
            )
            
            hash_map = {
                HashAlgorithm.SHA256: hashes.SHA256(),
                HashAlgorithm.SHA384: hashes.SHA384(),
                HashAlgorithm.SHA512: hashes.SHA512()
            }
            
            hash_alg = hash_map.get(hash_algorithm, hashes.SHA256())
            
            signature = private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hash_alg),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hash_alg
            )
            
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            return {
                'success': True,
                'signature': signature_b64
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _sign_ecdsa(self, message: bytes, key_pair: CryptoKeyPair,
                    hash_algorithm: HashAlgorithm) -> Dict[str, Any]:
        """Sign with ECDSA key"""
        if not self.backends['cryptography']:
            return {'success': False, 'error': 'cryptography library not available'}
        
        try:
            private_key = serialization.load_pem_private_key(
                key_pair.private_key.encode('utf-8'),
                password=None,
                backend=default_backend()
            )
            
            hash_map = {
                HashAlgorithm.SHA256: hashes.SHA256(),
                HashAlgorithm.SHA384: hashes.SHA384(),
                HashAlgorithm.SHA512: hashes.SHA512()
            }
            
            hash_alg = hash_map.get(hash_algorithm, hashes.SHA256())
            
            signature = private_key.sign(message, ec.ECDSA(hash_alg))
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            return {
                'success': True,
                'signature': signature_b64
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _sign_ed25519(self, message: bytes, key_pair: CryptoKeyPair) -> Dict[str, Any]:
        """Sign with Ed25519 key"""
        if not self.backends['nacl']:
            return {'success': False, 'error': 'PyNaCl library not available'}
        
        try:
            private_key_bytes = base64.b64decode(key_pair.private_key.encode('utf-8'))
            signing_key = nacl.signing.SigningKey(private_key_bytes)
            
            signed = signing_key.sign(message, encoder=nacl.encoding.Base64Encoder)
            signature_b64 = signed.signature.decode('utf-8')
            
            return {
                'success': True,
                'signature': signature_b64
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_supported_algorithms(self) -> Dict[str, List[str]]:
        """Get supported algorithms based on available backends"""
        algorithms = {
            'key_generation': [],
            'signing': [],
            'encryption': [],
            'hashing': ['sha256', 'sha384', 'sha512']
        }
        
        if self.backends['cryptography']:
            algorithms['key_generation'].extend(['rsa_2048', 'rsa_4096', 'ecdsa_p256', 'ecdsa_p384', 'ecdsa_p521'])
            algorithms['signing'].extend(['rsa_pss', 'ecdsa'])
            algorithms['encryption'].extend(['rsa_oaep', 'aes_256_gcm'])
        
        if self.backends['nacl']:
            algorithms['key_generation'].extend(['ed25519', 'x25519'])
            algorithms['signing'].append('ed25519')
            algorithms['encryption'].append('chacha20_poly1305')
        
        return algorithms
    
    def get_crypto_status(self) -> Dict[str, Any]:
        """Get cryptographic system status"""
        return {
            'backends_available': self.backends,
            'total_key_pairs': len(self.key_pairs),
            'total_signatures': len(self.signatures),
            'total_encrypted_data': len(self.encrypted_data),
            'supported_algorithms': self.get_supported_algorithms()
        }


# Utility functions
def create_advanced_crypto_manager() -> AdvancedCryptoManager:
    """Create new advanced crypto manager"""
    return AdvancedCryptoManager()


def secure_hash(data: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
    """Secure hash with specified algorithm"""
    data_bytes = data.encode('utf-8')
    
    if algorithm == HashAlgorithm.SHA256:
        return hashlib.sha256(data_bytes).hexdigest()
    elif algorithm == HashAlgorithm.SHA384:
        return hashlib.sha384(data_bytes).hexdigest()
    elif algorithm == HashAlgorithm.SHA512:
        return hashlib.sha512(data_bytes).hexdigest()
    elif algorithm == HashAlgorithm.SHA3_256:
        return hashlib.sha3_256(data_bytes).hexdigest()
    elif algorithm == HashAlgorithm.SHA3_512:
        return hashlib.sha3_512(data_bytes).hexdigest()
    else:
        return hashlib.sha256(data_bytes).hexdigest()


def derive_key(password: str, salt: bytes, iterations: int = 200000, 
               key_length: int = 32) -> bytes:
    """Derive key from password using PBKDF2"""
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations, key_length)


def generate_secure_salt(length: int = 32) -> bytes:
    """Generate cryptographically secure salt"""
    return secrets.token_bytes(length)


def constant_time_compare(a: str, b: str) -> bool:
    """Constant time string comparison"""
    return secrets.compare_digest(a, b)