#!/usr/bin/env python3
"""
Blockchain Utilities

This module provides utility functions for:
- Cryptographic operations
- Data validation
- Network helpers
- Configuration management
- Logging setup
"""

import hashlib
import json
import time
import secrets
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Logging setup
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# Cryptographic utilities
def hash_data(data: Union[str, bytes, Dict[str, Any]]) -> str:
    """Hash data using SHA-256"""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()

def hash_multiple(*args: Union[str, bytes]) -> str:
    """Hash multiple pieces of data together"""
    hasher = hashlib.sha256()
    
    for arg in args:
        if isinstance(arg, str):
            arg = arg.encode('utf-8')
        hasher.update(arg)
    
    return hasher.hexdigest()

def generate_random_id(prefix: str = "", length: int = 16) -> str:
    """Generate a random ID"""
    random_part = secrets.token_hex(length // 2)
    return f"{prefix}{random_part}" if prefix else random_part

def generate_nonce() -> str:
    """Generate a cryptographic nonce"""
    return secrets.token_hex(16)

def merkle_hash(data_list: List[str]) -> str:
    """Calculate Merkle root hash"""
    if not data_list:
        return hash_data("")
    
    hashes = data_list.copy()
    
    while len(hashes) > 1:
        new_level = []
        for i in range(0, len(hashes), 2):
            if i + 1 < len(hashes):
                combined = hashes[i] + hashes[i + 1]
            else:
                combined = hashes[i] + hashes[i]
            new_level.append(hash_data(combined))
        hashes = new_level
    
    return hashes[0]

# Validation utilities
def validate_address(address: str) -> bool:
    """Validate blockchain address format"""
    if not isinstance(address, str):
        return False
    
    if not address.startswith("addr_"):
        return False
    
    hex_part = address[5:]
    if len(hex_part) != 40:  # 20 bytes in hex
        return False
    
    try:
        int(hex_part, 16)
        return True
    except ValueError:
        return False

def validate_transaction_data(tx_data: Dict[str, Any]) -> bool:
    """Validate transaction data structure"""
    required_fields = ['sender', 'recipient', 'amount', 'fee', 'nonce', 'timestamp']
    
    for field in required_fields:
        if field not in tx_data:
            return False
    
    # Validate types
    if not isinstance(tx_data['amount'], (int, float)) or tx_data['amount'] < 0:
        return False
    
    if not isinstance(tx_data['fee'], (int, float)) or tx_data['fee'] < 0:
        return False
    
    if not isinstance(tx_data['nonce'], int) or tx_data['nonce'] < 0:
        return False
    
    if not validate_address(tx_data['sender']) or not validate_address(tx_data['recipient']):
        return False
    
    return True

def validate_block_data(block_data: Dict[str, Any]) -> bool:
    """Validate block data structure"""
    required_fields = ['index', 'previous_hash', 'timestamp', 'transactions', 'authority']
    
    for field in required_fields:
        if field not in block_data:
            return False
    
    # Validate types
    if not isinstance(block_data['index'], int) or block_data['index'] < 0:
        return False
    
    if not isinstance(block_data['transactions'], list):
        return False
    
    # Validate each transaction
    for tx in block_data['transactions']:
        if not validate_transaction_data(tx):
            return False
    
    return True

def validate_json_data(data: str) -> bool:
    """Validate JSON data"""
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

# Network utilities
def format_peer_address(host: str, port: int) -> str:
    """Format peer address"""
    return f"{host}:{port}"

def parse_peer_address(address: str) -> tuple[str, int]:
    """Parse peer address into host and port"""
    try:
        host, port_str = address.split(':')
        port = int(port_str)
        return host, port
    except (ValueError, IndexError):
        raise ValueError(f"Invalid peer address: {address}")

def is_valid_ip(ip: str) -> bool:
    """Check if IP address is valid"""
    import socket
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False

def is_valid_port(port: int) -> bool:
    """Check if port number is valid"""
    return 1 <= port <= 65535

# File and data utilities
def save_json_file(data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """Save data to JSON file"""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to save JSON file {filepath}: {e}")
        return False

def load_json_file(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load data from JSON file"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to load JSON file {filepath}: {e}")
        return None

def ensure_directory(path: Union[str, Path]) -> bool:
    """Ensure directory exists"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to create directory {path}: {e}")
        return False

# Configuration utilities
class Config:
    """Configuration management"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration"""
        self.config_file = config_file
        self.data: Dict[str, Any] = self.load_default_config()
        
        if config_file:
            self.load_config(config_file)
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'blockchain': {
                'block_time': 5.0,
                'max_block_size': 1024 * 1024,  # 1MB
                'max_transactions_per_block': 1000,
                'min_transaction_fee': 0.001
            },
            'network': {
                'max_peers': 50,
                'connection_timeout': 30,
                'heartbeat_interval': 60,
                'sync_interval': 300
            },
            'wallet': {
                'auto_backup': True,
                'backup_interval': 3600,
                'default_fee_rate': 0.001
            },
            'consensus': {
                'authority_rotation_interval': 3600,
                'min_reputation_score': 10.0,
                'reputation_decay_rate': 0.1
            },
            'storage': {
                'data_directory': './blockchain_data',
                'max_log_file_size': 10 * 1024 * 1024,  # 10MB
                'log_retention_days': 30
            }
        }
    
    def load_config(self, config_file: str) -> bool:
        """Load configuration from file"""
        config_data = load_json_file(config_file)
        if config_data:
            self.data.update(config_data)
            return True
        return False
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """Save configuration to file"""
        file_path = config_file or self.config_file
        if file_path:
            return save_json_file(self.data, file_path)
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        data = self.data
        
        for k in keys[:-1]:
            if k not in data or not isinstance(data[k], dict):
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value

# Performance utilities
def time_function(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = get_logger(func.__module__)
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper

def format_bytes(bytes_count: int) -> str:
    """Format bytes as human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"

def format_duration(seconds: float) -> str:
    """Format duration as human readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"

# Data structures
class LRUCache:
    """Simple LRU Cache implementation"""
    
    def __init__(self, capacity: int):
        """Initialize LRU cache"""
        self.capacity = capacity
        self.cache: Dict[str, Any] = {}
        self.order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                oldest = self.order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            self.order.append(key)
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.order.clear()

class CircularBuffer:
    """Circular buffer for storing recent items"""
    
    def __init__(self, capacity: int):
        """Initialize circular buffer"""
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.index = 0
        self.full = False
    
    def append(self, item: Any):
        """Add item to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.index] = item
            self.full = True
        
        self.index = (self.index + 1) % self.capacity
    
    def get_all(self) -> List[Any]:
        """Get all items in chronological order"""
        if not self.full:
            return self.buffer.copy()
        
        return self.buffer[self.index:] + self.buffer[:self.index]
    
    def get_recent(self, count: int) -> List[Any]:
        """Get most recent items"""
        all_items = self.get_all()
        return all_items[-count:] if count <= len(all_items) else all_items
    
    def size(self) -> int:
        """Get current size"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.full

# Blockchain-specific utilities
def calculate_difficulty(target_time: float, actual_time: float, current_difficulty: int) -> int:
    """Calculate next difficulty based on block times"""
    adjustment_factor = target_time / actual_time
    
    # Limit adjustment to prevent wild swings
    adjustment_factor = max(0.25, min(4.0, adjustment_factor))
    
    new_difficulty = int(current_difficulty * adjustment_factor)
    return max(1, new_difficulty)

def is_valid_hash(hash_value: str, difficulty: int) -> bool:
    """Check if hash meets difficulty requirement"""
    if len(hash_value) != 64:  # SHA-256 produces 64 hex characters
        return False
    
    return hash_value.startswith('0' * difficulty)

def estimate_network_hashrate(difficulty: int, block_time: float) -> float:
    """Estimate network hashrate"""
    # Simplified calculation
    target_hashes = 16 ** difficulty  # Number of hashes needed on average
    return target_hashes / block_time

# Error handling utilities
class BlockchainError(Exception):
    """Base blockchain exception"""
    pass

class ValidationError(BlockchainError):
    """Validation error"""
    pass

class NetworkError(BlockchainError):
    """Network error"""
    pass

class ConsensusError(BlockchainError):
    """Consensus error"""
    pass

class WalletError(BlockchainError):
    """Wallet error"""
    pass

def handle_error(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = get_logger(func.__module__)
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

if __name__ == "__main__":
    # Demo utility functions
    print("Blockchain Utilities Demo")
    
    # Test hashing
    data = {"test": "data", "number": 123}
    hash_result = hash_data(data)
    print(f"Hash of data: {hash_result}")
    
    # Test Merkle hash
    data_list = ["tx1", "tx2", "tx3", "tx4"]
    merkle_root = merkle_hash(data_list)
    print(f"Merkle root: {merkle_root}")
    
    # Test address validation
    valid_addr = "addr_1234567890abcdef1234567890abcdef12345678"
    invalid_addr = "invalid_address"
    print(f"Valid address check: {validate_address(valid_addr)}")
    print(f"Invalid address check: {validate_address(invalid_addr)}")
    
    # Test configuration
    config = Config()
    print(f"Block time: {config.get('blockchain.block_time')}")
    print(f"Max peers: {config.get('network.max_peers')}")
    
    # Test cache
    cache = LRUCache(3)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # Should evict key1
    
    print(f"Cache key1: {cache.get('key1')}")  # Should be None
    print(f"Cache key2: {cache.get('key2')}")  # Should be value2
    
    # Test circular buffer
    buffer = CircularBuffer(3)
    for i in range(5):
        buffer.append(f"item{i}")
    
    print(f"Buffer contents: {buffer.get_all()}")
    print(f"Recent 2 items: {buffer.get_recent(2)}")
    
    print("All tests completed!")