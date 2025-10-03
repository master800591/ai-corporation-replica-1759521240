#!/usr/bin/env python3
"""
P2P Proof of Authority (PoA) Blockchain Module

This module implements a complete P2P blockchain system with:
- Core blockchain functionality (blocks, transactions, chain validation)
- Proof of Authority consensus mechanism
- P2P networking for distributed operation
- Smart contracts for AI model sharing and monetization
- Wallet management and cryptographic operations
- Utility functions and configuration management

The blockchain is designed specifically for AI model sharing, data monetization,
and model sharing in a peer-to-peer network.
"""

# Version compatibility check
import sys
if sys.version_info < (3, 8):
    raise ImportError("Python 3.8 or higher is required for the blockchain module")

# Optional dependency warnings
try:
    import cryptography
    CRYPTO_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn(
        "cryptography library not found. Wallet security will be reduced. "
        "Install with: pip install cryptography",
        ImportWarning
    )
    CRYPTO_AVAILABLE = False

try:
    import websockets
    import aiohttp
    P2P_AVAILABLE = True
except ImportError:
    import warnings  
    warnings.warn(
        "Optional P2P dependencies not found. Network functionality may be limited. "
        "Install with: pip install websockets aiohttp",
        ImportWarning
    )
    P2P_AVAILABLE = False

# Import core components with error handling
try:
    # Core blockchain components
    from .core import (
        Block,
        Transaction, 
        Blockchain
    )
    
    # Consensus mechanism
    from .consensus import (
        Authority,
        PoAConsensus,
        PoABlockProducer
    )
    
    # Smart contracts
    from .contracts import (
        ContractType,
        ModelInfo,
        DataSharingAgreement,
        SmartContract,
        AIModelContract,
        DataSharingContract,
        ContractManager
    )
    
    # Wallet management
    from .wallet import (
        WalletInfo,
        Wallet,
        WalletManager,
        generate_random_address,
        is_valid_address,
        calculate_transaction_fee
    )
    
    # Utilities
    from .utils import (
        get_logger,
        hash_data,
        hash_multiple,
        generate_random_id,
        generate_nonce,
        merkle_hash,
        validate_address,
        validate_transaction_data,
        validate_block_data,
        Config,
        LRUCache,
        CircularBuffer,
        BlockchainError,
        ValidationError,
        NetworkError,
        ConsensusError,
        WalletError
    )
    
    CORE_AVAILABLE = True
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some blockchain components failed to import: {e}", ImportWarning)
    CORE_AVAILABLE = False

# Import P2P networking (optional)
try:
    if P2P_AVAILABLE:
        from .network import (
            NetworkMessage,
            BlockchainP2PNode,
            create_blockchain_node
        )
    else:
        # Provide dummy classes for missing P2P functionality
        class NetworkMessage:
            pass
        class BlockchainP2PNode:
            pass
        def create_blockchain_node(*args, **kwargs):
            raise ImportError("P2P networking not available. Install websockets and aiohttp.")
except ImportError as e:
    import warnings
    warnings.warn(f"P2P networking unavailable: {e}", ImportWarning)
    
    # Provide dummy classes
    class NetworkMessage:
        pass
    class BlockchainP2PNode:
        pass
    def create_blockchain_node(*args, **kwargs):
        raise ImportError("P2P networking not available. Install websockets and aiohttp.")

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "P2P Proof of Authority Blockchain for AI Model Sharing"

# Package metadata - only include available components
__all__ = []

if CORE_AVAILABLE:
    __all__.extend([
        # Core
        "Block",
        "Transaction", 
        "Blockchain",
        
        # Consensus
        "Authority",
        "PoAConsensus",
        "PoABlockProducer",
        
        # Contracts
        "ContractType",
        "ModelInfo",
        "DataSharingAgreement", 
        "SmartContract",
        "AIModelContract",
        "DataSharingContract",
        "ContractManager",
        
        # Wallet
        "WalletInfo",
        "Wallet",
        "WalletManager",
        "generate_random_address",
        "is_valid_address", 
        "calculate_transaction_fee",
        
        # Utils
        "get_logger",
        "hash_data",
        "hash_multiple",
        "generate_random_id",
        "generate_nonce",
        "merkle_hash",
        "validate_address",
        "validate_transaction_data",
        "validate_block_data",
        "Config",
        "LRUCache", 
        "CircularBuffer",
        "BlockchainError",
        "ValidationError",
        "NetworkError",
        "ConsensusError",
        "WalletError"
    ])

# Add P2P components if available
__all__.extend([
    "NetworkMessage",
    "BlockchainP2PNode", 
    "create_blockchain_node"
])

# Module level functions for easy access
if CORE_AVAILABLE:
    def create_test_blockchain(authorities=None):
        """Create a test blockchain with default authorities"""
        if authorities is None:
            authorities = ["auth1", "auth2", "auth3"]
        return Blockchain(authorities)

    def create_test_wallet():
        """Create a test wallet with some balance"""
        wallet = Wallet()
        wallet.update_balance(1000.0)
        return wallet
else:
    def create_test_blockchain(authorities=None):
        raise ImportError("Core blockchain components not available")
    
    def create_test_wallet():
        raise ImportError("Core blockchain components not available")

# Feature flags for external code
FEATURES = {
    'core': CORE_AVAILABLE,
    'crypto': CRYPTO_AVAILABLE,
    'p2p': P2P_AVAILABLE
}