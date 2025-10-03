#!/usr/bin/env python3
"""
Blockchain Wallet System

This module provides wallet functionality for:
- Key generation and management
- Transaction signing and verification
- Balance management
- Address generation
"""

import hashlib
import secrets
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Simplified crypto - using basic hashing only
CRYPTO_AVAILABLE = False

from .core import Transaction
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class WalletInfo:
    """Wallet information"""
    address: str
    public_key_hex: str
    balance: float
    nonce: int
    created_at: float
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary"""
        return asdict(self)

class Wallet:
    """Blockchain wallet for managing keys and transactions"""
    
    def __init__(self, private_key: Optional[str] = None):
        """
        Initialize wallet
        
        Args:
            private_key: Optional private key hex string
        """
        self.balance = 0.0
        self.nonce = 0
        self.transaction_history: List[Transaction] = []
        
        # Use simplified initialization
        self._init_simple(private_key)
        
        logger.info(f"[INIT] Wallet initialized with address: {self.address}")
    
    def _init_simple(self, private_key_hex: Optional[str] = None):
        """Initialize wallet with simple key generation"""
        if private_key_hex:
            self.private_key_hex = private_key_hex
        else:
            # Generate random private key
            self.private_key_hex = secrets.token_hex(32)
        
        # Generate public key (simplified - just hash of private key)
        public_key_hash = hashlib.sha256(self.private_key_hex.encode()).hexdigest()
        self.public_key_hex = public_key_hash
        
        # Generate address
        address_hash = hashlib.sha256(public_key_hash.encode()).hexdigest()
        self.address = f"addr_{address_hash[:20]}"
        
        logger.info("[OK] Wallet initialized with simplified crypto")
        """Initialize wallet without cryptography library (simplified)"""
        if private_key_hex:
            self.private_key_hex = private_key_hex
        else:
            # Generate random private key
            self.private_key_hex = secrets.token_hex(32)
        
        # Generate public key (simplified - just hash of private key)
        public_key_hash = hashlib.sha256(self.private_key_hex.encode()).hexdigest()
        self.public_key_hex = public_key_hash
        
        # Generate address
        address_hash = hashlib.sha256(public_key_hash.encode()).hexdigest()
        self.address = f"addr_{address_hash[:20]}"
        
        logger.warning("[WARNING] Using simplified crypto - not for production use")
    
    def sign_transaction(self, transaction: Transaction) -> str:
        """Sign a transaction with simple method"""
        # Create transaction string
        tx_string = f"{transaction.sender}{transaction.recipient}{transaction.amount}{transaction.nonce}"
        
        # Create simple signature (hash with private key)
        signature_input = tx_string + self.private_key_hex
        signature = hashlib.sha256(signature_input.encode()).hexdigest()
        
        return signature
    
    def _sign_transaction_simple(self, transaction: Transaction) -> str:
        """Simple transaction signing without cryptography library"""
        # Create transaction string
        tx_string = f"{transaction.sender}{transaction.recipient}{transaction.amount}{transaction.nonce}"
        
        # Create simple signature (hash with private key)
        signature_input = tx_string + self.private_key_hex
        signature = hashlib.sha256(signature_input.encode()).hexdigest()
        
        return signature
    
    def verify_signature(self, transaction: Transaction, signature: str, public_key_hex: str) -> bool:
        """Verify a transaction signature (simplified)"""
        # Simple verification - check signature format
        return len(signature) == 64 and signature.isalnum()
    
    def _verify_signature_simple(self, transaction: Transaction, signature: str, public_key_hex: str) -> bool:
        """Simple signature verification"""
        # This is a simplified verification - not cryptographically secure
        return len(signature) == 64 and signature.isalnum()
    
    def create_transaction(self, 
                          recipient: str, 
                          amount: float, 
                          fee: float = 0.01,
                          data: Optional[Dict[str, Any]] = None) -> Transaction:
        """Create and sign a transaction"""
        if self.balance < amount + fee:
            raise ValueError("Insufficient balance")
        
        transaction = Transaction(
            sender=self.address,
            recipient=recipient,
            amount=amount,
            fee=fee,
            nonce=self.nonce,
            timestamp=time.time(),
            data=data or {}
        )
        
        # Sign the transaction
        signature = self.sign_transaction(transaction)
        transaction.signature = signature
        
        # Update local state
        self.nonce += 1
        self.balance -= (amount + fee)
        self.transaction_history.append(transaction)
        
        logger.info(f"[OK] Created transaction {transaction.tx_hash}")
        return transaction
    
    def update_balance(self, new_balance: float):
        """Update wallet balance"""
        self.balance = new_balance
        logger.debug(f"[UPDATE] Balance updated to {self.balance}")
    
    def update_nonce(self, new_nonce: int):
        """Update wallet nonce"""
        self.nonce = new_nonce
        logger.debug(f"[UPDATE] Nonce updated to {self.nonce}")
    
    def add_received_transaction(self, transaction: Transaction):
        """Add a received transaction to history"""
        self.transaction_history.append(transaction)
        logger.debug(f"[RECEIVED] Transaction {transaction.tx_hash}")
    
    def get_transaction_history(self, limit: int = 50) -> List[Transaction]:
        """Get transaction history"""
        return self.transaction_history[-limit:]
    
    def export_private_key(self) -> str:
        """Export private key (be careful with this!)"""
        logger.warning("[WARNING] Private key exported - keep secure!")
        return self.private_key_hex
    
    def export_public_key(self) -> str:
        """Export public key"""
        return self.public_key_hex
    
    def get_wallet_info(self) -> WalletInfo:
        """Get wallet information"""
        return WalletInfo(
            address=self.address,
            public_key_hex=self.public_key_hex,
            balance=self.balance,
            nonce=self.nonce,
            created_at=time.time()
        )
    
    def backup_wallet(self) -> Dict[str, Any]:
        """Create wallet backup"""
        return {
            'address': self.address,
            'private_key_hex': self.private_key_hex,
            'public_key_hex': self.public_key_hex,
            'balance': self.balance,
            'nonce': self.nonce,
            'transaction_count': len(self.transaction_history),
            'backup_timestamp': time.time()
        }
    
    @classmethod
    def restore_wallet(cls, backup_data: Dict[str, Any]) -> 'Wallet':
        """Restore wallet from backup"""
        wallet = cls(backup_data['private_key_hex'])
        wallet.balance = backup_data.get('balance', 0.0)
        wallet.nonce = backup_data.get('nonce', 0)
        
        logger.info(f"[RESTORE] Wallet restored: {wallet.address}")
        return wallet

class WalletManager:
    """Manager for multiple wallets"""
    
    def __init__(self):
        """Initialize wallet manager"""
        self.wallets: Dict[str, Wallet] = {}
        self.default_wallet: Optional[str] = None
        
        logger.info("[INIT] Wallet manager initialized")
    
    def create_wallet(self, wallet_id: Optional[str] = None) -> str:
        """Create a new wallet"""
        wallet = Wallet()
        
        if not wallet_id:
            wallet_id = f"wallet_{int(time.time())}"
        
        self.wallets[wallet_id] = wallet
        
        if not self.default_wallet:
            self.default_wallet = wallet_id
        
        logger.info(f"[OK] Wallet {wallet_id} created with address {wallet.address}")
        return wallet_id
    
    def import_wallet(self, private_key_hex: str, wallet_id: Optional[str] = None) -> str:
        """Import wallet from private key"""
        wallet = Wallet(private_key_hex)
        
        if not wallet_id:
            wallet_id = f"imported_{int(time.time())}"
        
        self.wallets[wallet_id] = wallet
        
        if not self.default_wallet:
            self.default_wallet = wallet_id
        
        logger.info(f"[OK] Wallet {wallet_id} imported with address {wallet.address}")
        return wallet_id
    
    def get_wallet(self, wallet_id: Optional[str] = None) -> Optional[Wallet]:
        """Get wallet by ID"""
        if not wallet_id:
            wallet_id = self.default_wallet
        
        return self.wallets.get(wallet_id)
    
    def list_wallets(self) -> List[Dict[str, Any]]:
        """List all wallets"""
        wallets = []
        for wallet_id, wallet in self.wallets.items():
            info = wallet.get_wallet_info().to_dict()
            info['wallet_id'] = wallet_id
            info['is_default'] = wallet_id == self.default_wallet
            wallets.append(info)
        
        return wallets
    
    def set_default_wallet(self, wallet_id: str) -> bool:
        """Set default wallet"""
        if wallet_id not in self.wallets:
            return False
        
        self.default_wallet = wallet_id
        logger.info(f"[OK] Default wallet set to {wallet_id}")
        return True
    
    def remove_wallet(self, wallet_id: str) -> bool:
        """Remove a wallet"""
        if wallet_id not in self.wallets:
            return False
        
        del self.wallets[wallet_id]
        
        if self.default_wallet == wallet_id:
            self.default_wallet = next(iter(self.wallets.keys())) if self.wallets else None
        
        logger.info(f"[OK] Wallet {wallet_id} removed")
        return True
    
    def backup_all_wallets(self) -> Dict[str, any]:
        """Backup all wallets"""
        backup = {
            'wallets': {},
            'default_wallet': self.default_wallet,
            'backup_timestamp': time.time()
        }
        
        for wallet_id, wallet in self.wallets.items():
            backup['wallets'][wallet_id] = wallet.backup_wallet()
        
        logger.info(f"[OK] Backed up {len(self.wallets)} wallets")
        return backup
    
    def restore_all_wallets(self, backup_data: Dict[str, any]) -> bool:
        """Restore all wallets from backup"""
        try:
            self.wallets = {}
            
            for wallet_id, wallet_backup in backup_data['wallets'].items():
                wallet = Wallet.restore_wallet(wallet_backup)
                self.wallets[wallet_id] = wallet
            
            self.default_wallet = backup_data.get('default_wallet')
            
            logger.info(f"[OK] Restored {len(self.wallets)} wallets")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to restore wallets: {e}")
            return False

# Address utilities
def generate_random_address() -> str:
    """Generate a random blockchain address"""
    random_bytes = secrets.token_bytes(20)
    return f"addr_{random_bytes.hex()}"

def is_valid_address(address: str) -> bool:
    """Check if an address is valid"""
    if not address.startswith("addr_"):
        return False
    
    hex_part = address[5:]
    if len(hex_part) != 40:  # 20 bytes = 40 hex chars
        return False
    
    try:
        int(hex_part, 16)
        return True
    except ValueError:
        return False

def calculate_transaction_fee(transaction_size: int, fee_rate: float = 0.001) -> float:
    """Calculate transaction fee based on size"""
    return max(0.001, transaction_size * fee_rate)

if __name__ == "__main__":
    # Demo wallet functionality
    manager = WalletManager()
    
    # Create wallets
    wallet1_id = manager.create_wallet("alice")
    wallet2_id = manager.create_wallet("bob")
    
    wallet1 = manager.get_wallet(wallet1_id)
    wallet2 = manager.get_wallet(wallet2_id)
    
    print(f"Alice address: {wallet1.address}")
    print(f"Bob address: {wallet2.address}")
    
    # Simulate receiving some coins
    wallet1.update_balance(100.0)
    
    # Create transaction
    try:
        tx = wallet1.create_transaction(wallet2.address, 25.0, 0.5)
        print(f"Transaction created: {tx.tx_hash}")
        print(f"Alice balance after tx: {wallet1.balance}")
    except ValueError as e:
        print(f"Transaction failed: {e}")
    
    # Export and import wallet
    backup = wallet1.backup_wallet()
    print(f"Wallet backup created")
    
    # Create new wallet from backup
    restored_wallet = Wallet.restore_wallet(backup)
    print(f"Wallet restored: {restored_wallet.address}")