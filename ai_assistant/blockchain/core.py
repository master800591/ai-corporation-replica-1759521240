#!/usr/bin/env python3
"""
Core Blockchain Components

This module contains the fundamental blockchain data structures:
- Block: Individual blocks in the chain
- Transaction: Individual transactions
- Blockchain: The main blockchain class
"""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Transaction:
    """Represents a blockchain transaction"""
    
    sender: str
    recipient: str
    amount: float
    fee: float
    nonce: int
    timestamp: float
    data: Dict[str, Any]
    signature: str = ""
    tx_hash: str = ""
    
    def __post_init__(self):
        """Calculate transaction hash after initialization"""
        if not self.tx_hash:
            self.tx_hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate the transaction hash"""
        tx_data = {
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount,
            'fee': self.fee,
            'nonce': self.nonce,
            'timestamp': self.timestamp,
            'data': self.data
        }
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary"""
        return cls(**data)
    
    def verify_signature(self, public_key: str) -> bool:
        """Verify transaction signature"""
        # Placeholder for signature verification
        # In a real implementation, this would use cryptographic verification
        return len(self.signature) > 0
    
    def is_valid(self) -> bool:
        """Check if transaction is valid"""
        if self.amount < 0 or self.fee < 0:
            return False
        if self.sender == self.recipient:
            return False
        if not self.verify_signature(self.sender):
            return False
        return True

@dataclass
class Block:
    """Represents a blockchain block"""
    
    index: int
    previous_hash: str
    timestamp: float
    transactions: List[Transaction]
    authority: str  # Address of the authority that created this block
    nonce: int = 0
    block_hash: str = ""
    merkle_root: str = ""
    
    def __post_init__(self):
        """Calculate block hash and merkle root after initialization"""
        if not self.merkle_root:
            self.merkle_root = self.calculate_merkle_root()
        if not self.block_hash:
            self.block_hash = self.calculate_hash()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions"""
        if not self.transactions:
            return hashlib.sha256(b'').hexdigest()
        
        # Simple Merkle tree implementation
        tx_hashes = [tx.tx_hash for tx in self.transactions]
        
        while len(tx_hashes) > 1:
            new_level = []
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                else:
                    combined = tx_hashes[i] + tx_hashes[i]
                new_level.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = new_level
        
        return tx_hashes[0]
    
    def calculate_hash(self) -> str:
        """Calculate the block hash"""
        block_data = {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'authority': self.authority,
            'nonce': self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'authority': self.authority,
            'nonce': self.nonce,
            'block_hash': self.block_hash,
            'merkle_root': self.merkle_root
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """Create block from dictionary"""
        transactions = [Transaction.from_dict(tx) for tx in data.get('transactions', [])]
        return cls(
            index=data['index'],
            previous_hash=data['previous_hash'],
            timestamp=data['timestamp'],
            transactions=transactions,
            authority=data['authority'],
            nonce=data.get('nonce', 0),
            block_hash=data.get('block_hash', ''),
            merkle_root=data.get('merkle_root', '')
        )
    
    def is_valid(self, previous_block: Optional['Block'] = None) -> bool:
        """Check if block is valid"""
        # Check hash integrity
        if self.block_hash != self.calculate_hash():
            logger.error(f"Invalid block hash for block {self.index}")
            return False
        
        # Check merkle root
        if self.merkle_root != self.calculate_merkle_root():
            logger.error(f"Invalid merkle root for block {self.index}")
            return False
        
        # Check previous hash
        if previous_block and self.previous_hash != previous_block.block_hash:
            logger.error(f"Invalid previous hash for block {self.index}")
            return False
        
        # Check all transactions
        for tx in self.transactions:
            if not tx.is_valid():
                logger.error(f"Invalid transaction in block {self.index}: {tx.tx_hash}")
                return False
        
        return True

class Blockchain:
    """Main blockchain class implementing PoA consensus"""
    
    def __init__(self, authorities: List[str] = None):
        """
        Initialize blockchain
        
        Args:
            authorities: List of authority addresses
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.authorities = authorities or []
        self.balances: Dict[str, float] = {}
        self.nonces: Dict[str, int] = {}
        
        # Create genesis block
        self.create_genesis_block()
        
        logger.info(f"[INIT] Blockchain initialized with {len(self.authorities)} authorities")
    
    def create_genesis_block(self) -> Block:
        """Create the genesis block"""
        genesis_transactions = []
        
        # Give initial balance to authorities
        for authority in self.authorities:
            self.balances[authority] = 1000000.0  # Initial balance
            self.nonces[authority] = 0
        
        genesis_block = Block(
            index=0,
            previous_hash="0" * 64,
            timestamp=time.time(),
            transactions=genesis_transactions,
            authority="genesis"
        )
        
        self.chain.append(genesis_block)
        logger.info("[OK] Genesis block created")
        return genesis_block
    
    def get_latest_block(self) -> Block:
        """Get the latest block in the chain"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to the pending pool"""
        if not transaction.is_valid():
            logger.error(f"[ERROR] Invalid transaction: {transaction.tx_hash}")
            return False
        
        # Check sender balance
        sender_balance = self.get_balance(transaction.sender)
        total_cost = transaction.amount + transaction.fee
        
        if sender_balance < total_cost:
            logger.error(f"[ERROR] Insufficient balance for {transaction.sender}")
            return False
        
        # Check nonce
        expected_nonce = self.nonces.get(transaction.sender, 0)
        if transaction.nonce != expected_nonce:
            logger.error(f"[ERROR] Invalid nonce for {transaction.sender}")
            return False
        
        self.pending_transactions.append(transaction)
        logger.info(f"[OK] Transaction added to pool: {transaction.tx_hash}")
        return True
    
    def create_block(self, authority: str) -> Optional[Block]:
        """Create a new block (only authorities can do this)"""
        if authority not in self.authorities:
            logger.error(f"[ERROR] {authority} is not an authority")
            return None
        
        if not self.pending_transactions:
            logger.warning("[WARNING] No pending transactions")
            return None
        
        # Select transactions for this block
        block_transactions = self.pending_transactions[:100]  # Limit to 100 transactions
        
        new_block = Block(
            index=len(self.chain),
            previous_hash=self.get_latest_block().block_hash,
            timestamp=time.time(),
            transactions=block_transactions,
            authority=authority
        )
        
        # Validate and add block
        if self.add_block(new_block):
            # Remove processed transactions
            self.pending_transactions = self.pending_transactions[100:]
            logger.info(f"[OK] Block {new_block.index} created by {authority}")
            return new_block
        
        return None
    
    def add_block(self, block: Block) -> bool:
        """Add a block to the chain"""
        if not self.is_block_valid(block):
            logger.error(f"[ERROR] Invalid block {block.index}")
            return False
        
        # Apply transactions
        for tx in block.transactions:
            self.apply_transaction(tx)
        
        self.chain.append(block)
        logger.info(f"[OK] Block {block.index} added to chain")
        return True
    
    def is_block_valid(self, block: Block) -> bool:
        """Validate a block"""
        previous_block = self.get_latest_block()
        
        # Basic block validation
        if not block.is_valid(previous_block):
            return False
        
        # PoA specific validation
        if block.authority not in self.authorities:
            logger.error(f"[ERROR] Block authority {block.authority} not in authorities list")
            return False
        
        return True
    
    def apply_transaction(self, transaction: Transaction):
        """Apply a transaction to the blockchain state"""
        # Update balances
        sender_balance = self.balances.get(transaction.sender, 0)
        recipient_balance = self.balances.get(transaction.recipient, 0)
        
        self.balances[transaction.sender] = sender_balance - transaction.amount - transaction.fee
        self.balances[transaction.recipient] = recipient_balance + transaction.amount
        
        # Update nonce
        self.nonces[transaction.sender] = transaction.nonce + 1
        
        logger.debug(f"[APPLY] Transaction {transaction.tx_hash} applied")
    
    def get_balance(self, address: str) -> float:
        """Get balance for an address"""
        return self.balances.get(address, 0.0)
    
    def get_nonce(self, address: str) -> int:
        """Get nonce for an address"""
        return self.nonces.get(address, 0)
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if not current_block.is_valid(previous_block):
                logger.error(f"[ERROR] Invalid block at index {i}")
                return False
        
        logger.info("[OK] Blockchain is valid")
        return True
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get blockchain information"""
        return {
            'length': len(self.chain),
            'latest_block': self.get_latest_block().to_dict(),
            'pending_transactions': len(self.pending_transactions),
            'authorities': self.authorities,
            'total_addresses': len(self.balances)
        }
    
    def export_chain(self) -> List[Dict[str, Any]]:
        """Export the entire blockchain"""
        return [block.to_dict() for block in self.chain]
    
    def import_chain(self, chain_data: List[Dict[str, Any]]) -> bool:
        """Import a blockchain"""
        try:
            new_chain = [Block.from_dict(block_data) for block_data in chain_data]
            
            # Validate the imported chain
            temp_blockchain = Blockchain(self.authorities)
            temp_blockchain.chain = new_chain
            
            if temp_blockchain.is_chain_valid():
                self.chain = new_chain
                # Recalculate balances and nonces
                self.recalculate_state()
                logger.info("[OK] Blockchain imported successfully")
                return True
            else:
                logger.error("[ERROR] Imported chain is invalid")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to import chain: {e}")
            return False
    
    def recalculate_state(self):
        """Recalculate balances and nonces from the chain"""
        self.balances = {}
        self.nonces = {}
        
        # Initialize authority balances
        for authority in self.authorities:
            self.balances[authority] = 1000000.0
            self.nonces[authority] = 0
        
        # Apply all transactions
        for block in self.chain[1:]:  # Skip genesis block
            for tx in block.transactions:
                self.apply_transaction(tx)
        
        logger.info("[OK] Blockchain state recalculated")

if __name__ == "__main__":
    # Demo blockchain usage
    authorities = ["auth1", "auth2", "auth3"]
    blockchain = Blockchain(authorities)
    
    # Create a sample transaction
    tx = Transaction(
        sender="auth1",
        recipient="user1",
        amount=100.0,
        fee=1.0,
        nonce=0,
        timestamp=time.time(),
        data={"purpose": "test transfer"}
    )
    
    # Add transaction and create block
    blockchain.add_transaction(tx)
    new_block = blockchain.create_block("auth1")
    
    print(f"Blockchain length: {len(blockchain.chain)}")
    print(f"User1 balance: {blockchain.get_balance('user1')}")
    print(f"Auth1 balance: {blockchain.get_balance('auth1')}")