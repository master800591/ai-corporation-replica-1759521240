#!/usr/bin/env python3
"""
Proof of Authority (PoA) Consensus Mechanism

This module implements the PoA consensus algorithm:
- Authority management and validation
- Block production sch    def add_authority(self, address: str, public_key: str):
        """Add a new authority"""hori        new_authority = Authority(
            address=address,
            public_key=public_key,
            blocks_produced=0,
            last_block_time=0.0,
            is_active=True
        )ion
- Consensus rules
"""

import time
import random
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from threading import Lock

from .core import Block, Transaction, Blockchain
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Authority:
    """Represents a blockchain authority"""
    address: str
    public_key: str
    blocks_produced: int = 0
    last_block_time: float = 0.0
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, any]:
        """Convert authority to dictionary"""
        return {
            'address': self.address,
            'public_key': self.public_key,
            'blocks_produced': self.blocks_produced,
            'last_block_time': self.last_block_time,
            'is_active': self.is_active
        }

class PoAConsensus:
    """Proof of Authority consensus mechanism"""
    
    def __init__(self, blockchain: Blockchain, block_time: float = 5.0):
        """
        Initialize PoA consensus
        
        Args:
            blockchain: The blockchain instance
            block_time: Target time between blocks in seconds
        """
        self.blockchain = blockchain
        self.block_time = block_time
        self.authorities: Dict[str, Authority] = {}
        self.authority_order: List[str] = []
        self.current_authority_index = 0
        self.last_block_time = time.time()
        self.lock = Lock()
        
        # Initialize authorities from blockchain
        self._initialize_authorities()
        
        logger.info(f"[INIT] PoA consensus initialized with {len(self.authorities)} authorities")
    
    def _initialize_authorities(self):
        """Initialize authorities from blockchain"""
        for auth_address in self.blockchain.authorities:
            authority = Authority(
                address=auth_address,
                public_key=f"pubkey_{auth_address}",  # Placeholder
                reputation=100.0,
                blocks_produced=0,
                last_block_time=0.0,
                is_active=True
            )
            self.authorities[auth_address] = authority
            self.authority_order.append(auth_address)
        
        # Shuffle initial order for fairness
        random.shuffle(self.authority_order)
        logger.info(f"[OK] Initialized {len(self.authorities)} authorities")
    
    def get_current_authority(self) -> Optional[str]:
        """Get the current authority who should produce the next block"""
        if not self.authority_order:
            return None
        
        with self.lock:
            current_time = time.time()
            
            # Check if it's time for the next block
            if current_time - self.last_block_time >= self.block_time:
                authority_address = self.authority_order[self.current_authority_index]
                
                # Check if authority is active
                authority = self.authorities.get(authority_address)
                if authority and authority.is_active:
                    return authority_address
                else:
                    # Skip inactive authority
                    self._next_authority()
                    return self.get_current_authority()
            
            return None
    
    def _next_authority(self):
        """Move to the next authority in rotation"""
        with self.lock:
            self.current_authority_index = (self.current_authority_index + 1) % len(self.authority_order)
    
    def can_produce_block(self, authority_address: str) -> bool:
        """Check if an authority can produce a block"""
        current_auth = self.get_current_authority()
        return current_auth == authority_address
    
    def validate_block(self, block: Block) -> bool:
        """Validate a block according to PoA rules"""
        # Check if authority is valid
        if block.authority not in self.authorities:
            logger.error(f"[ERROR] Unknown authority: {block.authority}")
            return False
        
        authority = self.authorities[block.authority]
        
        # Check if authority is active
        if not authority.is_active:
            logger.error(f"[ERROR] Inactive authority: {block.authority}")
            return False
        
        # Check block timing
        previous_block = self.blockchain.get_latest_block()
        time_diff = block.timestamp - previous_block.timestamp
        
        if time_diff < self.block_time * 0.8:  # Allow 20% tolerance
            logger.error(f"[ERROR] Block produced too early: {time_diff}s")
            return False
        
        # Check if it was this authority's turn
        expected_authority = self.get_authority_at_time(block.timestamp)
        if expected_authority != block.authority:
            logger.warning(f"[WARNING] Unexpected authority. Expected: {expected_authority}, Got: {block.authority}")
            # Allow it if the expected authority was inactive
            expected_auth_obj = self.authorities.get(expected_authority)
            if expected_auth_obj and expected_auth_obj.is_active:
                return False
        
        logger.info(f"[OK] Block validated for authority: {block.authority}")
        return True
    
    def get_authority_at_time(self, timestamp: float) -> Optional[str]:
        """Get which authority should be producing at a given time"""
        if not self.authority_order:
            return None
        
        # Calculate which authority based on time slots
        genesis_time = self.blockchain.chain[0].timestamp
        time_elapsed = timestamp - genesis_time
        slot_number = int(time_elapsed // self.block_time)
        authority_index = slot_number % len(self.authority_order)
        
        return self.authority_order[authority_index]
    
    def on_block_produced(self, block: Block):
        """Handle when a block is successfully produced"""
        authority = self.authorities.get(block.authority)
        if authority:
            authority.blocks_produced += 1
            authority.last_block_time = block.timestamp
            
            # Update reputation based on performance
            self._update_reputation(authority, True)
            
            # Move to next authority
            self._next_authority()
            self.last_block_time = block.timestamp
            
            logger.info(f"[OK] Block produced by {block.authority}, moving to next authority")
    
    def on_block_missed(self, authority_address: str):
        """Handle when an authority misses their block production slot"""
        authority = self.authorities.get(authority_address)
        if authority:
            self._update_reputation(authority, False)
            
            # Move to next authority
            self._next_authority()
            self.last_block_time = time.time()
            
            logger.warning(f"[WARNING] Authority {authority_address} missed their slot")
    
    def _update_reputation(self, authority: Authority, produced_block: bool):
        """Update authority reputation based on performance"""
        if produced_block:
            authority.reputation = min(100.0, authority.reputation + 0.1)
        else:
            authority.reputation = max(0.0, authority.reputation - 5.0)
            
            # Deactivate authority if reputation is too low
            if authority.reputation < 10.0:
                authority.is_active = False
                logger.warning(f"[WARNING] Authority {authority.address} deactivated due to low reputation")
    
    def add_authority(self, address: str, public_key: str) -> bool:
        """Add a new authority"""
        if address in self.authorities:
            logger.warning(f"[WARNING] Authority {address} already exists")
            return False
        
        authority = Authority(
            address=address,
            public_key=public_key,
            reputation=100.0,
            blocks_produced=0,
            last_block_time=0.0,
            is_active=True
        )
        
        with self.lock:
            self.authorities[address] = authority
            self.authority_order.append(address)
        
        logger.info(f"[OK] Authority {address} added")
        return True
    
    def remove_authority(self, address: str) -> bool:
        """Remove an authority"""
        if address not in self.authorities:
            logger.warning(f"[WARNING] Authority {address} not found")
            return False
        
        with self.lock:
            del self.authorities[address]
            self.authority_order.remove(address)
            
            # Adjust current index if needed
            if self.current_authority_index >= len(self.authority_order):
                self.current_authority_index = 0
        
        logger.info(f"[OK] Authority {address} removed")
        return True
    
    def activate_authority(self, address: str) -> bool:
        """Activate an authority"""
        authority = self.authorities.get(address)
        if not authority:
            logger.warning(f"[WARNING] Authority {address} not found")
            return False
        
        authority.is_active = True
        authority.reputation = max(50.0, authority.reputation)  # Give a second chance
        
        logger.info(f"[OK] Authority {address} activated")
        return True
    
    def deactivate_authority(self, address: str) -> bool:
        """Deactivate an authority"""
        authority = self.authorities.get(address)
        if not authority:
            logger.warning(f"[WARNING] Authority {address} not found")
            return False
        
        authority.is_active = False
        
        logger.info(f"[OK] Authority {address} deactivated")
        return True
    
    def get_authority_stats(self) -> Dict[str, any]:
        """Get statistics about authorities"""
        active_authorities = sum(1 for auth in self.authorities.values() if auth.is_active)
        total_blocks = sum(auth.blocks_produced for auth in self.authorities.values())
        
        return {
            'total_authorities': len(self.authorities),
            'active_authorities': active_authorities,
            'total_blocks_produced': total_blocks,
            'current_authority': self.get_current_authority(),
            'block_time': self.block_time,
            'authorities': {addr: auth.to_dict() for addr, auth in self.authorities.items()}
        }
    
    def reorder_authorities(self, strategy: str = "reputation"):
        """Reorder authorities based on different strategies"""
        with self.lock:
            if strategy == "reputation":
                # Order by reputation (highest first)
                self.authority_order.sort(
                    key=lambda addr: self.authorities[addr].reputation, 
                    reverse=True
                )
            elif strategy == "random":
                # Random shuffle
                random.shuffle(self.authority_order)
            elif strategy == "round_robin":
                # Rotate the list
                if self.authority_order:
                    self.authority_order = (
                        self.authority_order[1:] + [self.authority_order[0]]
                    )
            
            # Reset index
            self.current_authority_index = 0
            
            logger.info(f"[OK] Authorities reordered using {strategy} strategy")
    
    def get_next_block_time(self) -> float:
        """Get estimated time for next block"""
        current_time = time.time()
        time_since_last = current_time - self.last_block_time
        
        if time_since_last >= self.block_time:
            return 0.0  # Should produce now
        else:
            return self.block_time - time_since_last
    
    def is_authority_due(self, authority_address: str) -> bool:
        """Check if an authority is due to produce a block"""
        current_auth = self.get_current_authority()
        return current_auth == authority_address
    
    def force_next_authority(self):
        """Force advancement to next authority (for testing/emergency)"""
        with self.lock:
            self._next_authority()
            self.last_block_time = time.time()
        
        logger.warning("[WARNING] Forced advancement to next authority")

class PoABlockProducer:
    """Block producer for PoA consensus"""
    
    def __init__(self, consensus: PoAConsensus, authority_address: str):
        """
        Initialize block producer
        
        Args:
            consensus: PoA consensus instance
            authority_address: Address of this authority
        """
        self.consensus = consensus
        self.authority_address = authority_address
        self.is_running = False
        
        logger.info(f"[INIT] Block producer initialized for {authority_address}")
    
    def try_produce_block(self) -> Optional[Block]:
        """Try to produce a block if it's this authority's turn"""
        if not self.consensus.can_produce_block(self.authority_address):
            return None
        
        # Create block
        block = self.consensus.blockchain.create_block(self.authority_address)
        if block:
            self.consensus.on_block_produced(block)
            logger.info(f"[OK] Block {block.index} produced by {self.authority_address}")
            return block
        
        return None
    
    def start_production(self):
        """Start automatic block production (would run in separate thread)"""
        self.is_running = True
        logger.info(f"[OK] Block production started for {self.authority_address}")
    
    def stop_production(self):
        """Stop automatic block production"""
        self.is_running = False
        logger.info(f"[OK] Block production stopped for {self.authority_address}")

if __name__ == "__main__":
    # Demo PoA consensus
    from .core import Blockchain
    
    authorities = ["auth1", "auth2", "auth3"]
    blockchain = Blockchain(authorities)
    consensus = PoAConsensus(blockchain, block_time=3.0)
    
    # Create producers
    producers = [PoABlockProducer(consensus, auth) for auth in authorities]
    
    # Simulate block production
    for i in range(5):
        current_auth = consensus.get_current_authority()
        if current_auth:
            producer = next((p for p in producers if p.authority_address == current_auth), None)
            if producer:
                block = producer.try_produce_block()
                if block:
                    print(f"Block {block.index} produced by {current_auth}")
        
        time.sleep(3.1)  # Wait longer than block time
    
    # Print stats
    stats = consensus.get_authority_stats()
    print(f"Total blocks produced: {stats['total_blocks_produced']}")
    for addr, auth_stats in stats['authorities'].items():
        print(f"{addr}: {auth_stats['blocks_produced']} blocks, reputation: {auth_stats['reputation']:.1f}")