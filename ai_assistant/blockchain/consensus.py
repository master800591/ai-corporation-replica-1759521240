#!/usr/bin/env python3
"""
Proof of Authority (PoA) Consensus Mechanism

This module implements the PoA consensus algorithm:
- Authority management and validation
- Block production scheduling
- Consensus rules
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Authority:
    """Represents a blockchain authority"""
    address: str
    public_key: str
    blocks_produced: int = 0
    last_block_time: float = 0.0
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    def __init__(self, authorities: Optional[List[str]] = None, block_time: int = 5):
        """Initialize PoA consensus"""
        self.authorities: Dict[str, Authority] = {}
        self.authority_order: List[str] = []
        self.current_authority_index: int = 0
        self.block_time: int = block_time
        self.last_block_time: float = 0.0
        self.is_producing: bool = False
        self.lock = threading.Lock()
        
        # Initialize authorities
        if authorities:
            for address in authorities:
                auth = Authority(
                    address=address,
                    public_key=address,  # Simplified
                    blocks_produced=0,
                    last_block_time=0.0,
                    is_active=True
                )
                self.authorities[address] = auth
                self.authority_order.append(address)
        
        logger.info(f"[OK] PoA consensus initialized with {len(self.authorities)} authorities")
    
    def is_valid_authority(self, address: str) -> bool:
        """Check if address is a valid authority"""
        authority = self.authorities.get(address)
        return authority is not None and authority.is_active
    
    def get_current_authority(self) -> Optional[str]:
        """Get the authority that should produce the next block"""
        if not self.authority_order:
            return None
        
        active_authorities = [addr for addr in self.authority_order 
                             if self.authorities[addr].is_active]
        
        if not active_authorities:
            return None
        
        # Simple round-robin
        index = self.current_authority_index % len(active_authorities)
        return active_authorities[index]
    
    def advance_authority(self):
        """Move to the next authority in rotation"""
        with self.lock:
            self.current_authority_index += 1
    
    def can_produce_block(self, address: str, current_time: Optional[float] = None) -> bool:
        """Check if authority can produce a block at current time"""
        if current_time is None:
            current_time = time.time()
        
        # Check if it's this authority's turn
        expected_authority = self.get_current_authority()
        if expected_authority != address:
            return False
        
        # Check if enough time has passed since last block
        time_since_last = current_time - self.last_block_time
        if time_since_last < self.block_time:
            return False
        
        return self.is_valid_authority(address)
    
    def validate_block_producer(self, block_producer: str, block_timestamp: float) -> bool:
        """Validate that the block producer was authorized at the time"""
        # Get expected authority at block time
        expected_authority = self.get_current_authority()
        
        if expected_authority != block_producer:
            logger.warning(f"[WARNING] Invalid block producer: expected {expected_authority}, got {block_producer}")
            return False
        
        # Check if authority is valid
        if not self.is_valid_authority(block_producer):
            logger.warning(f"[WARNING] Block producer {block_producer} is not a valid authority")
            return False
        
        return True
    
    def on_block_produced(self, producer: str, block_timestamp: float):
        """Called when a block is successfully produced"""
        with self.lock:
            authority = self.authorities.get(producer)
            if authority:
                # Update block count
                authority.blocks_produced += 1
                authority.last_block_time = block_timestamp
                
                logger.info(f"[OK] Authority {authority.address} produced block")
            
            # Update timing and advance
            self.last_block_time = block_timestamp
            self.advance_authority()
    
    def on_block_missed(self, expected_authority: str):
        """Called when an authority misses their block production slot"""
        authority = self.authorities.get(expected_authority)
        if authority:
            logger.warning(f"[WARNING] Authority {expected_authority} failed to produce block in time")
            
            # Move to next authority
            self.advance_authority()
    
    def add_authority(self, address: str, public_key: str):
        """Add a new authority"""
        new_authority = Authority(
            address=address,
            public_key=public_key,
            blocks_produced=0,
            last_block_time=0.0,
            is_active=True
        )
        
        with self.lock:
            self.authorities[address] = new_authority
            if address not in self.authority_order:
                self.authority_order.append(address)
        
        logger.info(f"[OK] Authority {address} added")
    
    def remove_authority(self, address: str):
        """Remove an authority"""
        with self.lock:
            if address in self.authorities:
                del self.authorities[address]
            
            if address in self.authority_order:
                self.authority_order.remove(address)
        
        logger.info(f"[OK] Authority {address} removed")
    
    def deactivate_authority(self, address: str):
        """Deactivate an authority"""
        authority = self.authorities.get(address)
        if authority:
            authority.is_active = False
            logger.info(f"[OK] Authority {address} deactivated")
    
    def reactivate_authority(self, address: str):
        """Reactivate an authority"""
        authority = self.authorities.get(address)
        if authority:
            authority.is_active = True
            logger.info(f"[OK] Authority {address} reactivated")
    
    def get_authorities(self) -> List[Authority]:
        """Get all authorities"""
        return list(self.authorities.values())
    
    def get_active_authorities(self) -> List[Authority]:
        """Get only active authorities"""
        return [auth for auth in self.authorities.values() if auth.is_active]
    
    def reorder_authorities(self, strategy: str = "blocks_produced"):
        """Reorder authorities based on strategy"""
        active_authorities = [auth for auth in self.authorities.values() if auth.is_active]
        
        if strategy == "blocks_produced":
            # Sort by blocks produced (descending)
            active_authorities.sort(key=lambda a: a.blocks_produced, reverse=True)
        elif strategy == "random":
            # Randomize order
            import random
            random.shuffle(active_authorities)
        
        # Update authority order
        self.authority_order = [auth.address for auth in active_authorities]
    
    def get_authority_stats(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        return {
            'total_authorities': len(self.authorities),
            'active_authorities': len(self.get_active_authorities()),
            'current_authority': self.get_current_authority(),
            'block_time': self.block_time,
            'last_block_time': self.last_block_time,
            'authorities': [auth.to_dict() for auth in self.authorities.values()]
        }


class PoABlockProducer:
    """Block producer for PoA consensus"""
    
    def __init__(self, consensus: PoAConsensus, authority_address: str):
        """Initialize block producer"""
        self.consensus = consensus
        self.authority_address = authority_address
        self.is_running = False
        self.producer_thread = None
        
    def start_production(self):
        """Start block production"""
        if self.is_running:
            return
        
        self.is_running = True
        self.producer_thread = threading.Thread(target=self._production_loop)
        self.producer_thread.start()
        
        logger.info(f"[OK] Block production started for {self.authority_address}")
    
    def stop_production(self):
        """Stop block production"""
        self.is_running = False
        if self.producer_thread:
            self.producer_thread.join()
        
        logger.info(f"[OK] Block production stopped for {self.authority_address}")
    
    def _production_loop(self):
        """Main block production loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if we can produce a block
                if self.consensus.can_produce_block(self.authority_address, current_time):
                    self._produce_block(current_time)
                
                # Sleep for a short time before checking again
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"[ERROR] Block production error: {e}")
                time.sleep(1.0)
    
    def _produce_block(self, timestamp: float):
        """Produce a new block"""
        try:
            # This would normally create and broadcast a new block
            # For now, just simulate the process
            logger.info(f"[BLOCK] Authority {self.authority_address} producing block at {timestamp}")
            
            # Notify consensus of successful block production
            self.consensus.on_block_produced(self.authority_address, timestamp)
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to produce block: {e}")


# Utility functions for creating test setups
def create_test_consensus(authorities: Optional[List[str]] = None) -> PoAConsensus:
    """Create a test consensus setup for testing purposes"""
    if not authorities:
        authorities = ["auth1", "auth2", "auth3"]
    return PoAConsensus(authorities)

def create_test_producers(consensus: PoAConsensus) -> List[PoABlockProducer]:
    """Create block producers for testing"""
    producers = []
    for auth_addr in consensus.authority_order:
        producer = PoABlockProducer(consensus, auth_addr)
        producers.append(producer)
    return producers