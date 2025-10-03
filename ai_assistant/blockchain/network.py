#!/usr/bin/env python3
"""
P2P Blockchain Network Integration

This module integrates the blockchain with P2P networking:
- Peer discovery and communication
- Block and transaction propagation
- Network consensus coordination
- Blockchain synchronization
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict

from .core import Block, Transaction, Blockchain
from .consensus import PoAConsensus, PoABlockProducer
from ..p2p.node import P2PNode
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class NetworkMessage:
    """Represents a network message"""
    message_type: str
    sender: str
    data: Dict[str, Any]
    timestamp: float
    message_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkMessage':
        """Create from dictionary"""
        return cls(**data)

class BlockchainP2PNode:
    """P2P node with blockchain capabilities"""
    
    def __init__(self, 
                 node_id: str,
                 port: int,
                 authorities: List[str] = None,
                 is_authority: bool = False):
        """
        Initialize blockchain P2P node
        
        Args:
            node_id: Unique identifier for this node
            port: Port to listen on
            authorities: List of authority addresses
            is_authority: Whether this node is an authority
        """
        self.node_id = node_id
        self.port = port
        self.is_authority = is_authority
        self.authority_address = node_id if is_authority else None
        
        # Initialize blockchain components
        self.blockchain = Blockchain(authorities or [])
        self.consensus = PoAConsensus(self.blockchain, block_time=5.0)
        self.block_producer = None
        
        if self.is_authority and self.authority_address:
            self.block_producer = PoABlockProducer(self.consensus, self.authority_address)
        
        # Initialize P2P networking
        self.p2p_node = P2PNode(node_id, port)
        self.peers: Set[str] = set()
        self.known_blocks: Set[str] = set()
        self.known_transactions: Set[str] = set()
        
        # Message handlers
        self.message_handlers = {
            'new_block': self._handle_new_block,
            'new_transaction': self._handle_new_transaction,
            'request_chain': self._handle_chain_request,
            'chain_response': self._handle_chain_response,
            'peer_list': self._handle_peer_list,
            'authority_announcement': self._handle_authority_announcement
        }
        
        logger.info(f"[INIT] Blockchain P2P node {node_id} initialized on port {port}")
    
    async def start(self):
        """Start the P2P blockchain node"""
        try:
            # Start P2P networking
            await self.p2p_node.start()
            
            # Register message handlers
            for msg_type, handler in self.message_handlers.items():
                self.p2p_node.add_message_handler(msg_type, handler)
            
            # Start block production if authority
            if self.block_producer:
                self.block_producer.start_production()
                # Start block production loop
                asyncio.create_task(self._block_production_loop())
            
            # Start network maintenance tasks
            asyncio.create_task(self._network_maintenance_loop())
            asyncio.create_task(self._sync_loop())
            
            logger.info(f"[OK] Blockchain P2P node {self.node_id} started")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to start node: {e}")
            raise
    
    async def stop(self):
        """Stop the P2P blockchain node"""
        if self.block_producer:
            self.block_producer.stop_production()
        
        await self.p2p_node.stop()
        logger.info(f"[OK] Blockchain P2P node {self.node_id} stopped")
    
    async def connect_to_peer(self, peer_address: str, peer_port: int):
        """Connect to a peer"""
        await self.p2p_node.connect_to_peer(peer_address, peer_port)
        
        # Request chain from new peer
        await self._request_chain_from_peer(f"{peer_address}:{peer_port}")
    
    async def _block_production_loop(self):
        """Main block production loop for authorities"""
        while self.block_producer and self.block_producer.is_running:
            try:
                # Check if it's time to produce a block
                if self.consensus.can_produce_block(self.authority_address):
                    block = self.block_producer.try_produce_block()
                    if block:
                        # Broadcast the new block
                        await self._broadcast_block(block)
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"[ERROR] Block production error: {e}")
                await asyncio.sleep(5.0)
    
    async def _network_maintenance_loop(self):
        """Network maintenance and peer discovery"""
        while True:
            try:
                # Announce to peers if authority
                if self.is_authority:
                    await self._announce_authority()
                
                # Share peer lists
                await self._share_peer_list()
                
                await asyncio.sleep(30.0)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"[ERROR] Network maintenance error: {e}")
                await asyncio.sleep(30.0)
    
    async def _sync_loop(self):
        """Blockchain synchronization loop"""
        while True:
            try:
                # Request chain updates from random peers
                if self.peers:
                    import random
                    peer = random.choice(list(self.peers))
                    await self._request_chain_from_peer(peer)
                
                await asyncio.sleep(60.0)  # Every minute
                
            except Exception as e:
                logger.error(f"[ERROR] Sync loop error: {e}")
                await asyncio.sleep(60.0)
    
    async def submit_transaction(self, transaction: Transaction) -> bool:
        """Submit a transaction to the network"""
        # Add to local blockchain
        if self.blockchain.add_transaction(transaction):
            # Broadcast to network
            await self._broadcast_transaction(transaction)
            logger.info(f"[OK] Transaction {transaction.tx_hash} submitted")
            return True
        
        logger.error(f"[ERROR] Failed to submit transaction {transaction.tx_hash}")
        return False
    
    async def _broadcast_block(self, block: Block):
        """Broadcast a new block to all peers"""
        message = NetworkMessage(
            message_type='new_block',
            sender=self.node_id,
            data={'block': block.to_dict()},
            timestamp=time.time(),
            message_id=f"block_{block.block_hash}"
        )
        
        await self.p2p_node.broadcast_message(json.dumps(message.to_dict()))
        self.known_blocks.add(block.block_hash)
        
        logger.info(f"[OK] Broadcasted block {block.index}")
    
    async def _broadcast_transaction(self, transaction: Transaction):
        """Broadcast a new transaction to all peers"""
        message = NetworkMessage(
            message_type='new_transaction',
            sender=self.node_id,
            data={'transaction': transaction.to_dict()},
            timestamp=time.time(),
            message_id=f"tx_{transaction.tx_hash}"
        )
        
        await self.p2p_node.broadcast_message(json.dumps(message.to_dict()))
        self.known_transactions.add(transaction.tx_hash)
        
        logger.info(f"[OK] Broadcasted transaction {transaction.tx_hash}")
    
    async def _request_chain_from_peer(self, peer_id: str):
        """Request blockchain from a peer"""
        message = NetworkMessage(
            message_type='request_chain',
            sender=self.node_id,
            data={'from_block': len(self.blockchain.chain)},
            timestamp=time.time(),
            message_id=f"chain_req_{int(time.time())}"
        )
        
        await self.p2p_node.send_to_peer(peer_id, json.dumps(message.to_dict()))
    
    async def _announce_authority(self):
        """Announce this node as an authority"""
        if not self.is_authority:
            return
        
        message = NetworkMessage(
            message_type='authority_announcement',
            sender=self.node_id,
            data={
                'authority_address': self.authority_address,
                'chain_length': len(self.blockchain.chain),
                'latest_block_hash': self.blockchain.get_latest_block().block_hash
            },
            timestamp=time.time(),
            message_id=f"auth_announce_{int(time.time())}"
        )
        
        await self.p2p_node.broadcast_message(json.dumps(message.to_dict()))
    
    async def _share_peer_list(self):
        """Share known peers with the network"""
        peer_list = list(self.peers)
        
        message = NetworkMessage(
            message_type='peer_list',
            sender=self.node_id,
            data={'peers': peer_list},
            timestamp=time.time(),
            message_id=f"peers_{int(time.time())}"
        )
        
        await self.p2p_node.broadcast_message(json.dumps(message.to_dict()))
    
    # Message Handlers
    
    async def _handle_new_block(self, sender: str, data: Dict[str, Any]):
        """Handle incoming new block"""
        try:
            block_data = data.get('block')
            if not block_data:
                return
            
            block = Block.from_dict(block_data)
            
            # Skip if we already know this block
            if block.block_hash in self.known_blocks:
                return
            
            # Validate and add block
            if self.consensus.validate_block(block):
                if self.blockchain.add_block(block):
                    self.known_blocks.add(block.block_hash)
                    self.consensus.on_block_produced(block)
                    
                    # Re-broadcast to other peers
                    await self._broadcast_block(block)
                    
                    logger.info(f"[OK] Added block {block.index} from {sender}")
                else:
                    logger.warning(f"[WARNING] Failed to add block {block.index} from {sender}")
            else:
                logger.warning(f"[WARNING] Invalid block {block.index} from {sender}")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to handle new block: {e}")
    
    async def _handle_new_transaction(self, sender: str, data: Dict[str, Any]):
        """Handle incoming new transaction"""
        try:
            tx_data = data.get('transaction')
            if not tx_data:
                return
            
            transaction = Transaction.from_dict(tx_data)
            
            # Skip if we already know this transaction
            if transaction.tx_hash in self.known_transactions:
                return
            
            # Add transaction to pool
            if self.blockchain.add_transaction(transaction):
                self.known_transactions.add(transaction.tx_hash)
                
                # Re-broadcast to other peers
                await self._broadcast_transaction(transaction)
                
                logger.info(f"[OK] Added transaction {transaction.tx_hash} from {sender}")
            else:
                logger.warning(f"[WARNING] Failed to add transaction from {sender}")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to handle new transaction: {e}")
    
    async def _handle_chain_request(self, sender: str, data: Dict[str, Any]):
        """Handle blockchain request"""
        try:
            from_block = data.get('from_block', 0)
            
            # Send blockchain from requested block
            chain_data = self.blockchain.export_chain()[from_block:]
            
            response = NetworkMessage(
                message_type='chain_response',
                sender=self.node_id,
                data={
                    'chain': chain_data,
                    'from_block': from_block
                },
                timestamp=time.time(),
                message_id=f"chain_resp_{int(time.time())}"
            )
            
            await self.p2p_node.send_to_peer(sender, json.dumps(response.to_dict()))
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to handle chain request: {e}")
    
    async def _handle_chain_response(self, sender: str, data: Dict[str, Any]):
        """Handle blockchain response"""
        try:
            chain_data = data.get('chain', [])
            from_block = data.get('from_block', 0)
            
            if not chain_data:
                return
            
            # Validate and potentially replace our chain
            if len(chain_data) > len(self.blockchain.chain) - from_block:
                # Import the longer chain
                full_chain = self.blockchain.export_chain()[:from_block] + chain_data
                
                if self.blockchain.import_chain(full_chain):
                    logger.info(f"[OK] Updated blockchain from {sender}")
                else:
                    logger.warning(f"[WARNING] Invalid chain from {sender}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to handle chain response: {e}")
    
    async def _handle_peer_list(self, sender: str, data: Dict[str, Any]):
        """Handle peer list from another node"""
        try:
            peer_list = data.get('peers', [])
            
            for peer in peer_list:
                if peer != self.node_id and peer not in self.peers:
                    self.peers.add(peer)
                    logger.info(f"[OK] Added peer {peer} from {sender}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to handle peer list: {e}")
    
    async def _handle_authority_announcement(self, sender: str, data: Dict[str, Any]):
        """Handle authority announcement"""
        try:
            authority_address = data.get('authority_address')
            chain_length = data.get('chain_length', 0)
            
            if authority_address and authority_address not in self.blockchain.authorities:
                # Add as potential authority
                logger.info(f"[INFO] New authority announced: {authority_address} by {sender}")
            
            # If their chain is longer, request it
            if chain_length > len(self.blockchain.chain):
                await self._request_chain_from_peer(sender)
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to handle authority announcement: {e}")
    
    # Public API methods
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Get blockchain information"""
        return {
            'node_id': self.node_id,
            'is_authority': self.is_authority,
            'authority_address': self.authority_address,
            'blockchain_info': self.blockchain.get_chain_info(),
            'consensus_stats': self.consensus.get_authority_stats(),
            'peer_count': len(self.peers),
            'known_blocks': len(self.known_blocks),
            'known_transactions': len(self.known_transactions)
        }
    
    def get_balance(self, address: str) -> float:
        """Get balance for an address"""
        return self.blockchain.get_balance(address)
    
    def get_pending_transactions(self) -> List[Transaction]:
        """Get pending transactions"""
        return self.blockchain.pending_transactions.copy()
    
    def create_transaction(self, 
                          recipient: str, 
                          amount: float, 
                          fee: float = 0.01,
                          data: Dict[str, Any] = None) -> Transaction:
        """Create a new transaction"""
        if not self.authority_address:
            raise ValueError("Cannot create transaction without authority address")
        
        return Transaction(
            sender=self.authority_address,
            recipient=recipient,
            amount=amount,
            fee=fee,
            nonce=self.blockchain.get_nonce(self.authority_address),
            timestamp=time.time(),
            data=data or {}
        )

# Factory function for easy node creation
def create_blockchain_node(node_id: str, 
                          port: int, 
                          authorities: List[str] = None,
                          is_authority: bool = False) -> BlockchainP2PNode:
    """Create a blockchain P2P node"""
    return BlockchainP2PNode(node_id, port, authorities, is_authority)

if __name__ == "__main__":
    # Demo blockchain P2P network
    async def demo():
        authorities = ["auth1", "auth2", "auth3"]
        
        # Create authority nodes
        node1 = create_blockchain_node("auth1", 8001, authorities, True)
        node2 = create_blockchain_node("auth2", 8002, authorities, True)
        node3 = create_blockchain_node("peer1", 8003, authorities, False)
        
        try:
            # Start nodes
            await node1.start()
            await node2.start()
            await node3.start()
            
            # Connect nodes
            await node2.connect_to_peer("localhost", 8001)
            await node3.connect_to_peer("localhost", 8001)
            
            # Create and submit a transaction
            tx = node1.create_transaction("user1", 100.0, 1.0)
            await node1.submit_transaction(tx)
            
            # Wait for block production
            await asyncio.sleep(10)
            
            # Print blockchain info
            for node in [node1, node2, node3]:
                info = node.get_blockchain_info()
                print(f"{node.node_id}: {info['blockchain_info']['length']} blocks")
            
        finally:
            await node1.stop()
            await node2.stop()
            await node3.stop()
    
    # Run demo
    asyncio.run(demo())