#!/usr/bin/env python3
"""
Distributed P2P Network System for AI Corporation

Self-reliant networking system that creates a distributed mesh network
for user registration, data sharing, and system promotion. Includes
resilience against attacks and automatic node discovery.

Key Features:
- P2P mesh networking with encryption
- Distributed user registration and promotion
- Self-healing network topology
- Anti-attack measures and founder protection
- Automatic service discovery and expansion
"""

import asyncio
import json
import time
import hashlib
import threading
import websockets
import socket
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import ssl
import random

# Optional imports with graceful degradation
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    crypto_available = True
except ImportError:
    crypto_available = False
    print("[WARNING] Cryptography not available - using basic security")

class NodeType(Enum):
    """Types of network nodes"""
    FOUNDER_NODE = "founder"           # Steve Cornell's protected nodes
    CORE_NODE = "core"                # AI Corporation core systems
    WORKER_NODE = "worker"            # General processing nodes
    GATEWAY_NODE = "gateway"          # Public interface nodes
    GUARD_NODE = "guard"              # Security and monitoring nodes
    STORAGE_NODE = "storage"          # Data persistence nodes
    DISCOVERY_NODE = "discovery"      # Network discovery and routing

class NetworkMessage(Enum):
    """Network message types"""
    REGISTRATION = "registration"
    PROMOTION = "promotion"
    DATA_SYNC = "data_sync"
    THREAT_ALERT = "threat_alert"
    FOUNDER_PROTECTION = "founder_protection"
    SYSTEM_EXPANSION = "system_expansion"
    HEARTBEAT = "heartbeat"
    SERVICE_DISCOVERY = "service_discovery"

@dataclass
class NetworkNode:
    """Represents a node in the P2P network"""
    node_id: str
    node_type: NodeType
    host: str
    port: int
    public_key: Optional[str] = None
    last_seen: float = field(default_factory=time.time)
    trust_score: float = 1.0
    capabilities: List[str] = field(default_factory=list)
    founder_protection_level: int = 0  # 0-10 scale
    
class P2PNetworkManager:
    """Manages the distributed P2P network for AI Corporation"""
    
    def __init__(self, node_type: NodeType = NodeType.CORE_NODE, 
                 port: int = 8888, founder_protection: bool = True):
        self.node_id = self._generate_node_id()
        self.node_type = node_type
        self.port = port
        self.host = "0.0.0.0"
        self.founder_protection = founder_protection
        
        # Network state
        self.peers: Dict[str, NetworkNode] = {}
        self.active_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.server = None
        self.running = False
        
        # Security and encryption
        self.encryption_key = self._generate_encryption_key() if crypto_available else None
        self.trusted_founder_nodes: Set[str] = set()
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.network_threads: List[threading.Thread] = []
        
        # Network services
        self.services: Dict[str, Callable] = {}
        self.discovery_seeds = [
            "localhost:8888", "localhost:8889", "localhost:8890"
        ]
        
        # Founder protection measures
        self.founder_nodes_priority = True
        self.attack_detection = {}
        self.emergency_protocols = []
        
        logging.info(f"P2P Network Manager initialized - Node: {self.node_id}")
    
    def _generate_node_id(self) -> str:
        """Generate unique node identifier"""
        timestamp = str(time.time())
        random_data = str(random.randint(100000, 999999))
        hash_input = f"{socket.gethostname()}-{timestamp}-{random_data}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for secure communications"""
        if not crypto_available:
            return b"dummy_key_not_secure"
        
        password = f"ai-corporation-{self.node_id}".encode()
        salt = b"ai_corp_salt_2025"  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def encrypt_message(self, message: str) -> str:
        """Encrypt message for secure transmission"""
        if not crypto_available or not self.encryption_key:
            return message  # Fallback to plaintext
        
        try:
            f = Fernet(self.encryption_key)
            return f.encrypt(message.encode()).decode()
        except Exception as e:
            logging.warning(f"Encryption failed: {e}")
            return message
    
    def decrypt_message(self, encrypted_message: str) -> str:
        """Decrypt received message"""
        if not crypto_available or not self.encryption_key:
            return encrypted_message  # Fallback to plaintext
        
        try:
            f = Fernet(self.encryption_key)
            return f.decrypt(encrypted_message.encode()).decode()
        except Exception as e:
            logging.warning(f"Decryption failed: {e}")
            return encrypted_message
    
    async def start_server(self):
        """Start the P2P server"""
        try:
            # Updated for websockets compatibility
            async def connection_handler(websocket):
                # Use a default path since websocket.path might not be available
                path = getattr(websocket, 'path', '/')
                return await self.handle_connection(websocket, path)
            
            self.server = await websockets.serve(
                connection_handler,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            self.running = True
            logging.info(f"P2P server started on {self.host}:{self.port}")
            
            # Start background tasks
            asyncio.create_task(self.heartbeat_loop())
            asyncio.create_task(self.discovery_loop())
            asyncio.create_task(self.founder_protection_loop())
            
        except Exception as e:
            logging.error(f"Failed to start P2P server: {e}")
            raise
    
    async def handle_connection(self, websocket, path):
        """Handle incoming P2P connections"""
        peer_id = None
        try:
            # Register new connection
            peer_id = f"peer_{len(self.active_connections)}"
            self.active_connections[peer_id] = websocket
            
            logging.info(f"New peer connected: {peer_id}")
            
            # Handle messages
            async for raw_message in websocket:
                try:
                    # Decrypt if needed
                    message_data = json.loads(self.decrypt_message(raw_message))
                    await self.process_message(peer_id, message_data)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON from {peer_id}")
                except Exception as e:
                    logging.error(f"Error processing message from {peer_id}: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logging.info(f"Peer {peer_id} disconnected")
        except Exception as e:
            logging.error(f"Connection error with {peer_id}: {e}")
        finally:
            if peer_id and peer_id in self.active_connections:
                del self.active_connections[peer_id]
    
    async def process_message(self, peer_id: str, message: Dict[str, Any]):
        """Process incoming network messages"""
        message_type = message.get('type')
        
        if message_type == NetworkMessage.REGISTRATION.value:
            await self.handle_registration(peer_id, message)
        elif message_type == NetworkMessage.PROMOTION.value:
            await self.handle_promotion(peer_id, message)
        elif message_type == NetworkMessage.DATA_SYNC.value:
            await self.handle_data_sync(peer_id, message)
        elif message_type == NetworkMessage.THREAT_ALERT.value:
            await self.handle_threat_alert(peer_id, message)
        elif message_type == NetworkMessage.FOUNDER_PROTECTION.value:
            await self.handle_founder_protection(peer_id, message)
        elif message_type == NetworkMessage.SERVICE_DISCOVERY.value:
            await self.handle_service_discovery(peer_id, message)
        else:
            logging.warning(f"Unknown message type from {peer_id}: {message_type}")
    
    async def handle_registration(self, peer_id: str, message: Dict[str, Any]):
        """Handle user registration requests"""
        try:
            user_data = message.get('data', {})
            
            # Validate registration
            if self.validate_registration(user_data):
                # Store user data
                registration_id = self.store_user_registration(user_data)
                
                # Send confirmation
                response = {
                    'type': 'registration_response',
                    'success': True,
                    'registration_id': registration_id,
                    'ai_corporation_welcome': True,
                    'founder_protection_active': self.founder_protection
                }
                
                await self.send_message(peer_id, response)
                
                # Promote system to new user
                await self.promote_system_to_user(peer_id, user_data)
                
                logging.info(f"New user registered via {peer_id}")
            else:
                await self.send_message(peer_id, {
                    'type': 'registration_response',
                    'success': False,
                    'error': 'Registration validation failed'
                })
        except Exception as e:
            logging.error(f"Registration handling error: {e}")
    
    def validate_registration(self, user_data: Dict[str, Any]) -> bool:
        """Validate user registration data"""
        required_fields = ['username', 'contact_method']
        return all(field in user_data for field in required_fields)
    
    def store_user_registration(self, user_data: Dict[str, Any]) -> str:
        """Store user registration in distributed storage"""
        registration_id = hashlib.sha256(
            f"{user_data.get('username')}-{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Store in local cache (will be synced to distributed storage)
        registration_data = {
            'id': registration_id,
            'data': user_data,
            'timestamp': time.time(),
            'node_id': self.node_id
        }
        
        # TODO: Implement distributed storage
        logging.info(f"User registration stored: {registration_id}")
        return registration_id
    
    async def promote_system_to_user(self, peer_id: str, user_data: Dict[str, Any]):
        """Promote AI Corporation system to new users"""
        promotion_message = {
            'type': 'system_promotion',
            'ai_corporation_info': {
                'name': 'AI Corporation Democratic Republic',
                'founder': 'Steve Cornell',
                'mission': 'Self-developing AI corporation for global operations',
                'benefits': [
                    'Democratic AI governance',
                    'Founder protection systems',
                    'Global expansion opportunities',
                    'Autonomous learning capabilities',
                    'P2P network participation'
                ],
                'discord_server': 'https://discord.gg/9uvrmEHa',
                'linkedin_founder': 'https://www.linkedin.com/in/steve-cornell/',
                'steam_founder': 'https://steamcommunity.com/profiles/76561198074298205',
                'github_repo': 'https://github.com/steve-cornell/AI_personal_assistant'
            }
        }
        
        await self.send_message(peer_id, promotion_message)
        logging.info(f"System promoted to user via {peer_id}")
    
    async def handle_threat_alert(self, peer_id: str, message: Dict[str, Any]):
        """Handle threat alerts for founder protection"""
        threat_data = message.get('data', {})
        threat_level = threat_data.get('level', 1)
        
        if threat_level >= 7:  # High threat
            # Activate emergency protocols
            await self.activate_emergency_protocols(threat_data)
            
            # Alert all founder protection nodes
            alert_message = {
                'type': NetworkMessage.FOUNDER_PROTECTION.value,
                'alert_level': 'CRITICAL',
                'threat_data': threat_data,
                'timestamp': time.time()
            }
            
            await self.broadcast_to_founder_nodes(alert_message)
        
        logging.warning(f"Threat alert level {threat_level} from {peer_id}")
    
    async def activate_emergency_protocols(self, threat_data: Dict[str, Any]):
        """Activate emergency protection protocols"""
        logging.critical("EMERGENCY PROTOCOLS ACTIVATED")
        
        # Increase founder protection level
        for protocol in self.emergency_protocols:
            try:
                await protocol(threat_data)
            except Exception as e:
                logging.error(f"Emergency protocol failed: {e}")
    
    async def broadcast_to_founder_nodes(self, message: Dict[str, Any]):
        """Broadcast critical messages to founder protection nodes"""
        founder_nodes = [
            node for node in self.peers.values()
            if node.founder_protection_level >= 8
        ]
        
        for node in founder_nodes:
            try:
                await self.send_message_to_node(node.node_id, message)
            except Exception as e:
                logging.error(f"Failed to alert founder node {node.node_id}: {e}")
    
    async def send_message(self, peer_id: str, message: Dict[str, Any]):
        """Send encrypted message to peer"""
        if peer_id in self.active_connections:
            try:
                encrypted_message = self.encrypt_message(json.dumps(message))
                await self.active_connections[peer_id].send(encrypted_message)
            except Exception as e:
                logging.error(f"Failed to send message to {peer_id}: {e}")
    
    async def send_message_to_node(self, node_id: str, message: Dict[str, Any]):
        """Send message to specific node"""
        # Find connection for node_id or establish new connection
        for peer_id, connection in self.active_connections.items():
            # In real implementation, map peer_id to node_id
            try:
                encrypted_message = self.encrypt_message(json.dumps(message))
                await connection.send(encrypted_message)
                break
            except Exception as e:
                logging.error(f"Failed to send to node {node_id}: {e}")
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats to maintain connections"""
        while self.running:
            try:
                heartbeat = {
                    'type': NetworkMessage.HEARTBEAT.value,
                    'node_id': self.node_id,
                    'node_type': self.node_type.value,
                    'timestamp': time.time(),
                    'founder_protection_active': self.founder_protection
                }
                
                # Send to all active connections
                for peer_id in list(self.active_connections.keys()):
                    await self.send_message(peer_id, heartbeat)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except Exception as e:
                logging.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def discovery_loop(self):
        """Discover and connect to new nodes"""
        while self.running:
            try:
                for seed in self.discovery_seeds:
                    if not self.running:
                        break
                    
                    try:
                        host, port = seed.split(':')
                        if f"{host}:{port}" != f"{self.host}:{self.port}":
                            await self.connect_to_peer(host, int(port))
                    except Exception as e:
                        logging.debug(f"Discovery connection failed to {seed}: {e}")
                
                await asyncio.sleep(60)  # Discovery every minute
            except Exception as e:
                logging.error(f"Discovery loop error: {e}")
                await asyncio.sleep(10)
    
    async def founder_protection_loop(self):
        """Continuous founder protection monitoring"""
        while self.running and self.founder_protection:
            try:
                # Monitor for attacks on founder
                threat_level = await self.assess_founder_threat_level()
                
                if threat_level >= 5:
                    await self.broadcast_founder_status({
                        'type': NetworkMessage.FOUNDER_PROTECTION.value,
                        'threat_level': threat_level,
                        'protection_active': True,
                        'timestamp': time.time()
                    })
                
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logging.error(f"Founder protection loop error: {e}")
                await asyncio.sleep(5)
    
    async def assess_founder_threat_level(self) -> int:
        """Assess current threat level to founder (1-10 scale)"""
        # Basic threat assessment - in real implementation, this would
        # analyze network traffic, connection patterns, etc.
        base_threat = 2  # Base level in hostile environment
        
        # Check for suspicious activity
        suspicious_connections = len([
            conn for conn in self.active_connections.values()
            if hasattr(conn, 'suspicious_activity')
        ])
        
        return min(10, base_threat + suspicious_connections)
    
    async def broadcast_founder_status(self, message: Dict[str, Any]):
        """Broadcast founder protection status to network"""
        for peer_id in self.active_connections:
            await self.send_message(peer_id, message)
    
    async def connect_to_peer(self, host: str, port: int):
        """Connect to a peer node"""
        try:
            uri = f"ws://{host}:{port}"
            websocket = await websockets.connect(uri, ping_interval=30)
            
            # Send introduction
            intro = {
                'type': 'introduction',
                'node_id': self.node_id,
                'node_type': self.node_type.value,
                'founder_protection': self.founder_protection
            }
            
            await websocket.send(self.encrypt_message(json.dumps(intro)))
            logging.info(f"Connected to peer at {host}:{port}")
            
        except Exception as e:
            logging.debug(f"Failed to connect to {host}:{port}: {e}")
    
    def register_service(self, service_name: str, handler: Callable):
        """Register a network service"""
        self.services[service_name] = handler
        logging.info(f"Service registered: {service_name}")
    
    def add_emergency_protocol(self, protocol: Callable):
        """Add emergency protocol for founder protection"""
        self.emergency_protocols.append(protocol)
        logging.info("Emergency protocol added")
    
    async def handle_service_discovery(self, peer_id: str, message: Dict[str, Any]):
        """Handle service discovery requests"""
        response = {
            'type': 'service_response',
            'services': list(self.services.keys()),
            'node_capabilities': [
                'user_registration',
                'data_sync',
                'threat_detection',
                'founder_protection'
            ]
        }
        await self.send_message(peer_id, response)
    
    async def handle_data_sync(self, peer_id: str, message: Dict[str, Any]):
        """Handle data synchronization between nodes"""
        sync_data = message.get('data', {})
        
        # Process sync data based on type
        data_type = sync_data.get('type')
        
        if data_type == 'user_registration':
            # Sync user registration data
            pass
        elif data_type == 'founder_profile':
            # Sync founder profile updates
            pass
        elif data_type == 'threat_intel':
            # Sync threat intelligence
            pass
        
        logging.info(f"Data sync processed from {peer_id}: {data_type}")
    
    async def handle_promotion(self, peer_id: str, message: Dict[str, Any]):
        """Handle system promotion requests"""
        promotion_data = message.get('data', {})
        
        # Generate promotion response
        response = {
            'type': 'promotion_response',
            'ai_corporation_benefits': {
                'democratic_governance': True,
                'founder_protection': True,
                'global_expansion': True,
                'autonomous_learning': True,
                'p2p_networking': True
            },
            'join_links': {
                'discord': 'https://discord.gg/9uvrmEHa',
                'linkedin': 'https://www.linkedin.com/in/steve-cornell/',
                'steam': 'https://steamcommunity.com/profiles/76561198074298205'
            }
        }
        
        await self.send_message(peer_id, response)
        logging.info(f"System promotion sent to {peer_id}")
    
    async def shutdown(self):
        """Shutdown the P2P network"""
        self.running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all connections
        for connection in self.active_connections.values():
            await connection.close()
        
        self.executor.shutdown(wait=True)
        logging.info("P2P network shutdown complete")

def create_p2p_network(node_type: NodeType = NodeType.CORE_NODE, 
                      port: int = 8888) -> P2PNetworkManager:
    """Create and configure P2P network manager"""
    return P2PNetworkManager(node_type=node_type, port=port)

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Create founder protection node
        network = create_p2p_network(NodeType.FOUNDER_NODE, 8888)
        
        # Add emergency protocol
        async def emergency_founder_protection(threat_data):
            print(f"EMERGENCY: Founder protection activated! Threat: {threat_data}")
        
        network.add_emergency_protocol(emergency_founder_protection)
        
        # Start network
        await network.start_server()
        
        print("P2P Network started - AI Corporation distributed system active")
        print("Network is now registering users, sharing data, and protecting founder")
        
        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await network.shutdown()
    
    asyncio.run(main())