#!/usr/bin/env python3
"""
P2P Toolkit - Peer-to-Peer Communication and Model Sharing

This module provides P2P functionality for the AI Personal Assistant toolkit,
enabling distributed model sharing, collaborative AI workflows, and peer discovery.
"""

import asyncio
import json
import socket
import threading
import time
import logging
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures

# Set up logging with UTF-8 safe format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('p2p_toolkit.log', encoding='utf-8', errors='replace'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_p2p_imports():
    """Check availability of optional P2P dependencies"""
    components = {
        'asyncio': True,  # Built-in
        'websockets': False,
        'aiohttp': False,
        'upnp': False,
        'cryptography': False
    }
    
    try:
        import websockets
        components['websockets'] = True
        logger.info("[OK] WebSockets available for P2P communication")
    except ImportError:
        logger.warning("[MISSING] websockets not available - install with: pip install websockets")
    
    try:
        import aiohttp
        components['aiohttp'] = True
        logger.info("[OK] aiohttp available for HTTP P2P")
    except ImportError:
        logger.warning("[MISSING] aiohttp not available - install with: pip install aiohttp")
    
    try:
        import upnpclient
        components['upnp'] = True
        logger.info("[OK] UPnP available for NAT traversal")
    except ImportError:
        logger.warning("[MISSING] upnpclient not available - install with: pip install upnpclient")
    
    try:
        from cryptography.fernet import Fernet
        components['cryptography'] = True
        logger.info("[OK] Cryptography available for secure P2P")
    except ImportError:
        logger.warning("[MISSING] cryptography not available - install with: pip install cryptography")
    
    return components

@dataclass
class PeerInfo:
    """Information about a peer in the network"""
    peer_id: str
    host: str
    port: int
    name: str
    capabilities: List[str]
    models: List[str]
    last_seen: datetime
    public_key: Optional[str] = None
    status: str = "online"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['last_seen'] = self.last_seen.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PeerInfo':
        """Create from dictionary"""
        data['last_seen'] = datetime.fromisoformat(data['last_seen'])
        return cls(**data)

@dataclass
class P2PMessage:
    """P2P message structure"""
    message_id: str
    sender_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'P2PMessage':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class P2PNode:
    """P2P Node for distributed AI communication"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8888, name: str = None):
        """
        Initialize P2P Node
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            name: Friendly name for this node
        """
        self.host = host
        self.port = port
        self.name = name or f"AI-Node-{uuid.uuid4().hex[:8]}"
        self.peer_id = str(uuid.uuid4())
        
        # P2P state
        self.peers: Dict[str, PeerInfo] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        self.server = None
        
        # Check available components
        self.components = check_p2p_imports()
        
        # Model and capability tracking
        self.local_models: List[str] = []
        self.capabilities: List[str] = ["chat", "generate", "discover"]
        
        # Security (if cryptography available)
        self.encryption_key = None
        if self.components['cryptography']:
            self.setup_encryption()
        
        logger.info(f"[INIT] P2P Node initialized: {self.name} ({self.peer_id[:8]})")
    
    def setup_encryption(self):
        """Setup encryption for secure P2P communication"""
        try:
            from cryptography.fernet import Fernet
            self.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
            logger.info("[OK] Encryption setup complete")
        except Exception as e:
            logger.error(f"[ERROR] Failed to setup encryption: {e}")
    
    def encrypt_message(self, message: str) -> str:
        """Encrypt a message if encryption is available"""
        if self.encryption_key and self.components['cryptography']:
            try:
                return self.cipher.encrypt(message.encode()).decode()
            except Exception as e:
                logger.error(f"[ERROR] Encryption failed: {e}")
        return message
    
    def decrypt_message(self, encrypted_message: str) -> str:
        """Decrypt a message if encryption is available"""
        if self.encryption_key and self.components['cryptography']:
            try:
                return self.cipher.decrypt(encrypted_message.encode()).decode()
            except Exception as e:
                logger.error(f"[ERROR] Decryption failed: {e}")
        return encrypted_message
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler for specific message types"""
        self.message_handlers[message_type] = handler
        logger.info(f"[OK] Handler registered for message type: {message_type}")
    
    def add_peer(self, peer_info: PeerInfo):
        """Add a peer to the network"""
        self.peers[peer_info.peer_id] = peer_info
        logger.info(f"[OK] Peer added: {peer_info.name} ({peer_info.peer_id[:8]})")
    
    def remove_peer(self, peer_id: str):
        """Remove a peer from the network"""
        if peer_id in self.peers:
            peer = self.peers.pop(peer_id)
            logger.info(f"[OK] Peer removed: {peer.name} ({peer_id[:8]})")
    
    def get_peer_info(self) -> PeerInfo:
        """Get this node's peer info"""
        return PeerInfo(
            peer_id=self.peer_id,
            host=self.host,
            port=self.port,
            name=self.name,
            capabilities=self.capabilities,
            models=self.local_models,
            last_seen=datetime.now()
        )
    
    def discover_peers(self, broadcast_port: int = 8889) -> List[PeerInfo]:
        """Discover peers on the local network using UDP broadcast"""
        discovered_peers = []
        
        try:
            # Create UDP socket for broadcasting
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(5.0)
            
            # Broadcast discovery message
            discovery_msg = {
                "type": "peer_discovery",
                "sender": self.get_peer_info().to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            message = json.dumps(discovery_msg).encode()
            sock.sendto(message, ('<broadcast>', broadcast_port))
            logger.info(f"[OK] Broadcasting peer discovery on port {broadcast_port}")
            
            # Listen for responses
            start_time = time.time()
            while time.time() - start_time < 5.0:
                try:
                    data, addr = sock.recvfrom(1024)
                    response = json.loads(data.decode())
                    
                    if response.get("type") == "peer_response":
                        peer_data = response.get("peer")
                        if peer_data:
                            peer = PeerInfo.from_dict(peer_data)
                            if peer.peer_id != self.peer_id:  # Don't add ourselves
                                discovered_peers.append(peer)
                                self.add_peer(peer)
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"[ERROR] Error processing discovery response: {e}")
            
            sock.close()
            logger.info(f"[OK] Discovery complete: found {len(discovered_peers)} peers")
            
        except Exception as e:
            logger.error(f"[ERROR] Peer discovery failed: {e}")
        
        return discovered_peers
    
    def start_discovery_listener(self, broadcast_port: int = 8889):
        """Start listening for peer discovery broadcasts"""
        def listen_for_discoveries():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(('', broadcast_port))
                sock.settimeout(1.0)
                
                logger.info(f"[OK] Discovery listener started on port {broadcast_port}")
                
                while self.running:
                    try:
                        data, addr = sock.recvfrom(1024)
                        message = json.loads(data.decode())
                        
                        if message.get("type") == "peer_discovery":
                            sender_data = message.get("sender")
                            if sender_data:
                                peer = PeerInfo.from_dict(sender_data)
                                if peer.peer_id != self.peer_id:
                                    self.add_peer(peer)
                                    
                                    # Send response
                                    response = {
                                        "type": "peer_response",
                                        "peer": self.get_peer_info().to_dict()
                                    }
                                    response_data = json.dumps(response).encode()
                                    sock.sendto(response_data, addr)
                    
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.running:
                            logger.error(f"[ERROR] Discovery listener error: {e}")
                
                sock.close()
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to start discovery listener: {e}")
        
        # Start listener in background thread
        listener_thread = threading.Thread(target=listen_for_discoveries, daemon=True)
        listener_thread.start()
    
    async def send_message(self, peer_id: str, message_type: str, payload: Dict[str, Any]) -> bool:
        """Send a message to a specific peer"""
        if peer_id not in self.peers:
            logger.error(f"[ERROR] Peer {peer_id[:8]} not found")
            return False
        
        peer = self.peers[peer_id]
        
        # Create message
        message = P2PMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.peer_id,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now()
        )
        
        try:
            if self.components['websockets']:
                # Use WebSockets if available
                import websockets
                uri = f"ws://{peer.host}:{peer.port}/p2p"
                
                async with websockets.connect(uri) as websocket:
                    message_data = json.dumps(message.to_dict())
                    encrypted_data = self.encrypt_message(message_data)
                    await websocket.send(encrypted_data)
                    logger.info(f"[OK] Message sent to {peer.name} ({peer_id[:8]})")
                    return True
            else:
                # Fallback to basic socket communication
                logger.warning("[FALLBACK] Using basic socket communication")
                return self._send_message_socket(peer, message)
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to send message to {peer_id[:8]}: {e}")
            return False
    
    def _send_message_socket(self, peer: PeerInfo, message: P2PMessage) -> bool:
        """Fallback socket-based message sending"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(10.0)
                sock.connect((peer.host, peer.port))
                
                message_data = json.dumps(message.to_dict())
                encrypted_data = self.encrypt_message(message_data)
                
                # Send message length first, then message
                message_bytes = encrypted_data.encode()
                length = len(message_bytes)
                sock.sendall(length.to_bytes(4, byteorder='big'))
                sock.sendall(message_bytes)
                
                logger.info(f"[OK] Socket message sent to {peer.name}")
                return True
                
        except Exception as e:
            logger.error(f"[ERROR] Socket message failed: {e}")
            return False
    
    def broadcast_message(self, message_type: str, payload: Dict[str, Any]):
        """Broadcast a message to all connected peers"""
        for peer_id in self.peers:
            asyncio.create_task(self.send_message(peer_id, message_type, payload))
    
    def update_local_models(self, models: List[str]):
        """Update the list of locally available models"""
        self.local_models = models
        logger.info(f"[OK] Local models updated: {len(models)} models")
        
        # Broadcast model update to peers
        self.broadcast_message("model_update", {
            "peer_id": self.peer_id,
            "models": models
        })
    
    def find_model_peers(self, model_name: str) -> List[PeerInfo]:
        """Find peers that have a specific model"""
        peers_with_model = []
        for peer in self.peers.values():
            if model_name in peer.models:
                peers_with_model.append(peer)
        return peers_with_model
    
    async def request_model_inference(self, model_name: str, prompt: str) -> Optional[str]:
        """Request model inference from a peer that has the model"""
        peers_with_model = self.find_model_peers(model_name)
        
        if not peers_with_model:
            logger.warning(f"[MISSING] No peers found with model: {model_name}")
            return None
        
        # Try the first available peer
        peer = peers_with_model[0]
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "request_id": str(uuid.uuid4())
        }
        
        success = await self.send_message(peer.peer_id, "inference_request", payload)
        if success:
            logger.info(f"[OK] Inference request sent for model: {model_name}")
            # In a real implementation, you'd wait for the response
            return f"Inference requested from {peer.name}"
        
        return None
    
    def start(self):
        """Start the P2P node"""
        self.running = True
        
        # Start discovery listener
        self.start_discovery_listener()
        
        # Discover initial peers
        initial_peers = self.discover_peers()
        logger.info(f"[OK] P2P Node started with {len(initial_peers)} initial peers")
        
        # Register default handlers
        self.register_default_handlers()
    
    def stop(self):
        """Stop the P2P node"""
        self.running = False
        if self.server:
            self.server.close()
        logger.info("[OK] P2P Node stopped")
    
    def register_default_handlers(self):
        """Register default message handlers"""
        def handle_ping(message: P2PMessage):
            logger.info(f"[PING] Received ping from {message.sender_id[:8]}")
            # Could send pong back
        
        def handle_model_update(message: P2PMessage):
            peer_id = message.payload.get("peer_id")
            models = message.payload.get("models", [])
            if peer_id in self.peers:
                self.peers[peer_id].models = models
                logger.info(f"[OK] Updated models for peer {peer_id[:8]}: {len(models)} models")
        
        def handle_inference_request(message: P2PMessage):
            model = message.payload.get("model")
            prompt = message.payload.get("prompt")
            request_id = message.payload.get("request_id")
            logger.info(f"[REQUEST] Inference request for {model}: {prompt[:50]}...")
            # In a real implementation, you'd process this with local Ollama
        
        self.register_handler("ping", handle_ping)
        self.register_handler("model_update", handle_model_update)
        self.register_handler("inference_request", handle_inference_request)

class P2PModelSharing:
    """P2P Model Sharing functionality"""
    
    def __init__(self, node: P2PNode):
        self.node = node
        self.shared_models: Dict[str, Dict[str, Any]] = {}
        logger.info("[INIT] P2P Model Sharing initialized")
    
    def share_model(self, model_name: str, model_path: str, description: str = ""):
        """Share a model with the P2P network"""
        model_info = {
            "name": model_name,
            "path": model_path,
            "description": description,
            "size": self._get_model_size(model_path),
            "checksum": self._calculate_checksum(model_path),
            "shared_at": datetime.now().isoformat()
        }
        
        self.shared_models[model_name] = model_info
        
        # Broadcast model availability
        self.node.broadcast_message("model_shared", {
            "model": model_info,
            "peer_id": self.node.peer_id
        })
        
        logger.info(f"[OK] Model shared: {model_name}")
    
    def _get_model_size(self, model_path: str) -> int:
        """Get model file size"""
        try:
            return Path(model_path).stat().st_size
        except:
            return 0
    
    def _calculate_checksum(self, model_path: str) -> str:
        """Calculate model checksum for integrity verification"""
        try:
            with open(model_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return ""
    
    async def request_model(self, model_name: str, peer_id: str) -> bool:
        """Request a model from a specific peer"""
        payload = {
            "model_name": model_name,
            "requester_id": self.node.peer_id
        }
        
        return await self.node.send_message(peer_id, "model_request", payload)

# Convenience functions for easy P2P usage
def create_p2p_node(name: str = None, port: int = 8888) -> P2PNode:
    """Create and start a P2P node"""
    node = P2PNode(name=name, port=port)
    node.start()
    return node

def discover_ai_peers(timeout: int = 5) -> List[PeerInfo]:
    """Quick peer discovery without creating a persistent node"""
    temp_node = P2PNode()
    peers = temp_node.discover_peers()
    return peers

async def request_distributed_inference(model_name: str, prompt: str, node: P2PNode = None) -> Optional[str]:
    """Request inference from the P2P network"""
    if not node:
        node = create_p2p_node()
    
    return await node.request_model_inference(model_name, prompt)

# Integration with existing toolkit
def integrate_with_ollama_toolkit():
    """Integration point with existing Ollama toolkit"""
    try:
        from ollama_toolkit import OllamaToolkit
        logger.info("[OK] P2P integration with Ollama toolkit available")
        return True
    except ImportError:
        logger.warning("[MISSING] Ollama toolkit not available for P2P integration")
        return False

# Integration utilities only - no demos