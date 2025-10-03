#!/usr/bin/env python3
"""
P2P + Ollama Integration Example

This example shows how to integrate P2P functionality with the existing Ollama toolkit
for distributed AI model inference and sharing.
"""

import asyncio
import time
from typing import List, Optional

def check_integration_dependencies():
    """Check if both P2P and Ollama toolkits are available"""
    components = {
        'p2p': False,
        'ollama': False
    }
    
    try:
        from p2p_toolkit import P2PNode, P2PModelSharing
        components['p2p'] = True
        print("[OK] P2P Toolkit available")
    except ImportError:
        print("[MISSING] P2P Toolkit not available")
    
    try:
        from ollama_toolkit import OllamaToolkit, list_available_models
        components['ollama'] = True
        print("[OK] Ollama Toolkit available")
    except ImportError:
        print("[MISSING] Ollama Toolkit not available")
    
    return components

class DistributedOllamaNode:
    """Distributed Ollama node with P2P capabilities"""
    
    def __init__(self, name: str = None, port: int = 8888):
        """Initialize distributed Ollama node"""
        self.components = check_integration_dependencies()
        
        # Initialize P2P if available
        if self.components['p2p']:
            from p2p_toolkit import P2PNode, P2PModelSharing
            self.p2p_node = P2PNode(name=name or "Ollama-P2P-Node", port=port)
            self.model_sharing = P2PModelSharing(self.p2p_node)
        else:
            self.p2p_node = None
            self.model_sharing = None
        
        # Initialize Ollama if available
        if self.components['ollama']:
            from ollama_toolkit import OllamaToolkit
            self.ollama = OllamaToolkit()
        else:
            self.ollama = None
        
        print(f"[INIT] Distributed Ollama Node initialized")
        print(f"  P2P: {'✓' if self.p2p_node else '✗'}")
        print(f"  Ollama: {'✓' if self.ollama else '✗'}")
    
    def start(self):
        """Start the distributed node"""
        if self.p2p_node:
            # Register P2P message handlers
            self.setup_p2p_handlers()
            self.p2p_node.start()
            
            # Sync local models with P2P network
            self.sync_models_with_p2p()
            
            print("[OK] Distributed Ollama node started")
        else:
            print("[ERROR] Cannot start - P2P not available")
    
    def stop(self):
        """Stop the distributed node"""
        if self.p2p_node:
            self.p2p_node.stop()
            print("[OK] Distributed Ollama node stopped")
    
    def setup_p2p_handlers(self):
        """Setup P2P message handlers for Ollama operations"""
        if not self.p2p_node:
            return
        
        def handle_inference_request(message):
            """Handle remote inference requests"""
            try:
                model = message.payload.get('model')
                prompt = message.payload.get('prompt')
                request_id = message.payload.get('request_id')
                
                print(f"[P2P] Inference request: {model} - {prompt[:50]}...")
                
                if self.ollama and self.has_model_locally(model):
                    # Process inference locally
                    response = self.ollama.chat(model, prompt)
                    
                    # Send response back (simplified - would need proper response routing)
                    print(f"[P2P] Inference completed for request {request_id}")
                else:
                    print(f"[P2P] Model {model} not available locally")
                    
            except Exception as e:
                print(f"[ERROR] P2P inference error: {e}")
        
        def handle_model_discovery(message):
            """Handle model discovery requests"""
            try:
                requested_models = message.payload.get('models', [])
                available_models = self.get_local_models()
                
                matching_models = [m for m in requested_models if m in available_models]
                
                if matching_models:
                    # Send model availability response
                    response_payload = {
                        'available_models': matching_models,
                        'node_id': self.p2p_node.peer_id
                    }
                    
                    # Would send response back to requester
                    print(f"[P2P] Model discovery: {len(matching_models)} matches found")
                
            except Exception as e:
                print(f"[ERROR] Model discovery error: {e}")
        
        # Register handlers
        self.p2p_node.register_handler('inference_request', handle_inference_request)
        self.p2p_node.register_handler('model_discovery', handle_model_discovery)
        
        print("[OK] P2P handlers registered")
    
    def sync_models_with_p2p(self):
        """Sync local Ollama models with P2P network"""
        if not self.p2p_node or not self.ollama:
            return
        
        try:
            local_models = self.get_local_models()
            self.p2p_node.update_local_models(local_models)
            print(f"[OK] Synced {len(local_models)} models with P2P network")
        except Exception as e:
            print(f"[ERROR] Model sync failed: {e}")
    
    def get_local_models(self) -> List[str]:
        """Get list of locally available Ollama models"""
        if not self.ollama:
            return []
        
        try:
            from ollama_toolkit import list_available_models
            return list_available_models()
        except Exception as e:
            print(f"[ERROR] Failed to get local models: {e}")
            return []
    
    def has_model_locally(self, model_name: str) -> bool:
        """Check if model is available locally"""
        local_models = self.get_local_models()
        return model_name in local_models
    
    async def distributed_inference(self, model: str, prompt: str) -> Optional[str]:
        """Perform inference using distributed P2P network"""
        if not self.p2p_node:
            print("[ERROR] P2P not available for distributed inference")
            return None
        
        # First try local inference
        if self.ollama and self.has_model_locally(model):
            print(f"[LOCAL] Using local model: {model}")
            try:
                return self.ollama.chat(model, prompt)
            except Exception as e:
                print(f"[ERROR] Local inference failed: {e}")
        
        # Try distributed inference
        print(f"[P2P] Requesting distributed inference for: {model}")
        
        # Find peers with the model
        peers_with_model = self.p2p_node.find_model_peers(model)
        
        if not peers_with_model:
            print(f"[ERROR] No peers found with model: {model}")
            return None
        
        # Request inference from first available peer
        peer = peers_with_model[0]
        print(f"[P2P] Requesting inference from: {peer.name}")
        
        try:
            success = await self.p2p_node.send_message(
                peer.peer_id, 
                'inference_request', 
                {
                    'model': model,
                    'prompt': prompt,
                    'request_id': f"req_{int(time.time())}"
                }
            )
            
            if success:
                return f"Inference requested from {peer.name} (P2P response handling would be implemented here)"
            else:
                return None
                
        except Exception as e:
            print(f"[ERROR] Distributed inference failed: {e}")
            return None
    
    def discover_network_models(self) -> List[str]:
        """Discover all models available in the P2P network"""
        if not self.p2p_node:
            return []
        
        all_models = set()
        
        # Add local models
        local_models = self.get_local_models()
        all_models.update(local_models)
        
        # Add peer models
        for peer in self.p2p_node.peers.values():
            all_models.update(peer.models)
        
        network_models = list(all_models)
        print(f"[DISCOVERY] Found {len(network_models)} models in network")
        
        return network_models
    
    def get_network_status(self):
        """Get status of the P2P network"""
        if not self.p2p_node:
            return {"error": "P2P not available"}
        
        status = {
            "node_id": self.p2p_node.peer_id,
            "node_name": self.p2p_node.name,
            "peers_connected": len(self.p2p_node.peers),
            "local_models": len(self.get_local_models()),
            "network_models": len(self.discover_network_models()),
            "capabilities": self.p2p_node.capabilities
        }
        
        return status

async def demo_distributed_ollama():
    """Demo distributed Ollama functionality"""
    print("🔗 DISTRIBUTED OLLAMA DEMO")
    print("=" * 40)
    
    # Create distributed node
    node = DistributedOllamaNode("Demo-Distributed-Node")
    
    # Start the node
    node.start()
    
    # Show network status
    print("\n📊 Network Status:")
    status = node.get_network_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Discover network models
    print("\n🔍 Network Model Discovery:")
    network_models = node.discover_network_models()
    if network_models:
        print(f"  Found {len(network_models)} models:")
        for model in network_models[:5]:  # Show first 5
            print(f"    - {model}")
    else:
        print("  No models found (install some with: ollama pull llama3.2)")
    
    # Try distributed inference if models are available
    if network_models:
        print(f"\n🤖 Testing Distributed Inference:")
        test_model = network_models[0]
        test_prompt = "What is artificial intelligence?"
        
        print(f"  Model: {test_model}")
        print(f"  Prompt: {test_prompt}")
        
        result = await node.distributed_inference(test_model, test_prompt)
        if result:
            print(f"  Response: {result[:100]}...")
        else:
            print("  Inference failed or timed out")
    # Integration utility functions only - no demos