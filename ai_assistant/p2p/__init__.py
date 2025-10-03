"""
P2P (Peer-to-Peer) functionality for AI Assistant

This package provides peer-to-peer communication capabilities:
- P2P node management and discovery
- Distributed model sharing
- Integration with Ollama toolkit
"""

from .node import P2PNode, P2PModelSharing, create_p2p_node, discover_ai_peers

# Integration imports with graceful degradation
try:
    from .integration import DistributedOllamaNode
    __all__ = ["P2PNode", "P2PModelSharing", "create_p2p_node", "discover_ai_peers", "DistributedOllamaNode"]
except ImportError:
    __all__ = ["P2PNode", "P2PModelSharing", "create_p2p_node", "discover_ai_peers"]