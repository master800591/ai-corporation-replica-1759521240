#!/usr/bin/env python3
"""
P2P Utility Functions

Pure utility functions for peer-to-peer operations.
No demos, no interactive code, just tools.
"""

from typing import List, Dict, Any, Optional

try:
    from ..p2p.node import P2PNode, create_p2p_node
    P2P_AVAILABLE = True
except ImportError:
    P2P_AVAILABLE = False


def create_p2p_node_if_available(name: str = "P2P-Node", port: int = 8888):
    """Create P2P node if dependencies are available"""
    if not P2P_AVAILABLE:
        return None
    return create_p2p_node(name, port)


def check_p2p_availability() -> bool:
    """Check if P2P functionality is available"""
    return P2P_AVAILABLE


def get_p2p_status() -> Dict[str, Any]:
    """Get P2P system status"""
    return {
        'available': P2P_AVAILABLE,
        'dependencies': {
            'websockets': _check_dependency('websockets'),
            'aiohttp': _check_dependency('aiohttp'),
        }
    }


def _check_dependency(module_name: str) -> bool:
    """Check if a dependency is available"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False