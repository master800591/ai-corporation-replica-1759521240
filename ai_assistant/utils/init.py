#!/usr/bin/env python3
"""
Utils Package

Pure utility functions for the AI Personal Assistant.
No demos, no interactive code, just reusable tools.
"""

from .blockchain_utils import (
    create_blockchain,
    create_wallet_manager, 
    create_contract_manager,
    validate_blockchain,
    get_blockchain_stats,
    create_simple_transaction,
    deploy_ai_model_contract,
    deploy_data_sharing_contract,
    get_wallet_info,
    backup_wallet,
    restore_wallet_from_backup
)

from .ollama_utils import (
    create_ollama_toolkit,
    create_model_manager,
    check_ollama_availability,
    get_ollama_status,
    list_local_models,
    get_popular_models,
    pull_model
)

from .p2p_utils import (
    create_p2p_node_if_available,
    check_p2p_availability,
    get_p2p_status
)

from .platform_utils import (
    create_project_structure,
    get_project_info,
    list_projects,
    create_python_module,
    create_config_file
)

__all__ = [
    # Blockchain utilities
    'create_blockchain',
    'create_wallet_manager', 
    'create_contract_manager',
    'validate_blockchain',
    'get_blockchain_stats',
    'create_simple_transaction',
    'deploy_ai_model_contract',
    'deploy_data_sharing_contract',
    'get_wallet_info',
    'backup_wallet',
    'restore_wallet_from_backup',
    
    # Ollama utilities
    'create_ollama_toolkit',
    'create_model_manager',
    'check_ollama_availability',
    'get_ollama_status',
    'list_local_models',
    'get_popular_models',
    'pull_model',
    
    # P2P utilities
    'create_p2p_node_if_available',
    'check_p2p_availability',
    'get_p2p_status',
    
    # Platform utilities
    'create_project_structure',
    'get_project_info',
    'list_projects',
    'create_python_module',
    'create_config_file'
]