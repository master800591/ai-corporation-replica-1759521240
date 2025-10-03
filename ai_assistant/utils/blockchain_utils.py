#!/usr/bin/env python3
"""
Blockchain Utility Functions

Pure utility functions for blockchain operations.
No demos, no interactive code, just tools.
"""

from typing import List, Dict, Any, Optional
from ..blockchain.core import Blockchain, Transaction
from ..blockchain.consensus import PoAConsensus, create_test_consensus
from ..blockchain.wallet import Wallet, WalletManager
from ..blockchain.contracts import ContractManager, ContractType


def create_blockchain(authorities: List[str]) -> Blockchain:
    """Create a blockchain with given authorities"""
    return Blockchain(authorities)


def create_wallet_manager() -> WalletManager:
    """Create a wallet manager instance"""
    return WalletManager()


def create_contract_manager() -> ContractManager:
    """Create a contract manager instance"""
    return ContractManager()


def validate_blockchain(blockchain: Blockchain) -> bool:
    """Validate a blockchain"""
    return blockchain.is_valid_chain()


def get_blockchain_stats(blockchain: Blockchain) -> Dict[str, Any]:
    """Get blockchain statistics"""
    return {
        'height': len(blockchain.chain),
        'pending_transactions': len(blockchain.pending_transactions),
        'total_transactions': sum(len(block.transactions) for block in blockchain.chain),
        'authorities': len(blockchain.consensus.authorities),
        'is_valid': blockchain.is_valid_chain()
    }


def create_simple_transaction(sender_wallet: Wallet, recipient_addr: str, amount: float) -> Transaction:
    """Create a simple transaction"""
    return sender_wallet.create_transaction(recipient_addr, amount)


def deploy_ai_model_contract(contract_manager: ContractManager, owner: str) -> str:
    """Deploy an AI model contract"""
    return contract_manager.deploy_contract(ContractType.AI_MODEL, owner)


def deploy_data_sharing_contract(contract_manager: ContractManager, owner: str) -> str:
    """Deploy a data sharing contract"""
    return contract_manager.deploy_contract(ContractType.DATA_SHARING, owner)


def get_wallet_info(wallet: Wallet) -> Dict[str, Any]:
    """Get wallet information"""
    info = wallet.get_wallet_info()
    return info.to_dict()


def backup_wallet(wallet: Wallet) -> Dict[str, Any]:
    """Backup wallet data"""
    return wallet.backup_wallet()


def restore_wallet_from_backup(backup_data: Dict[str, Any]) -> Wallet:
    """Restore wallet from backup"""
    return Wallet.restore_wallet(backup_data)