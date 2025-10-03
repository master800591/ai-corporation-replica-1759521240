#!/usr/bin/env python3
"""
Smart Contract System for AI Model Sharing

This module implements a smart contract system for:
- AI model registration and sharing
- Model usage tracking and monetization
- Reputation-based access control
- Data sharing agreements
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from .core import Transaction
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ContractType(Enum):
    """Smart contract types"""
    AI_MODEL = "ai_model"
    DATA_SHARING = "data_sharing"
    PAYMENT = "payment"

@dataclass
class ModelInfo:
    """Information about an AI model"""
    model_id: str
    name: str
    description: str
    owner: str
    model_type: str  # e.g., "llm", "embedding", "vision"
    version: str
    size_mb: float
    accuracy_metrics: Dict[str, float]
    usage_price: float  # Price per inference
    sharing_price: float  # Price to download model
    license_terms: str
    tags: List[str]
    created_at: float
    updated_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class DataSharingAgreement:
    """Data sharing agreement between parties"""
    agreement_id: str
    data_provider: str
    data_consumer: str
    data_type: str
    data_description: str
    usage_terms: str
    price_per_gb: float
    max_usage_gb: float
    privacy_level: str  # "public", "private", "confidential"
    expiration_time: float
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSharingAgreement':
        """Create from dictionary"""
        return cls(**data)

# Reputation system removed - focusing on core blockchain functionality

class SmartContract:
    """Base smart contract class"""
    
    def __init__(self, contract_id: str, contract_type: ContractType, owner: str):
        """
        Initialize smart contract
        
        Args:
            contract_id: Unique contract identifier
            contract_type: Type of contract
            owner: Owner address
        """
        self.contract_id = contract_id
        self.contract_type = contract_type
        self.owner = owner
        self.created_at = time.time()
        self.state: Dict[str, Any] = {}
        self.is_active = True
        
        logger.info(f"[INIT] Smart contract {contract_id} created")
    
    def execute(self, method: str, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a contract method"""
        if not self.is_active:
            return {"success": False, "error": "Contract is not active"}
        
        method_name = f"_execute_{method}"
        if hasattr(self, method_name):
            try:
                result = getattr(self, method_name)(caller, params)
                logger.info(f"[OK] Contract {self.contract_id} method {method} executed by {caller}")
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"[ERROR] Contract execution failed: {e}")
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": f"Method {method} not found"}
    
    def get_state(self) -> Dict[str, Any]:
        """Get contract state"""
        return {
            "contract_id": self.contract_id,
            "contract_type": self.contract_type.value,
            "owner": self.owner,
            "created_at": self.created_at,
            "is_active": self.is_active,
            "state": self.state
        }
    
    def deactivate(self, caller: str) -> bool:
        """Deactivate the contract"""
        if caller != self.owner:
            return False
        
        self.is_active = False
        logger.info(f"[OK] Contract {self.contract_id} deactivated")
        return True

class AIModelContract(SmartContract):
    """Smart contract for AI model sharing and monetization"""
    
    def __init__(self, contract_id: str, owner: str):
        """Initialize AI model contract"""
        super().__init__(contract_id, ContractType.AI_MODEL, owner)
        self.state = {
            "models": {},           # model_id -> ModelInfo
            "usage_stats": {},      # model_id -> usage statistics
            "earnings": {},         # owner -> earnings
            "access_rights": {}     # user -> list of accessible models
        }
    
    def _execute_register_model(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new AI model"""
        model_info = ModelInfo.from_dict(params)
        model_info.owner = caller
        model_info.created_at = time.time()
        model_info.updated_at = time.time()
        
        self.state["models"][model_info.model_id] = model_info.to_dict()
        self.state["usage_stats"][model_info.model_id] = {
            "total_inferences": 0,
            "total_downloads": 0,
            "total_earnings": 0.0,
            "last_used": 0.0
        }
        
        return {"model_id": model_info.model_id, "registered": True}
    
    def _execute_update_model(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing model"""
        model_id = params.get("model_id")
        if not model_id or model_id not in self.state["models"]:
            raise ValueError("Model not found")
        
        model_data = self.state["models"][model_id]
        if model_data["owner"] != caller:
            raise ValueError("Only owner can update model")
        
        # Update allowed fields
        updatable_fields = ["description", "accuracy_metrics", "usage_price", "sharing_price", "tags"]
        for field in updatable_fields:
            if field in params:
                model_data[field] = params[field]
        
        model_data["updated_at"] = time.time()
        self.state["models"][model_id] = model_data
        
        return {"model_id": model_id, "updated": True}
    
    def _execute_purchase_access(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Purchase access to a model"""
        model_id = params.get("model_id")
        payment_amount = params.get("payment_amount", 0.0)
        
        if not model_id or model_id not in self.state["models"]:
            raise ValueError("Model not found")
        
        model_data = self.state["models"][model_id]
        required_payment = model_data["sharing_price"]
        
        if payment_amount < required_payment:
            raise ValueError(f"Insufficient payment. Required: {required_payment}")
        
        # Grant access
        if caller not in self.state["access_rights"]:
            self.state["access_rights"][caller] = []
        
        if model_id not in self.state["access_rights"][caller]:
            self.state["access_rights"][caller].append(model_id)
        
        # Update earnings
        owner = model_data["owner"]
        if owner not in self.state["earnings"]:
            self.state["earnings"][owner] = 0.0
        self.state["earnings"][owner] += payment_amount
        
        # Update stats
        self.state["usage_stats"][model_id]["total_downloads"] += 1
        self.state["usage_stats"][model_id]["total_earnings"] += payment_amount
        
        return {"access_granted": True, "model_id": model_id}
    
    def _execute_use_model(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record model usage"""
        model_id = params.get("model_id")
        usage_count = params.get("usage_count", 1)
        
        if not model_id or model_id not in self.state["models"]:
            raise ValueError("Model not found")
        
        # Check access rights
        if caller not in self.state["access_rights"] or model_id not in self.state["access_rights"][caller]:
            model_data = self.state["models"][model_id]
            if model_data["owner"] != caller:
                raise ValueError("Access denied")
        
        # Calculate payment
        model_data = self.state["models"][model_id]
        payment = model_data["usage_price"] * usage_count
        
        # Update earnings
        owner = model_data["owner"]
        if owner not in self.state["earnings"]:
            self.state["earnings"][owner] = 0.0
        self.state["earnings"][owner] += payment
        
        # Update stats
        self.state["usage_stats"][model_id]["total_inferences"] += usage_count
        self.state["usage_stats"][model_id]["total_earnings"] += payment
        self.state["usage_stats"][model_id]["last_used"] = time.time()
        
        return {"usage_recorded": True, "payment": payment}
    
    def _execute_get_model_info(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model information"""
        model_id = params.get("model_id")
        if not model_id or model_id not in self.state["models"]:
            raise ValueError("Model not found")
        
        model_data = self.state["models"][model_id].copy()
        # Add usage stats
        model_data["usage_stats"] = self.state["usage_stats"][model_id]
        
        return model_data
    
    def _execute_list_models(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available models"""
        filters = params.get("filters", {})
        
        models = []
        for model_id, model_data in self.state["models"].items():
            # Apply filters
            if filters.get("model_type") and model_data["model_type"] != filters["model_type"]:
                continue
            if filters.get("owner") and model_data["owner"] != filters["owner"]:
                continue
            if filters.get("max_price") and model_data["usage_price"] > filters["max_price"]:
                continue
            
            # Add usage stats
            model_summary = model_data.copy()
            model_summary["usage_stats"] = self.state["usage_stats"][model_id]
            models.append(model_summary)
        
        return {"models": models, "count": len(models)}

class DataSharingContract(SmartContract):
    """Smart contract for data sharing agreements"""
    
    def __init__(self, contract_id: str, owner: str):
        """Initialize data sharing contract"""
        super().__init__(contract_id, ContractType.DATA_SHARING, owner)
        self.state = {
            "agreements": {},       # agreement_id -> DataSharingAgreement
            "usage_records": {},    # agreement_id -> usage records
            "earnings": {}          # provider -> earnings
        }
    
    def _execute_create_agreement(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a data sharing agreement"""
        agreement = DataSharingAgreement.from_dict(params)
        agreement.data_provider = caller
        agreement.created_at = time.time()
        
        self.state["agreements"][agreement.agreement_id] = agreement.to_dict()
        self.state["usage_records"][agreement.agreement_id] = []
        
        return {"agreement_id": agreement.agreement_id, "created": True}
    
    def _execute_accept_agreement(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a data sharing agreement"""
        agreement_id = params.get("agreement_id")
        if not agreement_id or agreement_id not in self.state["agreements"]:
            raise ValueError("Agreement not found")
        
        agreement_data = self.state["agreements"][agreement_id]
        if agreement_data["data_consumer"] != caller:
            raise ValueError("Only designated consumer can accept")
        
        # Mark as accepted (you could add an accepted field to the agreement)
        agreement_data["status"] = "accepted"
        self.state["agreements"][agreement_id] = agreement_data
        
        return {"agreement_id": agreement_id, "accepted": True}
    
    def _execute_record_usage(self, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record data usage"""
        agreement_id = params.get("agreement_id")
        data_size_gb = params.get("data_size_gb", 0.0)
        
        if not agreement_id or agreement_id not in self.state["agreements"]:
            raise ValueError("Agreement not found")
        
        agreement_data = self.state["agreements"][agreement_id]
        if agreement_data["data_consumer"] != caller:
            raise ValueError("Only consumer can record usage")
        
        # Check usage limits
        current_usage = sum(record["data_size_gb"] for record in self.state["usage_records"][agreement_id])
        if current_usage + data_size_gb > agreement_data["max_usage_gb"]:
            raise ValueError("Usage exceeds agreement limits")
        
        # Calculate payment
        payment = data_size_gb * agreement_data["price_per_gb"]
        
        # Record usage
        usage_record = {
            "timestamp": time.time(),
            "data_size_gb": data_size_gb,
            "payment": payment,
            "consumer": caller
        }
        self.state["usage_records"][agreement_id].append(usage_record)
        
        # Update earnings
        provider = agreement_data["data_provider"]
        if provider not in self.state["earnings"]:
            self.state["earnings"][provider] = 0.0
        self.state["earnings"][provider] += payment
        
        return {"usage_recorded": True, "payment": payment}

# ReputationContract removed - focusing on core blockchain functionality

class ContractManager:
    """Manager for all smart contracts"""
    
    def __init__(self):
        """Initialize contract manager"""
        self.contracts: Dict[str, SmartContract] = {}
        self.contract_registry: Dict[ContractType, List[str]] = {
            ContractType.AI_MODEL: [],
            ContractType.DATA_SHARING: [],
            ContractType.PAYMENT: []
        }
        
        logger.info("[INIT] Contract manager initialized")
    
    def deploy_contract(self, contract_type: ContractType, owner: str, contract_id: Optional[str] = None) -> str:
        """Deploy a new smart contract"""
        if not contract_id:
            contract_id = f"{contract_type.value}_{int(time.time())}_{owner}"
        
        if contract_type == ContractType.AI_MODEL:
            contract = AIModelContract(contract_id, owner)
        elif contract_type == ContractType.DATA_SHARING:
            contract = DataSharingContract(contract_id, owner)
        else:
            raise ValueError(f"Unsupported contract type: {contract_type}")
        
        self.contracts[contract_id] = contract
        self.contract_registry[contract_type].append(contract_id)
        
        logger.info(f"[OK] Contract {contract_id} deployed")
        return contract_id
    
    def execute_contract(self, contract_id: str, method: str, caller: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a contract method"""
        if contract_id not in self.contracts:
            return {"success": False, "error": "Contract not found"}
        
        contract = self.contracts[contract_id]
        return contract.execute(method, caller, params)
    
    def get_contract(self, contract_id: str) -> Optional[SmartContract]:
        """Get a contract by ID"""
        return self.contracts.get(contract_id)
    
    def list_contracts(self, contract_type: Optional[ContractType] = None, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """List contracts with optional filters"""
        contracts = []
        
        for contract in self.contracts.values():
            if contract_type and contract.contract_type != contract_type:
                continue
            if owner and contract.owner != owner:
                continue
            
            contracts.append(contract.get_state())
        
        return contracts
    
    def create_contract_transaction(self, 
                                   contract_id: str,
                                   method: str,
                                   caller: str,
                                   params: Dict[str, Any]) -> Transaction:
        """Create a transaction for contract execution"""
        contract_data = {
            "contract_id": contract_id,
            "method": method,
            "params": params
        }
        
        return Transaction(
            sender=caller,
            recipient=contract_id,
            amount=0.0,  # Contract execution fee could be added
            fee=0.01,
            nonce=0,  # Would be set by the blockchain
            timestamp=time.time(),
            data=contract_data
        )

if __name__ == "__main__":
    # Demo smart contracts
    manager = ContractManager()
    
    # Deploy AI model contract
    ai_contract_id = manager.deploy_contract(ContractType.AI_MODEL, "user1")
    
    # Register a model
    model_params = {
        "model_id": "llama3.2_3b",
        "name": "Llama 3.2 3B",
        "description": "Efficient 3B parameter language model",
        "model_type": "llm",
        "version": "1.0",
        "size_mb": 1500.0,
        "accuracy_metrics": {"perplexity": 12.5, "accuracy": 0.85},
        "usage_price": 0.001,
        "sharing_price": 10.0,
        "license_terms": "Commercial use allowed",
        "tags": ["llm", "efficient", "3b"]
    }
    
    result = manager.execute_contract(ai_contract_id, "register_model", "user1", model_params)
    print(f"Model registration: {result}")
    
    # Purchase access
    purchase_params = {"model_id": "llama3.2_3b", "payment_amount": 10.0}
    result = manager.execute_contract(ai_contract_id, "purchase_access", "user2", purchase_params)
    print(f"Access purchase: {result}")
    
    # Use model
    usage_params = {"model_id": "llama3.2_3b", "usage_count": 5}
    result = manager.execute_contract(ai_contract_id, "use_model", "user2", usage_params)
    print(f"Model usage: {result}")
    
    # List models
    list_params = {"filters": {"model_type": "llm"}}
    result = manager.execute_contract(ai_contract_id, "list_models", "user3", list_params)
    print(f"Available models: {len(result['result']['models'])}")