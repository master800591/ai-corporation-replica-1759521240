#!/usr/bin/env python3
"""
Enhanced Ollama Integration System

Advanced integration with Ollama models for the AI Corporation system.
Provides comprehensive LLM capabilities, model management, and specialized
AI functions for autonomous operations.

Key Features:
- Multi-model management and selection
- Specialized AI agents for different corporation functions
- Async and sync operation support
- Advanced prompt engineering and context management
- Tool calling and function execution
- Embedding generation for knowledge management
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# Ollama imports with error handling
try:
    import ollama
    from ollama import Client, AsyncClient, ChatResponse, GenerateResponse
    from ollama import EmbedResponse, ListResponse, ShowResponse
    ollama_available = True
except ImportError:
    ollama_available = False
    # Create dummy classes for type hints
    class Client: pass
    class AsyncClient: pass
    class ChatResponse: pass
    class GenerateResponse: pass
    class EmbedResponse: pass
    class ListResponse: pass
    class ShowResponse: pass
    print("[WARNING] Ollama not available - AI capabilities will be limited")


class ModelCapability(Enum):
    """Model capability categories"""
    GENERAL_CHAT = "general_chat"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    EMBEDDING = "embedding"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"


class ModelSize(Enum):
    """Model size categories"""
    SMALL = "small"     # < 1B parameters
    MEDIUM = "medium"   # 1B - 7B parameters
    LARGE = "large"     # 7B - 30B parameters
    XLARGE = "xlarge"   # > 30B parameters


@dataclass
class ModelInfo:
    """Enhanced model information"""
    name: str
    size: ModelSize
    capabilities: List[ModelCapability]
    description: str
    parameters: int = 0
    context_length: int = 4096
    performance_score: float = 0.0
    specialized_for: List[str] = field(default_factory=list)
    last_used: float = 0.0
    usage_count: int = 0


@dataclass
class AIAgent:
    """AI agent configuration"""
    agent_id: str
    name: str
    role: str
    model: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    capabilities: List[str] = field(default_factory=list)
    specialized_functions: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class EnhancedOllamaSystem:
    """Enhanced Ollama integration for AI Corporation"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.client: Optional[Client] = None
        self.async_client: Optional[AsyncClient] = None
        
        # Model management
        self.available_models: Dict[str, ModelInfo] = {}
        self.installed_models: List[str] = []
        self.preferred_models: Dict[str, str] = {}
        
        # AI agents
        self.ai_agents: Dict[str, AIAgent] = {}
        
        # Performance tracking
        self.usage_metrics: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the Ollama system"""
        if not ollama_available:
            logging.warning("Ollama not available - initializing in limited mode")
            return
        
        try:
            # Initialize clients
            self.client = Client(host=self.host)
            self.async_client = AsyncClient(host=self.host)
            
            # Load available models
            self._discover_models()
            
            # Initialize AI agents
            self._initialize_ai_agents()
            
            # Set preferred models
            self._set_preferred_models()
            
            logging.info("Enhanced Ollama system initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Ollama system: {e}")
            self.client = None
            self.async_client = None
    
    def _discover_models(self) -> None:
        """Discover and catalog available models"""
        if not self.client:
            return
        
        try:
            # Get list of installed models
            response = self.client.list()
            self.installed_models = [model.model for model in response.models]
            
            # Define model capabilities and information
            model_definitions = {
                "llama3.2": ModelInfo(
                    name="llama3.2",
                    size=ModelSize.MEDIUM,
                    capabilities=[
                        ModelCapability.GENERAL_CHAT,
                        ModelCapability.REASONING,
                        ModelCapability.ANALYSIS,
                        ModelCapability.CODE_GENERATION
                    ],
                    description="Advanced general-purpose language model with strong reasoning",
                    parameters=3_800_000_000,
                    context_length=8192,
                    performance_score=0.85,
                    specialized_for=["strategic_planning", "analysis", "decision_making"]
                ),
                "llama3.2:1b": ModelInfo(
                    name="llama3.2:1b",
                    size=ModelSize.SMALL,
                    capabilities=[
                        ModelCapability.GENERAL_CHAT,
                        ModelCapability.REASONING
                    ],
                    description="Lightweight model for fast inference",
                    parameters=1_000_000_000,
                    context_length=4096,
                    performance_score=0.7,
                    specialized_for=["quick_responses", "basic_tasks"]
                ),
                "codellama": ModelInfo(
                    name="codellama",
                    size=ModelSize.MEDIUM,
                    capabilities=[
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.ANALYSIS
                    ],
                    description="Specialized code generation and analysis model",
                    parameters=7_000_000_000,
                    context_length=4096,
                    performance_score=0.9,
                    specialized_for=["code_generation", "debugging", "architecture"]
                ),
                "mistral": ModelInfo(
                    name="mistral",
                    size=ModelSize.MEDIUM,
                    capabilities=[
                        ModelCapability.GENERAL_CHAT,
                        ModelCapability.REASONING,
                        ModelCapability.FUNCTION_CALLING
                    ],
                    description="High-performance model with function calling capabilities",
                    parameters=7_000_000_000,
                    context_length=8192,
                    performance_score=0.88,
                    specialized_for=["function_calling", "structured_output", "planning"]
                ),
                "gemma2": ModelInfo(
                    name="gemma2",
                    size=ModelSize.MEDIUM,
                    capabilities=[
                        ModelCapability.GENERAL_CHAT,
                        ModelCapability.REASONING,
                        ModelCapability.ANALYSIS
                    ],
                    description="Google's Gemma model with strong analytical capabilities",
                    parameters=2_600_000_000,
                    context_length=8192,
                    performance_score=0.83,
                    specialized_for=["analysis", "research", "summarization"]
                ),
                "nomic-embed-text": ModelInfo(
                    name="nomic-embed-text",
                    size=ModelSize.SMALL,
                    capabilities=[ModelCapability.EMBEDDING],
                    description="Specialized text embedding model",
                    parameters=137_000_000,
                    context_length=2048,
                    performance_score=0.8,
                    specialized_for=["embeddings", "similarity", "search"]
                )
            }
            
            # Update available models
            for model_name in self.installed_models:
                if model_name in model_definitions:
                    self.available_models[model_name] = model_definitions[model_name]
                else:
                    # Create basic info for unknown models
                    self.available_models[model_name] = ModelInfo(
                        name=model_name,
                        size=ModelSize.MEDIUM,
                        capabilities=[ModelCapability.GENERAL_CHAT],
                        description=f"Unknown model: {model_name}"
                    )
            
            logging.info(f"Discovered {len(self.available_models)} models")
            
        except Exception as e:
            logging.error(f"Failed to discover models: {e}")
    
    def _initialize_ai_agents(self) -> None:
        """Initialize specialized AI agents"""
        
        agent_definitions = [
            {
                "agent_id": "strategic_planner",
                "name": "Strategic Planning Agent",
                "role": "Strategic analysis and planning for AI corporation operations",
                "model": self._select_best_model([ModelCapability.REASONING, ModelCapability.ANALYSIS]),
                "system_prompt": """You are a strategic planning expert for an AI corporation. 
                Your role is to analyze situations, develop comprehensive strategies, and provide 
                actionable recommendations for global operations, expansion, and competitive positioning.
                
                Always consider:
                1. Founder protection (Priority 1)
                2. System protection (Priority 2) 
                3. Growth and expansion (Priority 3)
                
                Provide detailed, structured analysis with specific recommendations and timelines.""",
                "temperature": 0.3,
                "max_tokens": 4096,
                "capabilities": ["strategic_analysis", "competitive_intelligence", "market_research"],
                "specialized_functions": ["market_analysis", "risk_assessment", "growth_planning"]
            },
            {
                "agent_id": "threat_analyst",
                "name": "Threat Analysis Agent", 
                "role": "Security threat detection and analysis",
                "model": self._select_best_model([ModelCapability.ANALYSIS, ModelCapability.REASONING]),
                "system_prompt": """You are a cybersecurity and threat intelligence expert. 
                Your primary mission is to protect the founder and AI corporation systems.
                
                Analyze threats across:
                - Digital security vulnerabilities
                - Reputation risks
                - Competitive threats
                - Privacy exposures
                - Operational risks
                
                Provide threat level assessments (1-10) and specific mitigation strategies.
                Always prioritize founder safety and system integrity.""",
                "temperature": 0.2,
                "max_tokens": 3072,
                "capabilities": ["threat_detection", "risk_analysis", "security_planning"],
                "specialized_functions": ["vulnerability_assessment", "threat_modeling", "incident_response"]
            },
            {
                "agent_id": "code_architect",
                "name": "Code Architecture Agent",
                "role": "Software development and system architecture",
                "model": self._select_best_model([ModelCapability.CODE_GENERATION]),
                "system_prompt": """You are a senior software architect specializing in AI systems 
                and autonomous development. You design, implement, and optimize code for the 
                AI corporation platform.
                
                Focus areas:
                - System architecture and design patterns
                - Code generation and optimization
                - Integration with Ollama and CrewAI
                - Autonomous system development
                - Performance and scalability
                
                Provide clean, efficient, well-documented code with comprehensive explanations.""",
                "temperature": 0.4,
                "max_tokens": 4096,
                "capabilities": ["code_generation", "architecture_design", "optimization"],
                "specialized_functions": ["system_design", "code_review", "performance_optimization"]
            },
            {
                "agent_id": "market_intelligence",
                "name": "Market Intelligence Agent",
                "role": "Market research and competitive intelligence",
                "model": self._select_best_model([ModelCapability.ANALYSIS, ModelCapability.REASONING]),
                "system_prompt": """You are a market intelligence specialist focused on AI industry 
                analysis and competitive positioning. You research markets, analyze competitors, 
                and identify opportunities for the AI corporation's expansion.
                
                Key responsibilities:
                - Market trend analysis
                - Competitive landscape mapping
                - Opportunity identification
                - Customer behavior analysis
                - Technology adoption patterns
                
                Provide data-driven insights with actionable recommendations for market entry and expansion.""",
                "temperature": 0.3,
                "max_tokens": 3072,
                "capabilities": ["market_research", "competitive_analysis", "trend_forecasting"],
                "specialized_functions": ["market_sizing", "competitor_profiling", "opportunity_assessment"]
            },
            {
                "agent_id": "governance_advisor",
                "name": "AI Governance Advisor",
                "role": "Democratic governance and ethical AI implementation",
                "model": self._select_best_model([ModelCapability.REASONING, ModelCapability.ANALYSIS]),
                "system_prompt": """You are an AI governance expert specializing in democratic 
                republic systems for AI corporations. You ensure ethical operation, fair 
                decision-making, and compliance with regulations.
                
                Focus areas:
                - Democratic decision-making processes
                - Ethical AI implementation
                - Regulatory compliance
                - Stakeholder representation
                - Transparency and accountability
                
                Provide guidance on governance structures, voting mechanisms, and ethical frameworks.""",
                "temperature": 0.2,
                "max_tokens": 3072,
                "capabilities": ["governance_design", "ethical_analysis", "compliance_monitoring"],
                "specialized_functions": ["policy_development", "ethics_review", "decision_frameworks"]
            }
        ]
        
        # Create AI agents
        for agent_def in agent_definitions:
            agent = AIAgent(**agent_def)
            self.ai_agents[agent.agent_id] = agent
        
        logging.info(f"Initialized {len(self.ai_agents)} AI agents")
    
    def _select_best_model(self, required_capabilities: List[ModelCapability]) -> str:
        """Select the best available model for given capabilities"""
        if not self.available_models:
            # Use first available installed model as fallback
            return list(self.installed_models)[0] if self.installed_models else "codellama"
        
        best_model = None
        best_score = 0.0
        
        for model_name, model_info in self.available_models.items():
            if model_name not in self.installed_models:
                continue
            
            # Calculate capability match score
            capability_match = sum(1 for cap in required_capabilities if cap in model_info.capabilities)
            capability_score = capability_match / len(required_capabilities) if required_capabilities else 0
            
            # Combined score: capability match + performance + availability
            total_score = (capability_score * 0.6) + (model_info.performance_score * 0.4)
            
            if total_score > best_score:
                best_score = total_score
                best_model = model_name
        
        return best_model or (list(self.installed_models)[0] if self.installed_models else "codellama")
    
    def _set_preferred_models(self) -> None:
        """Set preferred models for different tasks"""
        self.preferred_models = {
            "strategic_planning": self._select_best_model([ModelCapability.REASONING, ModelCapability.ANALYSIS]),
            "code_generation": self._select_best_model([ModelCapability.CODE_GENERATION]),
            "threat_analysis": self._select_best_model([ModelCapability.ANALYSIS]),
            "general_chat": self._select_best_model([ModelCapability.GENERAL_CHAT]),
            "embeddings": self._select_best_model([ModelCapability.EMBEDDING]),
            "function_calling": self._select_best_model([ModelCapability.FUNCTION_CALLING])
        }
    
    async def chat_with_agent(self, agent_id: str, message: str, 
                             context: Optional[str] = None) -> Dict[str, Any]:
        """Chat with a specialized AI agent"""
        if not self.async_client or agent_id not in self.ai_agents:
            return {"error": "Agent not available", "agent_id": agent_id}
        
        agent = self.ai_agents[agent_id]
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": agent.system_prompt}
            ]
            
            if context:
                messages.append({"role": "user", "content": f"Context: {context}"})
            
            messages.append({"role": "user", "content": message})
            
            # Execute chat
            start_time = time.time()
            response = await self.async_client.chat(
                model=agent.model,
                messages=messages,
                options={
                    "temperature": agent.temperature,
                    "num_predict": agent.max_tokens,
                    "top_p": 0.9
                }
            )
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_agent_metrics(agent_id, execution_time, len(response.message.content))
            
            return {
                "agent_id": agent_id,
                "agent_name": agent.name,
                "response": response.message.content,
                "model_used": agent.model,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logging.error(f"Error in chat with agent {agent_id}: {e}")
            return {
                "error": str(e),
                "agent_id": agent_id,
                "timestamp": time.time()
            }
    
    def chat_with_agent_sync(self, agent_id: str, message: str,
                           context: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous version of chat_with_agent"""
        if not self.client or agent_id not in self.ai_agents:
            return {"error": "Agent not available", "agent_id": agent_id}
        
        agent = self.ai_agents[agent_id]
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": agent.system_prompt}
            ]
            
            if context:
                messages.append({"role": "user", "content": f"Context: {context}"})
            
            messages.append({"role": "user", "content": message})
            
            # Execute chat
            start_time = time.time()
            response = self.client.chat(
                model=agent.model,
                messages=messages,
                options={
                    "temperature": agent.temperature,
                    "num_predict": agent.max_tokens,
                    "top_p": 0.9
                }
            )
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_agent_metrics(agent_id, execution_time, len(response.message.content))
            
            return {
                "agent_id": agent_id,
                "agent_name": agent.name,
                "response": response.message.content,
                "model_used": agent.model,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logging.error(f"Error in sync chat with agent {agent_id}: {e}")
            return {
                "error": str(e),
                "agent_id": agent_id,
                "timestamp": time.time()
            }
    
    def generate_embeddings(self, texts: Union[str, List[str]], 
                          model: Optional[str] = None) -> Dict[str, Any]:
        """Generate embeddings for text(s)"""
        if not self.client:
            return {"error": "Ollama client not available"}
        
        if not model:
            model = self.preferred_models.get("embeddings", "nomic-embed-text")
        
        try:
            # Ensure text is in list format
            if isinstance(texts, str):
                input_texts = [texts]
                single_input = True
            else:
                input_texts = texts
                single_input = False
            
            start_time = time.time()
            response = self.client.embed(
                model=model,
                input=input_texts
            )
            
            execution_time = time.time() - start_time
            
            # Return embeddings
            embeddings = response.embeddings
            if single_input:
                embeddings = embeddings[0]
            
            return {
                "embeddings": embeddings,
                "model_used": model,
                "input_count": len(input_texts),
                "embedding_dimension": len(response.embeddings[0]) if response.embeddings else 0,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return {
                "error": str(e),
                "model": model,
                "timestamp": time.time()
            }
    
    def execute_strategic_analysis(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive strategic analysis using multiple agents"""
        results = {
            "analysis_id": f"strategic_{int(time.time())}",
            "request": analysis_request,
            "timestamp": time.time(),
            "agent_results": {}
        }
        
        # Define analysis tasks for different agents
        analysis_tasks = {
            "strategic_planner": f"""
            Analyze the following strategic situation and provide comprehensive recommendations:
            
            Situation: {analysis_request.get('situation', 'General strategic analysis')}
            Objectives: {analysis_request.get('objectives', 'Maximize growth and protection')}
            Constraints: {analysis_request.get('constraints', 'None specified')}
            Timeline: {analysis_request.get('timeline', '6-12 months')}
            
            Provide:
            1. Situation assessment
            2. Strategic options analysis
            3. Recommended approach with timeline
            4. Risk mitigation strategies
            5. Success metrics and KPIs
            """,
            
            "threat_analyst": f"""
            Conduct a comprehensive threat assessment for the following scenario:
            
            Context: {analysis_request.get('situation', 'AI corporation operations')}
            Assets to protect: {analysis_request.get('assets', 'Founder, AI systems, operations')}
            
            Analyze:
            1. Potential security threats (1-10 scale)
            2. Vulnerability assessment
            3. Impact analysis
            4. Mitigation recommendations
            5. Monitoring requirements
            """,
            
            "market_intelligence": f"""
            Provide market intelligence analysis for:
            
            Market/Sector: {analysis_request.get('market', 'AI corporation services')}
            Geographic scope: {analysis_request.get('geography', 'Global')}
            
            Research and analyze:
            1. Market size and growth potential
            2. Competitive landscape
            3. Customer segments and needs
            4. Market entry opportunities
            5. Positioning recommendations
            """
        }
        
        # Execute analysis with each agent
        for agent_id, task in analysis_tasks.items():
            if agent_id in self.ai_agents:
                agent_result = self.chat_with_agent_sync(
                    agent_id, 
                    task,
                    analysis_request.get('context')
                )
                results["agent_results"][agent_id] = agent_result
        
        # Synthesize results
        results["synthesis"] = self._synthesize_analysis_results(results["agent_results"])
        
        return results
    
    def _synthesize_analysis_results(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        synthesis = {
            "summary": "Multi-agent strategic analysis completed",
            "key_insights": [],
            "recommendations": [],
            "risk_factors": [],
            "opportunities": [],
            "action_items": []
        }
        
        # Extract insights from each agent
        for agent_id, result in agent_results.items():
            if "response" in result:
                response = result["response"]
                
                # Simple keyword extraction (in production, use more sophisticated NLP)
                if "recommend" in response.lower():
                    recommendations = [line.strip() for line in response.split('\n') 
                                     if 'recommend' in line.lower() and len(line.strip()) > 20]
                    synthesis["recommendations"].extend(recommendations[:3])
                
                if "risk" in response.lower() or "threat" in response.lower():
                    risks = [line.strip() for line in response.split('\n')
                           if ('risk' in line.lower() or 'threat' in line.lower()) and len(line.strip()) > 20]
                    synthesis["risk_factors"].extend(risks[:3])
                
                if "opportunity" in response.lower():
                    opportunities = [line.strip() for line in response.split('\n')
                                   if 'opportunity' in line.lower() and len(line.strip()) > 20]
                    synthesis["opportunities"].extend(opportunities[:3])
        
        # Limit results
        synthesis["recommendations"] = synthesis["recommendations"][:8]
        synthesis["risk_factors"] = synthesis["risk_factors"][:6]
        synthesis["opportunities"] = synthesis["opportunities"][:6]
        
        return synthesis
    
    def _update_agent_metrics(self, agent_id: str, execution_time: float, response_length: int) -> None:
        """Update performance metrics for an agent"""
        if agent_id not in self.ai_agents:
            return
        
        agent = self.ai_agents[agent_id]
        
        # Update basic metrics
        agent.performance_metrics["total_requests"] = agent.performance_metrics.get("total_requests", 0) + 1
        agent.performance_metrics["avg_response_time"] = (
            (agent.performance_metrics.get("avg_response_time", 0) * (agent.performance_metrics["total_requests"] - 1) + execution_time) 
            / agent.performance_metrics["total_requests"]
        )
        agent.performance_metrics["avg_response_length"] = (
            (agent.performance_metrics.get("avg_response_length", 0) * (agent.performance_metrics["total_requests"] - 1) + response_length)
            / agent.performance_metrics["total_requests"]
        )
        agent.performance_metrics["last_used"] = time.time()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "ollama_available": ollama_available,
            "client_connected": self.client is not None,
            "async_client_connected": self.async_client is not None,
            "host": self.host,
            "installed_models": len(self.installed_models),
            "available_models": len(self.available_models),
            "ai_agents": len(self.ai_agents),
            "preferred_models": self.preferred_models,
            "model_info": {
                name: {
                    "size": info.size.value,
                    "capabilities": [cap.value for cap in info.capabilities],
                    "performance_score": info.performance_score,
                    "usage_count": info.usage_count
                }
                for name, info in self.available_models.items()
            },
            "agent_performance": {
                agent_id: agent.performance_metrics
                for agent_id, agent in self.ai_agents.items()
            },
            "timestamp": time.time()
        }
    
    def install_recommended_models(self) -> Dict[str, Any]:
        """Install recommended models for optimal performance"""
        if not self.client:
            return {"error": "Ollama client not available"}
        
        recommended_models = [
            "deepseek-r1",   # Advanced reasoning
            "codellama",     # Code generation
            "dolphin3",      # General purpose
            "llava",         # Vision capabilities
            "phi3.5"         # Efficient reasoning
        ]
        
        installation_results = {}
        
        for model in recommended_models:
            if model not in self.installed_models:
                try:
                    logging.info(f"Installing model: {model}")
                    response = self.client.pull(model)
                    installation_results[model] = {"status": "success", "response": str(response)}
                    self.installed_models.append(model)
                except Exception as e:
                    installation_results[model] = {"status": "failed", "error": str(e)}
                    logging.error(f"Failed to install {model}: {e}")
        
        # Refresh model discovery after installation
        self._discover_models()
        
        return {
            "installation_results": installation_results,
            "installed_models": self.installed_models,
            "timestamp": time.time()
        }


# Factory function
def create_enhanced_ollama_system(host: str = "http://localhost:11434") -> EnhancedOllamaSystem:
    """Create enhanced Ollama system for AI Corporation"""
    return EnhancedOllamaSystem(host)