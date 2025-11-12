"""
Agent loader for loading agents from JSON.
"""
import json
from typing import Dict, Any, Optional
from app.utils.logger import logger


class AgentLoader:
    """Loads agents from agents.json file."""
    
    _agents: Dict[str, Any] = {}
    _initialized = False

    @classmethod
    def initialize(cls, agents_file: str = "agents.json"):
        """Initialize and load agents from file."""
        try:
            with open(agents_file, "r") as f:
                cls._agents = json.load(f)
            cls._initialized = True
            logger.info(f"Loaded {len(cls._agents)} agents from {agents_file}", "AgentLoader")
        except Exception as e:
            logger.error(f"Error loading agents: {str(e)}", "AgentLoader", exc_info=True)
            cls._agents = {}

    @classmethod
    def get_agent(cls, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration by ID."""
        if not cls._initialized:
            cls.initialize()
        return cls._agents.get(agent_id)

    @classmethod
    def get_all_agents(cls) -> Dict[str, Any]:
        """Get all agents."""
        if not cls._initialized:
            cls.initialize()
        return cls._agents

    @classmethod
    def get_system_prompt(cls, agent_id: str) -> str:
        """Get system prompt for agent."""
        agent = cls.get_agent(agent_id)
        if agent:
            return agent.get("system_prompt", "")
        return ""

    @classmethod
    def get_first_response_message(cls, agent_id: str) -> Optional[str]:
        """Get first response message for agent."""
        agent = cls.get_agent(agent_id)
        if agent:
            return agent.get("first_response_message")
        return None

