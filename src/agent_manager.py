"""
Persistent Agent Manager

Multi-agent orchestration with KV cache persistence and LRU eviction.
Manages multiple agents with isolated contexts, persistent memory, and
automatic cache management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from .mlx_utils import MLXModelLoader
from .mlx_cache_extractor import MLXCacheExtractor
from .cache_persistence import CachePersistence

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Context for a single persistent agent.

    Attributes:
        agent_id: Unique identifier
        agent_type: Type/role (e.g., 'technical', 'business', 'coordinator')
        system_prompt: System-level instructions
        cache: KV cache (List[KVCache])
        cache_tokens: Number of tokens in cache
        last_access: Timestamp of last interaction
        conversation_history: List of message dicts
    """
    agent_id: str
    agent_type: str
    system_prompt: str
    cache: Optional[List[Any]] = None
    cache_tokens: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    def update_access(self):
        """Update last_access timestamp."""
        self.last_access = datetime.now()


class PersistentAgentManager:
    """
    Manages multiple persistent agents with KV cache isolation.

    Features:
    - Per-agent KV cache isolation
    - LRU eviction when max_agents exceeded
    - Automatic save/load to disk
    - Memory usage monitoring
    - Conversation history tracking
    """

    def __init__(
        self,
        model_name: str = "mlx-community/gemma-3-12b-it-4bit",
        max_agents: int = 3,
        cache_dir: str = "~/.agent_caches"
    ):
        """
        Initialize persistent agent manager.

        Args:
            model_name: HuggingFace model ID or local path
            max_agents: Maximum number of agents to keep in memory
            cache_dir: Directory for cache persistence
        """
        logger.info(f"Initializing PersistentAgentManager: {model_name}, max_agents={max_agents}")

        # Load model and tokenizer
        self.model, self.tokenizer = MLXModelLoader.load_model(model_name)
        self.model_name = model_name

        # Initialize components
        self.cache_extractor = MLXCacheExtractor(self.model, self.tokenizer)
        self.persistence = CachePersistence(cache_dir)

        # Agent tracking
        self.agents: Dict[str, AgentContext] = {}
        self.max_agents = max_agents

        # Warmup: Process dummy prompt to compile forward pass (avoids 7x slowdown on first agent)
        logger.info("Running warmup to compile MLX forward pass...")
        _ = self.cache_extractor.process_prompt("warmup")
        logger.info("Warmup complete")

        logger.info("PersistentAgentManager initialized successfully")

    def create_agent(
        self,
        agent_id: str,
        agent_type: str,
        system_prompt: str
    ) -> AgentContext:
        """
        Create a new agent with isolated KV cache.

        If max_agents exceeded, evicts least recently used agent.

        Args:
            agent_id: Unique identifier
            agent_type: Agent role/type
            system_prompt: System-level instructions

        Returns:
            AgentContext: The created agent
        """
        logger.info(f"Creating agent: {agent_id} (type={agent_type})")

        # Check if already exists
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already exists, returning existing")
            return self.agents[agent_id]

        # Evict LRU if needed
        if len(self.agents) >= self.max_agents:
            self.evict_lru()

        # Create agent context
        agent = AgentContext(
            agent_id=agent_id,
            agent_type=agent_type,
            system_prompt=system_prompt
        )

        # Process system prompt into cache
        agent.cache = self.cache_extractor.process_prompt(system_prompt)
        cache_info = self.cache_extractor.get_cache_info(agent.cache)
        agent.cache_tokens = cache_info['total_tokens']

        self.agents[agent_id] = agent

        logger.info(f"Agent created: {agent_id} ({agent.cache_tokens} tokens cached)")

        return agent

    def load_agent(self, agent_id: str) -> AgentContext:
        """
        Load agent from memory or disk, or create new if not found.

        Args:
            agent_id: Unique identifier

        Returns:
            AgentContext: The loaded agent
        """
        # Check if already in memory
        if agent_id in self.agents:
            logger.debug(f"Agent {agent_id} already in memory")
            self.agents[agent_id].update_access()
            return self.agents[agent_id]

        # Try loading from disk
        if self.persistence.agent_cache_exists(agent_id):
            logger.info(f"Loading agent from disk: {agent_id}")

            # Evict LRU if needed
            if len(self.agents) >= self.max_agents:
                self.evict_lru()

            # Load cache and metadata
            cache, metadata = self.persistence.load_agent_cache(agent_id)

            # Reconstruct agent context
            agent = AgentContext(
                agent_id=agent_id,
                agent_type=metadata.get('agent_type', 'unknown'),
                system_prompt=metadata.get('system_prompt', ''),
                cache=cache,
                cache_tokens=int(metadata.get('cache_tokens', 0))
            )

            self.agents[agent_id] = agent

            logger.info(f"Agent loaded: {agent_id} ({agent.cache_tokens} tokens)")

            return agent

        # Agent not found - caller should create
        raise ValueError(f"Agent {agent_id} not found in memory or on disk")

    def generate(
        self,
        agent_id: str,
        user_input: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate response using agent's cached context.

        Args:
            agent_id: Unique identifier
            user_input: User's input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            str: Generated response
        """
        # Get agent (load if needed)
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not in memory. Load or create first.")

        agent = self.agents[agent_id]
        agent.update_access()

        logger.debug(f"Generating for agent {agent_id}: '{user_input[:50]}...'")

        # Add user message to history
        agent.conversation_history.append({
            'role': 'user',
            'content': user_input
        })

        # Generate with existing cache
        response, updated_cache = self.cache_extractor.generate_with_cache(
            prompt=user_input,
            existing_cache=agent.cache,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        # Update agent's cache
        agent.cache = updated_cache
        cache_info = self.cache_extractor.get_cache_info(agent.cache)
        agent.cache_tokens = cache_info['total_tokens']

        # Add assistant response to history
        agent.conversation_history.append({
            'role': 'assistant',
            'content': response
        })

        logger.info(f"Generated for {agent_id}: {len(response)} chars, cache now {agent.cache_tokens} tokens")

        return response

    def save_agent(self, agent_id: str):
        """
        Save single agent to disk.

        Args:
            agent_id: Unique identifier
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not in memory, cannot save")
            return

        agent = self.agents[agent_id]

        metadata = {
            'agent_id': agent.agent_id,
            'agent_type': agent.agent_type,
            'system_prompt': agent.system_prompt,
            'cache_tokens': agent.cache_tokens,
            'model': self.model_name
        }

        self.persistence.save_agent_cache(agent_id, agent.cache, metadata)

        logger.info(f"Saved agent: {agent_id}")

    def save_all(self):
        """Save all agents in memory to disk."""
        logger.info(f"Saving all agents ({len(self.agents)} total)")

        for agent_id in self.agents:
            self.save_agent(agent_id)

        logger.info("All agents saved")

    def evict_lru(self):
        """
        Evict least recently used agent.

        Saves agent to disk before removing from memory.
        """
        if not self.agents:
            logger.warning("No agents to evict")
            return

        # Find LRU agent
        lru_agent_id = min(
            self.agents.keys(),
            key=lambda aid: self.agents[aid].last_access
        )

        logger.info(f"Evicting LRU agent: {lru_agent_id}")

        # Save before evicting
        self.save_agent(lru_agent_id)

        # Remove from memory
        del self.agents[lru_agent_id]

        logger.info(f"Agent {lru_agent_id} evicted")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Report memory usage for all agents.

        Returns:
            dict with keys:
            - model_memory_gb: Model memory usage
            - agents: Dict[agent_id, {cache_tokens, cache_mb}]
            - total_cache_mb: Total cache memory
            - total_gb: Model + caches
        """
        # Get model memory
        model_mem = MLXModelLoader.get_memory_usage()
        model_gb = model_mem['active_memory_gb']

        # Get cache memory per agent
        agent_usage = {}
        total_cache_bytes = 0

        for agent_id, agent in self.agents.items():
            cache_bytes = self.cache_extractor.get_cache_memory_bytes(agent.cache)
            total_cache_bytes += cache_bytes

            agent_usage[agent_id] = {
                'cache_tokens': agent.cache_tokens,
                'cache_mb': cache_bytes / (1024 ** 2)
            }

        total_cache_mb = total_cache_bytes / (1024 ** 2)

        return {
            'model_memory_gb': model_gb,
            'agents': agent_usage,
            'total_cache_mb': total_cache_mb,
            'total_gb': model_gb + (total_cache_mb / 1024)
        }

    def get_agent_cache(self, agent_id: str) -> Optional[List]:
        """
        Get agent's current KV cache (load from disk if needed).

        Used by BatchedGenerationEngine to get cache before submitting request.
        Ensures the engine always has the latest cache state.

        Args:
            agent_id: Unique identifier

        Returns:
            List: Agent's KV cache, or None if agent doesn't exist
        """
        # If agent in memory, return its cache
        if agent_id in self.agents:
            return self.agents[agent_id].cache

        # Try loading from disk
        if self.persistence.agent_cache_exists(agent_id):
            try:
                self.load_agent(agent_id)
                return self.agents[agent_id].cache
            except Exception as e:
                logger.warning(f"Failed to load agent {agent_id}: {e}")
                return None

        # Agent doesn't exist
        return None

    def update_agent_cache(self, agent_id: str, cache: List):
        """
        Update agent's KV cache after batch generation.

        Called by ConcurrentAgentManager after batch completes to update
        the agent's cache with new conversation state.

        Args:
            agent_id: Unique identifier
            cache: Updated KV cache from batch engine
        """
        if agent_id not in self.agents:
            logger.warning(f"Cannot update cache for unknown agent: {agent_id}")
            return

        agent = self.agents[agent_id]
        agent.cache = cache

        # Update cache token count
        cache_info = self.cache_extractor.get_cache_info(cache)
        agent.cache_tokens = cache_info['total_tokens']

        # Update last access
        agent.update_access()

        logger.debug(
            f"Updated cache for {agent_id}: {agent.cache_tokens} tokens"
        )
