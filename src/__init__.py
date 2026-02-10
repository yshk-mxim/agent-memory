"""Legacy src package - DEPRECATED.

All new code should use `agent_memory.*` imports, not `src.*`.

Example:
    # GOOD
    from agent_memory.domain.entities import KVBlock
    from agent_memory.application.batch_engine import BlockPoolBatchEngine

    # BAD (deprecated)
    from src.agent_manager import PersistentAgentManager
"""

__version__ = "0.1.0"

# No exports - all new code is in agent_memory.* package
__all__: list[str] = []
