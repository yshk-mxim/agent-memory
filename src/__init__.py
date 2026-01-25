"""Legacy src package - DEPRECATED.

All new code should use `semantic.*` imports, not `src.*`.

Example:
    # GOOD
    from semantic.domain.entities import KVBlock
    from semantic.application.batch_engine import BlockPoolBatchEngine

    # BAD (deprecated)
    from src.agent_manager import PersistentAgentManager
"""

__version__ = "0.1.0"

# No exports - all new code is in semantic.* package
__all__: list[str] = []
