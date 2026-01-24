"""Domain layer for block-pool memory management.

This package contains pure business logic with zero external dependencies.
All domain code uses only Python stdlib (typing, dataclasses, abc) and
internal semantic.domain imports.

Modules:
    entities: Domain entities (KVBlock, AgentBlocks)
    value_objects: Immutable value objects (ModelCacheSpec, GenerationResult, CacheKey)
    services: Domain services (BlockPool)
    errors: Domain exception hierarchy
"""
