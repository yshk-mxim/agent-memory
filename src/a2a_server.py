"""
A2A Protocol Integration

Implements Google's Agent-to-Agent (A2A) protocol using a2a-sdk.
Features persistent KV cache across A2A tasks and multi-agent delegation.

References:
- A2A Protocol: https://github.com/google/a2a
- a2a-sdk: https://github.com/google/a2a/tree/main/python/packages/a2a-sdk
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
import uvicorn

try:
    from a2a_sdk import (
        AgentCard,
        Skill,
        AgentExecutor,
        EventQueue,
        Task,
        MessageType
    )
except ImportError:
    # Fallback if a2a-sdk not available
    AgentCard = None
    Skill = None
    AgentExecutor = None
    EventQueue = None
    Task = None
    MessageType = None

from .agent_manager import PersistentAgentManager

logger = logging.getLogger(__name__)


class PersistentA2AExecutor:
    """
    A2A AgentExecutor with persistent KV cache support.

    Routes A2A tasks to appropriate cached agents based on skill.
    Maintains persistent cache across tasks for efficiency.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/gemma-3-12b-it-4bit",
        max_agents: int = 3,
        cache_dir: str = "~/.agent_caches"
    ):
        """
        Initialize A2A executor with agent manager.

        Args:
            model_name: HuggingFace model ID or local path
            max_agents: Maximum number of agents in memory
            cache_dir: Directory for cache persistence
        """
        logger.info(f"Initializing PersistentA2AExecutor: {model_name}")

        self.manager = PersistentAgentManager(
            model_name=model_name,
            max_agents=max_agents,
            cache_dir=cache_dir
        )

        # Skill-to-agent mapping
        self.skill_agents = {
            "technical_analysis": "a2a_tech",
            "business_analysis": "a2a_biz",
            "coordination": "a2a_coord"
        }

        # Pre-create agents for each skill
        self._initialize_agents()

        logger.info("PersistentA2AExecutor initialized")

    def _initialize_agents(self):
        """Create agents for each skill."""
        agent_configs = [
            {
                "agent_id": "a2a_tech",
                "agent_type": "technical",
                "system_prompt": (
                    "You are a technical analysis specialist. "
                    "Analyze technical details, architecture, and implementation. "
                    "Provide concise, accurate technical assessments."
                )
            },
            {
                "agent_id": "a2a_biz",
                "agent_type": "business",
                "system_prompt": (
                    "You are a business analysis specialist. "
                    "Analyze business value, ROI, and strategic implications. "
                    "Provide actionable business insights."
                )
            },
            {
                "agent_id": "a2a_coord",
                "agent_type": "coordinator",
                "system_prompt": (
                    "You are a coordination specialist. "
                    "Synthesize inputs from multiple sources and coordinate tasks. "
                    "Provide clear, organized summaries."
                )
            }
        ]

        for config in agent_configs:
            try:
                # Try loading from disk first
                self.manager.load_agent(config["agent_id"])
                logger.info(f"Loaded {config['agent_id']} from disk")
            except ValueError:
                # Create new if not found
                self.manager.create_agent(**config)
                logger.info(f"Created new agent: {config['agent_id']}")

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute A2A task using appropriate cached agent.

        Args:
            task: A2A task dict with skill and message

        Returns:
            dict: A2A response with result
        """
        skill = task.get("skill", "coordination")
        message = task.get("message", "")

        logger.info(f"Executing A2A task: skill={skill}, message='{message[:50]}...'")

        # Get agent for skill
        agent_id = self.skill_agents.get(skill, "a2a_coord")

        # Generate response with persistent cache
        response = self.manager.generate(
            agent_id=agent_id,
            user_input=message,
            max_tokens=300,
            temperature=0.7
        )

        logger.info(f"Task completed for {agent_id}: {len(response)} chars")

        return {
            "result": response,
            "agent_id": agent_id,
            "cache_tokens": self.manager.agents[agent_id].cache_tokens
        }


def create_agent_card() -> Dict[str, Any]:
    """
    Create A2A Agent Card.

    Defines available skills and capabilities.

    Returns:
        dict: Agent card JSON
    """
    return {
        "name": "Persistent Multi-Agent System",
        "description": (
            "Multi-agent system with persistent KV cache for technical analysis, "
            "business analysis, and coordination tasks."
        ),
        "version": "1.0.0",
        "skills": [
            {
                "id": "technical_analysis",
                "name": "Technical Analysis",
                "description": "Analyze technical details, architecture, and implementation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            },
            {
                "id": "business_analysis",
                "name": "Business Analysis",
                "description": "Analyze business value, ROI, and strategic implications",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            },
            {
                "id": "coordination",
                "name": "Coordination",
                "description": "Synthesize inputs and coordinate multi-agent tasks",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        ],
        "capabilities": {
            "streaming": True,
            "persistent_cache": True,
            "multi_agent": True
        }
    }


# Global executor instance
_executor_instance: Optional[PersistentA2AExecutor] = None


def get_executor(
    model_name: str = "mlx-community/gemma-3-12b-it-4bit",
    max_agents: int = 3,
    cache_dir: str = "~/.agent_caches"
) -> PersistentA2AExecutor:
    """Get or create global executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = PersistentA2AExecutor(
            model_name=model_name,
            max_agents=max_agents,
            cache_dir=cache_dir
        )
    return _executor_instance


def create_a2a_app() -> Starlette:
    """
    Create A2A protocol Starlette app.

    Returns:
        Starlette: Configured app with A2A endpoints
    """

    async def agent_card_handler(request):
        """Serve agent card at /.well-known/agent.json"""
        card = create_agent_card()
        return JSONResponse(card)

    async def execute_task_handler(request):
        """Handle A2A task execution POST /tasks"""
        try:
            task = await request.json()
            executor = get_executor()
            result = executor.execute_task(task)
            return JSONResponse(result)
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return JSONResponse(
                {"error": str(e)},
                status_code=500
            )

    async def health_handler(request):
        """Health check endpoint"""
        executor = get_executor()
        return JSONResponse({
            "status": "healthy",
            "agents_in_memory": len(executor.manager.agents),
            "skills": list(executor.skill_agents.keys())
        })

    # Create Starlette app with routes
    app = Starlette(
        routes=[
            Route("/.well-known/agent.json", agent_card_handler),
            Route("/tasks", execute_task_handler, methods=["POST"]),
            Route("/health", health_handler)
        ]
    )

    return app


def demo_multi_agent_delegation():
    """
    Demo showing multi-agent A2A communication with persistent cache.

    Agent A (coordinator) delegates to Agent B (technical) via A2A.
    Both maintain persistent KV caches across tasks.
    """
    logger.info("=== A2A Multi-Agent Delegation Demo ===")

    executor = get_executor()

    # Task 1: Technical analysis (Agent B)
    logger.info("\n1. Agent B (Technical) analyzes architecture...")
    tech_task = {
        "skill": "technical_analysis",
        "message": "Analyze the architecture of a persistent multi-agent memory system"
    }
    tech_result = executor.execute_task(tech_task)
    print(f"Technical Agent Response:\n{tech_result['result']}\n")
    print(f"Cache tokens: {tech_result['cache_tokens']}")

    # Task 2: Business analysis (another agent)
    logger.info("\n2. Agent C (Business) analyzes value proposition...")
    biz_task = {
        "skill": "business_analysis",
        "message": "What is the business value of persistent KV cache for local LLMs?"
    }
    biz_result = executor.execute_task(biz_task)
    print(f"Business Agent Response:\n{biz_result['result']}\n")
    print(f"Cache tokens: {biz_result['cache_tokens']}")

    # Task 3: Coordination (Agent A synthesizes)
    logger.info("\n3. Agent A (Coordinator) synthesizes inputs...")
    coord_task = {
        "skill": "coordination",
        "message": (
            f"Technical perspective: {tech_result['result'][:100]}... "
            f"Business perspective: {biz_result['result'][:100]}... "
            "Synthesize these perspectives into a recommendation."
        )
    }
    coord_result = executor.execute_task(coord_task)
    print(f"Coordinator Response:\n{coord_result['result']}\n")
    print(f"Cache tokens: {coord_result['cache_tokens']}")

    # Show cache persistence benefit
    logger.info("\n4. Follow-up to Technical Agent (uses cached context)...")
    followup_task = {
        "skill": "technical_analysis",
        "message": "Based on your previous analysis, what are the key implementation challenges?"
    }
    followup_result = executor.execute_task(followup_task)
    print(f"Follow-up Response:\n{followup_result['result']}\n")
    print(f"Cache tokens (now includes conversation): {followup_result['cache_tokens']}")

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="A2A Protocol Server")
    parser.add_argument("--demo", action="store_true", help="Run multi-agent demo")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", default=8001, type=int, help="Server port")

    args = parser.parse_args()

    if args.demo:
        demo_multi_agent_delegation()
    else:
        logger.info("Starting A2A protocol server...")
        app = create_a2a_app()
        uvicorn.run(app, host=args.host, port=args.port)
