"""
Anthropic-Compatible API Server

FastAPI server implementing Anthropic Messages API (/v1/messages)
with persistent KV cache support via PersistentAgentManager.

Features:
- Non-streaming and streaming (SSE) responses
- Agent persistence via system prompt hash
- Token counting and usage tracking
- Compatible with Claude Code CLI
"""

import hashlib
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging

from .concurrent_manager import ConcurrentAgentManager

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Single message in conversation."""
    role: str
    content: str


class MessagesRequest(BaseModel):
    """Anthropic Messages API request format."""
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    system: Optional[str] = None
    stream: bool = False


class ContentBlock(BaseModel):
    """Content block in response."""
    type: str = "text"
    text: str


class Usage(BaseModel):
    """Token usage information."""
    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """Anthropic Messages API response format."""
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[ContentBlock]
    model: str
    stop_reason: str = "end_turn"
    usage: Usage


class APIServer:
    """
    Anthropic-compatible API server with persistent KV cache.

    Maps system prompts to agents via MD5 hash for persistence.
    Supports streaming and non-streaming responses.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/gemma-3-12b-it-4bit",
        max_agents: int = 3,
        cache_dir: str = "~/.agent_caches",
        max_batch_size: int = 5,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64
    ):
        """
        Initialize API server with agent manager.

        Args:
            model_name: HuggingFace model ID or local path
            max_agents: Maximum number of agents in memory
            cache_dir: Directory for cache persistence
            max_batch_size: Maximum sequences in a batch
            kv_bits: Optional KV cache quantization (2-8 bits, None=no quantization)
            kv_group_size: Group size for quantization (default 64)
        """
        logger.info(
            f"Initializing APIServer with model: {model_name}, "
            f"batch_size={max_batch_size}, kv_bits={kv_bits}"
        )

        self.model_name = model_name
        self.manager = ConcurrentAgentManager(
            model_name=model_name,
            max_agents=max_agents,
            cache_dir=cache_dir,
            max_batch_size=max_batch_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size
        )

        logger.info("APIServer initialized successfully")

    def _system_to_agent_id(self, system_prompt: Optional[str]) -> str:
        """
        Convert system prompt to agent_id via MD5 hash.

        Args:
            system_prompt: System-level instructions (or None)

        Returns:
            str: Agent ID (12-char MD5 hash)
        """
        if not system_prompt:
            system_prompt = "You are a helpful assistant."

        hash_obj = hashlib.md5(system_prompt.encode())
        return hash_obj.hexdigest()[:12]

    def _ensure_agent(self, agent_id: str, system_prompt: str) -> None:
        """
        Ensure agent exists (create if needed, load if on disk).

        Args:
            agent_id: Unique identifier
            system_prompt: System-level instructions
        """
        # Check if already in memory
        if agent_id in self.manager.agents:
            logger.debug(f"Agent {agent_id} already in memory")
            return

        # Try loading from disk
        try:
            self.manager.load_agent(agent_id)
            logger.info(f"Loaded agent {agent_id} from disk")
            return
        except ValueError:
            # Agent doesn't exist - create new
            logger.info(f"Creating new agent {agent_id}")
            self.manager.create_agent(
                agent_id=agent_id,
                agent_type="api_agent",
                system_prompt=system_prompt
            )

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tokenizer.

        Args:
            text: Input text

        Returns:
            int: Token count
        """
        tokens = self.manager.tokenizer.encode(text)
        return len(tokens)

    def _build_prompt_from_messages(self, messages: List[Message]) -> str:
        """
        Build single prompt from message list.

        For now, concatenates messages. Could be improved with
        proper chat template formatting.

        Args:
            messages: List of message dicts

        Returns:
            str: Formatted prompt
        """
        prompt = ""
        for msg in messages:
            role = msg.role.capitalize()
            prompt += f"{role}: {msg.content}\n"

        # Add assistant prefix for response
        prompt += "Assistant:"

        return prompt

    async def handle_messages(self, request: MessagesRequest) -> MessagesResponse:
        """
        Handle non-streaming Messages API request.

        Args:
            request: MessagesRequest object

        Returns:
            MessagesResponse: Generated response with usage
        """
        # Get or create agent
        system_prompt = request.system or "You are a helpful assistant."
        agent_id = self._system_to_agent_id(system_prompt)
        self._ensure_agent(agent_id, system_prompt)

        # Build prompt
        prompt = self._build_prompt_from_messages(request.messages)

        # Count input tokens
        input_tokens = self._count_tokens(prompt)

        # Generate response (async with concurrent manager)
        response_text = await self.manager.generate(
            agent_id=agent_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Count output tokens
        output_tokens = self._count_tokens(response_text)

        # Build response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            content=[ContentBlock(text=response_text)],
            model=self.model_name,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        )

    async def handle_messages_stream(
        self,
        request: MessagesRequest
    ) -> AsyncGenerator[str, None]:
        """
        Handle streaming Messages API request with SSE events.

        Generates proper event sequence:
        - message_start
        - content_block_start
        - content_block_delta (one per token)
        - content_block_stop
        - message_delta
        - message_stop

        Args:
            request: MessagesRequest object

        Yields:
            str: SSE-formatted event strings
        """
        # Get or create agent
        system_prompt = request.system or "You are a helpful assistant."
        agent_id = self._system_to_agent_id(system_prompt)
        self._ensure_agent(agent_id, system_prompt)

        # Build prompt
        prompt = self._build_prompt_from_messages(request.messages)
        input_tokens = self._count_tokens(prompt)

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Event 1: message_start
        yield self._sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model_name,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        })

        # Event 2: content_block_start
        yield self._sse_event("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })

        # Generate response (async with concurrent manager)
        response_text = await self.manager.generate(
            agent_id=agent_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Event 3: content_block_delta (simulate streaming)
        # Split into chunks to simulate token-by-token streaming
        chunk_size = 5  # characters per chunk
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i+chunk_size]
            yield self._sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk}
            })

        # Event 4: content_block_stop
        yield self._sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": 0
        })

        # Count output tokens
        output_tokens = self._count_tokens(response_text)

        # Event 5: message_delta
        yield self._sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": output_tokens}
        })

        # Event 6: message_stop
        yield self._sse_event("message_stop", {
            "type": "message_stop"
        })

    def _sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Format SSE event.

        Args:
            event_type: Event type string
            data: Event data dict

        Returns:
            str: SSE-formatted event
        """
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# Global server instance
_server_instance: Optional[APIServer] = None


def get_server(
    model_name: str = "mlx-community/gemma-3-12b-it-4bit",
    max_agents: int = 3,
    cache_dir: str = "~/.agent_caches",
    max_batch_size: int = 5,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64
) -> APIServer:
    """Get or create global server instance."""
    global _server_instance
    if _server_instance is None:
        _server_instance = APIServer(
            model_name=model_name,
            max_agents=max_agents,
            cache_dir=cache_dir,
            max_batch_size=max_batch_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size
        )
    return _server_instance


# FastAPI app
app = FastAPI(
    title="Persistent Agent API",
    description="Anthropic-compatible Messages API with persistent KV cache",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Start the concurrent manager's background worker."""
    server = get_server()
    await server.manager.start()
    logger.info("Concurrent manager worker started")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the concurrent manager's background worker."""
    server = get_server()
    await server.manager.stop()
    logger.info("Concurrent manager worker stopped")


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    """
    Create a message (Anthropic Messages API compatible).

    Supports both streaming and non-streaming responses.
    """
    server = get_server()

    if request.stream:
        # Return streaming response
        return StreamingResponse(
            server.handle_messages_stream(request),
            media_type="text/event-stream"
        )
    else:
        # Return non-streaming response
        response = await server.handle_messages(request)
        return response


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": get_server().model_name,
        "agents_in_memory": len(get_server().manager.agents)
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Anthropic-compatible API server...")

    uvicorn.run(app, host="0.0.0.0", port=8000)
