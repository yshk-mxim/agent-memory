#!/bin/bash
# Start agent-memory with Gemma 4 26B-A4B (llama.cpp backend)
# Requires: llama-server running on port 8001 (use start_gemma_26b.sh or start_gemma_31b.sh)

source ~/vllm-env/bin/activate
cd ~/agent-memory

pkill -f "uvicorn.*agent_memory" 2>/dev/null
sleep 2

echo "Starting agent-memory (Gemma 4, llamacpp backend) on port 8000..."
nohup env \
    SEMANTIC_BACKEND=llamacpp \
    SEMANTIC_LLAMACPP_BASE_URL=http://localhost:8001 \
    SEMANTIC_LLAMACPP_MODEL_ID=gemma-4-26b-a4b \
    SEMANTIC_LLAMACPP_TOKENIZER_ID=google/gemma-4-26B-A4B-it \
    SEMANTIC_LLAMACPP_N_SLOTS=4 \
    SEMANTIC_SERVER_SEARXNG_URL=http://localhost:8080 \
    SEMANTIC_SERVER_JINA_READER_URL=http://localhost:3000 \
    python -m uvicorn agent_memory.entrypoints.api_server:create_app \
        --factory --host 0.0.0.0 --port 8000 \
    >> ~/agent-memory-serve.log 2>&1 &

echo "agent-memory PID: $!"
echo "Log: ~/agent-memory-serve.log"
echo "Health: curl http://localhost:8000/health"
