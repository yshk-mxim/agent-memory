#!/bin/bash
# Start llama-server with Gemma 4 26B-A4B (MoE, fast) on Thor
# No speculative decoding — MoE is already 51 t/s

MODEL=~/models/gemma4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf
TEMPLATE=~/agent-memory/config/chat_templates/gemma4-26b-merged.jinja
PORT=8001
SLOTS=4
CTX=262144

mkdir -p ~/.agent_memory/llamacpp_slots

pkill -f "llama-server" 2>/dev/null
sleep 2

echo "Starting llama-server (Gemma 4 26B-A4B) on port $PORT..."
nohup ~/llama.cpp-build/build/bin/llama-server \
    -m "$MODEL" \
    --port $PORT \
    --host 0.0.0.0 \
    -ngl 99 \
    --ctx-size $CTX \
    -np $SLOTS \
    --slot-save-path ~/.agent_memory/llamacpp_slots \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --reasoning auto --reasoning-format deepseek \
    -fa on \
    --jinja --chat-template-file "$TEMPLATE" \
    > ~/llamacpp-serve.log 2>&1 &

echo "llama-server PID: $!"
echo "Log: ~/llamacpp-serve.log"
echo "Health: curl http://localhost:$PORT/health"
