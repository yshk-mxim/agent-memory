#!/bin/bash
# Start llama-server with Gemma 4 31B (Dense) + E2B Q3 speculative decoding on Thor
# E2B Q3 draft: 1.76x speedup (think ON), 2.10x (think OFF)

MODEL=~/models/gemma4-31b/gemma-4-31B-it-Q4_K_M.gguf
DRAFT=~/models/gemma4-e2b/gemma-4-E2B-it-UD-Q3_K_XL.gguf
TEMPLATE=~/agent-memory/config/chat_templates/gemma4-26b-merged.jinja
PORT=8001
SLOTS=1           # 1 slot with spec decode
CTX=131072        # 131K — peak ~74GB fits in 128GB unified

mkdir -p ~/.agent_memory/llamacpp_slots

pkill -f "llama-server" 2>/dev/null
sleep 2

echo "Starting llama-server (Gemma 4 31B + E2B Q3 spec decode) on port $PORT..."
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
    --model-draft "$DRAFT" \
    --gpu-layers-draft 99 \
    --draft 4 \
    --jinja --chat-template-file "$TEMPLATE" \
    > ~/llamacpp-serve.log 2>&1 &

echo "llama-server PID: $!"
echo "Log: ~/llamacpp-serve.log"
echo "Health: curl http://localhost:$PORT/health"
