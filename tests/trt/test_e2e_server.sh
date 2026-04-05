#!/bin/bash
# SPDX-License-Identifier: MIT
# End-to-end test: TRT backend server on Thor
#
# Starts the agent-memory server with SEMANTIC_BACKEND=trt,
# sends Anthropic Messages API requests, verifies responses.
#
# Usage (inside Docker container on Thor):
#   export EDGELLM_PLUGIN_PATH=...
#   export REAL_ENGINE_DIR=...
#   export REAL_INTERACTIVE_BIN=...
#   bash tests/trt/test_e2e_server.sh

set -euo pipefail

ENGINE_DIR="${REAL_ENGINE_DIR:?Set REAL_ENGINE_DIR}"
INTERACTIVE_BIN="${REAL_INTERACTIVE_BIN:?Set REAL_INTERACTIVE_BIN}"
PORT=8199  # Use non-standard port to avoid conflicts

echo "=== Starting server (SEMANTIC_BACKEND=trt) ==="

# Create wrapper script for the adapter
WRAPPER_SCRIPT=$(mktemp /tmp/trt_wrapper_XXXX.sh)
cat > "$WRAPPER_SCRIPT" << WEOF
#!/bin/sh
exec $INTERACTIVE_BIN --engineDir $ENGINE_DIR "\$@"
WEOF
chmod +x "$WRAPPER_SCRIPT"

# Start server in background
SEMANTIC_BACKEND=trt \
SEMANTIC_TRT_ENGINE_PATH="$ENGINE_DIR" \
SEMANTIC_TRT_LLM_INFERENCE_BIN="$WRAPPER_SCRIPT" \
SEMANTIC_TRT_MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct" \
SEMANTIC_SERVER_PORT=$PORT \
SEMANTIC_SERVER_LOG_LEVEL=WARNING \
python -m uvicorn agent_memory.entrypoints.api_server:app \
    --host 0.0.0.0 --port $PORT --log-level warning &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server (PID $SERVER_PID)..."
for i in $(seq 1 30); do
    if curl -s http://localhost:$PORT/health/live > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# Check if server is alive
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "FAIL: Server died"
    exit 1
fi

echo ""
echo "=== Test 1: Health check ==="
HEALTH=$(curl -s http://localhost:$PORT/health/live)
echo "Response: $HEALTH"
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok'" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL: Health check failed"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=== Test 2: Anthropic Messages API ==="
RESPONSE=$(curl -s http://localhost:$PORT/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
        "model": "SmolLM2-135M-Instruct",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "What is 1+1? Answer in one word."}
        ]
    }')
echo "Response: $RESPONSE"
if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d.get('content',[])) > 0; print('Text:', d['content'][0].get('text',''))" 2>/dev/null; then
    echo "PASS"
else
    echo "FAIL: No content in response"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=== Cleanup ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
rm -f "$WRAPPER_SCRIPT"

echo ""
echo "=== ALL TESTS PASSED ==="
