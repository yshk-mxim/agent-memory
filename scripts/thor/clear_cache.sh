#!/usr/bin/env bash
# Clear all agent-memory caches (memory + disk).
#
# Usage:
#   ./scripts/thor/clear_cache.sh
#   SEMANTIC_ADMIN_KEY=mykey ./scripts/thor/clear_cache.sh

set -euo pipefail

ADMIN_KEY="${SEMANTIC_ADMIN_KEY:-}"
BASE_URL="${SEMANTIC_SERVER_URL:-http://localhost:8000}"

if [ -z "$ADMIN_KEY" ]; then
    echo "ERROR: SEMANTIC_ADMIN_KEY not set"
    echo "Usage: SEMANTIC_ADMIN_KEY=<key> $0"
    exit 1
fi

echo "Clearing all caches at $BASE_URL..."
curl -s -X DELETE "$BASE_URL/admin/caches" \
    -H "X-Admin-Key: $ADMIN_KEY" | python3 -m json.tool

echo ""
echo "Done."
