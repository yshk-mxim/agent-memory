#!/bin/bash

# Papers to download from arXiv (using arxiv2pdf pattern)
PAPERS=(
  "2504.11765:RAG-DCache-Lee-2025"
  "2508.10395:XQuant-2025"
  "2511.11907:KVSwap-2025"
  "2512.03324:TRIM-KV-2025"
  "2512.14946:EvicPress-2025"
  "2601.17668:FastKVzip-2026"
  "2601.19139:vllm-mlx-Barrios-2026"
)

echo "Downloading papers as HTML from arXiv..."
for paper_id in "${PAPERS[@]}"; do
  ID=${paper_id%%:*}
  NAME=${paper_id##*:}
  echo "Downloading $NAME (arXiv:$ID)..."
  curl -L "https://arxiv.org/html/$ID" -o "${NAME}.html" 2>/dev/null
  echo "✓ Downloaded ${NAME}.html"
done

ls -lh *.html
