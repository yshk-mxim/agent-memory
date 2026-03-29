#!/usr/bin/env python3
"""Add debug fprintf to attention plugin before FMHA runner creation."""
from pathlib import Path

fp = Path("/workspace/agent-memory/vendor/TensorRT-Edge-LLM/cpp/plugins/attentionPlugin/attentionPlugin.cpp")
src = fp.read_text()

old = 'auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads,'
new = 'fprintf(stderr, "[PLUGIN] FMHA: SM=%d batch=%d seq=%d qH=%d kvH=%d headSz=%d\\n", mSMVersion, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads, mHeadSize);\n            auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads,'

if old in src and "[PLUGIN]" not in src:
    src = src.replace(old, new, 1)
    fp.write_text(src)
    print("Debug fprintf added")
else:
    print("Already patched or pattern not found")
