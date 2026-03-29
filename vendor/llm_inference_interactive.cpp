// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yakov Shkolnikov and contributors
// Based on TensorRT Edge-LLM examples/llm/llm_inference.cpp
// (Copyright NVIDIA CORPORATION & AFFILIATES, Apache-2.0)
//
// Interactive NDJSON wrapper around TensorRT Edge-LLM LLMInferenceRuntime
// with KV cache inject/extract support for agent-memory persistence.
//
// Protocol (stdin/stdout, one JSON per line):
//   Startup:        {"status": "ready"}
//   get_model_spec: {"n_layers":N, "n_kv_heads":N, "head_dim":N, ...}
//   generate:       {"text":"...", "tokens":[...], "finish_reason":"stop",
//                    "kv_cache_path":"/dev/shm/xxx.safetensors"}
//   extract_cache:  {"kv_cache_path":"/dev/shm/xxx.safetensors", "seq_len":N}
//   inject_cache:   {"status":"ok", "seq_len":N}
//   shutdown:       {"status":"shutdown"}

#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/linearKVCache.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{

void sendJson(Json const& j)
{
    std::cout << j.dump() << "\n";
    std::cout.flush();
}

void sendError(std::string const& msg)
{
    sendJson({{"error", msg}});
}

// ---- get_model_spec ----

Json handleGetModelSpec(rt::LLMEngineRunner& engineRunner)
{
    auto config = engineRunner.getEngineConfig();
    auto kvConfig = engineRunner.getLinearKVCache().getConfig();
    return {
        {"n_layers", config.numAttentionLayers},
        {"n_kv_heads", config.numKVHeads},
        {"head_dim", config.headDim},
        {"block_tokens", 256},
        {"vocab_size", config.vocabSize},
        {"max_seq_len", config.maxKVCacheCapacity},
        {"kv_dtype", kvConfig.kvCacheTypeTRT == nvinfer1::DataType::kFP8 ? "fp8" : "fp16"},
    };
}

// ---- extract_cache ----
// Copies per-layer K/V from GPU LinearKVCache to host and saves as safetensors.
// Layout per layer: K=[n_kv_heads, seq_len, head_dim], V=[n_kv_heads, seq_len, head_dim]

Json handleExtractCache(rt::LLMEngineRunner& engineRunner, Json const& cmd, cudaStream_t stream)
{
    auto config = engineRunner.getEngineConfig();
    auto& kvCache = engineRunner.getLinearKVCache();
    auto kvConfig = kvCache.getConfig();

    std::string outputPath = cmd.value("output_path", "/dev/shm/kv_cache.safetensors");

    // Get current sequence length from KV cache
    auto& kvLengths = kvCache.getKVCacheLengths();
    int32_t hostSeqLen = 0;
    cudaMemcpy(&hostSeqLen, kvLengths.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    if (hostSeqLen <= 0)
    {
        return {{"error", "KV cache is empty"}};
    }

    int64_t numLayers = kvConfig.numAttentionLayers;
    int64_t numKVHeads = kvConfig.numKVHeads;
    int64_t headDim = kvConfig.headDim;
    size_t elemSize = (kvConfig.kvCacheTypeTRT == nvinfer1::DataType::kFP8) ? 1 : 2; // FP8=1B, FP16=2B

    // Extract per-layer K and V tensors
    std::vector<rt::Tensor> tensors;
    for (int32_t layer = 0; layer < numLayers; ++layer)
    {
        auto [kGpu, vGpu] = kvCache.getSeparateKVCacheForDecoderLayer(layer);
        // GPU tensors have shape [batch=1, n_kv_heads, max_seq_len, head_dim]
        // We need to copy only the valid seq_len portion

        // Allocate host tensors with trimmed shape [n_kv_heads, seq_len, head_dim]
        rt::Coords kShape({numKVHeads, static_cast<int64_t>(hostSeqLen), headDim});

        rt::Tensor kHost(kShape, rt::DeviceType::kCPU, kvConfig.kvCacheTypeTRT,
            "L" + std::to_string(layer) + "_K");
        rt::Tensor vHost(kShape, rt::DeviceType::kCPU, kvConfig.kvCacheTypeTRT,
            "L" + std::to_string(layer) + "_V");

        // Copy from GPU (stride over max_seq_len) to host (packed seq_len)
        // GPU layout: [batch=1, n_kv_heads, max_seq_len, head_dim]
        // We copy head-by-head to handle the stride
        auto* kSrc = static_cast<char*>(kGpu.rawPointer());
        auto* vSrc = static_cast<char*>(vGpu.rawPointer());
        auto* kDst = static_cast<char*>(kHost.rawPointer());
        auto* vDst = static_cast<char*>(vHost.rawPointer());

        int64_t maxSeqLen = kvConfig.maxSequenceLength;
        for (int64_t h = 0; h < numKVHeads; ++h)
        {
            size_t srcOffset = h * maxSeqLen * headDim * elemSize;
            size_t dstOffset = h * hostSeqLen * headDim * elemSize;
            size_t copyBytes = hostSeqLen * headDim * elemSize;
            cudaMemcpyAsync(kDst + dstOffset, kSrc + srcOffset, copyBytes, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(vDst + dstOffset, vSrc + srcOffset, copyBytes, cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        tensors.push_back(std::move(kHost));
        tensors.push_back(std::move(vHost));
    }

    // Save to safetensors
    if (!rt::safetensors::saveSafetensors(outputPath, tensors, stream))
    {
        return {{"error", "Failed to save KV cache to " + outputPath}};
    }

    return {{"kv_cache_path", outputPath}, {"seq_len", hostSeqLen}};
}

// ---- inject_cache ----
// Loads safetensors from disk and copies per-layer K/V into GPU LinearKVCache.

Json handleInjectCache(rt::LLMEngineRunner& engineRunner, Json const& cmd, cudaStream_t stream)
{
    auto& kvCache = engineRunner.getLinearKVCache();
    auto kvConfig = kvCache.getConfig();

    std::string inputPath = cmd.value("input_path", "");
    if (inputPath.empty())
    {
        return {{"error", "input_path is required"}};
    }

    // Load tensors from safetensors
    std::vector<rt::Tensor> tensors;
    if (!rt::safetensors::loadSafetensors(inputPath, tensors, stream))
    {
        return {{"error", "Failed to load KV cache from " + inputPath}};
    }

    // Determine seq_len from first tensor shape: [n_kv_heads, seq_len, head_dim]
    if (tensors.empty())
    {
        return {{"error", "No tensors in safetensors file"}};
    }
    auto shape = tensors[0].getShape();
    int32_t seqLen = static_cast<int32_t>(shape[1]);

    // Reset KV cache for new sequence with reuse
    rt::Tensor reuseLen(rt::Coords({1}), rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
    *static_cast<int32_t*>(reuseLen.rawPointer()) = 0; // Start fresh
    kvCache.resetForNewSequences(reuseLen, stream);

    int64_t numKVHeads = kvConfig.numKVHeads;
    int64_t headDim = kvConfig.headDim;
    int64_t maxSeqLen = kvConfig.maxSequenceLength;
    size_t elemSize = (kvConfig.kvCacheTypeTRT == nvinfer1::DataType::kFP8) ? 1 : 2;

    // Copy each layer's K and V from host tensors into GPU cache
    for (size_t i = 0; i + 1 < tensors.size(); i += 2)
    {
        int32_t layer = static_cast<int32_t>(i / 2);
        auto [kGpu, vGpu] = kvCache.getSeparateKVCacheForDecoderLayer(layer);

        auto* kSrc = static_cast<char*>(tensors[i].rawPointer());
        auto* vSrc = static_cast<char*>(tensors[i + 1].rawPointer());
        auto* kDst = static_cast<char*>(kGpu.rawPointer());
        auto* vDst = static_cast<char*>(vGpu.rawPointer());

        // Copy head-by-head (host packed seq_len -> GPU strided max_seq_len)
        for (int64_t h = 0; h < numKVHeads; ++h)
        {
            size_t srcOffset = h * seqLen * headDim * elemSize;
            size_t dstOffset = h * maxSeqLen * headDim * elemSize;
            size_t copyBytes = seqLen * headDim * elemSize;
            cudaMemcpyAsync(kDst + dstOffset, kSrc + srcOffset, copyBytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(vDst + dstOffset, vSrc + srcOffset, copyBytes, cudaMemcpyHostToDevice, stream);
        }
    }

    // Commit the injected sequence length
    rt::Tensor newCtxLen(rt::Coords({1}), rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    int32_t hostLen = seqLen;
    cudaMemcpyAsync(newCtxLen.rawPointer(), &hostLen, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    kvCache.commitSequenceLength(newCtxLen, stream);
    cudaStreamSynchronize(stream);

    return {{"status", "ok"}, {"seq_len", seqLen}};
}

// ---- generate ----

Json handleGenerate(rt::LLMInferenceRuntime& runtime, rt::LLMEngineRunner& engineRunner,
    Json const& cmd, cudaStream_t stream)
{
    rt::LLMGenerationRequest genRequest;
    genRequest.maxGenerateLength = cmd.value("max_tokens", 256);
    genRequest.temperature = cmd.value("temperature", 0.7f);
    genRequest.topP = cmd.value("top_p", 0.95f);
    genRequest.topK = cmd.value("top_k", 40);

    rt::LLMGenerationRequest::Request req;
    if (cmd.contains("messages"))
    {
        for (auto const& msgJson : cmd["messages"])
        {
            rt::Message msg;
            msg.role = msgJson.value("role", "user");
            rt::Message::MessageContent mc;
            mc.type = "text";
            mc.content = msgJson.value("content", "");
            msg.contents.push_back(std::move(mc));
            req.messages.push_back(std::move(msg));
        }
    }
    else
    {
        rt::Message msg;
        msg.role = "user";
        rt::Message::MessageContent mc;
        mc.type = "text";
        mc.content = cmd.value("text", "Hello");
        msg.contents.push_back(std::move(mc));
        req.messages.push_back(std::move(msg));
    }
    genRequest.requests.push_back(std::move(req));

    if (cmd.contains("system_prompt"))
    {
        genRequest.saveSystemPromptKVCache = true;
    }

    rt::LLMGenerationResponse response;
    bool ok = runtime.handleRequest(genRequest, response, stream);

    if (!ok)
    {
        return {{"error", "Generation failed"}};
    }

    Json result;
    result["text"] = response.outputTexts.empty() ? "" : response.outputTexts[0];
    result["tokens"] = response.outputIds.empty() ? Json::array() : Json(response.outputIds[0]);
    result["finish_reason"] = "stop";

    // Optionally extract cache after generation
    if (cmd.value("extract_cache", false))
    {
        std::string cachePath = cmd.value("kv_cache_path", "/dev/shm/kv_out.safetensors");
        Json extractCmd = {{"output_path", cachePath}};
        auto cacheResult = handleExtractCache(engineRunner, extractCmd, stream);
        if (cacheResult.contains("kv_cache_path"))
        {
            result["kv_cache_path"] = cacheResult["kv_cache_path"];
        }
    }

    return result;
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    std::string engineDir;
    bool debug = false;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--engineDir=") == 0)
            engineDir = arg.substr(12);
        else if (arg == "--engineDir" && i + 1 < argc)
            engineDir = argv[++i];
        else if (arg.find("--engine-path=") == 0)
            engineDir = arg.substr(14);
        else if (arg == "--engine-path" && i + 1 < argc)
            engineDir = argv[++i];
        else if (arg == "--mode" && i + 1 < argc)
            ++i;
        else if (arg == "--debug")
            debug = true;
    }

    if (engineDir.empty())
    {
        std::cerr << "Usage: " << argv[0] << " --engineDir=<path> [--debug]" << std::endl;
        return 1;
    }

    try
    {
        // Load TRT plugin library (required for attention plugin deserialization)
        auto pluginHandles = loadEdgellmPluginLib();

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if (debug)
            std::cerr << "[interactive] Loading engine: " << engineDir << std::endl;

        std::unordered_map<std::string, std::string> emptyLoraMap;
        rt::LLMInferenceRuntime runtime(engineDir, "", emptyLoraMap, stream);

        if (debug)
            std::cerr << "[interactive] Ready" << std::endl;

        sendJson({{"status", "ready"}});

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty())
                continue;

            Json cmd;
            try
            {
                cmd = Json::parse(line);
            }
            catch (Json::parse_error const& e)
            {
                sendError(std::string("Invalid JSON: ") + e.what());
                continue;
            }

            auto action = cmd.value("cmd", std::string{});

            if (action == "get_model_spec")
                sendJson(handleGetModelSpec(runtime.getEngineRunner()));
            else if (action == "generate")
                sendJson(handleGenerate(runtime, runtime.getEngineRunner(), cmd, stream));
            else if (action == "extract_cache")
                sendJson(handleExtractCache(runtime.getEngineRunner(), cmd, stream));
            else if (action == "inject_cache")
                sendJson(handleInjectCache(runtime.getEngineRunner(), cmd, stream));
            else if (action == "shutdown")
            {
                sendJson({{"status", "shutdown"}});
                break;
            }
            else
                sendError("Unknown command: " + action);
        }

        cudaStreamDestroy(stream);
    }
    catch (std::exception const& e)
    {
        std::cerr << "Fatal: " << e.what() << std::endl;
        sendJson({{"status", "error"}, {"error", e.what()}});
        return 1;
    }

    return 0;
}
