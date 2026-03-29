// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yakov Shkolnikov and contributors
// Based on TensorRT Edge-LLM examples/llm/llm_inference.cpp
// (Copyright NVIDIA CORPORATION & AFFILIATES, Apache-2.0)
//
// Interactive NDJSON wrapper around TensorRT Edge-LLM LLMInferenceRuntime.
// Reads JSON commands from stdin, writes JSON responses to stdout.
//
// Protocol:
//   Startup: {"status": "ready"}\n
//   Commands:
//     {"cmd": "generate", "messages": [{"role":"user","content":"..."}],
//      "max_tokens": N, "temperature": T}
//     {"cmd": "get_model_spec"}
//     {"cmd": "shutdown"}

#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <cuda_runtime.h>
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

Json handleGetModelSpec(rt::LLMEngineRunner const& engineRunner)
{
    auto config = engineRunner.getEngineConfig();
    return {
        {"n_layers", config.numDecoderLayers},
        {"n_kv_heads", config.numKVHeads},
        {"head_dim", config.headDim},
        {"block_tokens", 256},
        {"vocab_size", config.vocabSize},
        {"max_seq_len", config.maxKVCacheCapacity},
    };
}

Json handleGenerate(rt::LLMInferenceRuntime& runtime, Json const& cmd, cudaStream_t stream)
{
    // Build LLMGenerationRequest from JSON
    rt::LLMGenerationRequest genRequest;
    genRequest.maxGenerateLength = cmd.value("max_tokens", 256);
    genRequest.temperature = cmd.value("temperature", 0.7f);
    genRequest.topP = cmd.value("top_p", 0.95f);
    genRequest.topK = cmd.value("top_k", 40);

    // Parse messages
    rt::LLMGenerationRequest::Request req;
    if (cmd.contains("messages"))
    {
        for (auto const& msgJson : cmd["messages"])
        {
            rt::Message msg;
            msg.role = msgJson.value("role", "user");
            msg.content = msgJson.value("content", "");
            req.messages.push_back(std::move(msg));
        }
    }
    else
    {
        // Fallback: wrap raw text as a user message
        std::string text = cmd.value("text", "Hello");
        rt::Message msg;
        msg.role = "user";
        msg.content = text;
        req.messages.push_back(std::move(msg));
    }

    genRequest.requests.push_back(std::move(req));

    // Handle system prompt caching
    if (cmd.contains("system_prompt"))
    {
        genRequest.saveSystemPromptKVCache = true;
    }

    // Run inference
    rt::LLMGenerationResponse response;
    bool ok = runtime.handleRequest(genRequest, response, stream);

    if (!ok)
    {
        return {{"error", "Generation failed"}};
    }

    // Build response
    Json result;
    if (!response.outputTexts.empty())
    {
        result["text"] = response.outputTexts[0];
    }
    else
    {
        result["text"] = "";
    }

    if (!response.outputIds.empty())
    {
        result["tokens"] = response.outputIds[0];
    }
    else
    {
        result["tokens"] = Json::array();
    }

    result["finish_reason"] = "stop";
    return result;
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    // Parse --engineDir argument
    std::string engineDir;
    bool debug = false;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--engineDir=") == 0)
        {
            engineDir = arg.substr(12);
        }
        else if (arg == "--engineDir" && i + 1 < argc)
        {
            engineDir = argv[++i];
        }
        else if (arg.find("--engine-path=") == 0)
        {
            engineDir = arg.substr(14);
        }
        else if (arg == "--engine-path" && i + 1 < argc)
        {
            engineDir = argv[++i];
        }
        else if (arg == "--mode" && i + 1 < argc)
        {
            ++i; // Skip mode value
        }
        else if (arg == "--debug")
        {
            debug = true;
        }
    }

    if (engineDir.empty())
    {
        std::cerr << "Usage: " << argv[0] << " --engineDir=<path> [--debug]" << std::endl;
        return 1;
    }

    try
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if (debug)
        {
            std::cerr << "[interactive] Loading engine from: " << engineDir << std::endl;
        }

        // Initialize runtime (no multimodal, no LoRA)
        std::unordered_map<std::string, std::string> emptyLoraMap;
        rt::LLMInferenceRuntime runtime(engineDir, "", emptyLoraMap, stream);

        if (debug)
        {
            std::cerr << "[interactive] Engine loaded, sending ready signal" << std::endl;
        }

        // Signal ready
        sendJson({{"status", "ready"}});

        // Main NDJSON loop
        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty())
            {
                continue;
            }

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
            {
                sendJson(handleGetModelSpec(*runtime.mLLMEngineRunner));
            }
            else if (action == "generate")
            {
                sendJson(handleGenerate(runtime, cmd, stream));
            }
            else if (action == "shutdown")
            {
                sendJson({{"status", "shutdown"}});
                break;
            }
            else
            {
                sendError("Unknown command: " + action);
            }
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
