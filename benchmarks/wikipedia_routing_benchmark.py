#!/usr/bin/env python3
"""Wikipedia Multi-Agent Routing Benchmark with Persistent KV Cache.

Demonstrates multi-agent routing where 10 expert agents are each primed
with a Wikipedia article on a statistics topic. Queries are routed to
relevant experts, and responses are synthesized by a reporter agent.

Measures:
- Cold prefill TTFT (priming phase)
- Warm/hot cache TTFT (query phase)
- Cache speedup ratios
- Semantic quality of expert responses

Usage:
    # Full benchmark against running server
    python benchmarks/wikipedia_routing_benchmark.py

    # Skip article download (use cached files)
    python benchmarks/wikipedia_routing_benchmark.py --skip-download

    # Custom port
    python benchmarks/wikipedia_routing_benchmark.py --port 8001

    # Verbose output
    python benchmarks/wikipedia_routing_benchmark.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTICLES_DIR = Path(__file__).resolve().parent / "data" / "wiki_articles"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Expert agents: name -> Wikipedia article title
EXPERT_AGENTS: dict[str, str] = {
    "bayesian_inference": "Bayesian_inference",
    "central_limit_theorem": "Central_limit_theorem",
    "regression_analysis": "Regression_analysis",
    "hypothesis_testing": "Statistical_hypothesis_testing",
    "markov_chain": "Markov_chain",
    "monte_carlo": "Monte_Carlo_method",
    "principal_component_analysis": "Principal_component_analysis",
    "time_series": "Time_series",
    "maximum_likelihood": "Maximum_likelihood_estimation",
    "anova": "Analysis_of_variance",
}

# Domain keywords per expert for quality checking
EXPERT_KEYWORDS: dict[str, list[str]] = {
    "bayesian_inference": ["bayes", "prior", "posterior", "likelihood", "probability"],
    "central_limit_theorem": ["distribution", "normal", "mean", "sample", "convergence"],
    "regression_analysis": ["regression", "variable", "coefficient", "predictor", "model"],
    "hypothesis_testing": ["hypothesis", "null", "test", "significance", "p-value"],
    "markov_chain": ["markov", "state", "transition", "probability", "chain"],
    "monte_carlo": ["monte carlo", "simulation", "random", "sampling", "estimate"],
    "principal_component_analysis": ["principal", "component", "variance", "dimension", "eigenvalue"],
    "time_series": ["time series", "forecast", "trend", "seasonal", "autoregressive"],
    "maximum_likelihood": ["likelihood", "estimator", "parameter", "maximum", "mle"],
    "anova": ["variance", "anova", "group", "factor", "mean"],
}

QUERIES = [
    {
        "question": "Compare Bayesian and frequentist approaches to hypothesis testing",
        "experts": ["bayesian_inference", "hypothesis_testing"],
    },
    {
        "question": "How do Markov chains relate to Monte Carlo methods?",
        "experts": ["markov_chain", "monte_carlo"],
    },
    {
        "question": "Explain PCA in the context of regression analysis",
        "experts": ["principal_component_analysis", "regression_analysis"],
    },
    {
        "question": "When should I use ANOVA vs regression?",
        "experts": ["anova", "regression_analysis"],
    },
    {
        "question": "How does the central limit theorem justify maximum likelihood estimation?",
        "experts": ["central_limit_theorem", "maximum_likelihood"],
    },
]

# Repeated queries for Phase 3 (hot cache benefit)
REPEATED_EXPERTS = ["bayesian_inference", "monte_carlo", "regression_analysis"]
REPEATED_QUERY = "Give a concise definition and one practical application."

MAX_TOKENS_PRIME = 256
MAX_TOKENS_QUERY = 256
MAX_TOKENS_SYNTHESIS = 512
TEMPERATURE = 0.3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingRecord:
    """Single API call timing measurement."""
    agent_name: str
    phase: str          # "prime", "query", "repeat", "synthesis"
    ttft_ms: float      # Time to first token (streaming)
    e2e_ms: float       # Total end-to-end time
    output_tokens: int  # Approximate output token count
    cache_state: str    # "cold", "warm", "hot"
    query: str = ""
    error: str | None = None


@dataclass
class QualityScore:
    """Semantic quality check for a response."""
    agent_name: str
    phase: str
    non_empty: bool
    sufficient_length: bool
    no_repetition: bool
    keyword_relevance: bool
    keyword_matches: list[str]
    text_length: int

    @property
    def passed(self) -> bool:
        return self.non_empty and self.sufficient_length and self.no_repetition and self.keyword_relevance


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    model_id: str
    timestamp: str
    git_sha: str
    server_port: int
    expert_count: int
    query_count: int
    timings: list[dict[str, Any]] = field(default_factory=list)
    quality: list[dict[str, Any]] = field(default_factory=list)
    agent_cache_state: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _sanitize_model_id(model_id: str) -> str:
    """Convert model ID to filesystem-safe string."""
    return model_id.replace("/", "_").replace(" ", "_")


def check_quality(text: str, agent_name: str, phase: str) -> QualityScore:
    """Check semantic quality of a response."""
    non_empty = len(text.strip()) > 0
    # Approximate token count: ~4 chars per token
    approx_tokens = len(text) / 4
    sufficient_length = approx_tokens > 50

    # Check for repetition loops: substring of 20+ chars repeated 3+ times
    no_repetition = True
    if len(text) > 60:
        # Check for any 20-char substring appearing 3+ times
        for i in range(len(text) - 60):
            substr = text[i:i + 20]
            if text.count(substr) >= 3:
                no_repetition = False
                break

    # Check keyword relevance
    keywords = EXPERT_KEYWORDS.get(agent_name, [])
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw in text_lower]
    keyword_relevance = len(matched) >= 2

    return QualityScore(
        agent_name=agent_name,
        phase=phase,
        non_empty=non_empty,
        sufficient_length=sufficient_length,
        no_repetition=no_repetition,
        keyword_relevance=keyword_relevance,
        keyword_matches=matched,
        text_length=len(text),
    )


# ---------------------------------------------------------------------------
# Article download
# ---------------------------------------------------------------------------

async def download_article(title: str, filename: str) -> str:
    """Download a Wikipedia article and cache locally.

    Tries the full extract API first, falls back to summary API.

    Args:
        title: Wikipedia article title (e.g. "Bayesian_inference")
        filename: Local filename to save (without directory)

    Returns:
        Article text content
    """
    filepath = ARTICLES_DIR / filename
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        if len(content) > 100:
            return content

    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    # Try full extract API first (longer articles)
    full_url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={title}&prop=extracts&explaintext=1&format=json"
    )
    # Fallback: summary API (shorter but more reliable)
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        text = ""

        # Attempt 1: full extract
        try:
            resp = await client.get(full_url, headers={"User-Agent": "SemanticBenchmark/1.0 (https://github.com/semantic-cache; academic-research; python-httpx)"})
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if page_id != "-1":
                        text = page_data.get("extract", "")
                        break
        except Exception as exc:
            print(f"  [WARN] Full extract failed for {title}: {exc}")

        # Attempt 2: summary if full extract was empty or failed
        if len(text) < 200:
            try:
                resp = await client.get(
                    summary_url,
                    headers={"User-Agent": "SemanticBenchmark/1.0 (https://github.com/semantic-cache; academic-research; python-httpx)"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("extract", "")
            except Exception as exc:
                print(f"  [WARN] Summary API failed for {title}: {exc}")

        if not text:
            raise RuntimeError(f"Failed to download article: {title}")

        # Truncate to ~3000 words to keep context manageable
        words = text.split()
        if len(words) > 3000:
            text = " ".join(words[:3000]) + "\n\n[Article truncated for benchmark context.]"

        filepath.write_text(text, encoding="utf-8")
        print(f"  Downloaded {title}: {len(words)} words -> {filepath.name}")
        return text


async def download_all_articles(skip_download: bool = False) -> dict[str, str]:
    """Download all expert articles.

    Args:
        skip_download: If True, only load from cache (fail if missing).

    Returns:
        Dict mapping expert_name -> article_text
    """
    articles: dict[str, str] = {}
    print("\n=== Downloading Wikipedia Articles ===\n")

    for expert_name, wiki_title in EXPERT_AGENTS.items():
        filename = f"{expert_name}.txt"
        filepath = ARTICLES_DIR / filename

        if skip_download:
            if not filepath.exists():
                print(f"  [ERROR] Missing cached article: {filepath}")
                sys.exit(1)
            articles[expert_name] = filepath.read_text(encoding="utf-8")
            word_count = len(articles[expert_name].split())
            print(f"  Loaded {expert_name}: {word_count} words (cached)")
        else:
            try:
                articles[expert_name] = await download_article(wiki_title, filename)
            except Exception as exc:
                print(f"  [ERROR] Failed to get article for {expert_name}: {exc}")
                sys.exit(1)

    print(f"\nAll {len(articles)} articles ready.")
    return articles


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------

async def check_server_ready(base_url: str, timeout: float = 10.0) -> bool:
    """Check if the server is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/health/ready")
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


async def get_model_id(base_url: str) -> str:
    """Get the model ID from the server."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/v1/models")
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id", "unknown")
    return "unknown"


async def get_agent_cache_state(base_url: str) -> dict[str, Any]:
    """Get current agent cache state from server."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/v1/agents/list")
            if resp.status_code == 200:
                return resp.json()

            resp = await client.get(f"{base_url}/v1/agents/stats")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


async def delete_agent(base_url: str, agent_id: str, evict_only: bool = False) -> None:
    """Delete or evict an agent from the cache."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{base_url}/v1/agents/{agent_id}"
            if evict_only:
                url += "?evict_only=true"
            await client.delete(url)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Streaming chat completion
# ---------------------------------------------------------------------------

async def stream_chat_completion(
    base_url: str,
    messages: list[dict[str, str]],
    session_id: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
    verbose: bool = False,
) -> tuple[str, TimingRecord]:
    """Send a streaming chat completion request and measure timing.

    Uses session_id to tie requests to a specific agent's KV cache.

    Args:
        base_url: Server base URL
        messages: Chat messages
        session_id: Session ID for cache routing (becomes agent_id = oai_{session_id})
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        verbose: Print debug info

    Returns:
        Tuple of (response_text, timing_record_without_phase_info)
    """
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {
        "X-Session-ID": session_id,
        "Content-Type": "application/json",
    }

    t_start = time.perf_counter()
    ttft = 0.0
    text = ""
    delta_count = 0
    output_tokens = 0
    error_msg: str | None = None

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    t_end = time.perf_counter()
                    error_body = ""
                    async for chunk in resp.aiter_text():
                        error_body += chunk
                    error_msg = f"HTTP {resp.status_code}: {error_body[:200]}"
                    return "", TimingRecord(
                        agent_name="",
                        phase="",
                        ttft_ms=0.0,
                        e2e_ms=(t_end - t_start) * 1000,
                        output_tokens=0,
                        cache_state="",
                        error=error_msg,
                    )

                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk.replace("\r\n", "\n")
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        for line in event_str.strip().split("\n"):
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                continue
                            try:
                                parsed = json.loads(payload)
                            except json.JSONDecodeError:
                                continue

                            # Extract usage from final chunk
                            chunk_usage = parsed.get("usage", {})
                            if chunk_usage:
                                output_tokens = chunk_usage.get(
                                    "completion_tokens", output_tokens
                                )

                            choices = parsed.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if ttft == 0.0:
                                    ttft = (time.perf_counter() - t_start) * 1000
                                text += content
                                delta_count += 1

    except Exception as exc:
        t_end = time.perf_counter()
        error_msg = str(exc)
        return "", TimingRecord(
            agent_name="",
            phase="",
            ttft_ms=0.0,
            e2e_ms=(t_end - t_start) * 1000,
            output_tokens=0,
            cache_state="",
            error=error_msg,
        )

    t_end = time.perf_counter()
    e2e_ms = (t_end - t_start) * 1000

    # Approximate tokens from text if usage not reported
    if output_tokens == 0:
        output_tokens = max(1, len(text.split()))

    if verbose and text:
        preview = text[:120].replace("\n", " ")
        print(f"    Response ({output_tokens} tok, TTFT={ttft:.0f}ms): {preview}...")

    return text, TimingRecord(
        agent_name="",
        phase="",
        ttft_ms=ttft,
        e2e_ms=e2e_ms,
        output_tokens=output_tokens,
        cache_state="",
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Benchmark phases
# ---------------------------------------------------------------------------

async def phase1_priming(
    base_url: str,
    articles: dict[str, str],
    verbose: bool = False,
) -> list[tuple[TimingRecord, QualityScore]]:
    """Phase 1: Prime each expert agent with its Wikipedia article.

    Sends the article as system message + "Summarize the key concepts" as user query.
    This is the cold prefill phase.

    Returns:
        List of (timing, quality) tuples for each expert
    """
    print("\n=== Phase 1: PRIMING (Cold Prefill) ===\n")
    results: list[tuple[TimingRecord, QualityScore]] = []

    for expert_name, article_text in articles.items():
        session_id = f"wiki_{expert_name}"
        agent_id = f"oai_{session_id}"

        # Delete any existing cache to ensure cold start
        await delete_agent(base_url, agent_id)
        await asyncio.sleep(0.5)  # Brief pause for cleanup

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert on {expert_name.replace('_', ' ')}. "
                    f"The following article is your knowledge base:\n\n{article_text}"
                ),
            },
            {
                "role": "user",
                "content": "Summarize the key concepts from this article in a clear, structured way.",
            },
        ]

        print(f"  Priming {expert_name}...", end=" ", flush=True)
        text, timing = await stream_chat_completion(
            base_url, messages, session_id,
            max_tokens=MAX_TOKENS_PRIME,
            verbose=verbose,
        )

        timing.agent_name = expert_name
        timing.phase = "prime"
        timing.cache_state = "cold"

        quality = check_quality(text, expert_name, "prime")

        status_str = "OK" if quality.passed else "WARN"
        print(
            f"TTFT={timing.ttft_ms:.0f}ms  E2E={timing.e2e_ms:.0f}ms  "
            f"tokens={timing.output_tokens}  quality={status_str}"
        )
        if timing.error:
            print(f"    ERROR: {timing.error}")

        results.append((timing, quality))

    return results


async def phase2_queries(
    base_url: str,
    articles: dict[str, str],
    verbose: bool = False,
) -> list[tuple[list[TimingRecord], list[QualityScore], TimingRecord | None]]:
    """Phase 2: Cross-topic queries routed to relevant experts.

    For each query:
    1. Send to 2-3 relevant experts (they should have warm/hot cache)
    2. Collect expert responses
    3. Send combined responses to a reporter agent for synthesis

    Returns:
        List of (expert_timings, expert_qualities, synthesis_timing) per query
    """
    print("\n=== Phase 2: QUERIES (Warm/Hot Cache) ===\n")
    all_results: list[tuple[list[TimingRecord], list[QualityScore], TimingRecord | None]] = []

    for qi, query_spec in enumerate(QUERIES):
        question = query_spec["question"]
        expert_names = query_spec["experts"]
        print(f"  Query {qi + 1}: {question}")
        print(f"  Experts: {', '.join(expert_names)}")

        expert_timings: list[TimingRecord] = []
        expert_qualities: list[QualityScore] = []
        expert_responses: list[tuple[str, str]] = []  # (name, response)

        for expert_name in expert_names:
            session_id = f"wiki_{expert_name}"
            article_text = articles[expert_name]

            # Re-send the same system context + new question
            # The server should detect cache hit on the system message prefix
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are an expert on {expert_name.replace('_', ' ')}. "
                        f"The following article is your knowledge base:\n\n{article_text}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Summarize the key concepts from this article in a clear, structured way.",
                },
                {
                    "role": "assistant",
                    "content": "[Previous summary acknowledged]",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]

            print(f"    -> {expert_name}...", end=" ", flush=True)
            text, timing = await stream_chat_completion(
                base_url, messages, session_id,
                max_tokens=MAX_TOKENS_QUERY,
                verbose=verbose,
            )

            timing.agent_name = expert_name
            timing.phase = "query"
            timing.cache_state = "warm_or_hot"
            timing.query = question

            quality = check_quality(text, expert_name, "query")
            status_str = "OK" if quality.passed else "WARN"
            print(
                f"TTFT={timing.ttft_ms:.0f}ms  E2E={timing.e2e_ms:.0f}ms  "
                f"tokens={timing.output_tokens}  quality={status_str}"
            )
            if timing.error:
                print(f"      ERROR: {timing.error}")

            expert_timings.append(timing)
            expert_qualities.append(quality)
            if text:
                expert_responses.append((expert_name, text))

        # Synthesis by reporter agent
        synthesis_timing: TimingRecord | None = None
        if expert_responses:
            reporter_session = f"wiki_reporter_q{qi}"
            reporter_agent_id = f"oai_{reporter_session}"
            # Clear reporter cache for fresh synthesis
            await delete_agent(base_url, reporter_agent_id)
            await asyncio.sleep(0.3)

            expert_text_block = "\n\n".join(
                f"=== Expert: {name} ===\n{resp}" for name, resp in expert_responses
            )
            synthesis_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a research synthesizer. Combine the expert analyses "
                        "below into a coherent, balanced answer. Highlight agreements, "
                        "disagreements, and connections between the perspectives."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nExpert Analyses:\n{expert_text_block}",
                },
            ]

            print(f"    -> reporter (synthesis)...", end=" ", flush=True)
            synth_text, synthesis_timing = await stream_chat_completion(
                base_url, synthesis_messages, reporter_session,
                max_tokens=MAX_TOKENS_SYNTHESIS,
                verbose=verbose,
            )
            synthesis_timing.agent_name = "reporter"
            synthesis_timing.phase = "synthesis"
            synthesis_timing.cache_state = "cold"
            synthesis_timing.query = question

            print(
                f"TTFT={synthesis_timing.ttft_ms:.0f}ms  "
                f"E2E={synthesis_timing.e2e_ms:.0f}ms  "
                f"tokens={synthesis_timing.output_tokens}"
            )
            if synthesis_timing.error:
                print(f"      ERROR: {synthesis_timing.error}")

        all_results.append((expert_timings, expert_qualities, synthesis_timing))
        print()

    return all_results


async def phase3_repeated(
    base_url: str,
    articles: dict[str, str],
    verbose: bool = False,
) -> list[tuple[TimingRecord, QualityScore]]:
    """Phase 3: Repeat queries to previously-queried experts to show hot cache benefit.

    These agents were already primed (Phase 1) and queried (Phase 2), so their
    KV cache should be in hot tier.

    Returns:
        List of (timing, quality) tuples
    """
    print("\n=== Phase 3: REPEATED QUERIES (Hot Cache) ===\n")
    results: list[tuple[TimingRecord, QualityScore]] = []

    for expert_name in REPEATED_EXPERTS:
        session_id = f"wiki_{expert_name}"
        article_text = articles[expert_name]

        # Build full conversation history to maximize cache reuse
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert on {expert_name.replace('_', ' ')}. "
                    f"The following article is your knowledge base:\n\n{article_text}"
                ),
            },
            {
                "role": "user",
                "content": "Summarize the key concepts from this article in a clear, structured way.",
            },
            {
                "role": "assistant",
                "content": "[Previous summary acknowledged]",
            },
            {
                "role": "user",
                "content": REPEATED_QUERY,
            },
        ]

        print(f"  Re-querying {expert_name}...", end=" ", flush=True)
        text, timing = await stream_chat_completion(
            base_url, messages, session_id,
            max_tokens=MAX_TOKENS_QUERY,
            verbose=verbose,
        )

        timing.agent_name = expert_name
        timing.phase = "repeat"
        timing.cache_state = "hot"
        timing.query = REPEATED_QUERY

        quality = check_quality(text, expert_name, "repeat")
        status_str = "OK" if quality.passed else "WARN"
        print(
            f"TTFT={timing.ttft_ms:.0f}ms  E2E={timing.e2e_ms:.0f}ms  "
            f"tokens={timing.output_tokens}  quality={status_str}"
        )
        if timing.error:
            print(f"    ERROR: {timing.error}")

        results.append((timing, quality))

    return results


# ---------------------------------------------------------------------------
# Summary and output
# ---------------------------------------------------------------------------

def compute_summary(
    prime_results: list[tuple[TimingRecord, QualityScore]],
    query_results: list[tuple[list[TimingRecord], list[QualityScore], TimingRecord | None]],
    repeat_results: list[tuple[TimingRecord, QualityScore]],
) -> dict[str, Any]:
    """Compute summary statistics across all phases."""

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        if n % 2 == 0:
            return (s[n // 2 - 1] + s[n // 2]) / 2
        return s[n // 2]

    # Phase 1 stats
    prime_ttfts = [t.ttft_ms for t, _ in prime_results if not t.error and t.ttft_ms > 0]
    prime_e2es = [t.e2e_ms for t, _ in prime_results if not t.error]
    prime_quality_pass = sum(1 for _, q in prime_results if q.passed)

    # Phase 2 stats (expert calls only, not synthesis)
    query_ttfts: list[float] = []
    query_e2es: list[float] = []
    query_quality_pass = 0
    query_quality_total = 0
    synthesis_ttfts: list[float] = []
    for expert_timings, expert_qualities, synth_timing in query_results:
        for t in expert_timings:
            if not t.error and t.ttft_ms > 0:
                query_ttfts.append(t.ttft_ms)
            if not t.error:
                query_e2es.append(t.e2e_ms)
        for q in expert_qualities:
            query_quality_total += 1
            if q.passed:
                query_quality_pass += 1
        if synth_timing and not synth_timing.error and synth_timing.ttft_ms > 0:
            synthesis_ttfts.append(synth_timing.ttft_ms)

    # Phase 3 stats
    repeat_ttfts = [t.ttft_ms for t, _ in repeat_results if not t.error and t.ttft_ms > 0]
    repeat_e2es = [t.e2e_ms for t, _ in repeat_results if not t.error]
    repeat_quality_pass = sum(1 for _, q in repeat_results if q.passed)

    # Speedup ratios
    avg_prime_ttft = avg(prime_ttfts)
    avg_query_ttft = avg(query_ttfts)
    avg_repeat_ttft = avg(repeat_ttfts)

    warm_speedup = avg_prime_ttft / avg_query_ttft if avg_query_ttft > 0 else 0.0
    hot_speedup = avg_prime_ttft / avg_repeat_ttft if avg_repeat_ttft > 0 else 0.0

    return {
        "phase1_priming": {
            "count": len(prime_results),
            "errors": sum(1 for t, _ in prime_results if t.error),
            "ttft_avg_ms": round(avg(prime_ttfts), 1),
            "ttft_median_ms": round(median(prime_ttfts), 1),
            "ttft_min_ms": round(min(prime_ttfts), 1) if prime_ttfts else 0,
            "ttft_max_ms": round(max(prime_ttfts), 1) if prime_ttfts else 0,
            "e2e_avg_ms": round(avg(prime_e2es), 1),
            "quality_pass": prime_quality_pass,
            "quality_total": len(prime_results),
        },
        "phase2_queries": {
            "query_count": len(QUERIES),
            "expert_calls": len(query_ttfts) + sum(
                1 for et, _, _ in query_results for t in et if t.error
            ),
            "errors": sum(1 for et, _, _ in query_results for t in et if t.error),
            "ttft_avg_ms": round(avg(query_ttfts), 1),
            "ttft_median_ms": round(median(query_ttfts), 1),
            "ttft_min_ms": round(min(query_ttfts), 1) if query_ttfts else 0,
            "ttft_max_ms": round(max(query_ttfts), 1) if query_ttfts else 0,
            "e2e_avg_ms": round(avg(query_e2es), 1),
            "quality_pass": query_quality_pass,
            "quality_total": query_quality_total,
            "synthesis_ttft_avg_ms": round(avg(synthesis_ttfts), 1),
        },
        "phase3_repeated": {
            "count": len(repeat_results),
            "errors": sum(1 for t, _ in repeat_results if t.error),
            "ttft_avg_ms": round(avg(repeat_ttfts), 1),
            "ttft_median_ms": round(median(repeat_ttfts), 1),
            "ttft_min_ms": round(min(repeat_ttfts), 1) if repeat_ttfts else 0,
            "ttft_max_ms": round(max(repeat_ttfts), 1) if repeat_ttfts else 0,
            "e2e_avg_ms": round(avg(repeat_e2es), 1),
            "quality_pass": repeat_quality_pass,
            "quality_total": len(repeat_results),
        },
        "speedup": {
            "warm_vs_cold": round(warm_speedup, 2),
            "hot_vs_cold": round(hot_speedup, 2),
            "cold_ttft_avg_ms": round(avg_prime_ttft, 1),
            "warm_ttft_avg_ms": round(avg_query_ttft, 1),
            "hot_ttft_avg_ms": round(avg_repeat_ttft, 1),
        },
    }


def print_summary_table(summary: dict[str, Any]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 78)
    print("WIKIPEDIA ROUTING BENCHMARK RESULTS")
    print("=" * 78)

    p1 = summary["phase1_priming"]
    p2 = summary["phase2_queries"]
    p3 = summary["phase3_repeated"]
    sp = summary["speedup"]

    print(f"\n{'Phase':<25} {'TTFT avg':>10} {'TTFT med':>10} {'TTFT min':>10} {'TTFT max':>10} {'E2E avg':>10}")
    print("-" * 78)
    print(
        f"{'1. Priming (cold)':<25} "
        f"{p1['ttft_avg_ms']:>9.0f}ms"
        f"{p1['ttft_median_ms']:>9.0f}ms"
        f"{p1['ttft_min_ms']:>9.0f}ms"
        f"{p1['ttft_max_ms']:>9.0f}ms"
        f"{p1['e2e_avg_ms']:>9.0f}ms"
    )
    print(
        f"{'2. Queries (warm/hot)':<25} "
        f"{p2['ttft_avg_ms']:>9.0f}ms"
        f"{p2['ttft_median_ms']:>9.0f}ms"
        f"{p2['ttft_min_ms']:>9.0f}ms"
        f"{p2['ttft_max_ms']:>9.0f}ms"
        f"{p2['e2e_avg_ms']:>9.0f}ms"
    )
    print(
        f"{'3. Repeated (hot)':<25} "
        f"{p3['ttft_avg_ms']:>9.0f}ms"
        f"{p3['ttft_median_ms']:>9.0f}ms"
        f"{p3['ttft_min_ms']:>9.0f}ms"
        f"{p3['ttft_max_ms']:>9.0f}ms"
        f"{p3['e2e_avg_ms']:>9.0f}ms"
    )
    print(
        f"{'   Synthesis (cold)':<25} "
        f"{p2['synthesis_ttft_avg_ms']:>9.0f}ms"
        f"{'':>10}{'':>10}{'':>10}{'':>10}"
    )

    print(f"\n{'Speedup Ratios':<25}")
    print("-" * 40)
    print(f"  Warm/Hot vs Cold TTFT:  {sp['warm_vs_cold']:.2f}x")
    print(f"  Hot vs Cold TTFT:       {sp['hot_vs_cold']:.2f}x")

    print(f"\n{'Quality Scores':<25}")
    print("-" * 40)
    print(f"  Phase 1 (priming):      {p1['quality_pass']}/{p1['quality_total']} passed")
    print(f"  Phase 2 (queries):      {p2['quality_pass']}/{p2['quality_total']} passed")
    print(f"  Phase 3 (repeated):     {p3['quality_pass']}/{p3['quality_total']} passed")

    # Errors
    total_errors = p1["errors"] + p2["errors"] + p3["errors"]
    if total_errors > 0:
        print(f"\n  ERRORS: {total_errors} total")
        if p1["errors"]:
            print(f"    Phase 1: {p1['errors']}")
        if p2["errors"]:
            print(f"    Phase 2: {p2['errors']}")
        if p3["errors"]:
            print(f"    Phase 3: {p3['errors']}")

    print()


def print_per_expert_table(
    prime_results: list[tuple[TimingRecord, QualityScore]],
    query_results: list[tuple[list[TimingRecord], list[QualityScore], TimingRecord | None]],
    repeat_results: list[tuple[TimingRecord, QualityScore]],
) -> None:
    """Print per-expert timing breakdown."""
    print(f"\n{'Per-Expert TTFT Breakdown':<40}")
    print("=" * 70)
    print(
        f"{'Expert':<30} {'Cold (P1)':>12} {'Warm (P2)':>12} {'Hot (P3)':>12}"
    )
    print("-" * 70)

    # Index query timings by expert name
    query_ttft_by_expert: dict[str, list[float]] = {}
    for expert_timings, _, _ in query_results:
        for t in expert_timings:
            if not t.error and t.ttft_ms > 0:
                query_ttft_by_expert.setdefault(t.agent_name, []).append(t.ttft_ms)

    # Index repeat timings by expert name
    repeat_ttft_by_expert: dict[str, float] = {}
    for t, _ in repeat_results:
        if not t.error and t.ttft_ms > 0:
            repeat_ttft_by_expert[t.agent_name] = t.ttft_ms

    for t_prime, _ in prime_results:
        name = t_prime.agent_name
        cold_str = f"{t_prime.ttft_ms:.0f}ms" if not t_prime.error and t_prime.ttft_ms > 0 else "ERR"

        query_vals = query_ttft_by_expert.get(name, [])
        if query_vals:
            avg_query = sum(query_vals) / len(query_vals)
            warm_str = f"{avg_query:.0f}ms"
        else:
            warm_str = "-"

        hot_val = repeat_ttft_by_expert.get(name)
        hot_str = f"{hot_val:.0f}ms" if hot_val else "-"

        print(f"  {name:<28} {cold_str:>12} {warm_str:>12} {hot_str:>12}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark."""
    base_url = f"http://localhost:{args.port}"

    # Check server readiness
    print(f"Checking server at {base_url}...")
    if not await check_server_ready(base_url, timeout=15.0):
        print(f"ERROR: Server not ready at {base_url}")
        print("Start the server first:")
        print(f"  python -m semantic.entrypoints.cli serve --port {args.port}")
        sys.exit(1)
    print("Server is ready.")

    # Get model info
    model_id = await get_model_id(base_url)
    print(f"Model: {model_id}")

    # Download articles
    articles = await download_all_articles(skip_download=args.skip_download)

    # Phase 1: Priming
    prime_results = await phase1_priming(base_url, articles, verbose=args.verbose)

    # Check cache state after priming
    cache_after_prime = await get_agent_cache_state(base_url)
    if cache_after_prime:
        agent_count = cache_after_prime.get("total", "?")
        print(f"\n  Cache state after priming: {agent_count} agents cached")

    # Phase 2: Cross-topic queries
    query_results = await phase2_queries(base_url, articles, verbose=args.verbose)

    # Phase 3: Repeated queries
    repeat_results = await phase3_repeated(base_url, articles, verbose=args.verbose)

    # Final cache state
    cache_final = await get_agent_cache_state(base_url)

    # Compute summary
    summary = compute_summary(prime_results, query_results, repeat_results)

    # Print results
    print_summary_table(summary)
    print_per_expert_table(prime_results, query_results, repeat_results)

    # Collect all timings and quality scores
    all_timings: list[dict[str, Any]] = []
    all_quality: list[dict[str, Any]] = []

    for t, q in prime_results:
        all_timings.append({
            "agent_name": t.agent_name, "phase": t.phase,
            "ttft_ms": round(t.ttft_ms, 1), "e2e_ms": round(t.e2e_ms, 1),
            "output_tokens": t.output_tokens, "cache_state": t.cache_state,
            "query": t.query, "error": t.error,
        })
        all_quality.append({
            "agent_name": q.agent_name, "phase": q.phase,
            "passed": q.passed, "non_empty": q.non_empty,
            "sufficient_length": q.sufficient_length,
            "no_repetition": q.no_repetition,
            "keyword_relevance": q.keyword_relevance,
            "keyword_matches": q.keyword_matches,
            "text_length": q.text_length,
        })

    for expert_timings, expert_qualities, synth_timing in query_results:
        for t in expert_timings:
            all_timings.append({
                "agent_name": t.agent_name, "phase": t.phase,
                "ttft_ms": round(t.ttft_ms, 1), "e2e_ms": round(t.e2e_ms, 1),
                "output_tokens": t.output_tokens, "cache_state": t.cache_state,
                "query": t.query, "error": t.error,
            })
        for q in expert_qualities:
            all_quality.append({
                "agent_name": q.agent_name, "phase": q.phase,
                "passed": q.passed, "non_empty": q.non_empty,
                "sufficient_length": q.sufficient_length,
                "no_repetition": q.no_repetition,
                "keyword_relevance": q.keyword_relevance,
                "keyword_matches": q.keyword_matches,
                "text_length": q.text_length,
            })
        if synth_timing:
            all_timings.append({
                "agent_name": synth_timing.agent_name,
                "phase": synth_timing.phase,
                "ttft_ms": round(synth_timing.ttft_ms, 1),
                "e2e_ms": round(synth_timing.e2e_ms, 1),
                "output_tokens": synth_timing.output_tokens,
                "cache_state": synth_timing.cache_state,
                "query": synth_timing.query,
                "error": synth_timing.error,
            })

    for t, q in repeat_results:
        all_timings.append({
            "agent_name": t.agent_name, "phase": t.phase,
            "ttft_ms": round(t.ttft_ms, 1), "e2e_ms": round(t.e2e_ms, 1),
            "output_tokens": t.output_tokens, "cache_state": t.cache_state,
            "query": t.query, "error": t.error,
        })
        all_quality.append({
            "agent_name": q.agent_name, "phase": q.phase,
            "passed": q.passed, "non_empty": q.non_empty,
            "sufficient_length": q.sufficient_length,
            "no_repetition": q.no_repetition,
            "keyword_relevance": q.keyword_relevance,
            "keyword_matches": q.keyword_matches,
            "text_length": q.text_length,
        })

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_safe = _sanitize_model_id(model_id)
    output_path = RESULTS_DIR / f"wiki_routing_{model_safe}.json"

    results = BenchmarkResults(
        model_id=model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(),
        server_port=args.port,
        expert_count=len(EXPERT_AGENTS),
        query_count=len(QUERIES),
        timings=all_timings,
        quality=all_quality,
        agent_cache_state=cache_final,
        summary=summary,
    )

    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2, default=str)

    print(f"Results saved to: {output_path}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wikipedia Multi-Agent Routing Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Use cached articles only (don't download from Wikipedia)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print response previews and debug info",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
