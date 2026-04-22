"""Common Gemini backend with rate limiting, retry, and cost tracking."""

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import (
    GEMINI_CONCURRENT_LIMIT,
    GEMINI_EMBED_MAX_RETRIES,
    GEMINI_EMBEDDING_DIMS,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_HTTP_TIMEOUT_MS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RPM_BURST_CAPACITY,
    GEMINI_RPM_LIMIT,
    GEMINI_USE_VERTEX_AI,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
)

load_dotenv()


def _tenacity_bump_retry(retry_state) -> None:
    """Increment retry counter on GeminiClient instances (best-effort)."""
    try:
        obj = retry_state.args[0]
        if hasattr(obj, "stats"):
            obj.stats.retries_after_error += 1
    except Exception:
        pass


@dataclass
class UsageStats:
    """Track token usage and estimated costs."""

    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    requests: int = 0
    errors: int = 0
    retries_after_error: int = 0  # incremented by tenacity before_sleep (best-effort)
    _start_time: float = field(default_factory=time.time)

    # Gemini 2.0 Flash pricing (per 1M tokens)
    INPUT_COST_PER_M = 0.10
    OUTPUT_COST_PER_M = 0.40
    EMBEDDING_COST_PER_M = 0.006

    @property
    def estimated_cost(self) -> float:
        return (
            self.input_tokens * self.INPUT_COST_PER_M / 1_000_000
            + self.output_tokens * self.OUTPUT_COST_PER_M / 1_000_000
            + self.embedding_tokens * self.EMBEDDING_COST_PER_M / 1_000_000
        )

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    def __str__(self) -> str:
        return (
            f"Requests: {self.requests} | Errors: {self.errors} | "
            f"Retries(log): {self.retries_after_error} | "
            f"Input: {self.input_tokens:,} | Output: {self.output_tokens:,} | "
            f"Embedding: {self.embedding_tokens:,} | "
            f"Cost: ${self.estimated_cost:.4f} | "
            f"Time: {self.elapsed_seconds:.1f}s"
        )


def _latency_summary(samples: deque[float]) -> str:
    if not samples:
        return "n=0"
    arr = sorted(samples)
    n = len(arr)

    def pct(p: float) -> float:
        idx = int(p * (n - 1))
        return arr[idx]

    b1 = sum(1 for x in arr if x < 2.0)
    b2 = sum(1 for x in arr if 2.0 <= x < 10.0)
    b3 = sum(1 for x in arr if 10.0 <= x < 60.0)
    b4 = sum(1 for x in arr if x >= 60.0)
    return (
        f"n={n} p50={pct(0.5):.2f}s p95={pct(0.95):.2f}s "
        f"buckets[<2s,{b1}][2-10s,{b2}][10-60s,{b3}][>=60s,{b4}]"
    )


class GeminiClient:
    """Async Gemini client with rate limiting, retry, and cost tracking."""

    def __init__(
        self,
        model: str = GEMINI_MODEL,
        rpm_limit: int | None = None,
        concurrent_limit: int | None = None,
    ):
        # Vertex AI (GCP / trial credits): Application Default Credentials + project/location.
        # See https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview
        if GEMINI_USE_VERTEX_AI:
            project = GOOGLE_CLOUD_PROJECT
            if not project:
                raise ValueError(
                    "Vertex AI is enabled (GOOGLE_GENAI_USE_VERTEXAI or GEMINI_USE_VERTEX) but "
                    "GOOGLE_CLOUD_PROJECT is not set. Add it to .env and run: "
                    "gcloud auth application-default login"
                )
            location = GOOGLE_CLOUD_LOCATION or "us-central1"
            # Vertex AI uses "v1" as its stable path for both generate_content and
            # embed_content (gemini-embedding-001 is served there). The Developer
            # API path below needs "v1beta" for embed_content to resolve.
            self._http_options = types.HttpOptions(
                api_version="v1",
                timeout=GEMINI_HTTP_TIMEOUT_MS,
            )
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=self._http_options,
            )
            self._auth_mode = "vertex"
        else:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key found. Set GEMINI_API_KEY in .env for Gemini Developer API, "
                    "or enable Vertex with GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT."
                )
            self._http_options = types.HttpOptions(
                api_version="v1beta",
                timeout=GEMINI_HTTP_TIMEOUT_MS,
            )
            self.client = genai.Client(api_key=api_key, http_options=self._http_options)
            self._auth_mode = "api_key"

        self.model = model
        self.embedding_model = GEMINI_EMBEDDING_MODEL
        self.stats = UsageStats()
        # Recent wall-clock latencies (seconds) for successful SDK calls (bounded).
        self._latency_generate: deque[float] = deque(maxlen=5000)
        self._latency_embed: deque[float] = deque(maxlen=2000)

        # Concurrency: max in-flight SDK calls. RPM: token bucket (burst + sustained average).
        # Constructor params override the module-level defaults so chunk_quality.py
        # can use the higher Tier-1 limits without polluting main-grid clients.
        effective_concurrent = concurrent_limit or GEMINI_CONCURRENT_LIMIT
        effective_rpm = rpm_limit or GEMINI_RPM_LIMIT
        self._semaphore = asyncio.Semaphore(effective_concurrent)
        self._rpm_capacity = float(GEMINI_RPM_BURST_CAPACITY)
        self._rpm_tokens = self._rpm_capacity
        self._rpm_refill_per_sec = effective_rpm / 60.0
        self._rpm_last_refill = time.monotonic()
        self._rate_lock = asyncio.Lock()

    async def _acquire_rpm_slot(self) -> None:
        """Token-bucket limiter: allows short bursts while averaging GEMINI_RPM_LIMIT RPM."""
        while True:
            async with self._rate_lock:
                now = time.monotonic()
                dt = now - self._rpm_last_refill
                self._rpm_last_refill = now
                self._rpm_tokens = min(
                    self._rpm_capacity,
                    self._rpm_tokens + dt * self._rpm_refill_per_sec,
                )
                if self._rpm_tokens >= 1.0:
                    self._rpm_tokens -= 1.0
                    return
                deficit = 1.0 - self._rpm_tokens
                wait = deficit / self._rpm_refill_per_sec if self._rpm_refill_per_sec > 0 else 0.25
            await asyncio.sleep(wait)

    @retry(
        stop=stop_after_attempt(GEMINI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=_tenacity_bump_retry,
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        response_mime_type: str | None = None,
    ) -> str:
        """Generate text with Gemini, with rate limiting and retry."""
        t0 = time.monotonic()
        async with self._semaphore:
            await self._acquire_rpm_slot()

            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                )
                if system_instruction:
                    config.system_instruction = system_instruction
                if response_mime_type:
                    config.response_mime_type = response_mime_type

                # Native async path — avoids thread-pool saturation that caps
                # concurrent HTTP calls at the default ThreadPoolExecutor size.
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                # Track usage
                self.stats.requests += 1
                if response.usage_metadata:
                    self.stats.input_tokens += response.usage_metadata.prompt_token_count or 0
                    self.stats.output_tokens += response.usage_metadata.candidates_token_count or 0

                dt = time.monotonic() - t0
                self._latency_generate.append(dt)
                return response.text

            except Exception as e:
                self.stats.errors += 1
                raise

    async def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Generate JSON output from Gemini."""
        return await self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        )

    async def batch_generate(
        self,
        prompts: list[str],
        system_instruction: str | None = None,
        temperature: float = 0.7,
        response_mime_type: str | None = None,
    ) -> list[str]:
        """Generate responses for multiple prompts concurrently."""
        tasks = [
            self.generate(
                prompt=p,
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type=response_mime_type,
            )
            for p in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    @retry(
        stop=stop_after_attempt(GEMINI_EMBED_MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=2, max=45),
        retry=retry_if_exception_type((ClientError, OSError, TimeoutError)),
        before_sleep=_tenacity_bump_retry,
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        t0 = time.monotonic()
        async with self._semaphore:
            await self._acquire_rpm_slot()

            try:
                response = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=self.embedding_model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=GEMINI_EMBEDDING_DIMS,
                    ),
                )

                self.stats.requests += 1
                # Approximate token count for embeddings
                self.stats.embedding_tokens += sum(len(t.split()) for t in texts)

                dt = time.monotonic() - t0
                self._latency_embed.append(dt)
                return [e.values for e in response.embeddings]

            except Exception as e:
                self.stats.errors += 1
                raise

    def print_stats(self):
        """Print current usage statistics."""
        mode = getattr(self, "_auth_mode", "unknown")
        print(f"\nGemini backend: {mode} | Usage: {self.stats}")
        print(f"Latency generate: {_latency_summary(self._latency_generate)}")
        print(f"Latency embed: {_latency_summary(self._latency_embed)}")
