"""Common Gemini backend with rate limiting, retry, and cost tracking."""

import asyncio
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import (
    GEMINI_CONCURRENT_LIMIT,
    GEMINI_EMBEDDING_DIMS,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_HTTP_TIMEOUT_MS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RPM_LIMIT,
    GEMINI_USE_VERTEX_AI,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
)

load_dotenv()


@dataclass
class UsageStats:
    """Track token usage and estimated costs."""

    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    requests: int = 0
    errors: int = 0
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
            f"Input: {self.input_tokens:,} | Output: {self.output_tokens:,} | "
            f"Embedding: {self.embedding_tokens:,} | "
            f"Cost: ${self.estimated_cost:.4f} | "
            f"Time: {self.elapsed_seconds:.1f}s"
        )


class GeminiClient:
    """Async Gemini client with rate limiting, retry, and cost tracking."""

    def __init__(self, model: str = GEMINI_MODEL):
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
                api_version="v1",
                timeout=GEMINI_HTTP_TIMEOUT_MS,
            )
            self.client = genai.Client(api_key=api_key, http_options=self._http_options)
            self._auth_mode = "api_key"

        self.model = model
        self.embedding_model = GEMINI_EMBEDDING_MODEL
        self.stats = UsageStats()

        # Rate limiting: token bucket for RPM
        self._semaphore = asyncio.Semaphore(GEMINI_CONCURRENT_LIMIT)
        self._rpm_interval = 60.0 / GEMINI_RPM_LIMIT
        self._last_request_time = 0.0
        self._rate_lock = asyncio.Lock()

    async def _wait_for_rate_limit(self):
        """Enforce rate limiting between requests."""
        async with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._rpm_interval:
                await asyncio.sleep(self._rpm_interval - elapsed)
            self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(GEMINI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
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
        async with self._semaphore:
            await self._wait_for_rate_limit()

            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                )
                if system_instruction:
                    config.system_instruction = system_instruction
                if response_mime_type:
                    config.response_mime_type = response_mime_type

                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                # Track usage
                self.stats.requests += 1
                if response.usage_metadata:
                    self.stats.input_tokens += response.usage_metadata.prompt_token_count or 0
                    self.stats.output_tokens += response.usage_metadata.candidates_token_count or 0

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
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type((ClientError, Exception)),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        async with self._semaphore:
            await self._wait_for_rate_limit()

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

                return [e.values for e in response.embeddings]

            except Exception as e:
                self.stats.errors += 1
                raise

    def print_stats(self):
        """Print current usage statistics."""
        mode = getattr(self, "_auth_mode", "unknown")
        print(f"\nGemini backend: {mode} | Usage: {self.stats}")
