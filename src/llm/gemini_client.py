"""Common Gemini backend with rate limiting, retry, and cost tracking."""

import asyncio
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import (
    GEMINI_CONCURRENT_LIMIT,
    GEMINI_EMBEDDING_DIMS,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RPM_LIMIT,
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
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment. Set it in .env")

        self.client = genai.Client(api_key=api_key)
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
        stop=stop_after_attempt(GEMINI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
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
        print(f"\nGemini Usage: {self.stats}")
