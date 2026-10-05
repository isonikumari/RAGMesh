import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter


_rate_limiter = InMemoryRateLimiter(
    requests_per_second=float(os.getenv("GEMINI_REQUESTS_PER_SECOND", "0.5")),
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def get_mistral_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Create a Gemini client with retries for temporary rate limits."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "5")),
        rate_limiter=_rate_limiter,
    )


def is_rate_limit_error(error: Exception) -> bool:
    return "429" in str(error) or "rate limit" in str(error).lower() or "rate_limited" in str(error).lower()
