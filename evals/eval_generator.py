
"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR.

Faithfulness: Checks whether generated answers are supported
by the given context.

Answer Relevancy: Checks whether generated answers are relevant
to the user's questions.

ISOLATION: Uses GOLDEN context instead of retriever output.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import GeminiModel

from core.rag_engine import generate_answer


GOLDEN_PATH = ROOT_DIR / "golden" / "generator_goldens.json"
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
THRESHOLD = 0.7
THROTTLE_SECONDS = 30
MAX_CASES = int(os.getenv("EVAL_MAX_CASES", "0"))


def validate_judge_model(model: str) -> None:
    placeholders = {
        "your-available-model-id",
        "paste-a-listed-model-id-here",
    }
    if model.strip().lower() in placeholders:
        raise ValueError(
            "DEEPEVAL_JUDGE_MODEL is still set to an example placeholder. "
            "Remove it with 'Remove-Item Env:DEEPEVAL_JUDGE_MODEL' to use "
            "GEMINI_MODEL from .env, or set it to a model ID supported by your API key."
        )


def run():
    validate_judge_model(JUDGE_MODEL)
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY in .env to run the evaluation.")
    if MAX_CASES < 0:
        raise ValueError("EVAL_MAX_CASES must be zero or a positive integer.")

    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{GOLDEN_PATH} must contain a non-empty JSON list.")
    if MAX_CASES:
        goldens = goldens[:MAX_CASES]
    print(f"Generator evaluation cases: {len(goldens)}")

    test_cases = []

    for index, g in enumerate(goldens):
        query = g["query"]
        context = g["ideal_context"]

        answer = generate_answer(query, context)
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=context,
            )
        )
        if index < len(goldens) - 1:
            time.sleep(THROTTLE_SECONDS)

    judge = GeminiModel(
        model=JUDGE_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
            async_mode=False,
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            throttle_value=THROTTLE_SECONDS,
            max_concurrent=1,
        ),
    )

    return result


if __name__ == "__main__":
    run()
