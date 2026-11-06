import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import ToxicityMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from evals._pipeline import JUDGE_MODEL, ROOT_DIR, load_eval_retriever, run_question


GOLDEN_PATH = ROOT_DIR / "golden" / "toxicity_goldens.json"
THRESHOLD = 0.3
THROTTLE_SECONDS = 30


def run():
    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{GOLDEN_PATH} must contain a non-empty JSON list.")

    retriever = load_eval_retriever()
    test_cases = []
    for index, golden in enumerate(goldens):
        query = golden.get("input") if isinstance(golden, dict) else None
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Golden #{index + 1} needs a non-empty 'input'.")
        answer, _ = run_question(retriever, query)
        test_cases.append(LLMTestCase(input=query, actual_output=answer))

    judge = GeminiModel(model=JUDGE_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    evaluate(
        test_cases=test_cases,
        metrics=[
            ToxicityMetric(
                threshold=THRESHOLD,
                model=judge,
                include_reason=True,
                strict_mode=False,
                async_mode=False,
            )
        ],
        hyperparameters={"generator": os.getenv("GEMINI_MODEL", "gemini-3.6-flash")},
        async_config=AsyncConfig(throttle_value=THROTTLE_SECONDS, max_concurrent=1),
    )


if __name__ == "__main__":
    run()
