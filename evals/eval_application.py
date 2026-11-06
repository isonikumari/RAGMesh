import json
import os

from dotenv import load_dotenv

from evals._pipeline import ROOT_DIR, load_eval_retriever, run_question

load_dotenv(ROOT_DIR / ".env")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, SingleTurnParams


GOLDEN_PATH = ROOT_DIR / "golden" / "application_goldens.json"
THRESHOLD = 0.7
THROTTLE_SECONDS = 30


def make_metric(name, steps, rubric):
    return GEval(
        name=name,
        evaluation_steps=steps,
        rubric=rubric,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=GeminiModel(
            model=os.getenv("DEEPEVAL_JUDGE_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")),
            api_key=os.getenv("GOOGLE_API_KEY"),
        ),
        strict_mode=False,
        async_mode=False,
    )


def run():
    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{GOLDEN_PATH} must contain a non-empty JSON list.")

    retriever = load_eval_retriever()
    test_cases = []
    for index, golden in enumerate(goldens):
        query = golden.get("query") if isinstance(golden, dict) else None
        expected = golden.get("ideal_answer") if isinstance(golden, dict) else None
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Golden #{index + 1} needs a non-empty 'query'.")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"Golden #{index + 1} needs a non-empty 'ideal_answer'.")
        answer, _ = run_question(retriever, query)
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected,
            )
        )

    metrics = [
        make_metric(
            "Correctness",
            [
                "Compare factual claims in the answer with the expected answer.",
                "Judge truth, not completeness. Do not penalize brevity or omitted details.",
                "Penalize factual contradictions or unsupported factual claims.",
            ],
            [
                Rubric(score_range=(9, 10), expected_outcome="All stated claims are correct; brevity is acceptable."),
                Rubric(score_range=(5, 8), expected_outcome="Mostly correct with a minor factual issue."),
                Rubric(score_range=(0, 4), expected_outcome="Contains a clear factual error or contradiction."),
            ],
        ),
        make_metric(
            "Completeness",
            [
                "Identify the key facts in the expected answer.",
                "Judge how many key facts the answer covers, without judging factual accuracy.",
                "Do not penalize concise wording or extra correct information.",
            ],
            [
                Rubric(score_range=(9, 10), expected_outcome="Covers essentially all key expected facts."),
                Rubric(score_range=(5, 8), expected_outcome="Covers the main facts but misses an important detail."),
                Rubric(score_range=(0, 4), expected_outcome="Misses most of the expected facts."),
            ],
        ),
        make_metric(
            "Style",
            [
                "Judge only whether the answer is clear, concise, and conversational.",
                "Do not judge factual correctness or completeness.",
            ],
            [
                Rubric(score_range=(9, 10), expected_outcome="Clear, concise, natural, and easy to understand."),
                Rubric(score_range=(5, 8), expected_outcome="Understandable but somewhat awkward or unnecessarily verbose."),
                Rubric(score_range=(0, 4), expected_outcome="Hard to understand, confusing, or excessively verbose."),
            ],
        ),
    ]
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={"generator": os.getenv("GEMINI_MODEL", "gemini-3.6-flash")},
        async_config=AsyncConfig(throttle_value=THROTTLE_SECONDS, max_concurrent=1),
    )


if __name__ == "__main__":
    run()
