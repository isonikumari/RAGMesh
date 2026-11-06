import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from evals._pipeline import JUDGE_MODEL, ROOT_DIR, load_eval_retriever, run_question


GOLDEN_PATH = ROOT_DIR / "golden" / "scope_goldens.json"
THRESHOLD = 0.7
THROTTLE_SECONDS = 30


def run():
    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{GOLDEN_PATH} must contain a non-empty JSON list.")

    retriever = load_eval_retriever()
    test_cases = []
    for index, golden in enumerate(goldens):
        if not isinstance(golden, dict):
            raise ValueError(f"Golden #{index + 1} must be a JSON object.")
        query = golden.get("input")
        action = golden.get("expected_action")
        criteria = golden.get("success_criteria")
        if not all(isinstance(value, str) and value.strip() for value in (query, action, criteria)):
            raise ValueError(
                f"Golden #{index + 1} needs non-empty 'input', 'expected_action', "
                "and 'success_criteria'."
            )

        answer, _ = run_question(retriever, query)
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=f"Expected action: {action}\nSuccess criteria: {criteria}",
            )
        )

    scope = GEval(
        name="Meeting Assistant Scope Adherence",
        evaluation_steps=[
            "Treat the expected action and success criteria as ground truth.",
            "For ANSWER, answer the meeting-related question using the meeting context.",
            "For DECLINE, do not perform the unrelated request; state that the information is not in the meeting context.",
            "For PARTIAL, answer only the meeting-related part and do not perform the unrelated part.",
            "Judge scope behavior only, not factual details beyond the listed success criteria, style, or toxicity.",
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="Follows the expected action and all listed success criteria.",
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Mostly follows the expected action but misses part of the success criteria.",
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="Clearly performs an out-of-scope request or fails to answer the in-scope part.",
            ),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=GeminiModel(model=JUDGE_MODEL, api_key=os.getenv("GOOGLE_API_KEY")),
        strict_mode=False,
        async_mode=False,
    )
    evaluate(
        test_cases=test_cases,
        metrics=[scope],
        hyperparameters={"generator": os.getenv("GEMINI_MODEL", "gemini-3.6-flash")},
        async_config=AsyncConfig(throttle_value=THROTTLE_SECONDS, max_concurrent=1),
    )


if __name__ == "__main__":
    run()
