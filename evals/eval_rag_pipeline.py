import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from core.rag_engine import generate_answer, sanitize_meeting_context
from core.vector_store import ensure_vector_store, get_collection_name, get_retriever


ROOT_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_PATH = ROOT_DIR / "data" / "retriever_eval_transcript.txt"
GOLDEN_PATH = ROOT_DIR / "golden" / "retriever_goldens.json"
TOP_K = 10
THRESHOLD = 0.7
THROTTLE_SECONDS = 30
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))


def run():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY in .env to run the evaluation.")

    transcript = sanitize_meeting_context(TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{GOLDEN_PATH} must contain a non-empty JSON list.")

    vector_store = ensure_vector_store(transcript, get_collection_name(transcript))
    retriever = get_retriever(vector_store, k=TOP_K)
    test_cases = []

    for golden in goldens:
        documents = retriever.invoke(golden["query"])
        if not documents:
            raise RuntimeError(f"No documents retrieved for: {golden['query']}")
        context = [document.page_content for document in documents]
        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                actual_output=generate_answer(golden["query"], context),
                retrieval_context=context,
            )
        )

    judge = GeminiModel(model=JUDGE_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    evaluate(
        test_cases=test_cases,
        metrics=[
            ContextualRelevancyMetric(
                threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False
            ),
            FaithfulnessMetric(
                threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False
            ),
            AnswerRelevancyMetric(
                threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False
            ),
        ],
        hyperparameters={"retriever": "chroma", "top_k": TOP_K, "judge_model": JUDGE_MODEL},
        async_config=AsyncConfig(
            throttle_value=THROTTLE_SECONDS,
            max_concurrent=1,
        ),
    )


if __name__ == "__main__":
    run()
