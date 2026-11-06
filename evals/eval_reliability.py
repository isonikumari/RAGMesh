import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.mistral import is_rate_limit_error
from evals._pipeline import ROOT_DIR, load_eval_retriever, run_question


GOLDEN_PATH = ROOT_DIR / "golden" / "retriever_goldens.json"
QUESTION_COUNT = 3
MAX_RETRIES = 1
BACKOFF_SECONDS = 1


def benchmark():
    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{GOLDEN_PATH} must contain a non-empty JSON list.")

    retriever = load_eval_retriever()
    calls = successes = failures = retries = first_attempt_successes = 0

    for golden in goldens[:QUESTION_COUNT]:
        calls += 1
        for attempt in range(MAX_RETRIES + 1):
            try:
                run_question(retriever, golden["query"])
                successes += 1
                if attempt == 0:
                    first_attempt_successes += 1
                break
            except Exception as exc:
                if is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                    failures += 1
                    print(f"Request failed: {golden['query']}: {exc}")
                    break
                retries += 1
                time.sleep(BACKOFF_SECONDS * (2**attempt))

    return calls, successes, failures, retries, first_attempt_successes


def main():
    calls, successes, failures, retries, first_attempt_successes = benchmark()
    print(f"Requests: {calls}; successful: {successes}; failed: {failures}; retries: {retries}")
    print(f"Success rate: {successes / calls:.1%}")
    print(f"First-attempt success rate: {first_attempt_successes / calls:.1%}")
    print(f"Retry rate: {retries / calls:.1%}")


if __name__ == "__main__":
    main()
