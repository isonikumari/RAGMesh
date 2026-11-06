import json
import time
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from evals._pipeline import ROOT_DIR, load_eval_retriever
from core.rag_engine import generate_answer


GOLDEN_PATH = ROOT_DIR / "golden" / "retriever_goldens.json"
QUESTION_COUNT = 3


def benchmark():
    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{GOLDEN_PATH} must contain a non-empty JSON list.")

    retriever = load_eval_retriever()
    totals, retrieval_times, generation_times = [], [], []
    for golden in goldens[:QUESTION_COUNT]:
        start = time.perf_counter()
        documents = retriever.invoke(golden["query"])
        retrieval_done = time.perf_counter()
        if not documents:
            raise RuntimeError(f"No documents retrieved for: {golden['query']}")
        generate_answer(golden["query"], [document.page_content for document in documents])
        finished = time.perf_counter()

        retrieval_times.append((retrieval_done - start) * 1000)
        generation_times.append((finished - retrieval_done) * 1000)
        totals.append((finished - start) * 1000)

    return totals, retrieval_times, generation_times


def report(label, values):
    print(
        f"{label:<12} n={len(values)} "
        f"mean={mean(values):.1f} ms median={median(values):.1f} ms "
        f"max={max(values):.1f} ms"
    )


def main():
    total, retrieval, generation = benchmark()
    print("Latency (single-user, milliseconds; no streaming/TTFT in this app)")
    report("end-to-end", total)
    report("retrieval", retrieval)
    report("generation", generation)


if __name__ == "__main__":
    main()
