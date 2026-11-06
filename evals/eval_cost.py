import json
import os
import time

from dotenv import load_dotenv

from evals._pipeline import ROOT_DIR, load_eval_retriever

load_dotenv(ROOT_DIR / ".env")

from core.rag_engine import build_answer_prompt, get_llm


GOLDEN_PATH = ROOT_DIR / "golden" / "application_goldens.json"
THROTTLE_SECONDS = 30


def optional_price(name: str) -> float | None:
    value = os.getenv(name)
    if value is None:
        return None
    price = float(value)
    if price < 0:
        raise ValueError(f"{name} must not be negative.")
    return price


def run():
    input_price = optional_price("GEMINI_INPUT_USD_PER_1M")
    output_price = optional_price("GEMINI_OUTPUT_USD_PER_1M")
    cached_input_price = optional_price("GEMINI_CACHED_INPUT_USD_PER_1M")

    goldens = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{GOLDEN_PATH} must contain a non-empty JSON list.")

    retriever = load_eval_retriever()
    measured_chain = build_answer_prompt() | get_llm()
    samples = []

    for index, golden in enumerate(goldens):
        documents = retriever.invoke(golden["query"])
        if not documents:
            raise RuntimeError(f"No documents retrieved for: {golden['query']}")
        response = measured_chain.invoke({
            "question": golden["query"],
            "context": "\n\n".join(document.page_content for document in documents),
        })
        usage = response.usage_metadata
        if not usage or not isinstance(usage.get("input_tokens"), int) or not isinstance(
            usage.get("output_tokens"), int
        ):
            raise RuntimeError("Gemini response did not provide input/output token usage.")

        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        details = usage.get("input_token_details") or {}
        cached_tokens = details.get("cache_read", 0) or 0
        cached_tokens = min(cached_tokens, input_tokens)
        sample = {
            "input": input_tokens,
            "output": output_tokens,
            "cached": cached_tokens,
        }

        if input_price is not None and output_price is not None:
            cached_price = (
                cached_input_price if cached_input_price is not None else input_price
            )
            sample["cost_usd"] = (
                (input_tokens - cached_tokens) * input_price
                + cached_tokens * cached_price
                + output_tokens * output_price
            ) / 1_000_000
        samples.append(sample)

        if index < len(goldens) - 1:
            time.sleep(THROTTLE_SECONDS)

    print(f"Generation samples: {len(samples)}")
    print(f"Average input tokens: {sum(item['input'] for item in samples) / len(samples):.1f}")
    print(f"Average output tokens: {sum(item['output'] for item in samples) / len(samples):.1f}")
    print(f"Average cached input tokens: {sum(item['cached'] for item in samples) / len(samples):.1f}")
    if input_price is None or output_price is None:
        print(
            "Cost estimate skipped. Set GEMINI_INPUT_USD_PER_1M and "
            "GEMINI_OUTPUT_USD_PER_1M in .env to your model's current prices."
        )
    else:
        average_cost = sum(item["cost_usd"] for item in samples) / len(samples)
        print(f"Estimated generation cost per query: ${average_cost:.8f}")
        if cached_input_price is None:
            print("Cached tokens were conservatively priced at the regular input rate.")
    print("Estimate excludes embedding/retrieval, judging, and other provider charges.")


if __name__ == "__main__":
    run()
