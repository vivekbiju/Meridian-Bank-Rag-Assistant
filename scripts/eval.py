import os
import sys
import json
import time
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://meridian-bank-rag-assistant-1.onrender.com/chat")

def run_evaluation():
    if not os.path.exists("golden_set.json"):
        print("[ERROR] golden_set.json not found.")
        sys.exit(1)

    with open("golden_set.json", "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    total_cases = len(golden_set)
    print(f"Running evaluation harness across {total_cases} test cases...\n")

    retrieved_hits = 0
    total_evaluatable_retrieval = 0
    reciprocal_ranks = []
    correct_refusals = 0
    total_unanswerable = 0
    generation_hits = 0

    for item in golden_set:
        question = item["question"]
        expected_sec = item["expected_section"]
        should_refuse = item["should_refuse"]

        if not question.strip():
            # Skip empty payload API call locally
            continue

        try:
            res = requests.post(API_URL, json={"message": question}, timeout=30)
            data = res.json()
        except Exception as e:
            print(f"API Error on case {item.get('id', 'unknown')}: {e}")
            continue

        reply = data.get("reply", "")
        sources = data.get("sources", [])

        # 1. Retrieval Metrics
        if expected_sec:
            total_evaluatable_retrieval += 1
            retrieved_sections = [s["section"] for s in sources]
            if expected_sec in retrieved_sections:
                retrieved_hits += 1
                rank = retrieved_sections.index(expected_sec) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        # 2. Refusal / Generation Metrics
        if should_refuse:
            total_unanswerable += 1
            reply_lower = reply.lower()
            if (
                "does not contain information" in reply_lower 
                or "cannot" in reply_lower 
                or "do not have" in reply_lower 
                or data.get("retrieved_k") == 0
            ):
                correct_refusals += 1
        else:
            if reply and not ("does not contain information" in reply):
                generation_hits += 1

    recall_at_k = (retrieved_hits / total_evaluatable_retrieval) if total_evaluatable_retrieval > 0 else 0.0
    mrr = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0
    refusal_rate = (correct_refusals / total_unanswerable) if total_unanswerable > 0 else 0.0
    correctness = (generation_hits / (total_cases - total_unanswerable)) if (total_cases - total_unanswerable) > 0 else 0.0

    print("=== EVALUATION RESULTS ===")
    print(f"Recall@k:             {recall_at_k:.4f}")
    print(f"MRR:                  {mrr:.4f}")
    print(f"Refusal Correctness:  {refusal_rate:.4f}")
    print(f"Answer Correctness:   {correctness:.4f}")

    # Parse command line flags for threshold enforcement
    config_file = None
    args = sys.argv[1:]

    for i, arg in enumerate(args):
        if arg in ["--offline-fail-under-config", "-c"]:
            if i + 1 < len(args):
                config_file = args[i + 1]
        elif not arg.startswith("-") and config_file is None:
            config_file = arg

    # Enforce CI Gate Threshold Check if a config file is provided or requested
    if config_file or "--offline-fail-under-config" in sys.argv:
        target_config = config_file or "thresholds.yaml"

        if not os.path.exists(target_config):
            print(f"[ERROR] Config file '{target_config}' missing!")
            sys.exit(1)
            
        with open(target_config, "r") as cf:
            thresholds = yaml.safe_load(cf)

        min_recall = thresholds.get("retrieval", {}).get("recall_at_k", 0.80)
        min_refusal = thresholds.get("generation", {}).get("refusal_correctness", 0.85)

        failed = False

        if recall_at_k < min_recall:
            print(f"[ERROR] Recall@k ({recall_at_k:.4f}) is below threshold ({min_recall:.4f})")
            failed = True
        else:
            print(f"[SUCCESS] Recall@k ({recall_at_k:.4f}) meets threshold ({min_recall:.4f})")

        if refusal_rate < min_refusal:
            print(f"[ERROR] Refusal Correctness ({refusal_rate:.4f}) is below threshold ({min_refusal:.4f})")
            failed = True
        else:
            print(f"[SUCCESS] Refusal Correctness ({refusal_rate:.4f}) meets threshold ({min_refusal:.4f})")

        if failed:
            print("\n❌ CI GATE FAILED: Evaluation metrics dropped below allowed thresholds!")
            sys.exit(1)
        else:
            print("\n✅ CI GATE PASSED: All metrics met defined thresholds.")
            sys.exit(0)

if __name__ == "__main__":
    run_evaluation()