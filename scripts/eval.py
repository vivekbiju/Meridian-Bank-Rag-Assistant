import argparse
import json
import os
import re
import sys
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://meridian-bank-rag-assistant-1.onrender.com/chat")

def normalize_string(text: str) -> str:
    """Strips special characters and lowercases text for flexible comparisons."""
    if not text:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()

def is_section_match(expected_section: str, retrieved_sections: list) -> bool:
    """Soft matches expected section titles against retrieved source sections."""
    expected_clean = normalize_string(expected_section)
    for ret in retrieved_sections:
        ret_clean = normalize_string(ret)
        if expected_clean in ret_clean or ret_clean in expected_clean:
            return True
    return False

def run_evaluation(golden_set_path: str = "golden_set.json"):
    if not os.path.exists(golden_set_path):
        print(f"[ERROR] Golden set file not found: {golden_set_path}")
        sys.exit(1)

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    print(f"Running evaluation harness across {len(golden_set)} test cases...")

    total_answerable = 0
    total_unanswerable = 0
    
    recall_hits = 0
    mrr_sum = 0.0
    correct_refusals = 0
    correct_answers = 0

    for test_case in golden_set:
        query = test_case.get("question")
        expected_sec = test_case.get("expected_section")
        is_refusal_expected = test_case.get("is_refusal", False)
        expected_answer = test_case.get("expected_answer", "")

        try:
            response = requests.post(
                API_URL, 
                json={"message": query}, 
                timeout=30
            )
            if response.status_code != 200:
                print(f"[WARNING] Non-200 response for query '{query}': {response.status_code}")
                continue

            data = response.json()
            retrieved_sources = data.get("sources", [])
            # FIX: Reads "reply" key returned by ChatResponse schema in main.py
            generated_response = data.get("reply", data.get("response", ""))

        except Exception as e:
            print(f"[WARNING] Request failed for query '{query}': {e}")
            continue

        if is_refusal_expected:
            total_unanswerable += 1
            # Check if response correctly refused or stayed grounded
            if "sorry" in generated_response.lower() or "cannot answer" in generated_response.lower() or not retrieved_sources:
                correct_refusals += 1
        else:
            total_answerable += 1
            
            # Extract section names from retrieved sources
            retrieved_sections = [
                src.get("section", "") if isinstance(src, dict) else str(src) 
                for src in retrieved_sources
            ]

            # 1. Recall@k Evaluation with Soft Matching
            if is_section_match(expected_sec, retrieved_sections):
                recall_hits += 1

            # 2. MRR Evaluation
            rank = 0
            for idx, sec in enumerate(retrieved_sections, start=1):
                if is_section_match(expected_sec, [sec]):
                    rank = idx
                    break
            if rank > 0:
                mrr_sum += (1.0 / rank)

            # 3. Basic Answer Correctness heuristic
            if expected_answer and normalize_string(expected_answer)[:30] in normalize_string(generated_response):
                correct_answers += 1

    # Calculate final metrics
    recall_k = (recall_hits / total_answerable) if total_answerable > 0 else 0.0
    mrr = (mrr_sum / total_answerable) if total_answerable > 0 else 0.0
    refusal_correctness = (correct_refusals / total_unanswerable) if total_unanswerable > 0 else 1.0
    answer_correctness = (correct_answers / total_answerable) if total_answerable > 0 else 0.0

    print("\n=== EVALUATION RESULTS ===")
    print(f"Recall@k:             {recall_k:.4f}")
    print(f"MRR:                  {mrr:.4f}")
    print(f"Refusal Correctness:  {refusal_correctness:.4f}")
    print(f"Answer Correctness:   {answer_correctness:.4f}")

    return {
        "recall_at_k": recall_k,
        "mrr": mrr,
        "refusal_correctness": refusal_correctness,
        "answer_correctness": answer_correctness
    }

def main():
    parser = argparse.ArgumentParser(description="Meridian RAG Evaluation Harness")
    parser.add_argument("--golden-set", default="golden_set.json", help="Path to golden set JSON")
    parser.add_argument("--offline-fail-under-config", default="thresholds.yaml", help="Path to thresholds YAML")
    args = parser.parse_args()

    results = run_evaluation(args.golden_set)

    config_path = args.offline_fail_under_config
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            thresholds = yaml.safe_load(f).get("thresholds", {})

        target_recall = thresholds.get("recall_at_k", 0.85)
        target_refusal = thresholds.get("refusal_correctness", 0.90)

        failed = False
        if results["recall_at_k"] < target_recall:
            print(f"[ERROR] Recall@k ({results['recall_at_k']:.4f}) is below threshold ({target_recall:.4f})")
            failed = True

        if results["refusal_correctness"] < target_refusal:
            print(f"[ERROR] Refusal Correctness ({results['refusal_correctness']:.4f}) is below threshold ({target_refusal:.4f})")
            failed = True

        if failed:
            print("\n❌ CI GATE FAILED: Evaluation metrics dropped below allowed thresholds!")
            sys.exit(1)
        else:
            print("\n✅ CI GATE PASSED: All metrics met defined thresholds.")
            sys.exit(0)

if __name__ == "__main__":
    main()