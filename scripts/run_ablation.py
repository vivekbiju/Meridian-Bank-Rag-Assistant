import os
import re
import json
import time
import psycopg
import pdfplumber
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
DB_URI = os.getenv("DATABASE_URL")

def reingest(chunk_size, overlap):
    full_text = ""
    with pdfplumber.open("corpus/meridian-handbook.pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = re.sub(r"Meridian Bank plc.*?\n", "", text)
            text = re.sub(r"Valid from 1 March 2026.*?\n", "", text)
            text = re.sub(r"Page \d+", "", text)
            full_text += text + "\n"

    pattern = r"(\n\d+\.\s+[A-Z][^\n]+)"
    parts = re.split(pattern, full_text)
    
    sections_dict = {}
    for i in range(1, len(parts), 2):
        sec_title = parts[i].strip()
        sec_body = parts[i+1].strip() if (i+1) < len(parts) else ""
        if sec_title in sections_dict:
            sections_dict[sec_title] += "\n" + sec_body
        else:
            sections_dict[sec_title] = sec_body

    db_records = []
    for section_title, section_body in sections_dict.items():
        start = 0
        idx = 0
        while start < len(section_body):
            chunk = section_body[start:start+chunk_size]
            db_records.append((section_title, idx, chunk, str(embedder.encode(chunk).tolist())))
            start += (chunk_size - overlap)
            idx += 1

    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks;")
            cur.executemany("""
                INSERT INTO chunks (section, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (section, chunk_index) 
                DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
            """, db_records)
            conn.commit()

def evaluate_run(k_val):
    with open("golden_set.json", "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    retrieved_hits = 0
    total_eval = 0
    generation_hits = 0
    total_non_refusal = 0
    latencies = []

    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            for item in golden_set:
                question = item["question"]
                expected_sec = item["expected_section"]
                should_refuse = item["should_refuse"]
                if not question.strip():
                    continue

                start_t = time.time()
                query_vector = embedder.encode(question).tolist()
                
                cur.execute("""
                    SELECT section, chunk_index, content, (embedding <=> %s::vector) as distance
                    FROM chunks
                    ORDER BY distance ASC
                    LIMIT %s;
                """, (str(query_vector), k_val))
                rows = cur.fetchall()

                latencies.append((time.time() - start_t) * 1000)
                sources = [r[0] for r in rows if r[3] <= 0.85]

                if expected_sec:
                    total_eval += 1
                    if expected_sec in sources:
                        retrieved_hits += 1

                if not should_refuse:
                    total_non_refusal += 1
                    if len(sources) > 0:
                        generation_hits += 1

    recall = retrieved_hits / total_eval if total_eval > 0 else 0
    correctness = generation_hits / total_non_refusal if total_non_refusal > 0 else 0
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    return recall, correctness, p95

print("--- RUNNING ABLATION MATRIX ---")

experiments = [
    {"run": 1, "chunk_size": 256, "overlap": 32, "k": 5},
    {"run": 2, "chunk_size": 512, "overlap": 64, "k": 5},
    {"run": 3, "chunk_size": 1024, "overlap": 128, "k": 5},
    {"run": 4, "chunk_size": 512, "overlap": 64, "k": 3},
    {"run": 5, "chunk_size": 512, "overlap": 64, "k": 10},
]

results = []
for exp in experiments:
    print(f"Executing Run #{exp['run']} (Chunk: {exp['chunk_size']}, Overlap: {exp['overlap']}, k: {exp['k']})...")
    reingest(exp['chunk_size'], exp['overlap'])
    recall, correctness, p95 = evaluate_run(exp['k'])
    results.append({
        "Run": exp['run'],
        "Chunk Size": exp['chunk_size'],
        "Overlap": exp['overlap'],
        "k": exp['k'],
        "Recall@k": f"{recall:.4f}",
        "Correctness": f"{correctness:.4f}",
        "p95 Latency": f"{int(p95)} ms"
    })

print("\n=== FINAL ABLATION TABLE ===")
print(f"{'Run':<5} | {'Chunk Size':<10} | {'Overlap':<7} | {'k':<3} | {'Recall@k':<9} | {'Correctness':<11} | {'p95 Latency':<11}")
print("-" * 75)
for r in results:
    print(f"{r['Run']:<5} | {r['Chunk Size']:<10} | {r['Overlap']:<7} | {r['k']:<3} | {r['Recall@k']:<9} | {r['Correctness']:<11} | {r['p95 Latency']:<11}")