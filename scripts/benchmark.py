import time
import statistics
import requests

RENDER_URL = "https://meridian-bank-rag-assistant-1.onrender.com/chat"  

test_query = {"message": "How long does a replacement card take to arrive?"}

print("=== STARTING PRODUCTION BENCHMARKS ===")

# 1. Cold Start Test
print("Measuring Cold Start Latency...")
start_cold = time.time()
try:
    res_cold = requests.post(RENDER_URL, json=test_query, timeout=60)
    cold_latency = (time.time() - start_cold) * 1000
    print(f"Cold Start Response Time: {int(cold_latency)} ms (Status: {res_cold.status_code})")
except Exception as e:
    print(f"Cold start failed: {e}")

# 2. Warm Load Test (50 Requests)
print("\nExecuting 50 Sequential Requests for Warm Metrics...")
latencies = []
errors = 0

for i in range(1, 51):
    start_req = time.time()
    try:
        res = requests.post(RENDER_URL, json=test_query, timeout=30)
        if res.status_code == 200:
            latencies.append((time.time() - start_req) * 1000)
        else:
            errors += 1
            if errors <= 3:  
                print(f"Request {i} failed with Status {res.status_code}: {res.text[:100]}")
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"Request {i} exception: {e}")
    
    time.sleep(2)
    
if latencies:
    latencies.sort()
    p50 = statistics.median(latencies)
    p95_idx = int(len(latencies) * 0.95) - 1
    p95 = latencies[p95_idx]
    avg_lat = statistics.mean(latencies)
    error_rate = (errors / 50) * 100

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Total Requests:     50")
    print(f"Successful Calls:   {len(latencies)}")
    print(f"Failed Calls:       {errors}")
    print(f"Error Rate:         {error_rate:.2f}%")
    print(f"p50 Latency:        {int(p50)} ms")
    print(f"p95 Latency:        {int(p95)} ms")
    print(f"Mean Latency:       {int(avg_lat)} ms")
else:
    print("\nAll requests failed. Please check the endpoint URL and server logs.")
