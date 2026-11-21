# Few-shot latency (qwen2.5:7b, T=0.0, seed=42, n=10)

| k | prompt chars | cold wall ms | warm median wall ms | cold prefill ms | warm median prefill ms | warm median decode ms |
|---|---|---|---|---|---|---|
| 0 | 2830 | 667 | 615 | 107 | 89 | 516 |
| 2 | 3234 | 653 | 552 | 161 | 90 | 456 |
| 4 | 3681 | 514 | 492 | 162 | 89 | 372 |
| 8 | 4528 | 567 | 546 | 207 | 89 | 427 |
