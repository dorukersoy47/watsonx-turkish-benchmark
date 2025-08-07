# Token Usage Estimation for Turkish Benchmark Test

This document provides a detailed estimation of token usage for the Turkish LLM benchmarking project.

## Dataset Configuration

- **2 datasets**: MLSUM and XLSum Turkish
- **250 examples each** (as configured in `split="test[:250]"`)
- **4 models** to test:
  - `ibm/granite-3-8b-instruct`
  - `meta-llama/llama-3-3-70b-instruct`
  - `meta-llama/llama-3-2-90b-vision-instruct`
  - `meta-llama/llama-3-2-11b-vision-instruct`
- **Total generations**: 4 models × 2 datasets × 250 examples = **2,000 summaries**

## Token Analysis per Summary

### Input Tokens (per summary)
- **Turkish news articles**: Typically 500-2000 words
- **Prompt overhead**: ~50 tokens for Turkish prompt template
- **Estimated average input**: ~1,200 tokens per article
- **Total input tokens**: 2,000 × 1,200 = **~2.4M input tokens**

### Output Tokens (per summary)
- **MAX_NEW_TOKENS**: 150 (configured limit)
- **MIN_NEW_TOKENS**: 30 (configured minimum)
- **Expected average output**: ~80-100 tokens (for 25-30 word summaries)
- **Total output tokens**: 2,000 × 90 = **~180K output tokens**

## Total Token Estimation

| Token Type | Count | Usage Pattern |
|------------|-------|---------------|
| **Input Tokens** | ~2.4M | Reading articles + prompts |
| **Output Tokens** | ~180K | Generated summaries |
| **Total Tokens** | **~2.58M** | Complete benchmark |

## Breakdown by Model

Each model will process:
- **Input**: ~600K tokens (500 articles × 1,200 tokens)
- **Output**: ~45K tokens (500 summaries × 90 tokens)
- **Per model total**: ~645K tokens

## Cost Considerations

### Model Cost Hierarchy (Typical)
1. **Granite-3-8b**: Lowest cost (smaller model)
2. **Llama-3-2-11b**: Medium cost
3. **Llama-3-3-70b**: High cost (larger model)
4. **Llama-3-2-90b**: Highest cost (largest model)

### Token Pricing Pattern
- **Input tokens**: Lower cost rate
- **Output tokens**: 3-5x higher cost than input tokens
- **Total cost**: Primarily driven by the 90b model

## Time Estimation

- **Delay overhead**: 2,000 × 0.1s = 3.3 minutes
- **API response time**: ~1-3 seconds per call = 33-100 minutes
- **Total estimated time**: **35-105 minutes**

## Recommendations for Cost Optimization

### 1. Start with Smaller Test
```python
# Reduce to 50 examples per dataset
mlsum_dataset = load_dataset("mlsum", "tu", split="test[:50]")
xlsum_dataset = load_dataset("csebuetnlp/xlsum", "turkish", split="test[:50]")
```
**Result**: ~130K total tokens (95% cost reduction)

### 2. Test One Model First
```python
models = {
    "granite-3-8b": ModelInference(...)  # Start with cheapest model
    # Comment out others initially
}
```
**Result**: ~645K total tokens (75% cost reduction)

### 3. Reduce Output Length
```python
params = {
    GenParams.MAX_NEW_TOKENS: 100,  # Reduce from 150
    GenParams.MIN_NEW_TOKENS: 20,   # Reduce from 30
}
```
**Result**: ~40% reduction in output token costs

### 4. Progressive Testing Strategy

#### Phase 1: Quick Test (5 minutes, low cost)
- 1 model (Granite-3-8b)
- 10 examples per dataset
- **Tokens**: ~26K total

#### Phase 2: Model Comparison (20 minutes, medium cost)
- All 4 models
- 50 examples per dataset
- **Tokens**: ~520K total

#### Phase 3: Full Benchmark (90 minutes, full cost)
- All 4 models
- 250 examples per dataset
- **Tokens**: ~2.6M total

## Token Usage by Component

| Component | Input Tokens | Output Tokens | Total |
|-----------|-------------|---------------|-------|
| MLSUM Dataset | 1.2M | 90K | 1.29M |
| XLSum Dataset | 1.2M | 90K | 1.29M |
| **Grand Total** | **2.4M** | **180K** | **2.58M** |

## Monitoring and Optimization

### During Execution
- Monitor API response times
- Track failed generations (retry overhead)
- Watch for rate limiting

### After Execution
- Analyze actual vs. estimated token usage
- Compare model performance vs. cost
- Optimize parameters for future runs

## Emergency Stop Strategy

If costs become concerning during execution:
1. **Ctrl+C** to stop the process
2. Results are saved incrementally
3. Partial results can still be analyzed
4. Resume with adjusted parameters

---

**Note**: These are estimates based on typical Turkish news article lengths. Actual usage may vary by ±20% depending on article complexity and model behavior.