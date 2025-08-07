# Turkish LLM Benchmark for Text Summarization

This project evaluates multiple Large Language Models (LLMs) on Turkish text summarization using ROUGE metrics. The benchmark tests models on two Turkish datasets: MLSUM and XLSum, comparing their summarization capabilities.

## Overview

The benchmark evaluates 4 different models:
- **IBM Granite-3-8B-Instruct**: IBM's instruction-tuned model
- **Meta Llama-3.3-70B-Instruct**: Large Meta model with strong reasoning
- **Meta Llama-3.2-90B-Vision-Instruct**: Multi-modal Meta model
- **Meta Llama-3.2-11B-Vision-Instruct**: Smaller multi-modal Meta model

## ROUGE Metrics Explained

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the quality of summaries by comparing them to reference summaries. We use three ROUGE variants:

### ROUGE-1 (Unigram Overlap)
- **What it measures**: Overlap of individual words between generated and reference summaries
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 0.0-0.2: Poor - Very few words match
  - 0.2-0.4: Fair - Some word overlap
  - 0.4-0.6: Good - Substantial word overlap
  - 0.6-0.8: Very Good - High word overlap
  - 0.8-1.0: Excellent - Nearly perfect word overlap

### ROUGE-2 (Bigram Overlap)
- **What it measures**: Overlap of two-word sequences between summaries
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Generally lower than ROUGE-1 since bigram matching is stricter
  - 0.0-0.1: Poor - Few phrase matches
  - 0.1-0.2: Fair - Some phrase similarity
  - 0.2-0.3: Good - Reasonable phrase overlap
  - 0.3-0.4: Very Good - Strong phrase overlap
  - 0.4+: Excellent - High phrase similarity

### ROUGE-L (Longest Common Subsequence)
- **What it measures**: Longest common subsequence between summaries (preserves word order)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Captures fluency and structure preservation
  - Similar ranges to ROUGE-1 but considers word order
  - Higher ROUGE-L with similar ROUGE-1 indicates better structure

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- IBM Watson AI API credentials

### Installation
1. **Create virtual environment**: 
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install dependencies**: 
   ```bash
   pip install datasets ibm-watsonx-ai python-dotenv rouge-score tqdm matplotlib
   ```

3. **Set up environment variables** (create `.env` file):
   ```
   API_KEY=your_ibm_watsonx_api_key
   PROJECT_ID=your_project_id
   REGION_URL=your_region_url
   ```

## Usage

### Quick Test (10 examples)
```bash
python tests/tr_benchmark_test.py
```

### Full Benchmark (1000 examples per dataset)
```bash
python tr_benchmark.py
```

### Generate Visualization
```bash
python plot_results.py rouge_evaluation_results_YYYYMMDD_HHMMSS.json
```

## Output

The benchmark generates:
1. **JSON results file**: Detailed scores and sample summaries
2. **Console output**: Real-time progress and summary table
3. **Visualization**: Line graph comparing model performance
4. **Success rates**: Percentage of successful summary generations

## Dataset Information

- **MLSUM Turkish**: 12,775 total test examples (news articles with summaries)
- **XLSum Turkish**: 3,397 total test examples (news articles with summaries)
- **Current benchmark**: Uses 1000 examples from each dataset for faster evaluation

## Interpreting Results

**Good ROUGE scores for Turkish summarization:**
- ROUGE-1: 0.35+ (indicates good content overlap)
- ROUGE-2: 0.15+ (indicates good phrase preservation)
- ROUGE-L: 0.30+ (indicates good structure preservation)

**Model comparison tips:**
- Higher ROUGE-1: Better content coverage
- Higher ROUGE-2: Better phrase accuracy
- Higher ROUGE-L: Better fluency and structure
- Success rate: Model reliability (aim for 95%+)

## File Structure

```
turkish-benchmark-test/
├── tr_benchmark.py          # Main benchmark script
├── plot_results.py          # Visualization generator
├── tests/
│   ├── tr_benchmark_test.py # Quick test with 10 examples
│   └── datasets_test.py     # Dataset loading test
├── env/                     # Virtual environment
├── .env                     # Environment variables (you create this)
└── README.md               # This file
```