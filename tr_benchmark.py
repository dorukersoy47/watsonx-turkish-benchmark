import os
import json
import time
import random
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from datasets import load_dataset
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from rouge_score import rouge_scorer
from tqdm import tqdm
from plot_results import plot_rouge_results, print_summary_stats

# Support functions
def finish_sentence(text: str) -> str:
    last_dot = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    return text[: last_dot + 1] if last_dot != -1 else text

def create_prompt(text: str) -> str:
    return f"""Lütfen bu metni 25–30 kelime arasında, net bir cümleyle özetle. Özet açık, kısa ve ana noktaları içermelidir:\n {text} \n\n Özet:"""

def safe_generate_summary(model: ModelInference, text: str, max_retries: int = 3) -> str:
    """Generate summary with retry logic for API failures."""
    prompt = create_prompt(text)
    
    for attempt in range(max_retries):
        try:
            raw_summary = model.generate_text(prompt=prompt)
            return finish_sentence(raw_summary)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5 seconds before retry
    
    return f"ERROR: Failed to generate summary after {max_retries} attempts"

def evaluate_rouge_scores(generated_summaries: List[str], reference_summaries: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for a list of summaries."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for gen_sum, ref_sum in zip(generated_summaries, reference_summaries):
        if gen_sum.startswith("ERROR:"):
            continue  # Skip failed generations
            
        scores = scorer.score(ref_sum, gen_sum)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
        'num_successful': len(rouge1_scores)
    }

# Loading datasets
mlsum_dataset = load_dataset("mlsum", "tu", split="test[:250]")
xlsum_dataset = load_dataset("csebuetnlp/xlsum", "turkish", split="test[:250]")

# Configure Watsonx credentials
load_dotenv()
api_key = os.getenv("API_KEY")        
project_id = os.getenv("PROJECT_ID")  
service_url = os.getenv("REGION_URL") 

if not api_key:
    raise ValueError("API_KEY environment variable is required")
if not project_id:
    raise ValueError("PROJECT_ID environment variable is required")
if not service_url:
    raise ValueError("REGION_URL environment variable is required")

creds = Credentials(api_key=api_key, url=service_url)

# Optimized params for consistent summarization of full text
params = {
    GenParams.MAX_NEW_TOKENS: 150,              # Increased for longer summaries
    GenParams.MIN_NEW_TOKENS: 30,               # Ensure minimum summary length
    GenParams.TEMPERATURE: 0.1,                 # Very low randomness for consistency
    GenParams.STOP_SEQUENCES: ["\n\n", "###"],  # Stop on double newlines or separators
    # GenParams.TOP_P: 0.8,                     # Reduced for more focused responses
    # GenParams.TOP_K: 10,                      # Reduced for more deterministic choices
    # GenParams.REPETITION_PENALTY: 1.05,       # Slight penalty to avoid repetition
    # GenParams.RANDOM_SEED: 42,                # Fixed seed for reproducibility
    # GenParams.DECODING_METHOD: "greedy",      # Most deterministic method
    # GenParams.TRUNCATE_INPUT_TOKENS: 16000,   # Increased to handle full articles
}

# Define models to test in a dictionary
models = {
    "granite-3-8b": ModelInference(
        model_id="ibm/granite-3-8b-instruct",
        project_id=project_id,
        params=params,
        credentials=creds
    ),
    "llama-3-3-70b": ModelInference(
        model_id="meta-llama/llama-3-3-70b-instruct",
        project_id=project_id,
        params=params,
        credentials=creds
    ),
    "llama-3-2-90b": ModelInference(
        model_id="meta-llama/llama-3-2-90b-vision-instruct",
        project_id=project_id,
        params=params,
        credentials=creds
    ),
    "llama-3-2-11b": ModelInference(
        model_id="meta-llama/llama-3-2-11b-vision-instruct",
        project_id=project_id,
        params=params,
        credentials=creds
    )
}

# Prepare datasets for evaluation
datasets_to_evaluate = {
    "mlsum": {
        "texts": [str(example['text']) for example in mlsum_dataset],  # type: ignore
        "summaries": [str(example['summary']) for example in mlsum_dataset]  # type: ignore
    },
    "xlsum": {
        "texts": [str(example['text']) for example in xlsum_dataset],  # type: ignore
        "summaries": [str(example['summary']) for example in xlsum_dataset]  # type: ignore
    }
}

# Results storage
results = {
    "timestamp": datetime.now().isoformat(),
    "model_results": {}
}

print(f"Starting ROUGE evaluation for {len(models)} models on {len(datasets_to_evaluate)} datasets...")
print(f"Each dataset contains 250 examples.")

# Evaluate each model on each dataset
for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*50}")
    
    results["model_results"][model_name] = {}
    
    for dataset_name, dataset_data in datasets_to_evaluate.items():
        print(f"\nEvaluating on {dataset_name} dataset...")
        
        texts = dataset_data["texts"]
        reference_summaries = dataset_data["summaries"]
        generated_summaries = []
        
        # Generate summaries with progress bar
        for text in tqdm(texts, desc=f"Generating summaries for {dataset_name}"):
            summary = safe_generate_summary(model, text)
            generated_summaries.append(summary)
            time.sleep(0.5)  # Small delay to prevent rate limiting while keeping speed
        
        # Calculate ROUGE scores
        rouge_scores = evaluate_rouge_scores(generated_summaries, reference_summaries)
        
        # Store results
        results["model_results"][model_name][dataset_name] = {
            "rouge_scores": rouge_scores,
            "sample_summaries": {
                "generated": generated_summaries[:5],  # Store first 5 examples
                "reference": reference_summaries[:5]
            }
        }
        
        # Print results
        print(f"\nROUGE Scores for {model_name} on {dataset_name}:")
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        print(f"Successful generations: {rouge_scores['num_successful']}/{len(texts)}")

# Save results to JSON file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_filename = f"rouge_evaluation_results_{timestamp}.json"

with open(results_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"Evaluation complete! Results saved to: {results_filename}")
print(f"{'='*50}")

# Print summary table
print("\nSUMMARY TABLE:")
print("-" * 80)
print(f"{'Model':<20} {'Dataset':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Success':<10}")
print("-" * 80)

for model_name, model_results in results["model_results"].items():
    for dataset_name, dataset_results in model_results.items():
        scores = dataset_results["rouge_scores"]
        success_rate = scores['num_successful'] / 1000 * 100
        print(f"{model_name:<20} {dataset_name:<10} {scores['rouge1']:<10.4f} {scores['rouge2']:<10.4f} {scores['rougeL']:<10.4f} {success_rate:<9.1f}%")

# Generate visualization and summary
print("\nGenerating visualization...")
plot_rouge_results(results_filename, save_plot=True)
print_summary_stats(results_filename)