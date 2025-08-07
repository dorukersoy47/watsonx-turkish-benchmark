import os
import random
from dotenv import load_dotenv
from datasets import load_dataset
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

# Load Turkish datasets
print("Loading datasets...")

# Load MLSUM dataset (more samples for variety)
mlsum_dataset = load_dataset("mlsum", "tu", split="test[:50]")
print("MLSUM loaded successfully - 50 samples available")

# Load XLSum dataset (more samples for variety)
xlsum_dataset = load_dataset("csebuetnlp/xlsum", "turkish", split="test[:50]")
print("XLSum loaded successfully - 50 samples available")

# Randomly select articles for testing
random.seed()  # Use current time for different results each run
mlsum_index = random.randint(0, 49)
xlsum_index = random.randint(0, 49)
print(f"Randomly selected MLSUM article #{mlsum_index}")
print(f"Randomly selected XLSum article #{xlsum_index}")

# Configure Watsonx credentials
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
    GenParams.MAX_NEW_TOKENS: 200,           # Increased for longer summaries
    GenParams.MIN_NEW_TOKENS: 50,            # Ensure minimum summary length
    GenParams.TEMPERATURE: 0.1,              # Very low randomness for consistency
    GenParams.TOP_P: 0.8,                    # Reduced for more focused responses
    GenParams.TOP_K: 10,                     # Reduced for more deterministic choices
    GenParams.REPETITION_PENALTY: 1.05,     # Slight penalty to avoid repetition
    GenParams.RANDOM_SEED: 42,               # Fixed seed for reproducibility
    GenParams.STOP_SEQUENCES: ["\n\n", "###"], # Stop on double newlines or separators
    GenParams.DECODING_METHOD: "greedy",     # Most deterministic method
    GenParams.TRUNCATE_INPUT_TOKENS: 16000,  # Increased to handle full articles
}

model = ModelInference(
    model_id="ibm/granite-3-8b-instruct",
    project_id=project_id,
    params=params,
    credentials=creds
)

# Test with first MLSUM article
print("\n=== Testing with MLSUM article ===")
mlsum_text = str(mlsum_dataset[0]['text'])  # type: ignore
print(f"Original text length: {len(mlsum_text)} characters")
print(f"Original text preview: {mlsum_text[:200]}...")

# Enhanced prompt for better summarization
prompt = f"""Lütfen aşağıdaki Türkçe metni özetleyin. Özet açık, kısa ve ana noktaları içermelidir:

{mlsum_text}

Özet:"""

generated_text = model.generate_text(prompt=prompt)
print("Generated summary:", generated_text)

# Test with first XLSum article  
print("\n=== Testing with XLSum article ===")
xlsum_text = str(xlsum_dataset[0]['text'])  # type: ignore
print(f"Original text length: {len(xlsum_text)} characters")
print(f"Original text preview: {xlsum_text[:200]}...")

prompt = f"""Lütfen aşağıdaki Türkçe metni özetleyin. Özet açık, kısa ve ana noktaları içermelidir:

{xlsum_text}

Özet:"""

generated_text = model.generate_text(prompt=prompt)
print("Generated summary:", generated_text)