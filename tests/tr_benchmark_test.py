import os
import json
import time
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from datasets import load_dataset
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from rouge_score import rouge_scorer
from tqdm import tqdm

# Support functions
def finish_sentence(text: str) -> str:
    """Truncate text at the last full sentence terminator (., !, ?)."""
    last_dot = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    return text[: last_dot + 1] if last_dot != -1 else text

def create_prompt(text: str) -> str:
    """Create a Turkish summarization prompt."""
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

def test_environment_setup():
    """Test if environment variables are properly set."""
    print("🔧 Testing environment setup...")
    
    load_dotenv()
    api_key = os.getenv("API_KEY")        
    project_id = os.getenv("PROJECT_ID")  
    service_url = os.getenv("REGION_URL") 

    if not api_key:
        print("❌ API_KEY environment variable is missing!")
        return False
    if not project_id:
        print("❌ PROJECT_ID environment variable is missing!")
        return False
    if not service_url:
        print("❌ REGION_URL environment variable is missing!")
        return False
    
    print("✅ Environment variables are properly set")
    return True

def test_dataset_loading():
    """Test if datasets can be loaded successfully."""
    print("\n📊 Testing dataset loading...")
    
    try:
        mlsum_test = load_dataset("mlsum", "tu", split="test[:5]")
        xlsum_test = load_dataset("csebuetnlp/xlsum", "turkish", split="test[:5]")
        
        print(f"✅ MLSUM dataset loaded: {len(mlsum_test)} examples")  # type: ignore
        print(f"✅ XLSum dataset loaded: {len(xlsum_test)} examples")  # type: ignore
        
        # Test data structure
        mlsum_example = mlsum_test[0]  # type: ignore
        xlsum_example = xlsum_test[0]  # type: ignore
        
        if 'text' in mlsum_example and 'summary' in mlsum_example:
            print("✅ MLSUM data structure is correct")
        else:
            print("❌ MLSUM data structure is incorrect")
            return False
            
        if 'text' in xlsum_example and 'summary' in xlsum_example:
            print("✅ XLSum data structure is correct")
        else:
            print("❌ XLSum data structure is incorrect")
            return False
            
        return True, mlsum_test, xlsum_test
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False, None, None

def test_model_connection():
    """Test if model connection works."""
    print("\n🤖 Testing model connection...")
    
    load_dotenv()
    api_key = os.getenv("API_KEY")        
    project_id = os.getenv("PROJECT_ID")  
    service_url = os.getenv("REGION_URL") 

    try:
        creds = Credentials(api_key=api_key, url=service_url)
        params = {
            GenParams.MAX_NEW_TOKENS: 50,
            GenParams.MIN_NEW_TOKENS: 10,
            GenParams.TEMPERATURE: 0.1,
        }

        # Test with just one model
        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            project_id=project_id,
            params=params,
            credentials=creds
        )
        
        # Test generation
        test_prompt = "Bu kısa bir test metnidir. Lütfen özetleyin."
        summary = model.generate_text(prompt=test_prompt)
        
        print(f"✅ Model connection successful")
        print(f"   Test generation: {summary[:50]}...")
        return True, model
        
    except Exception as e:
        print(f"❌ Model connection failed: {e}")
        return False, None

def test_rouge_calculation():
    """Test ROUGE score calculation."""
    print("\n📈 Testing ROUGE calculation...")
    
    try:
        # Test data
        generated = ["Bu kısa bir özettir.", "Başka bir özet örneği."]
        reference = ["Bu kısa özet örneğidir.", "Farklı bir özet metni."]
        
        scores = evaluate_rouge_scores(generated, reference)
        
        print(f"✅ ROUGE calculation successful")
        print(f"   ROUGE-1: {scores['rouge1']:.4f}")
        print(f"   ROUGE-2: {scores['rouge2']:.4f}")
        print(f"   ROUGE-L: {scores['rougeL']:.4f}")
        print(f"   Successful: {scores['num_successful']}/2")
        
        return True
        
    except Exception as e:
        print(f"❌ ROUGE calculation failed: {e}")
        return False

def run_mini_benchmark():
    """Run a mini benchmark with 3 examples and 1 model."""
    print("\n🏃‍♂️ Running mini benchmark (3 examples, 1 model)...")
    
    # Load environment and setup
    load_dotenv()
    api_key = os.getenv("API_KEY")        
    project_id = os.getenv("PROJECT_ID")  
    service_url = os.getenv("REGION_URL")
    
    # Load tiny datasets
    mlsum_dataset = load_dataset("mlsum", "tu", split="test[:3]")
    xlsum_dataset = load_dataset("csebuetnlp/xlsum", "turkish", split="test[:3]")
    
    # Setup model
    creds = Credentials(api_key=api_key, url=service_url)
    params = {
        GenParams.MAX_NEW_TOKENS: 100,
        GenParams.MIN_NEW_TOKENS: 20,
        GenParams.TEMPERATURE: 0.1,
        GenParams.STOP_SEQUENCES: ["\n\n", "###"],
    }
    
    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct",
        project_id=project_id,
        params=params,
        credentials=creds
    )
    
    # Test on MLSUM
    print("  Testing MLSUM dataset...")
    mlsum_texts = [str(example['text']) for example in mlsum_dataset]  # type: ignore
    mlsum_refs = [str(example['summary']) for example in mlsum_dataset]  # type: ignore
    
    mlsum_generated = []
    for text in mlsum_texts:
        summary = safe_generate_summary(model, text)
        mlsum_generated.append(summary)
        time.sleep(0.1)
    
    mlsum_scores = evaluate_rouge_scores(mlsum_generated, mlsum_refs)
    
    print(f"    MLSUM ROUGE-1: {mlsum_scores['rouge1']:.4f}")
    print(f"    MLSUM ROUGE-2: {mlsum_scores['rouge2']:.4f}")
    print(f"    MLSUM ROUGE-L: {mlsum_scores['rougeL']:.4f}")
    print(f"    Success rate: {mlsum_scores['num_successful']}/3")
    
    # Test on XLSum
    print("  Testing XLSum dataset...")
    xlsum_texts = [str(example['text']) for example in xlsum_dataset]  # type: ignore
    xlsum_refs = [str(example['summary']) for example in xlsum_dataset]  # type: ignore
    
    xlsum_generated = []
    for text in xlsum_texts:
        summary = safe_generate_summary(model, text)
        xlsum_generated.append(summary)
        time.sleep(0.1)
    
    xlsum_scores = evaluate_rouge_scores(xlsum_generated, xlsum_refs)
    
    print(f"    XLSum ROUGE-1: {xlsum_scores['rouge1']:.4f}")
    print(f"    XLSum ROUGE-2: {xlsum_scores['rouge2']:.4f}")
    print(f"    XLSum ROUGE-L: {xlsum_scores['rougeL']:.4f}")
    print(f"    Success rate: {xlsum_scores['num_successful']}/3")
    
    return True

def main():
    """Run all tests."""
    print("🧪 TURKISH BENCHMARK TEST SUITE")
    print("="*50)
    
    all_tests_passed = True
    
    # Test 1: Environment setup
    if not test_environment_setup():
        all_tests_passed = False
        print("\n❌ Environment test failed. Please check your .env file.")
        return
    
    # Test 2: Dataset loading
    dataset_result = test_dataset_loading()
    if isinstance(dataset_result, tuple):
        dataset_success, mlsum_data, xlsum_data = dataset_result
        if not dataset_success:
            all_tests_passed = False
            print("\n❌ Dataset test failed. Please check your internet connection.")
            return
    else:
        all_tests_passed = False
        print("\n❌ Dataset test failed. Please check your internet connection.")
        return

    # Test 3: Model connection
    model_result = test_model_connection()
    if isinstance(model_result, tuple):
        model_success, model = model_result
        if not model_success:
            all_tests_passed = False
            print("\n❌ Model connection test failed. Please check your API credentials.")
            return
    else:
        all_tests_passed = False
        print("\n❌ Model connection test failed. Please check your API credentials.")
        return
    
    # Test 4: ROUGE calculation
    if not test_rouge_calculation():
        all_tests_passed = False
        print("\n❌ ROUGE calculation test failed.")
        return
    
    # Test 5: Mini benchmark
    try:
        run_mini_benchmark()
        print("\n✅ Mini benchmark completed successfully!")
    except Exception as e:
        print(f"\n❌ Mini benchmark failed: {e}")
        all_tests_passed = False
    
    # Final result
    print("\n" + "="*50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! You're ready to run the full benchmark.")
        print("   Run: python tr_benchmark.py")
    else:
        print("❌ Some tests failed. Please fix the issues before running the full benchmark.")
    print("="*50)

if __name__ == "__main__":
    main()
