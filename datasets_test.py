from datasets import load_dataset

# Test MLSUM dataset
print("Testing MLSUM dataset...")
try:
    mlsum_dataset = load_dataset("mlsum", "tu", split="test[:5]")
    print("MLSUM loaded successfully!")
    # Safely access the first item
    first_item = mlsum_dataset[0]  # type: ignore
    text = str(first_item['text'])  # type: ignore
    print(f"Sample MLSUM article preview: {text[:200]}...")
    print(f"MLSUM columns: {mlsum_dataset.column_names}")
    print()
except Exception as e:
    print(f"Error loading MLSUM: {e}")
    print()

# Test XLSum dataset
print("Testing XLSum dataset...")
try:
    xlsum_dataset = load_dataset("csebuetnlp/xlsum", "turkish", split="test[:5]")
    print("XLSum loaded successfully!")
    # Safely access the first item
    first_item = xlsum_dataset[0]  # type: ignore
    text = str(first_item['text'])  # type: ignore
    print(f"Sample XLSum article preview: {text[:200]}...")
    print(f"XLSum columns: {xlsum_dataset.column_names}")
    print()
except Exception as e:
    print(f"Error loading XLSum: {e}")
    print()