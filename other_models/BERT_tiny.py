from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load BERT tiny model and tokenizer from Hugging Face
model_name = "prajjwal1/bert-tiny"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("Model loaded successfully!")

# Example text to process
texts = [
    "The stock market is performing well today.",
    "This is a great investment opportunity.",
    "The company's earnings report was disappointing."
]

print("\n" + "="*50)
print("Tokenizing and encoding texts...")
print("="*50)

# Tokenize and encode the texts
for i, text in enumerate(texts, 1):
    print(f"\nText {i}: {text}")
    
    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(f"Token IDs: {tokens['input_ids']}")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**tokens)
        # Get the [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        print(f"CLS embedding shape: {cls_embedding.shape}")
        print(f"CLS embedding (first 10 values): {cls_embedding[0][:10].numpy()}")

print("\n" + "="*50)
print("Batch processing example...")
print("="*50)

# Process all texts in a batch
batch_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
print(f"Batch input shape: {batch_tokens['input_ids'].shape}")

with torch.no_grad():
    batch_outputs = model(**batch_tokens)
    batch_cls_embeddings = batch_outputs.last_hidden_state[:, 0, :]
    print(f"Batch CLS embeddings shape: {batch_cls_embeddings.shape}")

# Compute similarity between embeddings (using cosine similarity)
print("\n" + "="*50)
print("Computing similarity between texts...")
print("="*50)

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=1).item()

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(
            batch_cls_embeddings[i:i+1],
            batch_cls_embeddings[j:j+1]
        )
        print(f"Similarity between text {i+1} and text {j+1}: {sim:.4f}")

print("\n" + "="*50)
print("Model information:")
print("="*50)
print(f"Model type: {type(model)}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Number of attention heads: {model.config.num_attention_heads}")



