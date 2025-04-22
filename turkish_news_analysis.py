import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset
dataset = load_dataset("batubayk/TR-News")

# Downsample the dataset for training and evaluation
def downsample_dataset(dataset, num_samples=1000):
    downsampled = dataset.shuffle(seed=42).select(range(num_samples))
    return downsampled

dataset["train"] = downsample_dataset(dataset["train"], num_samples=55600)
dataset["validation"] = downsample_dataset(dataset["validation"], num_samples=2920)
dataset["test"] = downsample_dataset(dataset["test"], num_samples=3080)

# Encode labels
label_encoder = LabelEncoder()
dataset_labels = dataset["train"]["topic"]
label_encoder.fit(dataset_labels)

def preprocess_text(example):
    text = "[CLS] " + example["abstract"] + " [SEP] " + example["content"] + " [SEP]"
    return {"text": text, "label": label_encoder.transform([example["topic"]])[0]}  # Encode labels

dataset = dataset.map(preprocess_text, remove_columns=["author", "title", "date", "source", "tags", "title", "url"])

# Define model names
models = {
    "mBERT": "bert-base-multilingual-cased",  # Uses WordPiece
    "XLM-R": "xlm-roberta-base",  # Uses SentencePiece
    "BERTurk": "dbmdz/bert-base-turkish-cased"  # Uses WordPiece
}

# Load tokenizers
tokenizers = {name: AutoTokenizer.from_pretrained(model) for name, model in models.items()}

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_datasets = {
    name: dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, num_proc=4)
    for name, tokenizer in tokenizers.items()
}

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained models
num_labels = len(set(dataset["train"]["label"]))
models = {
    name: AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels).to(device)
    for name, model in models.items()
}

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "crossloss": F.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
    }

trainers = {
    name: Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[name]["train"],
        eval_dataset=tokenized_datasets[name]["validation"],
        tokenizer=tokenizers[name],
        compute_metrics=compute_metrics
    )
    for name, model in models.items()
}

# Train models
for name, trainer in trainers.items():
    print(f"Training {name}...")
    trainer.train()

# Evaluate models
results = {name: trainer.evaluate() for name, trainer in trainers.items()}

# Display results
for name, res in results.items():
    print(f"\n{name} Evaluation Results:")
    print(f"Accuracy: {res['eval_accuracy']:.4f}")
    print(f"F1-score: {res['eval_f1']:.4f}")
    print(f"Cross-Entropy Loss: {res['eval_crossloss']:.4f}")


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (name, trainer) in enumerate(trainers.items()):
    result = trainer.predict(tokenized_datasets[name]["test"])
    y_true = np.array(dataset["test"]["label"])
    y_pred = np.argmax(result.predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(ax=axes[idx], xticks_rotation=45, colorbar=False)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("True")
    axes[idx].set_title(f"{name}")

plt.suptitle("Confusion Matrices for All Models", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("confusion_matrices_combined.png")
plt.show()

def calculate_tokenization_metrics(texts, tokenizer):
    total_words = 0
    total_tokens = 0
    total_chars = 0
    oov_tokens = 0

    for text in texts:
        words = text.split()
        total_words += len(words)
        total_chars += sum(len(w) for w in words)

        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)

        # Estimate OOV by counting unknown tokens if the tokenizer uses them
        if tokenizer.unk_token:
            oov_tokens += tokens.count(tokenizer.unk_token)

    avg_tokens_per_word = total_tokens / total_words
    token_to_char_ratio = total_tokens / total_chars
    compression_rate = total_chars / total_tokens
    oov_rate = oov_tokens / total_tokens if total_tokens > 0 else 0

    return {
        "Avg Tokens/Word": avg_tokens_per_word,
        "Token-to-Char Ratio": token_to_char_ratio,
        "Compression Rate": compression_rate,
        "OOV Rate": oov_rate
    }

# Evaluate tokenization metrics for each model
print("\nTokenization Metrics:\n")
for name, tokenizer in tokenizers.items():
    texts = dataset["test"]["text"]
    metrics = calculate_tokenization_metrics(texts, tokenizer)
    print(f"{name} Tokenization Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print()