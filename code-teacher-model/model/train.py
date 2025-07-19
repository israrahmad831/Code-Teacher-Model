import json
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Load dataset
with open("data/code_tutorials.json") as f:
    raw_data = json.load(f)

# Prepare text in "Q: question A: answer" format
texts = [f"Q: {item['question']} A: {item['answer']}" for item in raw_data]
dataset = Dataset.from_dict({"text": texts})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

# Fix: GPT-2 tokenizer has no pad_token by default, so set it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(config["model_name"])

# Tokenize function for dataset mapping
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=config["max_length"]
    )
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply tokenization to dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments setup
training_args = TrainingArguments(
    output_dir=config["save_dir"],
    overwrite_output_dir=True,
    num_train_epochs=config["epochs"],
    per_device_train_batch_size=config["batch_size"],
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
trainer.save_model(config["save_dir"])
tokenizer.save_pretrained(config["save_dir"])
