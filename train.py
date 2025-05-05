import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
import wandb
import os

# Optional: Initialize Weights & Biases
wandb.init(project="725-P", name="gemma-lora")

# Paths
MODEL_NAME = "google/gemma-2b-it"
DATASET_PATH = "./processed_dataset"  # from dataprep.py (use Dataset.save_to_disk)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

# LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_from_disk(DATASET_PATH)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]


# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=1e-4,
    #fp16=True,
    resume_from_checkpoint=True,
    report_to=["wandb"],
    run_name="gemma-lora-run",
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("./gemma-lora-checkpoint")
tokenizer.save_pretrained("./gemma-lora-checkpoint")

# Finish W&B
wandb.finish()
