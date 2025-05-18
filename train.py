import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from dataprep import RISCImageCaptionDataset
import os
import wandb

# Initialize Weights & Biases
wandb.init(project="725-P", name="3shorter-paligemma-lora-run", job_type="training")


# Paths
# Set paths relative to the script location or project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "RISCM", "resized")
caption_csv = os.path.join(BASE_DIR, "RISCM", "captions.csv")


# Load processor and model
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224")


# Freeze encoder (SigLIP)
model.vision_tower.requires_grad_(False)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model.language_model = get_peft_model(model.language_model, peft_config)

model.gradient_checkpointing_enable()

model.config.use_cache = False  # Important for training

# Load datasets
train_dataset = RISCImageCaptionDataset(caption_csv, image_dir, split="train", processor=processor)
val_dataset = RISCImageCaptionDataset(caption_csv, image_dir, split="val", processor=processor)



# Training Arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-5,
    max_steps=400,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    fp16=True,
    report_to="wandb",
    run_name="3shorter-paligemma-lora-run",
    logging_dir="./logs",
    logging_steps=100,
    eval_steps=200,
    save_steps=200
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    data_collator=lambda data: {
        key: torch.stack([f[key] for f in data]) for key in data[0]
    }
)
# Start training
trainer.train()
model.language_model.save_pretrained("./lora_adapter")  # <-- only LoRA adapter
processor.save_pretrained("./lora_adapter")


#up0load model from checkpoint to wandb as artifact
artifact = wandb.Artifact("3paligemma-lora-checkpoint", type="model")
artifact.add_dir("./output")
wandb.log_artifact(artifact)

wandb.finish()