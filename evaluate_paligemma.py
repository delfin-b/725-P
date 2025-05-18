import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from dataprep import RISCImageCaptionDataset
import evaluate
from tqdm import tqdm
import PIL
import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load processor and model
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
model.eval()

# Load validation set (subset for quick evaluation)
val_dataset = RISCImageCaptionDataset(
    csv_path=r"C:\Users\airlab\Desktop\New folder\Lora2\RISCM\captions.csv",
    image_dir=r"C:\Users\airlab\Desktop\New folder\Lora2\RISCM\resized",
    processor=processor,
    split="val"
)
val_dataset = Subset(val_dataset, range(100))  # First 100 samples

val_loader = DataLoader(val_dataset, batch_size=1)
'''
for batch in val_loader:
    print("Batch keys:", batch.keys())
    break
'''

# Metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

predictions, references = [], []

for batch in tqdm(val_loader, desc="Evaluating"):
    pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)

    inputs = {
        "input_ids": batch["input_ids"].to("cuda"),
        "attention_mask": batch["attention_mask"].to("cuda"),
        "pixel_values": pixel_values
    }

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    generated_caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    true_caption = processor.tokenizer.decode(batch["labels"][0][batch["labels"][0] != -100], skip_special_tokens=True)

    predictions.append(generated_caption)
    references.append([true_caption])



# Compute metrics
results = {
    "BLEU": bleu.compute(predictions=predictions, references=references),
    "METEOR": meteor.compute(predictions=predictions, references=references)
}

print("\nEvaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value}")

# Save example captions
with open("predictions.txt", "w", encoding="utf-8") as f:
    for pred, ref in zip(predictions, references):
        f.write(f"Predicted: {pred}\nReference: {ref[0]}\n\n")
