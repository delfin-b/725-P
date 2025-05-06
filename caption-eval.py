import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from torchvision import transforms
from evaluate import load as load_metric
import nltk

# Step 0: Download nltk resources
nltk.download('punkt')
nltk.download('wordnet')

# Load metrics
bleu = load_metric("bleu")
meteor = load_metric("meteor")

# Paths
CSV_PATH = "RISCM/captions.csv"
IMAGE_DIR = "RISCM/resized"
CHECKPOINT_PATH = "./gemma-lora-checkpoint"

# Load data
df = pd.read_csv(CSV_PATH)
df_val = df[df["split"] == "val"]
samples = df_val.sample(n=50, random_state=42)

# Load tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH, device_map="auto")
model.eval()

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Store results
preds = []
refs = []

# Optional prompt
prompt = "Aerial view of "

for idx, row in samples.iterrows():
    image_path = f"{IMAGE_DIR}/{row['image']}"
    img = Image.open(image_path).convert("RGB")
    pixel_values = image_transform(img).unsqueeze(0).to("cuda")

    input_text = prompt
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, max_length=64)
        generated_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    preds.append(generated_caption)
    refs.append([row['caption_1']])  # you can average over all 5 if you want

# Tokenize for evaluation
tokenized_preds = [nltk.word_tokenize(p.lower()) for p in preds]
tokenized_refs = [[nltk.word_tokenize(r.lower()) for r in ref_list] for ref_list in refs]

# Compute BLEU & METEOR (raw strings, not tokenized)
bleu_score = bleu.compute(predictions=preds, references=refs)
meteor_score = meteor.compute(predictions=preds, references=[r[0] for r in refs])


print(f"\nBLEU Score: {bleu_score['bleu']:.4f}")
print(f"METEOR Score: {meteor_score['meteor']:.4f}")
