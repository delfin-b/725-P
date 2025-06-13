import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- Config ---
MODEL_ID = "google/paligemma-3b-mix-224"
DATA_PATH = "RISCM/captions.csv"  
IMAGE_FOLDER = "RISCM/resized"          
OUTPUT_CSV = "inference/predictions.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model and processor ---
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
df["predicted_caption"] = ""

# --- Inference ---
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating captions"):
    image_path = Path(IMAGE_FOLDER) / row["image"]
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        continue

    image = Image.open(image_path).convert("RGB")

    # You can adjust the prompt to guide the model, like "caption" or "caption en"
    prompt = "caption"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=100)
        generated_ids = outputs[0][input_len:]
        decoded = processor.decode(generated_ids, skip_special_tokens=True)

    df.at[idx, "predicted_caption"] = decoded

# --- Save predictions ---
# Ensure output folder exists
Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to: {OUTPUT_CSV}")
