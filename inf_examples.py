import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# load main csv
main_df = pd.read_csv("RISCM/captions.csv")

# Load LoRA predictions from predictions.txt
lora_preds = []
lora_refs = []

with open("predictions.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i in range(0, len(lines), 3):
        pred = lines[i].replace("Predicted:", "").strip()
        ref = lines[i + 1].replace("Reference:", "").strip()
        lora_preds.append(pred)
        lora_refs.append(ref)

val_images = main_df[main_df["split"] == "val"].reset_index(drop=True)["image"].tolist()

lora_df = pd.DataFrame({
    "image": val_images[:len(lora_preds)],
    "lora_caption": lora_preds
})

baseline_df = pd.read_csv("inference/predictions.csv")
baseline_df = baseline_df[["image", "predicted_caption"]].rename(columns={"predicted_caption": "baseline_caption"})

# Merge all into one df
df = main_df.merge(lora_df, on="image", how="left")
df = df.merge(baseline_df, on="image", how="left")

# Filter only rows that have both baseline and LoRA predictions
valid_df = df[
    df["baseline_caption"].notna() & df["lora_caption"].notna()
]

# Sample
#sampled = df[df["split"] == "val"].sample(3, random_state=42)
# Sample from only the valid ones
sampled = valid_df.sample(3, random_state=42)

# Set up subplot with 2 columns (image | text)
fig, axs = plt.subplots(len(sampled), 2, figsize=(12, 12))  # width increased for text
fig.subplots_adjust(hspace=0.4)

for idx, (_, row) in enumerate(sampled.iterrows()):
    image_path = Path("RISCM/resized") / row["image"]
    image = Image.open(image_path)

    # Plot image
    axs[idx, 0].imshow(image)
    axs[idx, 0].axis("off")

    # Prepare wrapped text
    ref = textwrap.fill(f"Reference: {row['caption_1']}", width=50)
    baseline = textwrap.fill(f"Baseline: {row['baseline_caption'] if pd.notna(row['baseline_caption']) else 'No prediction'}", width=50)
    lora = textwrap.fill(f"LoRA Fine-tuned: {row['lora_caption'] if pd.notna(row['lora_caption']) else 'No prediction'}", width=50)


    caption_text = f"{ref}\n\n{baseline}\n\n{lora}"

    axs[idx, 1].text(0, 1, caption_text, fontsize=10, va="top", ha="left", wrap=True)
    axs[idx, 1].axis("off")

# Save and show
plt.tight_layout()
plt.savefig("inference_examples_side.png", dpi=300)
plt.show()
