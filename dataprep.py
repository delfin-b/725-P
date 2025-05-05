import os
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
from torch.utils.data import Dataset

# Paths
BASE_DIR = "RISCM"
CSV_PATH = os.path.join(BASE_DIR, "captions.csv")
DATA_DIR = os.path.join(BASE_DIR, "resized")
TOKENIZER_NAME = "google/gemma-2b-it"  # Update if using a different Gemma variant

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Define image transform (as used by SigLIP)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # assuming 3-channel
])

# Load CSV and flatten to image-caption pairs
def load_caption_data(csv_path):
    df = pd.read_csv(csv_path)
    caption_cols = [col for col in df.columns if "caption" in col]
    rows = []

    for _, row in df.iterrows():
        image_name = row['image']
        for col in caption_cols:
            caption = row[col]
            if pd.notnull(caption):
                rows.append({"image": image_name, "caption": caption, "split": row["split"]})

    return pd.DataFrame(rows)

# Dataset class
class RISCTextImageDataset(Dataset):
    def __init__(self, df, image_root, tokenizer, transform, max_length=64):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image"])
        image = self.transform(Image.open(img_path).convert("RGB"))

        text = row["caption"]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "text": text
        }

# usage
if __name__ == "__main__":
    df = load_caption_data(CSV_PATH)
    print("Loaded", len(df), "image-caption pairs")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    print("Train set:", len(train_df), "Val set:", len(val_df))

    train_dataset = RISCTextImageDataset(train_df, DATA_DIR, tokenizer, image_transform)
    val_dataset = RISCTextImageDataset(val_df, DATA_DIR, tokenizer, image_transform)
    sample = train_dataset[0]
    print("Sample caption:", sample["text"])
    print("Token IDs shape:", sample["input_ids"].shape)
    print("Image shape:", sample["pixel_values"].shape)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    from datasets import Dataset, DatasetDict

    # Convert to HuggingFace Datasets
    train_hf = Dataset.from_pandas(train_df[["caption"]])
    val_hf = Dataset.from_pandas(val_df[["caption"]])

    # Tokenize
    def tokenize_fn(example):
        return tokenizer(
            example["caption"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )

    train_hf = train_hf.map(tokenize_fn, batched=True)
    val_hf = val_hf.map(tokenize_fn, batched=True)

    # Create DatasetDict and save
    dataset_dict = DatasetDict({
        "train": train_hf,
        "validation": val_hf
    })

    dataset_dict.save_to_disk("processed_dataset")
    print("Saved HuggingFace dataset to 'processed_dataset'")
