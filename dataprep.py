from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
from transformers import AutoProcessor

class RISCImageCaptionDataset(Dataset):
    def __init__(self, csv_path, image_dir, split="train", processor=None, max_target_length=32):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split]
        self.image_dir = image_dir
        self.processor = processor
        self.max_target_length = max_target_length

        # Flatten: each image-caption pair is one item
        self.samples = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.image_dir, row['image'])
            for i in range(1, 6):  # caption_1 to caption_5
                cap = row[f'caption_{i}']
                self.samples.append((img_path, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        # Insert <image> token explicitly before the caption
        image_token = self.processor.tokenizer.image_token if hasattr(self.processor.tokenizer, "image_token") else "<image>"
        caption_with_token = f"{image_token} {caption}"

        inputs = self.processor(
            images=image,
            text=caption_with_token,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt"
        )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]
        return inputs

