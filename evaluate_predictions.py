import pandas as pd
import evaluate
import nltk
nltk.download("punkt")
nltk.download("wordnet")

# Load predictions CSV
df = pd.read_csv("inference/predictions.csv")

# Extract predictions
preds = df["predicted_caption"].tolist()

# Prepare list of 5 references per example ***
refs = df[["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]].values.tolist()

# Evaluate BLEU
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=preds, references=refs)

# Evaluate METEOR
meteor = evaluate.load("meteor")
# METEOR only accepts a single reference per prediction, so we use caption_1 as a proxy
meteor_score = meteor.compute(predictions=preds, references=df["caption_1"].tolist())

# Print results
print("\n=== Evaluation Results ===")
print(f"BLEU: {bleu_score}")
print(f"METEOR: {meteor_score}")

# save to a text file
with open("inference/eval_results.txt", "w", encoding="utf-8") as f:
    f.write("=== Evaluation Results ===\n")
    f.write(f"BLEU: {bleu_score}\n")
    f.write(f"METEOR: {meteor_score}\n")
