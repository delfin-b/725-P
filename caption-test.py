from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./gemma-lora-checkpoint")
model = AutoModelForCausalLM.from_pretrained("./gemma-lora-checkpoint").to("cuda")

prompts = [
    "This image shows",
    "In the satellite photo, we observe",
    "Aerial view of",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=40, num_return_sequences=3, do_sample=True, top_k=50)
    print(f"Prompt: {prompt}")
    for i, output in enumerate(outputs):
        print(f"  Caption {i+1}: {tokenizer.decode(output, skip_special_tokens=True)}")
    print()
