import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Load Cornell data ──────────────────────────────────────
from utils.data_loader import load_cornell_data

samples = load_cornell_data("data/cornell", num_samples=2000)
split = int(0.8 * len(samples))
train_data = samples[:split]
test_data  = samples[split:]

# save test data for evaluation later
with open("data/gpt2_test_data.json", "w") as f:
    json.dump(test_data, f)

print(f"Train pairs: {len(train_data)}, Test pairs: {len(test_data)}")

# ── Tokenizer ─────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ── Dataset ───────────────────────────────────────────────
class DialogDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=64):
        self.examples = []
        for src, tgt in pairs:
            # format: "input: {src} response: {tgt} <|endoftext|>"
            text = f"input: {src} response: {tgt} {tokenizer.eos_token}"
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"].squeeze(),
            "attention_mask": self.examples[idx]["attention_mask"].squeeze()
        }

train_dataset = DialogDataset(train_data, tokenizer)
train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ── Model ─────────────────────────────────────────────────
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# ── Training ──────────────────────────────────────────────
EPOCHS = 3  # 3 epochs is enough for fine-tuning
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batches = 0

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    avg_loss = total_loss / batches
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} | loss {avg_loss:.4f}")

# ── Save ──────────────────────────────────────────────────
model.save_pretrained("models/gpt2_finetuned")
tokenizer.save_pretrained("models/gpt2_finetuned")
print("GPT-2 saved to models/gpt2_finetuned/")

# ── Save loss curve ───────────────────────────────────────
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="GPT-2 fine-tune loss", color="steelblue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GPT-2 fine-tuning — Cornell Movie Dialogs")
plt.legend()
plt.tight_layout()
plt.savefig("results/gpt2_loss_curve.png")
print("Loss curve saved.")