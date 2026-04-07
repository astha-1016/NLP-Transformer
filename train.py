import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import time

from models.transformer import TransformerSystem, Lexicon
from utils.data_loader import load_cornell_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Data ──────────────────────────────────────────────────
samples = load_cornell_data("data/cornell", num_samples=2000)
split = int(0.8 * len(samples))
train_data = samples[:split]
test_data  = samples[split:]

vocab = Lexicon()
vocab.prepare(train_data)
print("Vocab size:", vocab.total)

# ── Encode once, not every epoch ──────────────────────────
encoded_src = [vocab.encode(x[0]) for x in train_data]
encoded_trg = [vocab.encode(x[1]) for x in train_data]

def pad_batch(data):
    mx = max(len(x) for x in data)
    return torch.tensor([x + [0] * (mx - len(x)) for x in data])

# ── Mini-batch generator ───────────────────────────────────
BATCH_SIZE = 32

def get_batches(src_data, trg_data, batch_size):
    for i in range(0, len(src_data), batch_size):
        src_batch = pad_batch(src_data[i:i+batch_size]).to(device)
        trg_batch = pad_batch(trg_data[i:i+batch_size]).to(device)
        yield src_batch, trg_batch

# ── Model ──────────────────────────────────────────────────
model = TransformerSystem(vocab.total).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

# ── Training loop ──────────────────────────────────────────
EPOCHS = 150   # was 1000

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batches = 0
    t0 = time.time()

    for src, trg in get_batches(encoded_src, encoded_trg, BATCH_SIZE):
        decoder_in = trg[:, :-1]
        target_out = trg[:, 1:]

        pred, _ = model(src, decoder_in)
        loss = criterion(pred.reshape(-1, vocab.total), target_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevents exploding gradients
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    avg_loss = total_loss / batches
    scheduler.step(avg_loss)

    if epoch % 10 == 0:
        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d} | loss {avg_loss:.4f} | {elapsed:.1f}s/epoch")

# ── Save ───────────────────────────────────────────────────
torch.save(model.state_dict(), "models/model_v2.pth")
with open("models/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("data/test_data.json", "w") as f:
    json.dump(test_data, f)

print("Done!")