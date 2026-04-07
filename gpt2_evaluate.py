import json
import torch
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from utils.inference import predict
from gpt2_inference import load_gpt2, predict_gpt2
from models.transformer import TransformerSystem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smoothie = SmoothingFunction().method4

# ── Load custom transformer ────────────────────────────────
with open("models/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
custom_model = TransformerSystem(vocab.total).to(device)
custom_model.load_state_dict(torch.load("models/model_v2.pth", map_location=device))
custom_model.eval()

# ── Load GPT-2 ─────────────────────────────────────────────
gpt2_model, gpt2_tokenizer = load_gpt2()

# ── Load test data ─────────────────────────────────────────
with open("data/gpt2_test_data.json") as f:
    test_data = json.load(f)

custom_refs, custom_hyps = [], []
gpt2_refs,   gpt2_hyps   = [], []

print("\n--- Comparison ---\n")
for inp, ref in test_data[:20]:
    custom_pred, _, _ = predict(inp, custom_model, vocab, device)
    gpt2_pred         = predict_gpt2(inp, gpt2_model, gpt2_tokenizer)

    custom_refs.append([ref.split()])
    custom_hyps.append(custom_pred.split())
    gpt2_refs.append([ref.split()])
    gpt2_hyps.append(gpt2_pred.split())

    print(f"Input:          {inp}")
    print(f"Custom model:   {custom_pred}")
    print(f"GPT-2:          {gpt2_pred}")
    print(f"Expected:       {ref}")
    print("-" * 50)

# ── BLEU scores ────────────────────────────────────────────
custom_bleu = corpus_bleu(custom_refs, custom_hyps,
                           weights=(1, 0, 0, 0))
gpt2_bleu   = corpus_bleu(gpt2_refs, gpt2_hyps,
                           weights=(1, 0, 0, 0),
                           smoothing_function=smoothie)

print("\n=== Final Results ===")
print(f"Custom Transformer BLEU-1 : {custom_bleu:.4f}")
print(f"Fine-tuned GPT-2   BLEU-1 : {gpt2_bleu:.4f}")