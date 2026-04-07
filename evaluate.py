import torch
import pickle
import json
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from utils.inference import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.transformer import TransformerSystem
with open("models/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = TransformerSystem(vocab.total).to(device)
model.load_state_dict(torch.load("models/model_v2.pth", map_location=device))
model.eval()

with open("data/test_data.json") as f:
    test_data = json.load(f)

references = []
hypotheses = []
smoothie = SmoothingFunction().method4

print("\n--- Sample predictions ---\n")
for inp, ref in test_data[:20]:
    pred, _, _ = predict(inp, model, vocab, device)
    references.append([ref.split()])
    hypotheses.append(pred.split())
    print(f"Input:     {inp}")
    print(f"Predicted: {pred}")
    print(f"Expected:  {ref}")
    print("-" * 40)

# BLEU-1 and BLEU-2 are more meaningful for short responses
bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
bleu4 = corpus_bleu(references, hypotheses,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothie)

print(f"\nBLEU-1: {bleu1:.4f}")
print(f"BLEU-2: {bleu2:.4f}")
print(f"BLEU-4 (smoothed): {bleu4:.4f}")