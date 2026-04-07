import torch
from nltk.tokenize import word_tokenize

def predict(sentence, model, vocab, device, max_len=25):
    model.eval()
    
    tokens = word_tokenize(sentence.lower())
    source = torch.tensor([vocab.encode(sentence)]).to(device)
    generated = [1]

    last_attention = None

    with torch.no_grad():
        for _ in range(max_len):
            target = torch.tensor([generated]).to(device)
            pred, attention = model(source, target)

            logits = pred[:, -1] / 0.9
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumulative - sorted_probs > 0.9] = 0
            sorted_probs /= sorted_probs.sum()

            nxt = sorted_idx[0][torch.multinomial(sorted_probs[0], 1)].item()
            last_attention = attention

            if nxt == 2:
                break
            generated.append(nxt)

    response = vocab.decode(generated[1:])
    return response, last_attention, tokens