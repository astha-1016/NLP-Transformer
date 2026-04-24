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


def predict_beam(sentence, model, vocab, device, beam_width=3, max_len=25):
    model.eval()
    source = torch.tensor([vocab.encode(sentence)]).to(device)
    
    beams = [([1], 0.0)]
    completed = []

    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                target = torch.tensor([seq]).to(device)
                pred, _ = model(source, target)
                logits = pred[:, -1]
                log_probs = torch.log_softmax(logits, dim=-1)[0]

                top_probs, top_ids = torch.topk(log_probs, beam_width)
                for prob, idx in zip(top_probs, top_ids):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    if idx.item() == 2:
                        completed.append((new_seq, new_score))
                    else:
                        new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if not beams:
                break

    all_candidates = completed if completed else beams
    best_seq, _ = max(all_candidates, key=lambda x: x[1])
    return vocab.decode(best_seq[1:])