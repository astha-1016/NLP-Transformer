# Seq2Seq Transformer Chatbot — From Scratch

Built a sequence-to-sequence transformer entirely in PyTorch without using `nn.Transformer`.  
Trained on the Cornell Movie Dialogs corpus and deployed with Streamlit.

## Demo
> Type a message → get a response + see attention weights as a heatmap

![Loss Curve](results/loss_curve.png)

## Architecture

| Component | Detail |
|-----------|--------|
| Model dim | 128 |
| Attention heads | 4 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Positional encoding | Sinusoidal |
| Activation | GELU |
| Decoding | Nucleus sampling (p=0.9) |
| Regularization | Dropout 0.1, Label smoothing 0.1 |

## What I built from scratch
- Multi-head self attention and cross attention
- Sinusoidal positional encoding
- Encoder and decoder stacks
- Custom tokenizer and vocabulary (Lexicon)
- Nucleus sampling inference
- Attention heatmap visualizer in Streamlit


## Dataset
- **Cornell Movie Dialogs Corpus** — 220,579 conversational exchanges from 617 movies
- Published in Danescu-Niculescu-Mizil & Lee, ACL 2011
- Filtered to 2,000 short pairs (2–20 words) for CPU training
- 80/20 train/test split

## Results

| Metric | Score |
|--------|-------|
| Final training loss | ~1.5 (epoch 150) |

## Model Comparison

| Model | BLEU-1 | Parameters | Training time |
|-------|--------|------------|---------------|
| Custom Transformer (from scratch) | 0.064 | ~2M | ~25 mins CPU |
| Fine-tuned GPT-2 (HuggingFace) | 0.025 | 117M | ~40 mins CPU |

> Custom transformer scores higher BLEU due to overfitting on short patterns.  
> GPT-2 produces more fluent and diverse responses qualitatively,  
> but BLEU penalizes diversity — a known limitation of the metric.  
> A semantic metric like BERTScore would be more appropriate for dialog evaluation.

## Limitations and what would improve it
- 2,000 training pairs is too small for generalization
- Increasing to 50,000+ pairs would significantly improve responses
- Beam search decoding would improve response quality over nucleus sampling
- BERTScore would be a better evaluation metric than BLEU for dialog

## Run locally
```bash
git clone https://github.com/YOURUSERNAME/nlp-transformer
cd nlp-transformer
pip install -r requirements.txt
python train.py
python gpt2_train.py
streamlit run app.py
```
## Project structure
```
nlp-transformer/
├── app.py                  # Streamlit UI with attention heatmap
├── train.py                # Custom transformer training
├── gpt2_train.py           # GPT-2 fine-tuning
├── gpt2_evaluate.py        # BLEU comparison of both models
├── gpt2_inference.py       # GPT-2 inference
├── models/
│   └── transformer.py      # Full architecture from scratch
├── utils/
│   ├── inference.py        # Nucleus sampling + attention output
│   └── data_loader.py      # Cornell corpus loader
├── data/
│   └── cornell/            # Raw corpus files
└── results/
    ├── loss_curve.png       # Custom transformer training loss
    └── gpt2_loss_curve.png  # GPT-2 fine-tuning loss
```

## Requirements
```
torch
streamlit
nltk
seaborn
matplotlib
numpy
transformers
accelerate
```
## References
- Vaswani et al. (2017) — Attention Is All You Need
- Danescu-Niculescu-Mizil & Lee (2011) — Cornell Movie Dialogs Corpus
- Radford et al. (2019) — GPT-2: Language Models are Unsupervised Multitask Learners