import torch
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.inference import predict, predict_beam  # ← added predict_beam

# ── Load model ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    from models.transformer import TransformerSystem
    with open("models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    model = TransformerSystem(vocab.total).to(device)
    model.load_state_dict(torch.load("models/model_v2.pth", map_location=device))
    model.eval()
    return model, vocab

model, vocab = load_model()

# ── UI ────────────────────────────────────────────────────
st.title("Transformer Chatbot")
st.caption("Seq2seq transformer built from scratch · Trained on Cornell Movie Dialogs")

if "chat" not in st.session_state:
    st.session_state.chat = []

# ── NEW: Decoding mode selector ───────────────────────────
mode = st.radio("Decoding mode:", ["Nucleus Sampling", "Beam Search"], horizontal=True)

user_input = st.text_input("You:", key="input")

if user_input:
    # ── NEW: Choose decoding based on mode ────────────────
    if mode == "Beam Search":
        response = predict_beam(user_input, model, vocab, device)
        attention = None
        tokens = user_input.split()
    else:
        response, attention, tokens = predict(user_input, model, vocab, device)

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", response))

    # ── Attention heatmap (only for Nucleus Sampling) ─────
    if attention is not None:
        st.subheader("Attention weights")  # ← fixed indent
        try:
            gen_tokens = response.split() if response.strip() else ["<response>"]
            attn = attention[0].mean(dim=0).cpu().detach().numpy()

            min_rows = min(attn.shape[0], len(gen_tokens))
            min_cols = min(attn.shape[1], len(tokens))
            attn = attn[:min_rows, :min_cols]
            gen_tokens = gen_tokens[:min_rows]
            tokens_trimmed = tokens[:min_cols]

            fig, ax = plt.subplots(figsize=(max(4, len(tokens_trimmed)), max(3, len(gen_tokens))))
            sns.heatmap(attn, xticklabels=tokens_trimmed, yticklabels=gen_tokens,
                        cmap="YlOrRd", ax=ax, linewidths=0.5)
            ax.set_xlabel("Input tokens")
            ax.set_ylabel("Output tokens")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.info(f"Attention map unavailable: {e}")

# ── Chat history ──────────────────────────────────────────
st.subheader("Conversation")
for sender, msg in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")