import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_finetuned")
    model     = GPT2LMHeadModel.from_pretrained("models/gpt2_finetuned")
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return model, tokenizer

def predict_gpt2(sentence, model, tokenizer, max_new_tokens=20):
    prompt = f"input: {sentence} response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # extract only the response part
    if "response:" in decoded:
        response = decoded.split("response:")[-1].strip()
    else:
        response = decoded.strip()

    return response