from fastapi import FastAPI
from pydantic import BaseModel
import torch
import sentencepiece as spm
from model import GPT, GPTConfig


sp = spm.SentencePieceProcessor()
sp.load('sentencepiece/token_model.model')


device = "cuda" if torch.cuda.is_available() else "cpu"
config = GPTConfig(vocab_size=sp.get_piece_size())
model = GPT(config).to(device)
model.eval()  

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

@app.post("/generate")
def generate_text(request: GenerateRequest):
    tokens = sp.encode(request.prompt, out_type=int)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=request.max_new_tokens)
    generated_tokens = output_ids[0].tolist()
    generated_text = sp.decode(generated_tokens)
    return {"generated_text": generated_text}
