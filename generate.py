from model import GPT, GPTConfig
import torch
import tiktoken

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(GPTConfig())
model.load_state_dict(torch.load('trained_model.pt'))  # tren etdikdən sonra
model = model.to(device)
model.eval()

enc = tiktoken.get_encoding('gpt2')
prompt = "Bu gün hava çox"
ids = enc.encode(prompt)
input_ids = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # (1, T)

generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
out = enc.decode(generated[0].tolist())

print(out)
