import torch
from model import GPT, GPTConfig
from tokenizer import DataLoaderLite
import sentencepiece as spm
import math

sp = spm.SentencePieceProcessor()
sp.load("c:/Users/User/Desktop/gpt-2/sentencepiece/token_model.model")


def get_lr(step, max_lr=6e-4, min_lr=6e-5, warmup_steps=10, max_steps=50):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T = 16, 128
    total_batch_size = 524288
    grad_accum_steps = total_batch_size // (B * T)

   



    train_loader = DataLoaderLite(B, T, sp)


    model = GPT(GPTConfig()).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)

    max_steps = 50

    for step in range(max_steps):
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        print(f"Step {step} | Loss: {loss.item():.4f}")

    prompt = "Azərbaycanın paytaxtı"
    ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    generated = model.generate(input_ids, max_new_tokens=50)
    output_text = sp.decode(generated[0].tolist())
    print(output_text)

if __name__ == '__main__':
    main()


   