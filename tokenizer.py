import torch
class DataLoaderLite:
    def __init__(self, B, T, sp):  
        self.B = B
        self.T = T
        
        text = "salam Ümid edirəm hər şey qaydasindadır" * 50000  
        tokens = sp.encode(text, out_type=int)  
        
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f'loaded: {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens)//(B*T)} batches')
        
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y
