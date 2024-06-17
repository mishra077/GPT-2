from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------------#
### x = x + f(x)
    
    
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        
        # key, query, value projections for all heads.
        self.c_attn = nn.Linear(config.n_embed , 3 * config.n_embed, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y
        
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4, bias = config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias = config.bias)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
"""
    x ----> LayerNorm ---->  Attention ----->  y --------> LayerNorm --------> MLP ------> Output
    |                                       |  |                                |
    |                                       |  |                                |
    ----------------------------------------   ---------------------------------
"""

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size:int = 50257
    n_layer:int = 12
    n_head:int = 12
    n_embed:int = 768
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster.

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed, bias= config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)
    
    def forward(self, idx):
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embed)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embed)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
     
    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model weights from hugging face"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)
        
        # n_layer, n_head and n_embed are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head = 12, n_embed=768),         # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024),           # 350M params
            'gpt2-large':   dict(n_layer=36, n_head =20, n_embed=1280),          # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600),           # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked.bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys : {len(sd_keys_hf) != len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        return model
        
        
# ---------------------------------------------------------------------------------------------------------------------------------
num_return_sequences = 5
max_length = 50

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')


import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
"""

tensor([[15496,    11,   314,  1101,   257,  3303,  2746],
        [15496,    11,   314,  1101,   257,  3303,  2746],
        [15496,    11,   314,  1101,   257,  3303,  2746],
        [15496,    11,   314,  1101,   257,  3303,  2746],
        [15496,    11,   314,  1101,   257,  3303,  2746]])

"""
x = tokens.to('cuda')

B, T = x.shape # {B: 5, T: 7}
print(f"shape of the input: {B}, {T}")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length: # 7->30
    with torch.no_grad():
        logits = model(x) # {num_return_sequences, T, vocab_size}
        logits = logits[:, -1, :] # taking last column's logit which is of size (1, 50257)
        probs=  F.softmax(logits, dim = -1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim = -1) # prob with their indices
        # topk_indices : shape {num_return_sequences, 50}
        ix = torch.multinomial(topk_probs, 1)
        # shape {B: num_return_sequences, 1} 
        xcol = torch.gather(topk_indices, -1, ix) # shape : {B: num of samples, 1}
        x = torch.cat((x, xcol), dim = 1) 

# final shape of x : {num_return_sequences, max_length}

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    print(tokens)
    decoded = enc.decode(tokens)
    print(">", decoded)