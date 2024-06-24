from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import inspect
# -----------------------------------------------------------------------------------#

wandb.init(
    # set the wandb project where this run will be logged
    project="GPT2",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 3e-4,
    "architecture": "GPT2",
    "dataset": "tinyshakespeare",
    "batch_size": 4,
    "epochs": 50,
    }
)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim = -1)
        # y = att @ v
        
        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y
        
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4, bias = config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias = config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        
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

def ce(y, logits):
    
    """
    
    Here is the implementation of cross-entropy function
    
    In pytorch, we pass logits (unormalized final scores of your model) and target classes
    Now for next token prediction for traget we have indices of vocab, which indicates at what position we have which tokens.
    
    Cross Entropy function is basically applying softmax function with Negative Log Likelihood.
    But logits can be very large which can make their exponents even larger, thus for numerical stability and faster calculation,
    pytorch has adopted log softmax 
    
    softmax(x) = exp(x_i) / sigma(exp(x_j)) 
               => exp(x_i - b) * exp(b) / sigma(exp(x_j - b) * exp(b))
               => exp(x_i - b) / sigma(exp(x_j - b)) , where b = max(x)

    log(softmax(x)) = log(exp(x_i) / sigma(exp(x_j)))
                    => log(exp(x_i)) - log(sigma(exp(x_j)))
                    => x_i - log(sigma(exp(x_j)))
                    => x_i - log(sigma(exp(x_j - c) * exp(c)))
                    => x_i - log(sigma(exp(x_j - c))) - log(exp(c))
                    => x_i - c - log(sigma(exp(x_j - c))), where c = max(x)
                    
                    
    After computing log softmax, we computer negative log likelihood
    Now given log_probs from log-softmax we need to find the vocab prob wrt target class.
    log_probs.gather(1, y.unsqueeze(1)) -> reteriving the value of probability coressponding to its target index which is y
    now take the negative mean for all the batches and thats you nn.CrossEntropy loss function.
    
    """
    
    
    with torch.no_grad():
        
        logits = logits.view(-1, logits.size(-1)) # shape = [B * T, vocab_size]
        y = y.view(-1)
    
        batch_size, num_classes = logits.shape

        # Compute log softmax
        logits_max = logits.max(dim=1, keepdim=True)[0]  # Compute max along class dimension, shape = [B*T, 1]
        
        logits_exp = torch.exp(logits - logits_max)  # Subtract max for numerical , shape = [B*T, vocab_size]
        logits_sum = logits_exp.sum(dim=1, keepdim=True)  # Sum along class dimension, shape = [B* T, 1]
        log_probs = logits - logits_max - torch.log(logits_sum)  # Log softmax shape = [B*T, vocab_size]
        
        # Compute negative log likelihood
        log_probs_targets = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)  # Gather log probs for targets, shape = [B*T, 1]
        
        nll_loss = -log_probs_targets.mean()  # Compute negative log likelihood loss
    
        print(nll_loss)
        
        # log_probs = F.log_softmax(logits, dim=1)  # Compute log softmax along class dimension
        # nll_loss = F.nll_loss(log_probs, y, reduction='mean')  # Compute negative log likelihood loss
    
    # print(nll_loss)


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
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = (2 * self.config.n_layer) ** -0.5 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
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
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that required grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed otherwise No.
        # i.e., all wt tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tenosrs: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        """
        
        Fused AdamW optimizer is an optimized implementation of the AdamW (Adam with weight decay) algorithm that aims to improve performance, especially on GPUs. 
        
        Here are the key points about fused AdamW:
            1. Performance Improvement: Fused AdamW is designed to be faster than the standard AdamW implementation, particularly for large models and when training on GPUs.
            2. Fusion Techniques:
                It fuses multiple operations into a single CUDA kernel, reducing memory bandwidth and improving computational efficiency.
                Specifically, it implements two main fusions:
                    a) Fusion of the Adam update's elementwise operations.
                    b) A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches
        
        """
        return optimizer
        
        
# ---------------------------------------------------------------------------------------------------------------------------------

import tiktoken
import numpy as np
enc = tiktoken.get_encoding('gpt2')
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}, "split can be either train or val"
        
        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        # with open('input.txt', 'r') as f:
        #     text = f.read()
        self.reset()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"Loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} samples")
        
        
        # state, init at shard zero
        # self.current_position = self.B * self.T * self.process_rank
        # self.current_shard = 0
        # self.tokens = load_tokens(self.shards[self.current_shard])
    
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B,T) # inputs
        y = buf[1:].view(B, T) # targets
        
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if(self.current_position + B * T * self.num_processes + 1 > len(self.tokens)):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
            
        return x, y

# ---------------------------------------------------------------------------------------------------------------------------------
import time
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# set up the DDP (Distributed Data Parallel)
# torch run command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to the RANK
    assert torch.cuda.is_available(), "CUDA is required for DDP at the moment!"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging and checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    
    # attempt to autodetect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps" # for mac books
    print(f"using device: {device}")
    



torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B=2 # micro batch size per gpu
T=1024 # sequence length per gpu
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps= total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# get a data batch
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# set the mat mul precision
torch.set_float32_matmul_precision("high")

# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

"""
In distributed data parallelism, the model is repleated to all GPUs available.
Now during forward pass, the calculations are identical.
But, during back-propogation the gradient cacluated in all the GPU will be taken out
to calculate the avg gradient using ALL-REDUCE method. The avg gradients will be broadcasted
again to all GPUs for weight updation.

 In distributed data parallelism, particularly in frameworks like PyTorch's DistributedDataParallel (DDP), the optimization process involves synchronizing gradients across multiple GPUs. This synchronization is typically done using an operation called All-Reduce. Here's a detailed explanation of how this works and how it can be optimized:
Gradient Synchronization in Distributed Data Parallelism
Gradient Calculation:
Each GPU computes the gradients for its portion of the data independently.
All-Reduce Operation:
After computing the gradients, an All-Reduce operation is performed to aggregate these gradients across all GPUs.
The All-Reduce operation sums the gradients from all GPUs and then distributes the summed gradients back to each GPU.
Each GPU then divides the summed gradients by the number of GPUs to get the average gradient.
Optimization by Overlapping Computation and Communication
To optimize this process, the gradient synchronization can be overlapped with the backward pass computation. This is often referred to as gradient bucketing or bucketed all-reduce. Here's how it works:
Layer-wise Gradient Calculation:
As each layer's gradients are computed during the backward pass, they are immediately scheduled for the All-Reduce operation.
This means that instead of waiting for the entire backward pass to complete, the gradients for earlier layers can start being synchronized while later layers are still being computed.
Bucketed All-Reduce:
Gradients are grouped into buckets, and the All-Reduce operation is performed on these buckets as soon as they are ready.
This reduces the idle time for GPUs and overlaps communication with computation, leading to better utilization of resources and reduced overall training time.
Benefits
Reduced Latency: By overlapping gradient synchronization with gradient computation, the overall latency is reduced.
Improved Resource Utilization: GPUs spend less time waiting for synchronization to complete, leading to better utilization of computational resources.
Scalability: This approach scales better with the number of GPUs, as the communication overhead is distributed throughout the backward pass rather than being concentrated at the end.

"""
if ddp:
    model = DDP(model , device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model
wandb.watch(raw_model)
# logits, loss = model(x, y)

# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

def get_lr(it):
    # 1. linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    # 2. if it > lr_decay_iters, return minimum learning rate
    if it > max_steps:
        return min_lr
    
    # 3. in between use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    """
    1. The Cosine Function:
        > The cosine function cos(x) oscillates between -1 and 1.
        > When x = 0, cos(x) = 1
        > When x = π (pi), cos(x) = -1
    2. The Role of decay_ratio:
        > decay_ratio goes from 0 to 1 as training progresses.
        > When multiplied by π, it scales the input to the cosine function from 0 to π.
    3. The Math.cos(math.pi * decay_ratio) Part:
        > At the start of decay (decay_ratio = 0): cos(0) = 1
        > At the end of decay (decay_ratio = 1): cos(π) = -1
    4. The 1.0 + ... Part:
        > This shifts the cosine output from the range [-1, 1] to [0, 2]
    5. The 0.5 * ... Part:
        > This scales the result to the range [0, 1]
    """
    
    return min_lr + coeff * (max_lr - min_lr)

# optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas= (0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device = device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    # once in a while evaluate our validation loss
    if step > 0 and step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
    
        if ddp:
            dist.all_reduce(val_loss_accum, op = dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss : {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
    
    # once in a while generate from the model (except step 0, which is noisy)
    if step > 0 and step % 100 == 0 and False:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model, ")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    
    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    # Gradient Accumulation Step for ~0.5M batch size
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation
        # because the gradients just add on each successive backward()
        # addition of gradients coressponds to a SUM in the objective, but
        # intead of SUM we want MEAN. Scale the loss here so it comes out right. 
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) 
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine the lr and set it
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the gpu to finish the work
    t1 = time.time()
    dt = (t1 - t0)*1000 # time diff in secs
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    wandb.log({
        "loss": loss_accum.item(),
        "time_elapsed": dt,
        "gpu_memory": torch.cuda.max_memory_allocated(device) / 1024 ** 2  # in MB
    })
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | lr: {lr:.4e} | time elapsed: {dt:.2f} ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

"""
>>> logits.dtype
torch.bfloat16
>>> model.transformer.wte
Embedding(50257, 768)
>>> model.transformer.wte.weight
Parameter containing:
tensor([[ 9.9743e-05,  1.6065e-03,  1.6118e-02,  ..., -2.3506e-02,
         -9.5092e-03,  8.6977e-04],
        [ 5.6103e-03, -8.9315e-04,  3.3858e-02,  ...,  1.8497e-02,
         -1.2024e-02,  4.2558e-03],
        [ 1.2853e-02,  8.0832e-03,  1.8367e-02,  ..., -2.2407e-02,
         -1.2174e-02, -1.2083e-02],
        ...,
        [ 6.8869e-03,  1.8946e-02,  2.7229e-02,  ..., -9.2498e-03,
         -1.6403e-02,  1.1806e-02],
        [-6.4153e-03,  6.4614e-03, -1.8471e-02,  ...,  3.3779e-04,
          8.5628e-03, -4.6225e-03],
        [ 4.5271e-03, -2.1883e-02,  2.6784e-02,  ..., -4.7267e-03,
         -1.2253e-02,  2.1918e-02]], device='cuda:0', requires_grad=True)
>>> model.transformer.wte.weight.dtype
torch.float32



"""

import sys; sys.exit(0)
num_return_sequences = 5
max_length = 50

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)

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
x = tokens.to(device)

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