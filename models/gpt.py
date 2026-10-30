"""A small decoder-only Transformer (GPT-2 style, pre-LayerNorm).

q/k/v/output projections are separate Linear layers so that Muon/Shampoo see
each projection as its own matrix. `tok_emb`, `pos_emb` and `head` are the
AdamW-handled input/output layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        shape = (B, T, self.n_head, C // self.n_head)
        q, k, v = (f(x).view(shape).transpose(1, 2) for f in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.fc_out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc_out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    AUX_PREFIXES = ("tok_emb.", "pos_emb.", "head.")

    def __init__(self, vocab_size, block_size=256, n_layer=6, n_head=6, n_embd=384):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 scaled init of residual projections
            if name.endswith(("proj.weight", "fc_out.weight")):
                nn.init.normal_(p, std=0.02 / (2 * n_layer) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
