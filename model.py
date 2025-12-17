import torch
import torch.nn as nn
import math
from tqdm import tqdm
from dataclasses import dataclass
import torch.nn.functional as F


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1);  (seq_len,) * (d_model/2,) --> not_work
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # x: (batch, seq_len, d_model)
        return self.dropout(x)



class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter(multiplicative)
        # shape: (d_model,); d_models == features
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter(additive)
        # shape: (d_model,)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        # (batch, seq_len, d_model)


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1: (d_ff, d_model) and b1: (d_ff,)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2: (d_model, d_ff) and b2: (d_model,)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        # x1 = xw1.T + b1 --> (batch, seq_len, d_model) * (d_model, d_ff) + (d_ff,) --> (batch, seq_len, d_ff)
        # x2 = x1W2.T + b2 --> (batch, seq_len, d_ff) * (d_ff, d_model) + (d_model,) --> (batch, seq_len, d_model)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, h:int, d_model:int, dropout:float):
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        # query, key, value: (batch, h, seq_len, d_k)
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # attention_score: (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        # (batch, h, seq_len, seq_len)
        # return attention scores which can be used for visualization
        


    def forward(self, q, k, v, mask):
        # q, k, v: (batch, seq_len, d_model)
        query = self.linear_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.linear_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.linear_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # query, key, value: (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # x: (batch, h, seq_len, d_k)
        # attention_score: (batch, h, seq_len, seq_len)

        # Combine all the heads together
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # x: (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)

        return self.linear_o(x)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)



class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            # x: (batch, seq_len, d_model)
            return x + self.dropout(sublayer(self.norm(x)))
            # (batch, seq_len, d_model)




class GPTBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        # x: (batch, tgt_len, d_model)
        # self-attention
               # Q = x: (batch, seq_len, d_model)
               # K = x: (batch, seq_len, d_model)
               # V = x: (batch, seq_len, d_model)
               # output: (batch, seq_len, d_model)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        # x: (batch, seq_len, d_model)
        
        x = self.residual_connections[1](x, self.feed_forward_block)
        # x: (batch, tgt_len, d_model)
        return x


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        # self.proj: (d_model -> vocab_size)

    def forward(self, x):
        # x: (batch, tgt_len, d_model)
        return self.proj(x)
        # (batch, tgt_len, vocab_size)


    
@dataclass
class Config:
    vocab_size: int = 1000
    embed_size: int = 512
    seq_len: int = 200
    h: int = 4
    d_ff: int = 2048
    n_layer: int = 6
    dropout: float = 0.1
    weight_decay: float = 0.2
    total_epochs: int = 1000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr: float = 0.001

class GPT(nn.Module):
    def __init__(self, config, input_embedding: InputEmbeddings, pos_encoding: PositionalEncoding, gpt_block: nn.ModuleList, proj_layer: ProjectionLayer):
        super().__init__()
        self.config = config
        self.input_emb = input_embedding
        self.pos_en = pos_encoding
        self.gpt_block = gpt_block
        self.proj = proj_layer
        self.final_norm = LayerNormalization(config.embed_size)

        self.apply(self.initialize_weights)
        
        self.proj.proj.weight = self.input_emb.embedding.weight


    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, y=None, mask=None):
        # x: (batch, seq_len)
        # mask: (1, 1, seq_len, seq_len)
        # Automatic Mask Generation
        if mask is None:
            # Create a triangle mask for the current sequence length
            seq_len = x.size(1)
            mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=x.device)).bool()
            
        x = self.input_emb(x)
        # x: (batch, seq_len, embed_dm)
        x = self.pos_en(x)
        # x: (batch, seq_len, embed_dm)

        for block in self.gpt_block:
            x = block(x, mask)
            # x: (batch, seq_len, embed_dm)
        x = self.final_norm(x)
        # x: (batch, seq_len, embed_dm)
        logits = self.proj(x)
        # logits: (batch, seq_len, embed_dim)

        loss = None
        if y is not None:
            # F.cross_entropy requires flattened shapes:
            # logits: (Batch * Seq_Len, Vocab_Size)
            # targets: (Batch * Seq_Len)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))

        return logits, loss
            


    def configure_optimizer(self):
        config = self.config
        train_params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in train_params.values() if p.dim() >= 2]
        non_decay_params = [p for p in train_params.values() if p.dim() < 2]

        opt_group = [{"params": decay_params, "weight_decay": config.weight_decay}, {"params": non_decay_params, "weight_decay": 0.0}]

        optimizer = torch.optim.AdamW(opt_group, lr = config.lr)
        return optimizer


    def train_gpt(self, train_loader):
        config = self.config
        self.to(config.device)
        self.train()
        
        optimizer = self.configure_optimizer()

        for epoch in range(config.total_epochs):
            total_loss = 0.0
            pbar = tqdm(train_loader, desc = f"{epoch}/{config.total_epochs}")
            for batch_idx, batch in enumerate(pbar):
                x, y = batch
                x, y = x.to(config.device), y.to(config.device)
    
                optimizer.zero_grad()
                logits, loss = self(x, y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())

                

            avg_loss = total_loss / len(train_loader)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}/{config.total_epochs} Loss: {avg_loss}")

    @torch.no_grad()
    def generate(self, idx, max_new_token: int, top_k: int = None):
        self.eval()
        # idx: (batch, seq_len)
        for _ in range(max_new_token):
            idx_filter = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
            logits, _ = self(idx_filter)
            # logits: (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]
            # logits: (batch, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # v: (batch, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim = -1)
            # probs: (batch, vocab_size)
            next_idx = torch.multinomial(probs, num_samples=1)
            # nex_idx: (batch, 1)
            idx = torch.cat((idx, next_idx), dim = 1)
            # idx: (batch, seq_len + 1)
        return idx
    


    @classmethod
    def build_gpt(cls, config):
        input_embed = InputEmbeddings(config.embed_size, config.vocab_size)
        pos_enc = PositionalEncoding(config.embed_size, config.seq_len, config.dropout)
        proj_layer = ProjectionLayer(config.embed_size, config.vocab_size)

        block = []
        for _ in range(config.n_layer):
            feed_forwd = FeedForwardBlock(config.embed_size, config.d_ff, config.dropout)
            self_attn = MultiHeadAttentionBlock(config.h, config.embed_size, config.dropout)
            block.append(GPTBlock(config.embed_size, self_attn, feed_forwd, config.dropout))

        return GPT(config, input_embed, pos_enc, nn.ModuleList(block), proj_layer)



