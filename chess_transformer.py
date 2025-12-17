import torch
from torch import nn
import math


class TransformerLayer(nn.Module):
    """Almost identical architecture as ViT"""
    def __init__(self, emb_dim=512, head_count=4, mlp_hidden_size=2048):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_count = head_count

        if emb_dim % head_count != 0:
            raise ValueError("emb_dim must be divisible by head_count")
        self.head_dim = self.emb_dim // self.head_count

        # layer norm before self attention
        self.first_layer_norm = nn.LayerNorm(self.emb_dim)

        self.q = nn.Linear(self.emb_dim, self.emb_dim)
        self.k = nn.Linear(self.emb_dim, self.emb_dim)
        self.v = nn.Linear(self.emb_dim, self.emb_dim)

        # output matrix of the self attention block
        self.out = nn.Linear(self.emb_dim, self.emb_dim)

        # layer norm before mlp
        self.second_layer_norm = nn.LayerNorm(self.emb_dim)

        # two linear transformations with gelu as mlp layer
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, self.emb_dim),
        )

    def forward(self, emb: torch.Tensor):
        is_batched = emb.dim() == 3
        if not is_batched:
            emb = emb.unsqueeze(0)
        # batch size, token count, emb_dim
        B, N, D = emb.shape

        # layer norm before self attention
        normed_emb = self.first_layer_norm(emb)

        # q, k, v shapes [B, Token count, self.emb_dim]
        q = self.q(normed_emb)
        k = self.k(normed_emb)
        v = self.v(normed_emb)

        # reshape for multihead - add head id as an extra "batch" dimension
        # resulting shape [B, head_count, token count, head_dim]
        q = q.view(B, N, self.head_count, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.head_count, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.head_count, self.head_dim).transpose(1, 2)

        # calculate scaled dot product similarity (scale by sqrt of embedding size (d))
        # q is shape [B, Token count, self.emb_dim], we multiply by kT - [B, self.emb_dim, token count]
        similarities = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        # similarities shape [B, token count, token count]
        # similarities in [..., i, j] is similarity of query i to key j -> softmax along -1
        softmaxes = nn.functional.softmax(similarities, dim=-1)
        # sum(softmaxes[b, i, :]) is now 1 for every b,i
        # every row has the softmax values for that query
        attention_adjusted_vals = softmaxes @ v

        # reshape back from multiple heads - "transpose back and flatten across heads"
        # [B, head_count, N, head_dim] -> [B, N, D]
        attention_adjusted_vals = attention_adjusted_vals.transpose(1, 2).contiguous().view(B, N, D)

        mha_out = self.out(attention_adjusted_vals)
        # residual connection
        mha_out += emb

        # layer norm before mlp
        out = self.second_layer_norm(mha_out)

        out = self.mlp(out)
        # residual connection
        out += mha_out

        if not is_batched:
            out = out.squeeze(0)

        return out


class ChessTransformer(nn.Module):
    # TODO: add some dropouts 0.1 and other potential enhancements
    def __init__(self, in_size, emb_dim=512, layer_count=4, head_hidden_dim=2048):
        """
        ViT-like trasnformer for chess

        in_size doesnt include position feature (it has to be the last one)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.max_tokens = 64  # 64 chess squares
        self.embedding = nn.Linear(in_size, self.emb_dim)

        # embeddings for piece positions on the chess board
        self.positional_encoding = nn.Embedding(self.max_tokens, self.emb_dim)

        # CLS token - use lower variance distribution (0.05)
        self.cls_token_embedding = nn.Parameter(torch.randn(self.emb_dim) * 0.05)

        self.multihead_layers = nn.Sequential(*(TransformerLayer(self.emb_dim) for _ in range(layer_count)))

        # layer norm for the result in the cls token
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.head = nn.Sequential(
            nn.Linear(emb_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )


    def forward(self, x: torch.Tensor()):
        # expects the last feature to be the position
        positions = x[...,-1].int()
        x = x[...,:-1]
        # embed x
        emb = self.embedding(x)
        # add positional encoding
        positional_encoding = self.positional_encoding(positions)

        # resulting embedding is embedding plus positional encoding
        emb += positional_encoding

        cls_token = self.cls_token_embedding
        cls_token = cls_token.unsqueeze(0)
        if x.ndim == 3:
            # batched
            cls_token = cls_token.unsqueeze(0)
            cls_token = cls_token.expand(x.shape[0], -1, -1)

        emb = torch.cat([cls_token, emb], dim=-2)

        #TODO: dropout 0.1 probably goes here

        # apply all the MSA+MLP layers
        attention_layers_out = self.multihead_layers(emb)

        # take only the cls token output for result
        cls_out = attention_layers_out[..., 0, :]
        cls_out = self.layer_norm(cls_out)

        out = self.head(cls_out)
        return out


if __name__ == "__main__":
    trans = ChessTransformer(in_size=1)

    x = torch.Tensor([[1, 3], [2, 4]])
    print(x)
    print(trans(x))
