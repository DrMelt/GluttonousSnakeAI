import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparameters
block_side = 10  # what is the maximum context length for predictions?
n_embd = 32
n_head = 8
n_layer = 8
dropout = 0.0
# ------------


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.register_buffer(
        #     "tril", torch.tril(torch.ones(block_size, block_size)).bool()
        # )

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T], float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(head_size * num_heads, n_embd, bias=False)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # out = self.dropout(out)
        out = self.proj(out)
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        # self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        # self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        # ln1 = self.ln1(x)
        sa = x + self.sa(x)  # 向原特征向量添加修饰
        # ln2 = self.ln2(x)
        x = x + self.ffwd(sa)  # 从新向量提取信息
        return x


class SnakeTransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.register_buffer("arange", torch.arange(block_side * block_side))
        self.position_embedding_table = nn.Embedding(block_side * block_side, n_embd)
        self.sequence_embedding_table = nn.Embedding(
            block_side * block_side + 1, n_embd
        )
        self.food_embedding_table = nn.Embedding(2, n_embd)

        self.blocks = nn.Sequential(
            *[AttentionBlock(n_embd, n_head=n_head) for _ in range(n_layer)],
            # nn.LayerNorm(n_embd)
        )

        self.quality_out = nn.Sequential(
            nn.Conv1d(n_embd, n_embd * 4, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(n_embd * 4, 4, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=block_side * block_side),
        )
        self.value_out = nn.Sequential(
            nn.Conv1d(n_embd, n_embd * 4, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(n_embd * 4, 1, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=block_side * block_side),
        )

        self.apply(self._init_weights)
        print(sum(p.numel() for p in self.parameters()) / 1e3, "K parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(
        self,
        snake_states: torch.Tensor,
        food_states: torch.Tensor,
    ):
        snake_states = snake_states.view(-1, block_side * block_side)
        food_states = food_states.view(-1, block_side * block_side)

        snake_emb = self.sequence_embedding_table(snake_states)
        food_emb = self.food_embedding_table(food_states)

        pos_emb = self.position_embedding_table(self.arange)  # (T,C)

        embed = snake_emb + food_emb + pos_emb  # (B,T,C)

        x = self.blocks(embed)  # (B,T,C)
        x = x.permute(0, 2, 1)  # (B,C,T)
        quality_out = self.quality_out(x).view(-1, 4)
        value_out = self.value_out(x).view(-1, 1)

        return quality_out, value_out

    def batch_out(
        self,
        snake_states_list,
        food_states_list,
        batch_size=512,
    ):
        total_length = len(snake_states_list)
        snake_states_array = np.array(snake_states_list)
        food_states_array = np.array(food_states_list)

        out_arrays = [None for _ in range(2)]
        start_idx = 0
        while start_idx < total_length:
            end_idx = min(start_idx + batch_size, total_length)

            snake_states_tensor = torch.tensor(
                snake_states_array[start_idx:end_idx],
                dtype=torch.int64,
                device=device,
            )
            snake_states_tensor = snake_states_tensor.view(
                -1, 1, block_side, block_side
            )

            food_states_tensor = torch.tensor(
                food_states_array[start_idx:end_idx],
                dtype=torch.int64,
                device=device,
            )
            food_states_tensor = food_states_tensor.view(-1, 1, block_side, block_side)

            batch_outs = self(
                snake_states_tensor,
                food_states_tensor,
            )

            # if not isinstance(batch_outs, tuple):
            #     batch_outs = [batch_outs]

            for i in range(2):
                out_cpu = batch_outs[i].cpu().numpy()

                if out_arrays[i] is None:
                    out_arrays[i] = out_cpu
                else:
                    out_arrays[i] = np.concatenate((out_arrays[i], out_cpu), axis=0)

            start_idx = end_idx

        return out_arrays


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_size = n_embd // n_head

        self.wei = nn.Sequential(
            nn.Conv2d(n_embd, n_head, kernel_size=1, bias=False),
            nn.Softmax(dim=-1),
        )
        self.attention = nn.ModuleList(
            nn.Conv2d(n_embd, self.head_size, kernel_size=1, bias=False)
            for _ in range(n_head)
        )

        self.value = nn.Linear(self.head_size * n_head, n_embd, bias=False)

        self.residual1 = nn.Sequential(
            nn.Conv2d(n_embd, n_embd * 2, kernel_size=1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(n_embd * 2, n_embd, kernel_size=1, bias=False),
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(n_embd, n_embd, kernel_size=5, padding=2, bias=False),
            nn.Conv2d(n_embd, n_embd * 2, kernel_size=1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(n_embd * 2, n_embd, kernel_size=1, bias=False),
        )

    def forward(self, x):
        attention = torch.cat(
            [
                att(x).view(-1, 1, self.head_size, block_side, block_side)
                for att in self.attention
            ],
            dim=1,
        )  # (B,n_head,hs,bs,bs)
        wei = self.wei(x).view(
            -1, n_head, 1, block_side, block_side
        )  # (B,n_head,1,bs,bs)

        value: torch.Tensor = (wei * attention).view(
            -1, self.head_size * n_head, block_side * block_side
        )  # (B,n_head*hs,T)

        value = value.sum(dim=-1)  # (B,n_head*hs)
        value = self.value(value).view(-1, n_embd, 1, 1)  # (B,C,1,1)

        residual1 = self.residual1(x + value)
        residual2 = self.residual2(x + residual1)
        return x + residual2


class SnakeConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("arange", torch.arange(block_side * block_side))
        self.position_embedding_table = nn.Embedding(block_side * block_side, n_embd)
        self.sequence_embedding_table = nn.Embedding(
            block_side * block_side + 1, n_embd
        )
        self.food_embedding_table = nn.Embedding(2, n_embd)

        self.blocks = nn.Sequential(
            *[ConvBlock() for _ in range(n_layer)],
        )

        self.quality_out = nn.Sequential(
            nn.Conv1d(n_embd, n_embd * 4, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(n_embd * 4, 4, kernel_size=1, bias=False),
        )
        self.quality_out_wei = nn.Sequential(
            nn.Conv1d(n_embd, 1, kernel_size=1, bias=False),
            nn.Softmax(dim=-1),
        )

        self.value_out = nn.Sequential(
            nn.Conv1d(n_embd, n_embd * 4, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(n_embd * 4, 1, kernel_size=1, bias=False),
        )
        self.value_out_wei = nn.Sequential(
            nn.Conv1d(n_embd, 1, kernel_size=1, bias=False),
            nn.Softmax(dim=-1),
        )

        self.apply(self._init_weights)
        print(sum(p.numel() for p in self.parameters()) / 1e3, "K parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(
        self,
        snake_states: torch.Tensor,
        food_states: torch.Tensor,
    ):
        snake_states = snake_states.view(-1, block_side * block_side)
        food_states = food_states.view(-1, block_side * block_side)

        snake_emb = self.sequence_embedding_table(snake_states)
        food_emb = self.food_embedding_table(food_states)
        pos_emb = self.position_embedding_table(self.arange)  # (T,C)
        embed = snake_emb + food_emb + pos_emb  # (B,T,C)

        embed = embed.permute(0, 2, 1)  # (B,C,T)
        embed = embed.view(-1, n_embd, block_side, block_side)
        x = self.blocks(embed)  # (B,C,bs,bs)
        x = x.view(-1, n_embd, block_side * block_side)  # (B,C,T)

        quality_out = self.quality_out(x)  # (B,4,T)
        quality_out_wei = self.quality_out_wei(x)  # (B,1,T)
        quality_out: torch.Tensor = quality_out * quality_out_wei  # (B,4,T)
        quality_out = quality_out.sum(dim=-1).view(-1, 4)

        value_out = self.value_out(x)  # (B,1,T)
        value_out_wei = self.value_out_wei(x)  # (B,1,T)
        value_out: torch.Tensor = value_out * value_out_wei  # (B,1,T)
        value_out = value_out.sum(dim=-1).view(-1, 1)

        return quality_out, value_out

    def batch_out(
        self,
        snake_states_list,
        food_states_list,
        batch_size=512,
    ):
        total_length = len(snake_states_list)
        snake_states_array = np.array(snake_states_list)
        food_states_array = np.array(food_states_list)

        out_arrays = [None for _ in range(2)]
        start_idx = 0
        while start_idx < total_length:
            end_idx = min(start_idx + batch_size, total_length)

            snake_states_tensor = torch.tensor(
                snake_states_array[start_idx:end_idx],
                dtype=torch.int64,
                device=device,
            )
            snake_states_tensor = snake_states_tensor.view(
                -1, 1, block_side, block_side
            )

            food_states_tensor = torch.tensor(
                food_states_array[start_idx:end_idx],
                dtype=torch.int64,
                device=device,
            )
            food_states_tensor = food_states_tensor.view(-1, 1, block_side, block_side)

            batch_outs = self(
                snake_states_tensor,
                food_states_tensor,
            )

            # if not isinstance(batch_outs, tuple):
            #     batch_outs = [batch_outs]

            for i in range(2):
                out_cpu = batch_outs[i].cpu().numpy()

                if out_arrays[i] is None:
                    out_arrays[i] = out_cpu
                else:
                    out_arrays[i] = np.concatenate((out_arrays[i], out_cpu), axis=0)

            start_idx = end_idx

        return out_arrays


if __name__ == "__main__":
    model = SnakeConvModel()

    quality_out, value_out = model(
        torch.randint(0, 2, (1, 1, block_side * block_side), dtype=torch.int64),
        torch.zeros(1, 1, block_side * block_side, dtype=torch.int64),
    )

    print(F.softmax(quality_out))
    print(value_out)
