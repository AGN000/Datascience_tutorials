# %% [markdown]
# ============================================================
# TinyGPT from Scratch (Single-File Notebook)
# ============================================================
# This notebook implements a very small GPT-like language model
# using PyTorch and Transformer blocks.
#
# Task: Next-token prediction
# Dataset: Small handcrafted corpus (food & travel)
# ============================================================

# %% [markdown]
# ## 1️⃣ Imports and Environment Check

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_blocks import Block

# %%
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# %% [markdown]
# ## 2️⃣ Create a Small Text Corpus (New Example)

# %%
corpus = [
    "i love south indian food",
    "we traveled by train to chennai",
    "the dosa was crispy and tasty",
    "tea tastes better in the evening",
    "we visited the beach at sunrise",
    "spicy food makes me happy",
    "the journey was long but fun",
    "coffee and snacks are perfect",
]

# Add end-of-sentence token
corpus = [s + " <END>" for s in corpus]

text = " ".join(corpus)
print(text)

# %% [markdown]
# ## 3️⃣ Vocabulary Construction

# %%
words = list(set(text.split()))
vocab_size = len(words)

print("Vocabulary:", words)
print("Vocab size:", vocab_size)

# %%
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}

print("word2idx:", word2idx)

# %% [markdown]
# ## 4️⃣ Encode Text as Token IDs

# %%
data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)

print("Encoded data:", data)
print("Total tokens:", len(data))

# %% [markdown]
# ## 5️⃣ Hyperparameters

# %%
block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
learning_rate = 1e-3
epochs = 1500
batch_size = 16

# %% [markdown]
# ## 6️⃣ Mini-batch Sampling Function

# %%
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# %% [markdown]
# ## 7️⃣ TinyGPT Model Definition

# %%
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
                reduction="none"
            )

            end_id = word2idx["<END>"]
            end_mask = (targets.view(-1) == end_id)

            loss[end_mask] *= 2.0
            loss = loss.mean()

        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_idx), dim=1)

            # 🔴 HARD STOP at <END>
            if next_idx.item() == word2idx["<END>"]:
                break

        return idx


# %% [markdown]
# ## 8️⃣ Model Initialization and Optimizer

# %%
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Total parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 9️⃣ Training Loop

# %%
for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}, Loss = {loss.item():.4f}")

# %% [markdown]
# ## 🔟 Text Generation

# %%
start_word = "i"
context = torch.tensor([[word2idx[start_word]]], dtype=torch.long)

generated = model.generate(context, max_new_tokens=15)

print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in generated[0]))
