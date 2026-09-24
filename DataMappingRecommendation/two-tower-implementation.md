# Two-Tower Recommender: Implementation Guide

This guide builds a two-tower retrieval model end to end, from a ratings dataset through training to serving:

**raw ratings → split by rating → training rows → towers → in-batch softmax (hard negatives + masks + logQ) → train → evaluate → ANN serving**

Stack: Python, pandas, PyTorch, FAISS.

---

## 0. Pipeline overview

```
ratings.csv (user_id, recipe_id, rating)
        │
        ▼
1. Split by rating        4–5★ positive │ 1–2★ hard negative │ 3★ neutral
        │
        ▼
2. Train/test split       hold out some positives per user for evaluation
        │
        ▼
3. Build training rows    one row per positive + one sampled hard negative
        │
        ▼
4. Batch + mask           collate_fn builds the [B, 2B] mask
        │
        ▼
5. Towers                 user tower → U,  item tower → I and H
        │
        ▼
6. Loss                   logits = [U·Iᵀ | U·Hᵀ]/τ − logQ → mask → cross-entropy
        │
        ▼
7. Evaluate               Recall@K on held-out positives
        │
        ▼
8. Serve                  precompute item embeddings → FAISS → top-K per user
```

---

## 1. Data preparation

### 1.1 Load and index

Map raw IDs to contiguous integers and **reserve index 0 as PAD**.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("ratings.csv")          # columns: user_id, recipe_id, rating

user2idx = {u: i for i, u in enumerate(df.user_id.unique())}
item2idx = {r: i + 1 for i, r in enumerate(df.recipe_id.unique())}   # 0 = PAD
PAD = 0

df["u"] = df.user_id.map(user2idx)
df["i"] = df.recipe_id.map(item2idx)
n_users, n_items = len(user2idx), len(item2idx) + 1
```

### 1.2 Check the rating distribution first

```python
print(df.rating.value_counts(normalize=True).sort_index())
```

If ratings are heavily skewed toward 5★ (common in food datasets), consider using **5★ = positive** instead of 4–5★, or normalize each rating against that user's mean.

### 1.3 Split by rating

| Rating | Role | Used as |
|---|---|---|
| 4–5★ | Positive | Training **row** |
| 3★ | Neutral | **Masked** |
| 1–2★ | Hard negative | Extra **column** |

```python
pos = df[df.rating >= 4]
neu = df[df.rating == 3]
neg = df[df.rating <= 2]
```

### 1.4 Train/test split (on positives only)

Hold out about 20% of each user's positives for evaluation.

```python
rng = np.random.default_rng(42)
pos = pos.sample(frac=1, random_state=42)
pos["rank"] = pos.groupby("u").cumcount()
pos["n"] = pos.groupby("u").u.transform("size")

test_pos  = pos[(pos.n >= 5) & (pos["rank"] < (pos.n * 0.2).astype(int))]
train_pos = pos.drop(test_pos.index)
```

### 1.5 Per-user lookup sets

```python
user_pos     = train_pos.groupby("u").i.apply(set).to_dict()
user_neg     = neg.groupby("u").i.apply(list).to_dict()
user_neutral = neu.groupby("u").i.apply(set).to_dict()
```

### 1.6 Item frequency (for logQ correction)

```python
item_freq = np.bincount(train_pos.i, minlength=n_items).astype(np.float32)
item_freq = (item_freq + 1) / (item_freq.sum() + n_items)     # smoothed probability
log_q = torch.log(torch.tensor(item_freq))
```

---

## 2. Dataset: one row per positive

- **Each positive interaction = one row.** A user with several positives appears in several rows.
- Each row gets **one randomly sampled hard negative** from that user's dislikes, or PAD if they have none.
- **Cap rows per user** each epoch so heavy users don't dominate.
- **Resample** every epoch.

```python
import torch
from torch.utils.data import Dataset

class TwoTowerDataset(Dataset):
    def __init__(self, train_pos, user_neg, max_per_user=50, seed=0):
        self.all_pos = train_pos[["u", "i"]]
        self.user_neg = user_neg
        self.max_per_user = max_per_user
        self.rng = np.random.default_rng(seed)
        self.resample()

    def resample(self):
        """Call at the start of every epoch."""
        capped = (self.all_pos
                  .sample(frac=1, random_state=int(self.rng.integers(1e9)))
                  .groupby("u").head(self.max_per_user))
        self.rows = capped.values                       # [[u, i], ...]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        u, i = self.rows[idx]
        negs = self.user_neg.get(u)
        h = self.rng.choice(negs) if negs else PAD
        return int(u), int(i), int(h)
```

Example rows:

| row | user | positive | hard_neg |
|---|---|---|---|
| 0 | u1 | r10 | r22 |
| 1 | u1 | r12 | r25 |
| 2 | u2 | r15 | r30 |
| 3 | u3 | r10 | PAD |

---

## 3. Batching and the mask

The mask is a `[B, 2B]` boolean matrix. Cells marked True are set to `-inf`, so they count as **neither positive nor negative**.

| Mask when a column is… | Why |
|---|---|
| Another positive of this row's user (including a duplicate of the row's own item) | Would be a false negative |
| A neutral (3★) item of this user | Neither liked nor disliked |
| PAD | Empty hard-negative slot |

**Never mask** the diagonal (the row's own positive) or the user's own dislikes (hard negatives).

Why is a positive item masked in another row? Each row is one-hot: only **one** column can be the answer. r12 can't also be correct in r10's row, and it must not be treated as wrong, so it's masked there and learned in its own row.

```python
def build_mask(user_ids, all_ids, user_pos, user_neutral):
    B, C = len(user_ids), len(all_ids)
    mask = torch.zeros(B, C, dtype=torch.bool)
    for r, u in enumerate(user_ids.tolist()):
        up, un = user_pos.get(u, set()), user_neutral.get(u, set())
        for c, item in enumerate(all_ids.tolist()):
            if item == PAD:
                mask[r, c] = True
            elif c != r and (item in up or item in un):
                mask[r, c] = True
    return mask

def make_collate(user_pos, user_neutral):
    def collate(batch):
        u, i, h = map(torch.tensor, zip(*batch))
        all_ids = torch.cat([i, h])                       # [2B]
        mask = build_mask(u, all_ids, user_pos, user_neutral)
        return u, i, h, mask
    return collate
```

The mask is built in the data loader on the CPU, so the GPU never waits on the Python loop.

---

## 4. The towers

Each tower maps an ID (a one-hot encoding implemented as an embedding lookup) to a vector. Outputs are **L2-normalized**, so the dot product equals cosine similarity.

```python
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, n, dim=64, hidden=128, padding_idx=None):
        super().__init__()
        self.emb = nn.Embedding(n, dim, padding_idx=padding_idx)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, ids):
        return F.normalize(self.mlp(self.emb(ids)), dim=-1)

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_tower = Tower(n_users, dim)
        self.item_tower = Tower(n_items, dim, padding_idx=PAD)
```

**Optional: content features in the item tower.** To handle cold start, concatenate a pretrained text embedding of the recipe (name, ingredients, cuisine) with the ID embedding before the MLP. New recipes then get useful vectors even without ratings.

---

## 5. Loss: in-batch softmax + hard negatives + logQ + mask

```
logits = [ U·Iᵀ | U·Hᵀ ] / τ          → [B, 2B]
         ─ in-batch ─  ─ hard negs ─
```

- **Positive:** the diagonal of the left block (`labels = arange(B)`).
- **Negatives:** every other column.
- **Hard-negative weight `w`:** add `log(w)` to the right block.
- **logQ correction:** subtract `log q(item)` from the **left block only**. Popular items appear as in-batch negatives more often, and this removes that bias.
- **Mask:** set masked cells to `-inf`.

```python
import math

def in_batch_softmax_loss(U, I, H, i_ids, mask, log_q, tau=0.05, w=1.0):
    B = U.size(0)
    left  = (U @ I.T) / tau - log_q[i_ids].unsqueeze(0)    # logQ on in-batch columns
    right = (U @ H.T) / tau + math.log(w)                  # hard-negative columns
    logits = torch.cat([left, right], dim=1)               # [B, 2B]
    logits = logits.masked_fill(mask, float("-inf"))
    labels = torch.arange(B, device=U.device)
    return F.cross_entropy(logits, labels)
```

Loss per row:

$$
L = -\log \frac{e^{s(u,i^+)}}{e^{s(u,i^+)} + \sum_{j \notin \text{mask}} e^{s(u,j)} + w \sum_{h \notin \text{mask}} e^{s(u,h)}}
$$

---

## 6. Training loop

```python
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TwoTower(n_users, n_items, dim=64).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
log_q = log_q.to(device)

ds = TwoTowerDataset(train_pos, user_neg, max_per_user=50)
collate = make_collate(user_pos, user_neutral)

for epoch in range(20):
    ds.resample()                                          # new rows + new hard negatives
    dl = DataLoader(ds, batch_size=512, shuffle=True, collate_fn=collate,
                    drop_last=True, num_workers=2)
    model.train()
    total = 0.0
    for u, i, h, mask in dl:
        u, i, h, mask = u.to(device), i.to(device), h.to(device), mask.to(device)
        U = model.user_tower(u)
        I = model.item_tower(i)
        H = model.item_tower(h)
        loss = in_batch_softmax_loss(U, I, H, i, mask, log_q, tau=0.05, w=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"epoch {epoch}: loss {total / len(dl):.4f}  recall@20 {recall_at_k(model, 20):.4f}")
```

**Starting hyperparameters:**

| Param | Start | Notes |
|---|---|---|
| batch size | 512 | Larger means more negatives; go as high as memory allows |
| τ (temperature) | 0.05 | Tune in the range 0.02–0.2 |
| w (hard-negative weight) | 1.0 | Try 2–5 if dislikes are informative |
| dim | 64 | 32–128 |
| max_per_user | 50 | Prevents heavy users from dominating |

---

## 7. Evaluation: Recall@K

For each test user, retrieve the top K items, excluding items already seen in training, and check how many held-out positives were found.

```python
@torch.no_grad()
def recall_at_k(model, k=20):
    model.eval()
    item_emb = model.item_tower(torch.arange(n_items, device=device))   # [n_items, d]
    test = test_pos.groupby("u").i.apply(set).to_dict()
    hits, total = 0, 0
    for u, targets in test.items():
        ue = model.user_tower(torch.tensor([u], device=device))
        scores = (ue @ item_emb.T).squeeze(0)
        scores[PAD] = -float("inf")
        seen = list(user_pos.get(u, ())) + list(user_neutral.get(u, ())) + user_neg.get(u, [])
        if seen:
            scores[torch.tensor(seen, device=device)] = -float("inf")   # don't re-recommend
        topk = set(scores.topk(k).indices.tolist())
        hits += len(topk & targets)
        total += len(targets)
    return hits / max(total, 1)
```

Compare against a **popularity baseline** (always recommending the most-rated recipes). If two-tower doesn't beat it, check the data, τ, and batch size before anything else.

---

## 8. Serving with ANN

At inference, **two-tower is just similarity**. Precompute every item embedding, index them, and query with the user embedding.

```python
import faiss

model.eval()
with torch.no_grad():
    item_emb = model.item_tower(torch.arange(1, n_items, device=device)).cpu().numpy()

index = faiss.IndexFlatIP(item_emb.shape[1])     # inner product = cosine (vectors normalized)
index.add(item_emb)

def recommend(user_idx, k=20):
    with torch.no_grad():
        ue = model.user_tower(torch.tensor([user_idx], device=device)).cpu().numpy()
    scores, ids = index.search(ue, k + 50)       # over-fetch, then filter
    seen = user_pos.get(user_idx, set())
    return [int(i) + 1 for i in ids[0] if int(i) + 1 not in seen][:k]   # +1 undoes the PAD offset
```

`IndexFlatIP` is exact and fine for up to around 1M items. Beyond that, switch to `IndexHNSWFlat` or `IndexIVFFlat`, or use pgvector if the data lives in Postgres.

---

## 9. Where this sits in the full system

```
two-tower retrieval (top ~200)
        │   (+ BM25 / content retrieval merged in, for hybrid search)
        ▼
ranker (LightGBM / small MLP)   ← rating used directly as the label here
        │
        ▼
re-rank / rules                 ← dietary constraints, diversity, price
        │
        ▼
final list
```

- **Two-tower:** ratings only decide rows, hard negatives, and masks. The output is a similarity score.
- **Ranker:** predicts the rating or like-probability directly, using richer features (price, nutrients, cuisine, time of day).

---

## 10. Checklist

- [ ] Reserve index 0 as PAD
- [ ] Check the rating distribution and choose positive/negative thresholds
- [ ] One row per positive; cap rows per user
- [ ] Sample one hard negative per row; resample each epoch
- [ ] Mask other positives of the same user, neutral items, and PAD
- [ ] Apply logQ correction to in-batch columns only
- [ ] L2-normalize tower outputs; tune τ
- [ ] Evaluate Recall@K against a popularity baseline
- [ ] Exclude already-seen items when recommending
