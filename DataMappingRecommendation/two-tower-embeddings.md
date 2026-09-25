# How Each Tower Builds Its Embedding

Each tower turns raw inputs into **one 64-dim vector** through the same steps:

**look up / project each input → concat → MLP → L2-normalize**

---

## Item tower (one recipe → 64-dim)

| Input | How it becomes a vector | Output dim |
|---|---|---|
| Recipe ID | `nn.Embedding` lookup (with ID dropout during training) | 32 |
| Ingredients (multi-hot) | `nn.EmbeddingBag`, mean of ingredient vectors | 32 |
| Nutrition | log1p → clip outliers → standardize → `nn.Linear` | 32 |

```
concat → [96] → MLP (96 → 128 → 64) → L2-normalize → item vector [64]
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemTower(nn.Module):
    def __init__(self, n_items, n_ingredients, n_nutrition, dim=64, part=32, id_dropout=0.3):
        super().__init__()
        self.id_emb   = nn.Embedding(n_items, part, padding_idx=0)
        self.ingr_emb = nn.EmbeddingBag(n_ingredients, part, mode="mean")
        self.nut_proj = nn.Linear(n_nutrition, part)
        self.mlp = nn.Sequential(
            nn.Linear(3 * part, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, dim)
        )
        self.id_dropout = id_dropout

    def forward(self, item_ids, ingr_ids, ingr_offsets, nutrition):
        id_vec = self.id_emb(item_ids)
        if self.training and self.id_dropout > 0:            # force reliance on content
            keep = (torch.rand(id_vec.size(0), 1, device=id_vec.device) > self.id_dropout).float()
            id_vec = id_vec * keep
        x = torch.cat([
            id_vec,                                          # [B, 32]
            self.ingr_emb(ingr_ids, ingr_offsets),           # [B, 32]
            self.nut_proj(nutrition),                        # [B, 32]
        ], dim=-1)                                           # [B, 96]
        return F.normalize(self.mlp(x), dim=-1)              # [B, 64]
```

### Nutrition preprocessing

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.log1p(nutrition.clip(lower=0))                   # tame right skew
X = np.clip(X, None, np.percentile(X, 99.5, axis=0))    # cap extreme outliers
scaler = StandardScaler().fit(X[train_idx])             # fit on training items only
X_scaled = scaler.transform(X)                          # reuse the same scaler at inference
```

Keep **raw** nutrition values separately for rule filters (e.g. low-calorie thresholds).

---

## User tower (one user → 64-dim)

A user is described by the recipes they liked, so the user tower **reuses the item tower**:

```
liked recipes in history (excluding the target)
    → item_tower each              → [n_hist, 64]
    → pool (rating-weighted mean) → [64]
    → concat context features (interaction history sequentially)
    → MLP (→ 128 → 64) → L2-normalize → user vector [64]
```

```python
class UserTower(nn.Module):
    def __init__(self, item_tower, dim=64):
        super().__init__()
        self.item_tower = item_tower                         # shared with the item side
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, dim)
        )

    def forward(self, hist_vecs, hist_mask, hist_weight):
        """
        hist_vecs:   [B, L, 64] history items encoded by item_tower
        hist_mask:   [B, L] 1 = real item, 0 = padding
        hist_weight: [B, L] e.g. rating-based weight
        """
        w = hist_weight * hist_mask
        pooled = (hist_vecs * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-6)
        return F.normalize(self.mlp(pooled), dim=-1)
```

**Users with no history:** `pooled` is all zeros, so fall back to popularity for them.

---

## Why this design

| Choice | Reason |
|---|---|
| No user ID embedding | Avoids memorizing users with ~2 ratings each |
| Shared item tower | Users and items live in the same space; new users work immediately from history |
| ID dropout on recipe ID | Forces reliance on ingredients and nutrition, which helps rarely-rated recipes |
| Separate projection per feature group | Keeps any one group from dominating the concat |
| L2-normalized outputs | Dot product equals cosine similarity, consistent with in-batch softmax and FAISS |
