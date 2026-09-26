# Spec: GRU + Cross-Attention Recipe Reranker

## Context for the implementer

You are implementing a **reranking model** for a recipe recommendation system. Given a user's rated recipe history and one candidate recipe, the model predicts how the user would rate that candidate (centered rating). Candidates come from an upstream retrieval stage; this model only scores them.

Core idea: do **not** compress the user into a single vector. A GRU produces one hidden state per history step, and the **candidate item cross-attends** over those states plus a static profile embedding.

Stack: Python, Pandas, PyTorch.

---

## Input data (assumed schema)

| Column | Type | Description |
|---|---|---|
| `user_id` | id | User identifier |
| `recipe_id` | id | Recipe identifier |
| `rating` | numeric | Raw rating given by user to recipe |
| `date` | datetime | Interaction timestamp (defines sequence order) |
| `product` | list[str] / str | Text tokens: product/ingredient words |
| `adj` | list[str] / str | Text tokens: adjectives |
| `verb` | list[str] / str | Text tokens: verbs (cooking actions) |
| `nutrient values` | list[float] | Per-recipe nutrient vector (fixed length) |
| `price` | float | Total recipe price |

---

## Phase 1: Preprocessing and profiling

### 1.1 Text normalization (applied to `product`, `adj`, `verb`)
1. Lowercase.
2. Strip punctuation.
3. Lemmatize.
4. Singularize.

### 1.2 Rating centering
- For each user: `mu_user = mean(rating)`.
- `rating_centered = rating - mu_user`.

### 1.3 Rating filter
- Rating 0-2 is classified into Negative
- Rating 3-5 is classfied into Positive
- I checked and there is a strong data imbalance. So for each user, for each Positive rating, you negative sample with one Negative rating (unobserved - set as 0)

### 1.4 Activity filter
- After the rating filter, group by `user_id` and drop all users with **fewer than 6** remaining ratings (and all their rows).
- Note: recompute `mu_user` / `rating_centered` after filtering if the filter removed rows used in the mean (state which choice was made).

### 1.5 Profiling plots (after filtering)
1. **User engagement histogram**: x = number of ratings per user, y = number of users.
2. **Class balance histogram**: x = `rating_centered` value, y = row count.

---

## Phase 2: Embeddings

### 2.1 Item embedding (dimension `d`)
1. **Text branch**
   - Build a vocabulary from normalized `product + adj + verb` tokens.
   - Token sequence → `nn.Embedding` → self-attention block (`nn.TransformerEncoderLayer` or `nn.MultiheadAttention` + residual/LayerNorm).
   - Pool to one vector: mean pooling (with padding mask) **or** a learned CLS token.
   - Output: `text_emb ∈ R^d`.
2. **Nutrient branch**
   - Standardize nutrient values (z-score, statistics from training split only).
   - `nn.Linear(n_nutrients, d)` → `nutr_emb ∈ R^d`.
3. **Fusion**
   - Option A: `concat(text_emb, nutr_emb)` → `nn.Linear(2d, d)`.
   - Option B: `text_emb + nutr_emb`.
   - Output: `item_emb ∈ R^d`.
4. **Caching**
   - Store `{recipe_id: item_emb}` for inference.
   - **Constraint:** the item encoder contains trainable layers. During training, item embeddings must be computed in the forward pass (or the encoder must be pre-trained and frozen before caching). Do not train against a stale cache.

### 2.2 User profile: Static profile embedding `U_profile ∈ R^d`**
- From the user's history (only interactions **before** the target — see Constraints):
  - average `price`
  - average nutrient vector
- `concat` → standardize → `nn.Linear(1 + n_nutrients, d)` (optionally a small MLP).

### 2.3 User history: Capture all hidden states from GRU capturing user history `H ∈ R^{T×d}`**
- User's history sorted by `date` (ascending), mapped to item embeddings → `(B, T, d)`.
- Omit rating from history
- Feed through `nn.GRU(batch_first=True)`.
- Keep **all** hidden states `H = [h_1, …, h_T]`, not only `h_T`.
- Use `pack_padded_sequence` or masks for variable-length histories; truncate to the most recent `T_max` interactions.

### 2.4 Ground truth: Refer to 1.3 - Rating filter
- Binary classification (0 - Negative, 1 - Positive)
- Use Sigmoid activation function for output layer

---

## Phase 3: Reranker (cross-attention)

### 3.1 Architecture

```
candidate item_emb ─────────────► Q  (B, 1, d)
                                   │
[U_profile ; h_1 … h_T] ─────────► K, V  (B, 1+T, d)
                                   │
                        MultiheadAttention (+ key_padding_mask)
                                   │
                        attn_out (B, d)
                                   │
                  MLP( concat(attn_out, item_emb) )   # concat recommended
                                   │
                  negative or positive classifcation (BCE loss)
```

1. **Query:** candidate `item_emb`, shape `(B, 1, d)`.
2. **Keys / Values:** `concat([U_profile.unsqueeze(1), H], dim=1)`, shape `(B, 1+T, d)`. Add a learned type embedding to distinguish the profile token from GRU states (recommended).
3. **Attention:** `nn.MultiheadAttention(d, n_heads, batch_first=True)` with `key_padding_mask` for padded history positions (never mask the profile token).
4. **Head:** MLP over `concat(attn_out, item_emb)` → scalar.

### 3.2 Training
- One training example = (user history before time t, target recipe at t, `rating_centered` at t).
- Loss: MSE (or Huber) on `rating_centered`.
- Split by time (e.g. per-user leave-last-out: last → test, second-to-last → val, earlier → train).
- Metrics: RMSE/MAE on ratings; for reranking quality also NDCG@10 over candidate lists (see Q3).

---

## Constraints (must hold)

1. **No temporal leakage.** For a target at time t, the GRU sequence, `U_profile`, and `mu_user` use only interactions **strictly before** t.
2. **Causal GRU states.** Since `H` is attended over, never include the target interaction (or anything after it) in `H`.
3. **Normalization stats** (nutrients, price) fit on the training split only.
4. **Padding** is masked in both the GRU (packing) and attention (`key_padding_mask`).
5. **Reproducibility:** fixed seeds; config values (d, n_heads, T_max, lr, batch size) in one config object.

---

