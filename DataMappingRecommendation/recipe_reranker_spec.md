# Spec: GRU + Cross-Attention Recipe Reranker

## Context for the implementer

You are implementing a **reranking model** for a recipe recommendation system. Given a user's rated recipe history and one candidate recipe, the model predicts **P(the user rates the candidate positively)**. Candidates come from an upstream retrieval stage; this model only scores them.

Core idea: do **not** compress the user into a single vector. A GRU produces one hidden state per history step, and the **candidate item cross-attends** over those states plus a static profile embedding.

The model is built from three encoder classes (`ItemEncoder`, `UserEncoder`, `HistoryEncoder`) composed by one reranker class (`GRUCrossAttentionReranker`). Each class has its own section below.

Stack: Python, Pandas, PyTorch.

---

## Input data

File: `input_stage2/recipe_user_ratings_prices.csv`. One row per rating; the recipe columns repeat on every row of that recipe. `(user_id, recipe_id)` is unique.

| Column | Type | Description |
|---|---|---|
| `user_id` | int | User identifier |
| `recipe_id` | int | Recipe identifier |
| `rating` | int, 0–5 | Rating given by the user. In Food.com, 0 means a review without stars. |
| `date` | date (day) | Interaction day. Defines sequence order. Many ratings share a day. |
| `product`, `adj`, `verb` | JSON string | One list of phrases per ingredient, e.g. `[["winter squash"], ["mexican seasoning"]]`. |
| `calories (g)`, `total fat (g)`, `sugar (g)`, `sodium (g)`, `protein (g)`, `saturated fat (g)`, `carbohydrates (g)` | float | Nutrients (`n_nutrients = 7`). Heavy-tailed: calories reach 8.7e8. |
| `total_prices` | float | Total recipe price (3.08–426, none missing). |

---

## Phase 1: Preprocessing and profiling

Run the steps in this order.

### 1.1 Text normalization (`product`, `adj`, `verb`)
1. Lowercase.
2. Strip punctuation (a hyphen splits words: `hard-boiled` → `hard boiled`), then split phrases into words.
3. Lemmatize with the column's part of speech (`product` noun, `adj` adjective, `verb` verb).
4. Singularize.
5. Keep each (token, field) pair once per recipe, in order; truncate to `max_tokens`.

### 1.2 Labels
- Rating 0–2 → Negative (label 0). Rating 3–5 → Positive (label 1).
- Rating 0 counts as negative (decision). In Food.com it usually means a review without stars, and it makes up about 62% of the negatives kept after 1.3.

### 1.3 Activity filter
- Drop all users with **fewer than 6** observed ratings (and all their rows).
- Count observed ratings only; sampled negatives (1.6) do not exist yet.
- Result: 10,450 of 126,032 users, 356,422 ratings, 96.4% positive.

### 1.4 Rating centering (profiling only)
- For each user: `mu_user = mean(rating)` over their observed ratings; `rating_centered = rating - mu_user`.
- The activity filter removes whole users, so kept users' `mu_user` is the same before and after it.
- Sampled negatives are excluded from `mu_user` and have no `rating_centered`.
- Neither value is a model input or target. The history ratings (2.3) are centered differently, on the user's mean before t, because `mu_user` also averages the target and later ratings.

### 1.5 Train / validation / test split
- Per user, order observed ratings by (`date`, `recipe_id`). Last → test, second-to-last → validation, the rest → train.
- Every user keeps at least 4 training ratings.

### 1.6 Negative sampling
There is a strong class imbalance (1.3), so each **positive** row gets one sampled negative:
- `recipe_id`: a recipe the user **never rated** (at any time). Draws for one user are distinct.
- Draw probability ∝ `popularity ** negative_sampling_power` (popularity = the recipe's ratings in the whole file, all users; power 1 by default, 0 is uniform). With uniform draws, popularity alone would separate the classes.
- Popularity is only a sampling distribution, never a model input, so counting the whole file does not leak labels. Counting the training split instead leaves most recipes near zero: on the full data, popularity alone then reaches HR@10 0.20 in 3.3 (random order is 0.10), against 0.14 with whole-file counts.
- `label = 0`, `date` and split copied from the positive, so both rows share the same history.
- `rating` stays empty (NaN), not 0: rating 0 already means an observed negative. Keep a `source` column: `positive`, `rated_negative`, `sampled_negative`.
- Sampled negatives are targets only. They never appear in any history (2.3) or profile (2.2).

### 1.7 Profiling plots
1. **User engagement histogram**: x = number of observed ratings per user, y = number of users.
2. **Class balance**: row count per label, stacked by `source`, after negative sampling.
3. **Rating centering**: histogram of `rating_centered` over observed rows, colored by label.

---

## Phase 2: Encoders

```
recipe tokens + [price, nutrients] ──► ItemEncoder ──► item_emb (d), one per recipe
                                            │
             ┌──────────────────────────────┴──────────────┐
             ▼                                             ▼
   candidate item_emb                   item_emb of the last T_max observed
   (the query, Phase 3)                 recipes before t, date order
                                                           │  + their normalized centered ratings
                                                           ▼
                                        HistoryEncoder (GRU) ──► H = h_1 … h_T (T, d)

   mean [price, nutrients] of all observed recipes before t
                         │
                         ▼
             UserEncoder (MLP) ──► U_profile (d)
```

Encoders take **tensors, not recipe ids**. The reranker (Phase 3) owns the catalog lookup, so the same `HistoryEncoder` runs on item embeddings computed in the forward pass (training) or read from the cache (serving).

### 2.1 `ItemEncoder`

```python
class ItemEncoder(nn.Module):
    def __init__(self, vocab_size: int, n_numeric: int, config): ...
    def forward(self, token_ids,   # LongTensor (N, L), 0 = padding
                field_ids,         # LongTensor (N, L), 1 product / 2 adj / 3 verb, 0 = padding
                numeric            # FloatTensor (N, n_numeric)
                ) -> Tensor:       # item_emb (N, d)
```

1. **Text branch**
   - Vocabulary from normalized tokens of **training** recipes: `<pad>` = 0, `<unk>` = 1, tokens in fewer than `min_token_count` recipes → `<unk>`.
   - `nn.Embedding(vocab_size, d, padding_idx=0)` + field embedding `nn.Embedding(4, d, padding_idx=0)` (recommended: `ground` as a verb and as a product differ).
   - Self-attention: `nn.TransformerEncoder(nn.TransformerEncoderLayer(d, n_heads, batch_first=True), text_layers)` with `src_key_padding_mask = token_ids == 0`. No positional encoding: ingredient order carries no meaning.
   - Pool: mean over non-padding tokens (default) **or** a learned CLS token → `text_emb ∈ R^d`.
   - Every recipe needs at least one token; give an empty recipe a single `<unk>`.
2. **Numeric branch**
   - `numeric = [price, nutrients]`, `n_numeric = 1 + n_nutrients = 8`.
   - `log1p`, then z-score with statistics of training recipes only.
   - `nn.Linear(n_numeric, d)` → `num_emb ∈ R^d`.
   - Price is included so the candidate can be compared with the profile's average price (2.2). Without it, the query cannot see price at all.
3. **Fusion**
   - Option A (default): `concat(text_emb, num_emb)` → `nn.Linear(2d, d)`.
   - Option B: `text_emb + num_emb`.
   - Output: `item_emb ∈ R^d`.
4. **Caching**
   - The encoder trains end to end: during training, `item_emb` is computed in the forward pass, for candidates and history steps alike. Do not train against a cache.
   - After training, store `{recipe_id: item_emb}` from the final weights (`model.eval()`) for inference (3.4).

### 2.2 `UserEncoder` (static profile `U_profile ∈ R^d`)

```python
class UserEncoder(nn.Module):
    def __init__(self, n_profile: int, config): ...
    def forward(self, profile        # FloatTensor (B, n_profile)
                ) -> Tensor:         # U_profile (B, d)
```

- **Input features** (preprocessing, not inside the module), for target day t:
  - mean `log1p(price)` and mean `log1p(nutrients)` over **all** of the user's observed ratings on days strictly before t (not truncated to `T_max`; sampled negatives excluded);
  - `concat` → `n_profile = 1 + n_nutrients = 8`;
  - z-score with statistics of training rows only.
- **Module:** `nn.Linear(n_profile, d)` → ReLU → `nn.Linear(d, d)` (or a single `nn.Linear(n_profile, d)`).

### 2.3 `HistoryEncoder` (GRU states `H ∈ R^{T×d}`)

```python
class HistoryEncoder(nn.Module):
    def __init__(self, config): ...
    def forward(self, history_emb,   # FloatTensor (B, T, d), item_emb per step, right-padded with zeros
                history_rating,      # FloatTensor (B, T), normalized centered rating per step, 0 at padded steps
                lengths              # LongTensor (B,) on CPU, 1 <= length <= T_max
                ) -> tuple[Tensor, Tensor]:
        # H (B, T, d), padding_mask (B, T) with True = padded step
```

- **Input:** the item embeddings and ratings of the user's last `T_max` observed recipes on days strictly before t, ascending by (`date`, `recipe_id`).
- **Rating per step** (preprocessing, not inside the module):
  - centered: `rating - mu_before_t`, where `mu_before_t` is the mean of **all** the user's observed ratings on days strictly before t (the same set as the profile, 2.2). Not `mu_user` (1.4), which averages the target and later ratings too;
  - normalized: divided by one global scale, the RMS of the centered ratings over the real steps of training rows. A per-user standard deviation is 0 for users who give every recipe the same rating.
- **Step input:** `item_emb + rating_projection(rating)`, with `rating_projection = nn.Linear(1, d)`.
- **Module:** `nn.GRU(d, d, batch_first=True)`. Hidden size must be `d` so states can serve as attention keys.
- `pack_padded_sequence(step_inputs, lengths, batch_first=True, enforce_sorted=False)` → GRU → `pad_packed_sequence(..., total_length=T)`.
- Keep **all** hidden states `H = [h_1, …, h_T]`, not only `h_T`.
- Return the padding mask with PyTorch's `key_padding_mask` convention (True = ignore).

---

## Phase 3: Reranker

### 3.1 `GRUCrossAttentionReranker`

```python
class GRUCrossAttentionReranker(nn.Module):
    def __init__(self, catalog, vocab_size: int, config):
        # catalog: token_ids (R, L), field_ids (R, L), numeric (R, 8), registered as buffers
        self.item_encoder = ItemEncoder(vocab_size, n_numeric=8, config)
        self.user_encoder = UserEncoder(n_profile=8, config)
        self.history_encoder = HistoryEncoder(config)
        self.key_type = nn.Embedding(2, d)  # 0 = profile token, 1 = GRU state
        self.cross_attention = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def encode_items(self, recipe_idx) -> Tensor: ...             # (N,) → item_emb (N, d)
    def user_keys(self, history_emb, history_rating, lengths, profile): ...  # → keys (B, 1+T, d), key_padding_mask (B, 1+T)
    def score_candidates(self, candidate_emb, keys, key_padding_mask): ...  # candidate_emb (B, C, d) → logits (B, C)
    def forward(self, candidate_idx, history_idx, history_rating, lengths, profile) -> Tensor:  # → logits (B,)
```

```
candidate item_emb ─────────────► Q  (B, 1, d)   (or (B, C, d) for C candidates)
                                   │
[U_profile ; h_1 … h_T] + type ──► K, V  (B, 1+T, d)
                                   │
                        MultiheadAttention (+ key_padding_mask)
                                   │
                        attn_out (B, d)
                                   │
                  MLP( concat(attn_out, item_emb) )
                                   │
                  logit ─► sigmoid ─► P(positive)
```

1. **`forward`:** encode each distinct recipe in `candidate_idx` and `history_idx` once with `item_encoder`, gather the candidate and history embeddings, then `user_keys` and `score_candidates`.
2. **Keys / values (`user_keys`):** `concat([U_profile.unsqueeze(1), H], dim=1)` plus `key_type` (0 for the profile, 1 for GRU states). `key_padding_mask = concat([False], history padding_mask)`: the profile token is never masked.
3. **Query (`score_candidates`):** candidate `item_emb`. Several candidates per user can share one set of keys: queries attend independently, so `(B, C, d)` queries give the same scores as C separate calls.
4. **Head:** MLP over `concat(attn_out, item_emb)` → one logit per candidate.
5. **Output:** return **logits**. Sigmoid gives P(positive) at inference; training uses `BCEWithLogitsLoss`, which applies the sigmoid inside and is numerically stable. Do not apply sigmoid and then `BCELoss`.

`d` must be divisible by `n_heads`.

### 3.2 Training
- One training example = (user, candidate recipe, day t, label). History and profile come from the user's observed ratings on days strictly before t.
- Drop rows with no earlier day (each user's first day): they have no history and no profile.
- Loss: `BCEWithLogitsLoss` on the label. Optimizer: AdamW. Early stopping on validation log loss; restore the best epoch's weights.
- Split: 1.5; a sampled negative belongs to its positive's split.

### 3.3 Evaluation (test split)
- **Classification:** log loss and ROC-AUC over all test rows, plus AUC of positives vs rated negatives (the hard case) and positives vs sampled negatives.
- **Ranking:** for each test user whose held-out rating is positive, rank that recipe among 99 recipes the user never rated, sampled as in 1.6. Report HR@10 and NDCG@10.
- **Baselines:** popularity (training rating count) and a constant (training positive rate).

### 3.4 Inference with the cache
- Score from `{recipe_id: item_emb}` (2.1): cached history embeddings and their normalized centered ratings → `HistoryEncoder`; profile features → `UserEncoder`; all candidates of a user in one `score_candidates` call.
- Check that the cached path reproduces the forward-pass probabilities on the test rows.

---

## Constraints (must hold)

1. **No temporal leakage.** For a target on day t, the GRU sequence (recipes and their ratings), the history rating centering and `U_profile` use only observed ratings on days **strictly before** t. About 35% of kept ratings share a day with another rating of the same user, and their order within a day is unknown, so same-day ratings are excluded.
2. **Causal GRU states.** Since `H` is attended over, never include the target interaction (or anything after it) in `H`.
3. **Training-split statistics only:** token vocabulary, numeric z-scores (2.1), profile z-scores (2.2), the history rating scale (2.3) and the popularity baseline (3.3).
4. **Padding is masked everywhere:** text self-attention (`src_key_padding_mask`) and mean pooling, the GRU (packing), and cross-attention (`key_padding_mask`).
5. **Sampled negatives are targets only**: never in a history, a profile or `mu_user`.
6. **Reproducibility:** fixed seeds; every config value in one config object: `seed`, `positive_min_rating = 3`, `min_user_ratings = 6`, `negative_sampling_power`, `min_token_count`, `max_tokens`, `d`, `n_heads`, `text_layers`, `T_max`, `dropout`, `lr`, `weight_decay`, `batch_size`, `max_epochs`, `patience`, `ranking_negatives = 99`, `ndcg_k = 10`.
