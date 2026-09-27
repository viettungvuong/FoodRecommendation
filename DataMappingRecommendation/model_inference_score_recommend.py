"""Recommend catalog recipes to a user: HNSW retrieval over item embeddings, then the GRU + cross-attention reranker.

1. Index (once per model and catalog): every recipe of price_mapped_nutrients.csv is embedded by the reranker's
   ItemEncoder, i.e. its product / adj / verb tokens concatenated with its standardized [price, nutrients], and the
   embeddings go into an HNSW index with cosine distance.
2. Query: the user's representative embedding is the average of their history recipes' embeddings.
3. Retrieval: HNSW returns the recipes closest to it by cosine similarity, the history's own recipes excluded.
4. User state: the history is cut into windows wherever two ratings are window_gap_days (about 6 months) or more
   apart; the GRU runs over each window from a fresh state, and each window's final hidden state is stored.
5. Reranking: each candidate attends over [user profile; window states] and gets P(like); candidates are sorted by it.

Usage:
    python model_inference_score_recommend.py build --catalog path/to/price_mapped_nutrients.csv
    python model_inference_score_recommend.py recommend --catalog path/to/price_mapped_nutrients.csv \\
        --user examples/retrieval_user.json
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from model_inference_reranker_avgemb import (DEFAULT_ARTIFACTS, THRESHOLD, ItemEncoder, Reranker as BaseReranker,
                                                     UserEncoder)

HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "input_stage2" / "price_mapped_nutrients.csv"
DEFAULT_INDEX_DIR = HERE / "model_artifacts" / "retrieval_index"
DEFAULT_USER = HERE / "examples" / "retrieval_user.json"
TEXT_COLUMNS = ("product", "adj", "verb")
# Reranker nutrient input key (NUTRIENT_INPUTS) → catalog column.
CATALOG_NUTRIENTS = {"calories_kcal": "energy_kcal", "fat_g": "fat_g", "sugar_g": "sugars_g",
                     "sodium_mg": "sodium_mg", "protein_g": "protein_g", "saturated_fat_g": "satfat_g",
                     "carbohydrates_g": "carb_g"}
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 200
SEED = 42


# ---------------------------------------------------------------------------------------------------------------
# History windows: the same functions as model_training_reranker_avgemb.ipynb, so serving cuts the history
# exactly as training did.
# ---------------------------------------------------------------------------------------------------------------
def window_starts(user_codes, days, gap_days):
    """Index of the first rating of each rating's window. Rows are sorted by user, then day. A window ends where
    the user changes or where the next rating comes gap_days or more after the previous one."""
    new_window = np.ones(len(days), dtype=bool)
    new_window[1:] = (user_codes[1:] != user_codes[:-1]) | (np.diff(days) >= gap_days)
    return np.maximum.accumulate(np.where(new_window, np.arange(len(days)), 0))


def history_windows(prior_end, user_start, mu_before, first_of_window, recipe_idx, ratings, max_windows, t_max):
    """Each row's history as windows, most recent window first.

    A row's history is ratings[user_start:prior_end]. Returns recipe indices (B, max_windows, t_max), -1 at padding;
    ratings centered on mu_before (B, max_windows, t_max), 0 at padding; window lengths (B, max_windows), 0 for
    missing windows. Each window keeps its last t_max ratings, in date order."""
    steps = np.arange(t_max)
    history_idx = np.full((len(prior_end), max_windows, t_max), -1, dtype=np.int64)
    history_rating = np.zeros((len(prior_end), max_windows, t_max), dtype=np.float32)
    window_len = np.zeros((len(prior_end), max_windows), dtype=np.int64)
    ends = np.asarray(prior_end, dtype=np.int64).copy()
    for k in range(max_windows):
        alive = ends > user_start
        if not alive.any():
            break
        # The window of the last rating before `ends`; it never reaches before user_start (a user change starts one).
        starts = np.where(alive, first_of_window[np.maximum(ends - 1, 0)], ends)
        length = np.minimum(ends - starts, t_max)
        positions = np.clip(ends[:, None] - length[:, None] + steps, 0, len(recipe_idx) - 1)
        real = steps < length[:, None]
        history_idx[:, k] = np.where(real, recipe_idx[positions], -1)
        history_rating[:, k] = np.where(real, ratings[positions] - mu_before[:, None], 0.0)
        window_len[:, k] = length
        ends = starts
    return history_idx, history_rating, window_len


def history_day(value):
    """Day number (days since 1970-01-01) of a history date; a missing date is day 0."""
    if value is None or value == "":
        return 0
    return int(np.datetime64(pd.Timestamp(value).date(), "D").astype(np.int64))


# ---------------------------------------------------------------------------------------------------------------
# The windowed model: the HistoryEncoder and GRUCrossAttentionReranker of the notebook (same parameter names).
# ---------------------------------------------------------------------------------------------------------------
class WindowHistoryEncoder(nn.Module):
    """History windows → the final GRU state of each window; the GRU restarts from zero in every window."""

    def __init__(self, config):
        super().__init__()
        self.rating_projection = nn.Linear(1, config["d"])
        self.gru = nn.GRU(config["d"], config["d"], batch_first=True)

    def forward(self, history_emb, history_rating, window_len):
        """Window states (B, K, d) and their padding mask (B, K), True = missing window."""
        batch, windows, steps, d = history_emb.shape
        steps_in = (history_emb + self.rating_projection(history_rating.unsqueeze(-1))).reshape(-1, steps, d)
        lengths = window_len.reshape(-1).cpu()
        real = lengths > 0
        packed = pack_padded_sequence(steps_in[real.to(steps_in.device)], lengths[real], batch_first=True,
                                      enforce_sorted=False)
        _, last = self.gru(packed)  # (1, windows, d): each window's state after its last real step.
        states = steps_in.new_zeros(batch * windows, d)
        states[real.to(states.device)] = last[0]
        return states.view(batch, windows, d), (window_len == 0).to(history_emb.device)


class WindowGRUCrossAttentionReranker(nn.Module):
    """Candidate item_emb cross-attends over [U_profile; w_1 … w_K], w_1 the most recent window's state."""

    def __init__(self, vocab_size, n_numeric, n_profile, n_fields, config):
        super().__init__()
        d = config["d"]
        self.item_encoder = ItemEncoder(vocab_size, n_numeric, n_fields, config)
        self.user_encoder = UserEncoder(n_profile, config)
        self.history_encoder = WindowHistoryEncoder(config)
        self.key_type = nn.Embedding(2, d)  # 0: profile token, 1: window state.
        self.window_position = nn.Embedding(config["max_windows"], d)  # 0: most recent window.
        self.cross_attention = nn.MultiheadAttention(d, config["n_heads"], dropout=config["dropout"],
                                                     batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(d, 1))

    def user_keys(self, history_emb, history_rating, window_len, profile):
        """Keys and values [U_profile; w_1 … w_K] (B, 1+K, d) and their padding mask (True = padded)."""
        states, padded = self.history_encoder(history_emb, history_rating, window_len)
        states = states + self.window_position.weight[:states.size(1)] + self.key_type.weight[1]
        profile_token = self.user_encoder(profile).unsqueeze(1) + self.key_type.weight[0]
        keys = torch.cat([profile_token, states], dim=1)
        return keys, torch.cat([padded.new_zeros(len(padded), 1), padded], dim=1)

    def score_candidates(self, candidate_emb, keys, key_padding_mask):
        """Logits (B, C) for candidates (B, C, d)."""
        attended, _ = self.cross_attention(candidate_emb, keys, keys, key_padding_mask=key_padding_mask,
                                           need_weights=False)
        return self.head(torch.cat([attended, candidate_emb], dim=-1)).squeeze(-1)


class Reranker(BaseReranker):
    """The reranker, scoring candidates from item embeddings.

    A config.json with max_windows is the windowed model of the current notebook: history cut into windows at gaps of
    window_gap_days or more, centered ratings. Otherwise it is the older model of
    model_inference_reranker_avgemb_approach.py: the last t_max recipes in one sequence."""

    def __init__(self, artifact_dir=DEFAULT_ARTIFACTS, tagger=None, device=None):
        artifact_dir = Path(artifact_dir)
        saved = json.loads((artifact_dir / "config.json").read_text())
        self.windowed = "max_windows" in saved["config"]
        if not self.windowed:
            super().__init__(artifact_dir, tagger, device)
            # Models trained with [0, 1] history ratings save offset 5 and scale 10; older ones only a scale.
            self.rating_offset = saved.get("history_rating_offset", 0.0)
            return
        # The windowed model: same attributes as the base class, another model class.
        self.config = saved["config"]
        self.text_fields = saved["text_fields"]
        self.nutrient_columns = saved["nutrient_columns"]
        self.token_lookup = {token: index for index, token in enumerate(saved["vocabulary"])}
        self.item_mean, self.item_std = np.array(saved["item_log1p_mean"]), np.array(saved["item_log1p_std"])
        self.profile_mean, self.profile_std = np.array(saved["profile_mean"]), np.array(saved["profile_std"])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        self.n_numeric = state["item_encoder.numeric_projection.weight"].shape[1]
        self.use_rating, self.rating_offset, self.rating_scale = True, 0.0, 1.0  # Centered, not scaled.
        self.model = WindowGRUCrossAttentionReranker(len(self.token_lookup), self.n_numeric,
                                                     state["user_encoder.mlp.0.weight"].shape[1],
                                                     len(self.text_fields), self.config)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.tagger = tagger

    def normalized_ratings(self, history):
        """History ratings as the GRU saw them in training (spec 2.3): centered on the mean of all of them, then
        (centered + offset) / scale. The windowed model uses them centered only (offset 0, scale 1).

        A recipe without a rating counts as the user's mean, i.e. centered 0."""
        if not self.use_rating:
            return np.zeros(len(history), dtype=np.float32)  # The model has no rating input.
        ratings = np.array([np.nan if recipe.get("rating") is None else recipe["rating"] for recipe in history],
                           dtype=np.float64)
        centered = np.zeros(len(history)) if np.isnan(ratings).all() else np.nan_to_num(ratings - np.nanmean(ratings))
        return ((centered + self.rating_offset) / self.rating_scale).astype(np.float32)

    def windows(self, history):
        """The windows of history (date order) that the GRU sees, most recent first: positions into history (K, T),
        -1 at padding; centered ratings (K, T); lengths (K,)."""
        centered = self.normalized_ratings(history).astype(np.float64)  # Centered, missing ratings 0.
        days = np.array([history_day(recipe.get("date")) for recipe in history], dtype=np.int64)
        count = len(history)
        positions, ratings, lengths = history_windows(
            np.array([count]), np.array([0]), np.array([0.0]),
            window_starts(np.zeros(count, dtype=np.int64), days, self.config["window_gap_days"]), np.arange(count),
            centered, self.config["max_windows"], self.config["t_max"])
        windows, steps = int((lengths[0] > 0).sum()), int(lengths[0].max())
        return positions[0, :windows, :steps], ratings[0, :windows, :steps], lengths[0, :windows]

    def _profile(self, history):
        profile = np.mean([self.log_numeric(recipe) for recipe in history], axis=0)[-len(self.profile_mean):]
        profile = (profile - self.profile_mean) / self.profile_std
        return torch.as_tensor(profile, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _window_inputs(self, history, history_emb):
        positions, ratings, lengths = self.windows(history)
        positions = torch.from_numpy(positions)
        window_emb = history_emb[positions.clamp(min=0)] * (positions >= 0).unsqueeze(-1)
        return (window_emb.unsqueeze(0).to(self.device), torch.from_numpy(ratings).unsqueeze(0).to(self.device),
                torch.from_numpy(lengths).unsqueeze(0))

    @torch.no_grad()
    def window_states(self, history, history_emb):
        """The hidden state (K, d) of each window of the windowed model, most recent first."""
        states, _ = self.model.history_encoder(*self._window_inputs(history, history_emb))
        return states[0].cpu()

    @torch.no_grad()
    def user_keys(self, history, history_emb):
        """The user's attention keys and their padding mask, for history (date order) and its item embeddings."""
        if self.windowed:
            return self.model.user_keys(*self._window_inputs(history, history_emb), self._profile(history))
        # Older model: the last t_max history recipes in date order go to the GRU as one sequence.
        t_max = self.config["t_max"]
        recent = history_emb[-t_max:].to(self.device).unsqueeze(0)
        history_rating = torch.as_tensor(self.normalized_ratings(history)[-t_max:], device=self.device).unsqueeze(0)
        return self.model.user_keys(recent, history_rating, torch.tensor([recent.size(1)]), self._profile(history))

    @torch.no_grad()
    def score(self, history, history_emb, candidate_emb):
        """P(like) (C,) of candidates for the user who rated history, from item embeddings.

        history is in date order and history_emb (len(history), d) holds its item embeddings; candidate_emb is
        (C, d). Both come from encode_items or from the index's cache of it. The profile is the mean over all."""
        keys, key_padding_mask = self.user_keys(history, history_emb)
        # Each candidate is a separate query, so scoring them together equals scoring them one at a time.
        logits = self.model.score_candidates(candidate_emb.to(self.device).unsqueeze(0), keys, key_padding_mask)[0]
        return torch.sigmoid(logits).cpu()


def load_catalog(path):
    """price_mapped_nutrients.csv → recipes in the reranker's input format, in file order.

    product / adj / verb are already NER output, so each phrase becomes one ingredient of "entities" and no tagging
    is needed. total_price is 0 whenever an item is unpriced, so price is the sum of item_prices instead (as in
    recipe_clustering.ipynb). Missing nutrients (sugars_g for 13% of rows) take the catalog median."""
    frame = pd.read_csv(path)
    nutrients = frame[list(CATALOG_NUTRIENTS.values())]
    nutrients = nutrients.fillna(nutrients.median())
    prices = frame["item_prices"].map(lambda value: float(sum(ast.literal_eval(value).values())))
    recipes = []
    for row, nutrient_row, price in zip(frame.to_dict("records"), nutrients.to_dict("records"), prices):
        recipes.append({"fdc_id": int(row["fdc_id"]), "name": row["description"], "price": price,
                        "nutrients": {key: float(nutrient_row[column]) for key, column in CATALOG_NUTRIENTS.items()},
                        "entities": {field: [[phrase] for phrase in ast.literal_eval(row[field])]
                                     for field in TEXT_COLUMNS}})
    return recipes


def fingerprint(*paths):
    """sha256 over the files' bytes: the index is stale when the model or the catalog changes."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


@torch.no_grad()
def encode_catalog(reranker, recipes, batch_size=1024):
    """item_emb (N, d) of every recipe, in catalog order."""
    return torch.cat([reranker.encode_items(recipes[start:start + batch_size])[0].cpu()
                      for start in range(0, len(recipes), batch_size)])


class RecipeIndex:
    """Item embeddings of the catalog and an HNSW (cosine) index over them. Labels are catalog row positions."""

    def __init__(self, fdc_ids, embeddings, index):
        self.fdc_ids = list(fdc_ids)
        self.embeddings = embeddings  # (N, d) float32, not normalized: the reranker takes them as they are.
        self.index = index

    @classmethod
    def build(cls, reranker, recipes):
        embeddings = encode_catalog(reranker, recipes).numpy()
        index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
        index.init_index(max_elements=len(recipes), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M, random_seed=SEED)
        index.add_items(embeddings, np.arange(len(recipes)))
        return cls([recipe["fdc_id"] for recipe in recipes], embeddings, index)

    def save(self, index_dir, source_fingerprint):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "item_embeddings.npy", self.embeddings)
        self.index.save_index(str(index_dir / "hnsw.bin"))
        (index_dir / "meta.json").write_text(json.dumps({
            "fingerprint": source_fingerprint, "dim": int(self.embeddings.shape[1]), "M": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION, "fdc_ids": self.fdc_ids}))

    @classmethod
    def load(cls, index_dir):
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        index = hnswlib.Index(space="cosine", dim=meta["dim"])
        index.load_index(str(index_dir / "hnsw.bin"), max_elements=len(meta["fdc_ids"]))
        return cls(meta["fdc_ids"], np.load(index_dir / "item_embeddings.npy"), index)

    @classmethod
    def load_or_build(cls, reranker, recipes, index_dir, source_fingerprint, rebuild=False):
        """The saved index when it was built from the same model and catalog; otherwise a new one, saved."""
        meta_path = Path(index_dir) / "meta.json"
        if not rebuild and meta_path.exists() and json.loads(meta_path.read_text())["fingerprint"] == source_fingerprint:
            return cls.load(index_dir)
        built = cls.build(reranker, recipes)
        built.save(index_dir, source_fingerprint)
        return built

    def search(self, query, k, exclude=(), ef=HNSW_EF_SEARCH):
        """Up to k (row, cosine similarity) nearest to query, most similar first, rows in exclude left out."""
        exclude = set(exclude)
        # Ask for enough neighbours that k remain after dropping the excluded rows.
        n = min(k + len(exclude), self.index.get_current_count())
        if k <= 0 or n == 0:
            return []
        self.index.set_ef(max(ef, n))
        rows, distances = self.index.knn_query(np.asarray(query, dtype=np.float32).reshape(1, -1), k=n)
        return [(int(row), 1.0 - float(distance)) for row, distance in zip(rows[0], distances[0])
                if row not in exclude][:k]


class RetrievalPipeline:
    """User profile → HNSW candidates → reranked recommendations."""

    def __init__(self, reranker, recipes, recipe_index):
        if [recipe["fdc_id"] for recipe in recipes] != recipe_index.fdc_ids:
            raise ValueError("The index was built from a different catalog; rebuild it.")
        self.reranker = reranker
        self.recipes = recipes
        self.recipe_index = recipe_index
        self.row_of = {fdc_id: row for row, fdc_id in enumerate(recipe_index.fdc_ids)}
        self.user_states = {}  # user_id → windows and their GRU hidden states, from the last recommend call.

    @classmethod
    def from_paths(cls, catalog=DEFAULT_CATALOG, artifacts=DEFAULT_ARTIFACTS, index_dir=DEFAULT_INDEX_DIR,
                   rebuild=False, device=None):
        reranker = Reranker(artifacts, device=device)
        recipes = load_catalog(catalog)
        source = fingerprint(Path(artifacts) / "model.pt", Path(artifacts) / "config.json", catalog)
        return cls(reranker, recipes, RecipeIndex.load_or_build(reranker, recipes, index_dir, source, rebuild))

    def resolve_history(self, history):
        """History recipes in date order and their item embeddings (len(history), d).

        An entry with "fdc_id" is that catalog recipe, with the entry's "rating" and "date"; its embedding comes from
        the index. Any other entry is a full recipe ("ingredients", "price", "nutrients", as in the inference
        script), tagged and encoded here."""
        # Stable sort: entries without a date keep their given order.
        history = sorted(history, key=lambda entry: str(entry.get("date", "")))
        resolved, embeddings, off_catalog = [], [None] * len(history), []
        for position, entry in enumerate(history):
            if "fdc_id" in entry:
                row = self.row_of.get(int(entry["fdc_id"]))
                if row is None:
                    raise KeyError(f"fdc_id {entry['fdc_id']} is not in the catalog.")
                resolved.append({**self.recipes[row], **entry, "fdc_id": int(entry["fdc_id"])})
                embeddings[position] = torch.from_numpy(self.recipe_index.embeddings[row])
            else:
                resolved.append(dict(entry))
                off_catalog.append(position)
        if off_catalog:
            recipes = [resolved[position] for position in off_catalog]
            self.reranker.tag_recipes(recipes)
            for position, embedding in zip(off_catalog, self.reranker.encode_items(recipes)[0].cpu()):
                embeddings[position] = embedding
        return resolved, torch.stack(embeddings)

    def user_state(self, history, history_emb):
        """The windows of a history (date order) and the GRU hidden state of each, most recent first."""
        positions, _, lengths = self.reranker.windows(history)
        states = self.reranker.window_states(history, history_emb)
        windows = []
        for window, (row, length, state) in enumerate(zip(positions, lengths, states), start=1):
            members = [history[position] for position in row[:length]]
            rated = [recipe["rating"] for recipe in members if recipe.get("rating") is not None]
            windows.append({"window": window, "from": str(members[0].get("date", "")),
                            "to": str(members[-1].get("date", "")), "recipes": int(length),
                            "fdc_ids": [recipe.get("fdc_id") for recipe in members],
                            "mean_rating": float(np.mean(rated)) if rated else None,
                            "state_norm": float(state.norm()), "state": state.tolist()})
        return {"windows": windows, "states": states}

    @staticmethod
    def user_embedding(history_emb):
        """Representative embedding: the average of the history's embeddings, each L2-normalized first so every
        recipe weighs the same under cosine similarity."""
        return F.normalize(history_emb, dim=1).mean(0)

    def recommend(self, user, n_candidates=100, top_k=10, log=None, show=20):
        """The top_k of n_candidates retrieved recipes by P(like), most likely first.

        With log (e.g. print), each step is reported: the history, the user embedding, the HNSW candidates (the first
        `show` of them) and how the reranker reorders them."""
        log = log or (lambda *_: None)
        if not user.get("history"):
            raise ValueError("The pipeline needs at least one recipe in the user's history.")

        history, history_emb = self.resolve_history(user["history"])
        ratings = self.reranker.normalized_ratings(history)
        t_max = self.reranker.config["t_max"]
        window_of = {}  # History position → window number (1 = most recent), for the recipes the GRU sees.
        if self.reranker.windowed:
            positions, _, _ = self.reranker.windows(history)
            window_of = {int(position): window + 1 for window, row in enumerate(positions) for position in row
                         if position >= 0}
            log(f"\n=== Step 1. History of user {user.get('user_id', '?')}: {len(history)} recipes, date order; a gap "
                f"of {self.reranker.config['window_gap_days']}+ days starts a window (the last "
                f"{self.reranker.config['max_windows']} windows, last {t_max} recipes each, feed the GRU) ===")
        else:
            log(f"\n=== Step 1. History of user {user.get('user_id', '?')}: {len(history)} recipes, date order "
                f"(the last {t_max} feed the GRU) ===")
        log(f"  {'date':<10}  {'fdc_id':>8}  {'recipe':<48} {'rating':>6} {'model rating':>12} {'window':>6} "
            f"{'|emb|':>6}")
        for position, (recipe, emb, rating) in enumerate(zip(history, history_emb, ratings)):
            window = window_of.get(position, "-") if self.reranker.windowed else ""
            log(f"  {str(recipe.get('date', '')):<10}  {recipe.get('fdc_id', 'custom'):>8}  "
                f"{str(recipe.get('name', ''))[:48]:<48} {str(recipe.get('rating', '-')):>6} {rating:>12.3f} "
                f"{window:>6} {emb.norm():>6.2f}")

        query = self.user_embedding(history_emb)
        member_cosine = F.cosine_similarity(history_emb, query.unsqueeze(0)).tolist()
        log(f"\n=== Step 2. Representative embedding: mean of the {len(history)} L2-normalized history embeddings "
            f"({query.numel()} dims, norm {query.norm():.3f}) ===")
        log("  first 8 dims: " + " ".join(f"{value:+.3f}" for value in query[:8].tolist()))
        for recipe, cosine in zip(history, member_cosine):
            log(f"  cosine to {str(recipe.get('name', ''))[:48]:<48} {cosine:.3f}")

        seen = {self.row_of[recipe["fdc_id"]] for recipe in history if "fdc_id" in recipe}
        candidates = self.recipe_index.search(query.numpy(), n_candidates, exclude=seen)
        log(f"\n=== Step 3. HNSW retrieval: {len(candidates)} nearest of {len(self.recipes):,} recipes by cosine "
            f"similarity ({len(seen)} history recipes excluded) ===")
        log(f"  {'rank':>4}  {'fdc_id':>8}  {'recipe':<56} {'cosine':>7}")
        for rank, (row, similarity) in enumerate(candidates[:show], start=1):
            log(f"  {rank:>4}  {self.recipes[row]['fdc_id']:>8}  {self.recipes[row]['name'][:56]:<56} {similarity:>7.3f}")
        if len(candidates) > show:
            log(f"  ... {len(candidates) - show} more")
        if not candidates:
            return []

        if self.reranker.windowed:
            self.user_states[user.get("user_id")] = state = self.user_state(history, history_emb)
            log(f"\n=== Step 4. User state: GRU hidden state of each window (restarted per window), stored in "
                f"pipeline.user_states ===")
            log(f"  {'window':>6}  {'from':<10}  {'to':<10} {'recipes':>7} {'mean rating':>11} {'|state|':>8}  first 6 dims")
            for window in state["windows"]:
                log(f"  {window['window']:>6}  {window['from']:<10}  {window['to']:<10} {window['recipes']:>7} "
                    f"{'-' if window['mean_rating'] is None else format(window['mean_rating'], '.2f'):>11} "
                    f"{window['state_norm']:>8.3f}  "
                    + " ".join(f"{value:+.2f}" for value in window["state"][:6]))

        rows = [row for row, _ in candidates]
        p_likes = self.reranker.score(history, history_emb, torch.from_numpy(self.recipe_index.embeddings[rows]))
        results = [{"fdc_id": self.recipes[row]["fdc_id"], "name": self.recipes[row]["name"],
                    "price": self.recipes[row]["price"], "p_like": p_like, "like": p_like >= THRESHOLD,
                    "similarity": similarity, "retrieval_rank": rank}
                   for rank, ((row, similarity), p_like) in enumerate(zip(candidates, p_likes.tolist()), start=1)]
        reranked = sorted(results, key=lambda result: -result["p_like"])

        keys = (f"the hidden states of {len(self.user_states[user.get('user_id')]['windows'])} windows"
                if self.reranker.windowed else f"GRU states of the last {min(len(history), t_max)} history recipes")
        log(f"\n=== Step 5. Reranker: each candidate attends over [user profile; {keys}] -> P(like) ===")
        log(f"  {'new':>4} {'was':>4} {'move':>5}  {'fdc_id':>8}  {'recipe':<48} {'cosine':>7} {'P(like)':>8}")
        for new_rank, result in enumerate(reranked[:show], start=1):
            move = result["retrieval_rank"] - new_rank
            log(f"  {new_rank:>4} {result['retrieval_rank']:>4} {move:>+5d}  {result['fdc_id']:>8}  "
                f"{result['name'][:48]:<48} {result['similarity']:>7.3f} {result['p_like']:>8.4f}")
        if len(reranked) > show:
            log(f"  ... {len(reranked) - show} more")
        log(f"\n=== Step 6. Top {min(top_k, len(reranked))} kept; {sum(r['like'] for r in reranked)} of "
            f"{len(reranked)} candidates have P(like) >= {THRESHOLD} ===")
        return reranked[:top_k]


def print_recommendations(user, results):
    print(f"\nRecommendations for user {user.get('user_id', '?')}: {len(user['history'])} history recipes")
    print(f"  {'fdc_id':>8}  {'recipe':<60} {'P(like)':>8} {'cosine':>7} {'retrieved':>9}")
    for result in results:
        print(f"  {result['fdc_id']:>8}  {result['name'][:60]:<60} {result['p_like']:>8.4f} "
              f"{result['similarity']:>7.3f} {'#' + str(result['retrieval_rank']):>9}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "recommend"],
                        help="build: embed the catalog and save the HNSW index; recommend: score one user")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="price_mapped_nutrients.csv")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="folder holding the reranker's model.pt and config.json")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="where the index is saved")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the index even if it is up to date")
    parser.add_argument("--user", type=Path, default=DEFAULT_USER,
                        help='JSON user profile {"user_id", "history": [{"fdc_id", "rating", "date"}, ...]}')
    parser.add_argument("--candidates", type=int, default=100, help="recipes retrieved by HNSW")
    parser.add_argument("--top-k", type=int, default=10, help="recommendations kept after reranking")
    parser.add_argument("--output", type=Path, help="also write the recommendations to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="print only the final recommendations, not each step")
    parser.add_argument("--show", type=int, default=20, help="rows printed per step")
    args = parser.parse_args()

    pipeline = RetrievalPipeline.from_paths(args.catalog, args.artifacts, args.index_dir,
                                            rebuild=args.rebuild or args.command == "build")
    if args.command == "build":
        print(f"Indexed {len(pipeline.recipes):,} recipes × {pipeline.recipe_index.embeddings.shape[1]} dims "
              f"in {args.index_dir}")
        return
    user = json.loads(args.user.read_text())
    results = pipeline.recommend(user, args.candidates, args.top_k, log=None if args.quiet else print,
                                 show=args.show)
    print_recommendations(user, results)
    if args.output:
        windows = pipeline.user_states.get(user.get("user_id"), {}).get("windows", [])
        args.output.write_text(json.dumps({"user_id": user.get("user_id"), "windows": windows,
                                           "recommendations": results}, indent=2))


if __name__ == "__main__":
    main()
