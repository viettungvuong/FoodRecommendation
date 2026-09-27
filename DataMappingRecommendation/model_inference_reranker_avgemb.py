"""Predict whether a user will like a recipe with the GRU + cross-attention reranker.

Input: a user (their rated recipes with dates, from which the model builds the profile and history) and one or
more recipes described by ingredients, price and nutrients. Output: P(like) for each recipe.

The model of model_training_reranker_avgemb.ipynb cuts the history into windows wherever two ratings are
window_gap_days (about 6 months) or more apart, runs the GRU over each window from a fresh state and lets each
recipe attend over [user profile; the final state of each window]. Ratings are centered on the user's mean.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

DEFAULT_ARTIFACTS = Path(__file__).resolve().parent / "model_artifacts" / "gru_xattn_reranker"
THRESHOLD = 0.5  # P(positive) at or above this is classified positive, as in the notebook's metrics.

# Recipes here give nutrients per serving in everyday units. The training data holds Food.com's % daily value times
# the daily value (map_recipe_nutrients.ipynb), which is 100 × the amount, except calories: Food.com gives those in
# kcal, not % daily value, and the notebook still multiplied them by 2000. Model column → (input key, scale).
NUTRIENT_INPUTS = {
    "calories (g)": ("calories_kcal", 2000.0),
    "total fat (g)": ("fat_g", 100.0),
    "sugar (g)": ("sugar_g", 100.0),
    "sodium (g)": ("sodium_mg", 100.0),
    "protein (g)": ("protein_g", 100.0),
    "saturated fat (g)": ("saturated_fat_g", 100.0),
    "carbohydrates (g)": ("carbohydrates_g", 100.0),
}

# ---------------------------------------------------------------------------------------------------------------
# Ingredient NER: GLiNER tags, then the rules of map_recipe_nutrients.ipynb split them into product / adj / verb.
# ---------------------------------------------------------------------------------------------------------------
INGREDIENT_NER_MODEL = "urchade/gliner_large-v2.1"
INGREDIENT_NER_LABELS = ["food", "color", "texture", "preparation", "adjective"]
INGREDIENT_NER_THRESHOLD = 0.3
INGREDIENT_NER_PREFIX = "The recipe calls for "
INGREDIENT_NER_SUFFIX = "."
# Numeric measures joined to a word, % or a hyphen ("10 inch", "1%", "2 oz"): never products, adj or verbs.
NUMBER_REGEX = re.compile(r"\d+(?:[./]\d+)?(?:%|(?:\s*[-–]\s*|\s+)?[A-Za-z]+)")
INGREDIENT_PREPARATION_WORDS = {
    "ground", "minced", "chopped", "diced", "sliced", "peeled", "grated",
    "shredded", "crushed", "mashed", "cooked", "uncooked", "boiled",
    "steamed", "fried", "roasted", "toasted", "baked", "grilled", "smoked",
    "canned", "drained", "rinsed", "sifted", "beaten", "whipped", "melted",
    "softened", "chilled", "frozen", "thawed", "dried", "dehydrated",
    "seeded", "deseeded", "pitted", "shelled", "trimmed", "cubed",
}
INGREDIENT_ADJECTIVE_WORDS = {
    "yellow", "red", "green", "white", "black", "purple", "brown",
    "fresh", "raw", "ripe", "unripe", "boneless", "skinless", "lean",
    "large", "small", "medium", "fine", "coarse", "organic", "unsalted",
    "salted", "sweet", "sour", "hot", "cold", "warm", "extra", "virgin",
}
PROTECTED_INGREDIENT_COMPOUNDS = {
    "sour cream", "sweet potato", "sweet corn", "hot dog", "ice cream",
    "cream cheese", "peanut butter", "baking powder", "baking soda",
    "red wine", "white wine", "brown sugar", "black bean", "green bean",
}

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
_lemmatizer = WordNetLemmatizer()
# map_recipe_nutrients.ipynb's STOP_WORDS: NLTK's English list without single letters and negations.
STOP_WORDS = set(stopwords.words("english")) - {"a", "d", "i", "m", "no", "nor", "not", "o", "s", "t", "y"}


def normalize_price_text(value):
    """Lowercase, drop "(...)" asides, punctuation and stop words, then singularize each word.

    map_recipe_nutrients.ipynb singularizes with its own plural rules; WordNet's noun lemma stands in for them
    here. The reranker singularizes with WordNet again anyway, so the tokens it sees are the same."""
    text = re.sub(r"\(.*?\)", " ", str(value).lower())
    text = re.sub(r"[^a-z0-9%\s]", " ", text)
    words = [word for word in text.split() if word not in STOP_WORDS and word.strip("%")]
    return " ".join(_lemmatizer.lemmatize(word, "n") for word in words)


def ingredient_entity_columns(phrase, predictions):
    """product / adj / verb lists of one ingredient phrase from its GLiNER entities (map_recipe_nutrients.ipynb)."""
    valid = [entity for entity in predictions if 0 <= entity["start"] < entity["end"] <= len(phrase)]
    # Strongest non-overlapping food spans, so "cream cheese" is not split into cream and cheese.
    foods = []
    for entity in sorted((e for e in valid if e["label"] == "food"),
                         key=lambda e: (-e["score"], -(e["end"] - e["start"]), e["start"])):
        if not any(entity["start"] < old["end"] and old["start"] < entity["end"] for old in foods):
            foods.append(entity)
    foods.sort(key=lambda e: e["start"])
    quantity_spans = [(match.start(), match.end()) for match in NUMBER_REGEX.finditer(phrase)]

    def overlaps_quantity(start, end):
        return any(start < quantity_end and quantity_start < end for quantity_start, quantity_end in quantity_spans)

    known_words = INGREDIENT_PREPARATION_WORDS | INGREDIENT_ADJECTIVE_WORDS
    modifiers = []
    label_columns = {"color": "adj", "texture": "adj", "adjective": "adj", "preparation": "verb"}
    for entity in valid:
        column = label_columns.get(entity["label"])
        if column is None:
            continue
        # Skip tags covering the whole phrase or a whole food span, and quantities.
        if entity["start"] == 0 and entity["end"] == len(phrase):
            continue
        if any(entity["start"] <= food["start"] and entity["end"] >= food["end"] for food in foods):
            continue
        if overlaps_quantity(entity["start"], entity["end"]):
            continue
        words = list(re.finditer(r"[A-Za-z]+", phrase[entity["start"]:entity["end"]]))
        if words and all(word.group().lower() in known_words for word in words):
            for word in words:
                text = word.group().lower()
                modifiers.append((entity["start"] + word.start(), entity["start"] + word.end(),
                                  "verb" if text in INGREDIENT_PREPARATION_WORDS else "adj", text))
        else:
            modifiers.append((entity["start"], entity["end"], column, phrase[entity["start"]:entity["end"]].lower()))
    for word in re.finditer(r"[A-Za-z]+", phrase):
        text = word.group().lower()
        if not overlaps_quantity(word.start(), word.end()) and text in known_words:
            modifiers.append((word.start(), word.end(),
                              "verb" if text in INGREDIENT_PREPARATION_WORDS else "adj", text))
    # Prefer a preparation verb over an overlapping adjective at the same span.
    distinct = {}
    for start, end, column, text in modifiers:
        if (start, end) not in distinct or column == "verb":
            distinct[start, end] = (start, end, column, text)
    modifiers = sorted(distinct.values())
    result = {"product": [], "adj": [], "verb": []}
    for _, _, column, text in modifiers:
        if text not in result[column]:
            result[column].append(text)
    for food in foods:
        surface = phrase[food["start"]:food["end"]]
        if normalize_price_text(surface) in PROTECTED_INGREDIENT_COMPOUNDS:
            core = normalize_price_text(surface)
        else:
            chars = list(surface)
            for start, end in [(start, end) for start, end, _, _ in modifiers] + quantity_spans:
                if food["start"] <= start and end <= food["end"]:
                    chars[start - food["start"]:end - food["start"]] = " " * (end - start)
            core = normalize_price_text("".join(chars))
        if core and core not in result["product"]:
            result["product"].append(core)
    return result


class IngredientTagger:
    """GLiNER over ingredient phrases, with the prompt and offset filter of map_recipe_nutrients.ipynb."""

    def __init__(self, model_name=INGREDIENT_NER_MODEL):
        from gliner import GLiNER  # Imported here so the rest of the module works without gliner installed.
        self.model = GLiNER.from_pretrained(model_name)

    def __call__(self, phrases):
        """One {"product", "adj", "verb"} dict per phrase."""
        texts = [INGREDIENT_NER_PREFIX + phrase + INGREDIENT_NER_SUFFIX for phrase in phrases]
        predictions = self.model.inference(texts, INGREDIENT_NER_LABELS, threshold=INGREDIENT_NER_THRESHOLD,
                                           flat_ner=False, multi_label=True)
        # Keep spans wholly inside the ingredient, with offsets into the phrase, so the prompt words and the
        # sentence punctuation never become entities.
        offset = len(INGREDIENT_NER_PREFIX)
        return [ingredient_entity_columns(phrase, [
            {**entity, "start": entity["start"] - offset, "end": entity["end"] - offset}
            for entity in entities if offset <= entity["start"] < entity["end"] <= offset + len(phrase)])
            for phrase, entities in zip(phrases, predictions)]


# ---------------------------------------------------------------------------------------------------------------
# The model: the classes of model_training_reranker_avgemb.ipynb.
# ---------------------------------------------------------------------------------------------------------------
class ItemEncoder(nn.Module):
    """Recipe tokens and [price, nutrients] → item_emb (spec 2.1)."""

    def __init__(self, vocab_size, n_numeric, n_fields, config):
        super().__init__()
        d = config["d"]
        self.token_embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.field_embedding = nn.Embedding(n_fields + 1, d, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d, config["n_heads"], dim_feedforward=2 * d,
                                           dropout=config["text_dropout"], batch_first=True, norm_first=True)
        self.self_attention = nn.TransformerEncoder(layer, config["text_layers"], enable_nested_tensor=False)
        self.numeric_projection = nn.Linear(n_numeric, d)
        self.fusion = nn.Linear(2 * d, d)

    def forward(self, token_ids, field_ids, numeric):
        padding = token_ids == 0
        tokens = self.self_attention(self.token_embedding(token_ids) + self.field_embedding(field_ids),
                                     src_key_padding_mask=padding)
        keep = (~padding).unsqueeze(-1).to(tokens.dtype)
        text_emb = (tokens * keep).sum(1) / keep.sum(1)  # Every recipe has at least one token.
        return self.fusion(torch.cat([text_emb, self.numeric_projection(numeric)], dim=-1))


class UserEncoder(nn.Module):
    """Standardized [mean log price, mean log nutrients] of the ratings before t → U_profile (spec 2.2)."""

    def __init__(self, n_profile, config):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_profile, config["d"]), nn.ReLU(), nn.Linear(config["d"], config["d"]))

    def forward(self, profile):
        return self.mlp(profile)


class HistoryEncoder(nn.Module):
    """Item embeddings and normalized centered ratings of the history in date order → every GRU state H (spec 2.3).

    use_rating=False rebuilds checkpoints trained before ratings were a history input."""

    def __init__(self, config, use_rating=True):
        super().__init__()
        self.rating_projection = nn.Linear(1, config["d"]) if use_rating else None
        self.gru = nn.GRU(config["d"], config["d"], batch_first=True)

    def forward(self, history_emb, history_rating, history_len):
        """H (B, T, d) and its padding mask (B, T), True = padded."""
        steps_in = history_emb
        if self.rating_projection is not None:
            steps_in = history_emb + self.rating_projection(history_rating.unsqueeze(-1))
        packed = pack_padded_sequence(steps_in, history_len.cpu(), batch_first=True, enforce_sorted=False)
        states, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=history_emb.size(1))
        steps = torch.arange(history_emb.size(1), device=history_emb.device)
        return states, steps >= history_len.to(history_emb.device).unsqueeze(1)


class GRUCrossAttentionReranker(nn.Module):
    """Candidate item_emb cross-attends over [U_profile; H] (spec 3.1).

    The notebook's version also holds the catalog as buffers; recipes here are new, so items are encoded
    directly with item_encoder."""

    def __init__(self, vocab_size, n_numeric, n_profile, n_fields, config, use_rating=True):
        super().__init__()
        d = config["d"]
        self.item_encoder = ItemEncoder(vocab_size, n_numeric, n_fields, config)
        self.user_encoder = UserEncoder(n_profile, config)
        self.history_encoder = HistoryEncoder(config, use_rating)
        self.key_type = nn.Embedding(2, d)  # 0: profile token, 1: GRU state.
        self.cross_attention = nn.MultiheadAttention(d, config["n_heads"], dropout=config["dropout"],
                                                     batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(d, 1))

    def user_keys(self, history_emb, history_rating, history_len, profile):
        """Keys and values [U_profile; h_1 … h_T] (B, 1+T, d) and their padding mask (True = padded)."""
        states, padded = self.history_encoder(history_emb, history_rating, history_len)
        profile_token = self.user_encoder(profile).unsqueeze(1)
        keys = torch.cat([profile_token + self.key_type.weight[0], states + self.key_type.weight[1]], dim=1)
        return keys, torch.cat([padded.new_zeros(len(padded), 1), padded], dim=1)

    def score_candidates(self, candidate_emb, keys, key_padding_mask):
        """Logits (B, C) for candidates (B, C, d)."""
        attended, _ = self.cross_attention(candidate_emb, keys, keys, key_padding_mask=key_padding_mask,
                                           need_weights=False)
        return self.head(torch.cat([attended, candidate_emb], dim=-1)).squeeze(-1)


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
    """Day number (days since 1970-01-01) of a history date such as "2024-01-31"; a missing date is day 0."""
    if value is None or value == "":
        return 0
    return int(np.datetime64(str(value)[:10], "D").astype(np.int64))


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


# Checkpoints saved before the encoders became separate classes, with nutrients only in the item embedding and no
# history ratings.
LEGACY_KEY_PREFIXES = {"item_encoder.nutrient_projection.": "item_encoder.numeric_projection.",
                       "profile_encoder.": "user_encoder.mlp.", "gru.": "history_encoder.gru."}


class Reranker:
    """The trained model with the vocabulary and normalization statistics saved next to it.

    A config.json with max_windows is the windowed model of model_training_reranker_avgemb.ipynb: history cut into
    windows at gaps of window_gap_days or more, centered ratings, one key per window. Otherwise it is an older model:
    the last t_max recipes in one sequence."""

    def __init__(self, artifact_dir=DEFAULT_ARTIFACTS, tagger=None, device=None):
        artifact_dir = Path(artifact_dir)
        saved = json.loads((artifact_dir / "config.json").read_text())
        self.config = saved["config"]
        self.text_fields = saved["text_fields"]  # {"product": ["n", 1], ...}: WordNet part of speech, field id.
        self.nutrient_columns = saved["nutrient_columns"]
        self.token_lookup = {token: index for index, token in enumerate(saved["vocabulary"])}
        self.item_mean, self.item_std = np.array(saved["item_log1p_mean"]), np.array(saved["item_log1p_std"])
        self.profile_mean, self.profile_std = np.array(saved["profile_mean"]), np.array(saved["profile_std"])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.windowed = "max_windows" in self.config

        state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        for old, new in LEGACY_KEY_PREFIXES.items():
            state = {new + key[len(old):] if key.startswith(old) else key: value for key, value in state.items()}
        # 8 inputs: [price, nutrients]. Legacy checkpoints take the 7 nutrients only.
        self.n_numeric = state["item_encoder.numeric_projection.weight"].shape[1]
        n_profile = state["user_encoder.mlp.0.weight"].shape[1]
        self.use_rating = "history_encoder.rating_projection.weight" in state
        if self.windowed:
            # History ratings are centered on the mean rating before t, not scaled.
            self.rating_offset, self.rating_scale = 0.0, 1.0
            self.model = WindowGRUCrossAttentionReranker(len(self.token_lookup), self.n_numeric, n_profile,
                                                         len(self.text_fields), self.config)
        else:
            # Older models: (centered + offset) / scale, offset 5 and scale 10 for [0, 1] ratings, else offset 0.
            self.rating_offset = saved.get("history_rating_offset", 0.0)
            self.rating_scale = saved.get("history_rating_scale")
            if self.use_rating and self.rating_scale is None:
                raise ValueError("model.pt uses history ratings, but config.json has no history_rating_scale.")
            self.model = GRUCrossAttentionReranker(len(self.token_lookup), self.n_numeric, n_profile,
                                                   len(self.text_fields), self.config, self.use_rating)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.tagger = tagger

    @lru_cache(maxsize=None)
    def _normalize_word(self, word, pos):
        """Lemmatize with the column's part of speech, then singularize (spec 1.1)."""
        return _lemmatizer.lemmatize(_lemmatizer.lemmatize(word, pos), "n")

    def tag_recipes(self, recipes):
        """Set each untagged recipe's "entities" (product / adj / verb, one list per ingredient) with one NER pass."""
        untagged = [recipe for recipe in recipes if "entities" not in recipe]
        if not untagged:
            return
        if self.tagger is None:
            self.tagger = IngredientTagger()
        tagged = iter(self.tagger([phrase for recipe in untagged for phrase in recipe["ingredients"]]))
        for recipe in untagged:
            ingredients = [next(tagged) for _ in recipe["ingredients"]]
            recipe["entities"] = {field: [ingredient[field] for ingredient in ingredients]
                                  for field in self.text_fields}

    def recipe_tokens(self, entities):
        """(token, field id) pairs, in order, each kept once (the notebook's recipe_tokens)."""
        tokens = {}
        for field, (pos, field_id) in self.text_fields.items():
            for ingredient in entities[field]:
                for phrase in ingredient:
                    for word in re.sub(r"[^a-z0-9]+", " ", phrase.lower()).split():
                        tokens.setdefault((self._normalize_word(word, pos), field_id), None)
        tokens = list(tokens)[:self.config["max_tokens"]]
        return tokens or [("<unk>", self.text_fields["product"][1])]  # An empty recipe gets one <unk>.

    def log_numeric(self, recipe):
        """log1p [price, nutrients] of one recipe, nutrients in the training data's units."""
        nutrients = [recipe["nutrients"][NUTRIENT_INPUTS[column][0]] * NUTRIENT_INPUTS[column][1]
                     for column in self.nutrient_columns]
        values = [recipe["price"], *nutrients]
        return np.log1p(np.asarray(values, dtype=np.float64))

    @torch.no_grad()
    def encode_items(self, recipes):
        """item_emb (N, d) and the tokens of each recipe."""
        tokens = [self.recipe_tokens(recipe["entities"]) for recipe in recipes]
        width = max(len(recipe_tokens) for recipe_tokens in tokens)
        token_ids = torch.zeros(len(recipes), width, dtype=torch.long)
        field_ids = torch.zeros_like(token_ids)
        for row, recipe_tokens in enumerate(tokens):
            token_ids[row, :len(recipe_tokens)] = torch.tensor([self.token_lookup.get(token, 1)
                                                                for token, _ in recipe_tokens])
            field_ids[row, :len(recipe_tokens)] = torch.tensor([field for _, field in recipe_tokens])
        numeric = (np.stack([self.log_numeric(recipe) for recipe in recipes]) - self.item_mean) / self.item_std
        numeric = torch.as_tensor(numeric[:, -self.n_numeric:], dtype=torch.float32)
        item_emb = self.model.item_encoder(token_ids.to(self.device), field_ids.to(self.device),
                                           numeric.to(self.device))
        return item_emb, tokens

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

    @torch.no_grad()
    def predict(self, history, recipes):
        """P(like) of each recipe for the user who rated the recipes in history.

        Every recipe needs "ingredients" (or "entities"), "price" and "nutrients" (keys of NUTRIENT_INPUTS); history
        recipes also need "date" and "rating". Returns one dict per recipe, in order."""
        if not history:
            raise ValueError("The reranker needs at least one recipe in the user's history.")
        history = sorted(history, key=lambda recipe: str(recipe.get("date", "")))
        self.tag_recipes([*history, *recipes])
        item_emb, tokens = self.encode_items([*history, *recipes])
        p_likes = self.score(history, item_emb[:len(history)], item_emb[len(history):])
        results = []
        for recipe, recipe_tokens, p_like in zip(recipes, tokens[len(history):], p_likes.tolist()):
            results.append({"recipe": recipe, "p_like": p_like, "like": p_like >= THRESHOLD,
                            "known_tokens": sum(token in self.token_lookup for token, _ in recipe_tokens),
                            "tokens": len(recipe_tokens)})
        return results


# ---------------------------------------------------------------------------------------------------------------
# Example users (x) and recipes to score (y). Nutrients are per serving; price is the ingredients' total in $,
# like total_prices in the training data (median about $33).
# ---------------------------------------------------------------------------------------------------------------
USERS = [
    {"user_id": "healthy_savoury",
     "history": [
         {"name": "lemon herb baked salmon", "date": "2012-01-14", "rating": 5, "price": 38.5,
          "ingredients": ["salmon fillets", "lemon", "fresh dill", "minced garlic", "olive oil", "sea salt"],
          "nutrients": {"calories_kcal": 310, "fat_g": 18, "sugar_g": 1, "sodium_mg": 420, "protein_g": 34,
                        "saturated_fat_g": 3, "carbohydrates_g": 3}},
         {"name": "garlic ginger chicken stir fry", "date": "2012-02-03", "rating": 5, "price": 29.0,
          "ingredients": ["boneless skinless chicken breasts", "soy sauce", "fresh ginger", "garlic cloves",
                          "red bell pepper", "broccoli florets", "vegetable oil"],
          "nutrients": {"calories_kcal": 280, "fat_g": 10, "sugar_g": 5, "sodium_mg": 890, "protein_g": 32,
                        "saturated_fat_g": 2, "carbohydrates_g": 14}},
         {"name": "black bean and corn salad", "date": "2012-03-20", "rating": 4, "price": 18.2,
          "ingredients": ["canned black beans", "frozen corn", "red onion", "fresh cilantro", "lime juice",
                          "extra virgin olive oil", "ground cumin"],
          "nutrients": {"calories_kcal": 220, "fat_g": 7, "sugar_g": 4, "sodium_mg": 310, "protein_g": 9,
                        "saturated_fat_g": 1, "carbohydrates_g": 33}},
         {"name": "classic banana bread", "date": "2012-04-08", "rating": 2, "price": 21.4,
          "ingredients": ["ripe bananas", "all-purpose flour", "white sugar", "baking soda", "eggs",
                          "melted butter", "vanilla extract"],
          "nutrients": {"calories_kcal": 330, "fat_g": 12, "sugar_g": 28, "sodium_mg": 260, "protein_g": 5,
                        "saturated_fat_g": 7, "carbohydrates_g": 52}},
         {"name": "roasted vegetable couscous", "date": "2012-05-11", "rating": 5, "price": 24.7,
          "ingredients": ["couscous", "zucchini", "red onion", "cherry tomatoes", "chickpeas", "olive oil",
                          "ground cumin", "fresh parsley"],
          "nutrients": {"calories_kcal": 360, "fat_g": 11, "sugar_g": 7, "sodium_mg": 380, "protein_g": 12,
                        "saturated_fat_g": 1.5, "carbohydrates_g": 54}},
     ]},
    {"user_id": "sweet_tooth_baker",
     "history": [
         {"name": "chocolate chip cookies", "date": "2011-11-02", "rating": 5, "price": 26.9,
          "ingredients": ["butter", "brown sugar", "white sugar", "eggs", "all-purpose flour", "baking soda",
                          "semi-sweet chocolate chips"],
          "nutrients": {"calories_kcal": 210, "fat_g": 11, "sugar_g": 18, "sodium_mg": 140, "protein_g": 2,
                        "saturated_fat_g": 6, "carbohydrates_g": 27}},
         {"name": "classic banana bread", "date": "2011-12-18", "rating": 5, "price": 21.4,
          "ingredients": ["ripe bananas", "all-purpose flour", "white sugar", "baking soda", "eggs",
                          "melted butter", "vanilla extract", "chopped walnuts"],
          "nutrients": {"calories_kcal": 360, "fat_g": 15, "sugar_g": 28, "sodium_mg": 260, "protein_g": 6,
                        "saturated_fat_g": 7, "carbohydrates_g": 52}},
         {"name": "new york cheesecake", "date": "2012-02-14", "rating": 5, "price": 41.3,
          "ingredients": ["cream cheese", "white sugar", "sour cream", "eggs", "graham cracker crumbs",
                          "melted butter", "vanilla extract", "lemon juice"],
          "nutrients": {"calories_kcal": 520, "fat_g": 38, "sugar_g": 32, "sodium_mg": 390, "protein_g": 9,
                        "saturated_fat_g": 22, "carbohydrates_g": 38}},
         {"name": "spicy beef tacos", "date": "2012-03-09", "rating": 3, "price": 33.6,
          "ingredients": ["lean ground beef", "taco seasoning", "flour tortillas", "shredded cheddar cheese",
                          "diced tomatoes", "sour cream", "shredded lettuce"],
          "nutrients": {"calories_kcal": 540, "fat_g": 29, "sugar_g": 4, "sodium_mg": 1120, "protein_g": 31,
                        "saturated_fat_g": 13, "carbohydrates_g": 36}},
         {"name": "apple crumble", "date": "2012-04-22", "rating": 4, "price": 19.8,
          "ingredients": ["granny smith apples", "rolled oats", "brown sugar", "all-purpose flour",
                          "cold butter", "ground cinnamon"],
          "nutrients": {"calories_kcal": 340, "fat_g": 13, "sugar_g": 33, "sodium_mg": 95, "protein_g": 3,
                        "saturated_fat_g": 8, "carbohydrates_g": 55}},
     ]},
]

TEST_RECIPES = [
    {"name": "grilled chicken quinoa bowl", "price": 31.5,
     "ingredients": ["boneless skinless chicken breasts", "quinoa", "baby spinach", "cherry tomatoes", "cucumber",
                     "feta cheese", "olive oil", "lemon juice"],
     "nutrients": {"calories_kcal": 450, "fat_g": 17, "sugar_g": 5, "sodium_mg": 520, "protein_g": 38,
                   "saturated_fat_g": 4, "carbohydrates_g": 36}},
    {"name": "double chocolate brownies", "price": 23.9,
     "ingredients": ["unsalted butter", "white sugar", "eggs", "cocoa powder", "all-purpose flour",
                     "semi-sweet chocolate chips", "vanilla extract", "salt"],
     "nutrients": {"calories_kcal": 290, "fat_g": 15, "sugar_g": 27, "sodium_mg": 110, "protein_g": 3,
                   "saturated_fat_g": 9, "carbohydrates_g": 38}},
    {"name": "thai green curry with tofu", "price": 27.4,
     "ingredients": ["firm tofu", "green curry paste", "coconut milk", "green beans", "red bell pepper",
                     "fish sauce", "fresh basil", "jasmine rice"],
     "nutrients": {"calories_kcal": 480, "fat_g": 26, "sugar_g": 6, "sodium_mg": 780, "protein_g": 17,
                   "saturated_fat_g": 18, "carbohydrates_g": 46}},
    {"name": "bacon cheeseburger", "price": 36.8,
     "ingredients": ["ground beef", "bacon", "cheddar cheese", "hamburger buns", "lettuce", "tomato",
                     "yellow onion", "ketchup", "mayonnaise"],
     "nutrients": {"calories_kcal": 820, "fat_g": 52, "sugar_g": 9, "sodium_mg": 1450, "protein_g": 45,
                   "saturated_fat_g": 20, "carbohydrates_g": 42}},
    {"name": "greek salad", "price": 19.6,
     "ingredients": ["cucumber", "tomatoes", "red onion", "kalamata olives", "feta cheese", "dried oregano",
                     "extra virgin olive oil", "red wine vinegar"],
     "nutrients": {"calories_kcal": 230, "fat_g": 19, "sugar_g": 5, "sodium_mg": 610, "protein_g": 6,
                   "saturated_fat_g": 6, "carbohydrates_g": 10}},
    {"name": "cinnamon rolls with cream cheese frosting", "price": 28.2,
     "ingredients": ["all-purpose flour", "active dry yeast", "milk", "brown sugar", "ground cinnamon",
                     "softened butter", "cream cheese", "powdered sugar"],
     "nutrients": {"calories_kcal": 480, "fat_g": 20, "sugar_g": 36, "sodium_mg": 320, "protein_g": 7,
                   "saturated_fat_g": 12, "carbohydrates_g": 69}},
]


def print_predictions(reranker, user, results):
    history = sorted(user["history"], key=lambda recipe: str(recipe.get("date", "")))
    window_of = {}  # History position → window number (1 = most recent), for the recipes the GRU sees.
    if reranker.windowed:
        positions, _, _ = reranker.windows(history)
        window_of = {int(position): window + 1 for window, row in enumerate(positions) for position in row
                     if position >= 0}
    print(f"User {user.get('user_id', '?')}: {len(history)} rated recipes")
    for position, recipe in enumerate(history):
        window = f"  window {window_of.get(position, '-')}" if reranker.windowed else ""
        print(f"  {recipe.get('date', '')}  rating {recipe.get('rating', '?')}{window}  {recipe['name']}")
    print(f"  {'recipe':<42} {'P(like)':>8}  prediction  known tokens")
    for result in sorted(results, key=lambda result: -result["p_like"]):
        print(f"  {result['recipe']['name']:<42} {result['p_like']:>8.4f}  {'like' if result['like'] else 'dislike':<10}"
              f"  {result['known_tokens']}/{result['tokens']}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="folder holding model.pt and config.json")
    parser.add_argument("--input", type=Path,
                        help='JSON file {"user": {"user_id", "history": [...]}, "recipes": [...]}; '
                             "defaults to the USERS and TEST_RECIPES of this file")
    args = parser.parse_args()

    reranker = Reranker(args.artifacts)
    if args.input:
        example = json.loads(args.input.read_text())
        users, recipes = [example["user"]], example["recipes"]
    else:
        users, recipes = USERS, TEST_RECIPES
    print(f"P(like) at or above {THRESHOLD} → like\n")
    for user in users:
        print_predictions(reranker, user, reranker.predict(user["history"], recipes))


if __name__ == "__main__":
    main()
