"""Classify one candidate recipe for one user with the GRU + cross-attention reranker.

The weights come from model_training_reranker_avgemb_approach.ipynb (model.pt and config.json in its artifact
folder). Every recipe, in the history and the candidate, is given as raw text: a name and its ingredient phrases.
GLiNER tags each ingredient (as in map_recipe_nutrients.ipynb) into product / adj / verb, the reranker's text
normalization turns those into tokens, and the model returns P(the user rates the candidate positively).

    python model_inference_reranker_avgemb_approach.py                      # a randomly generated user
    python model_inference_reranker_avgemb_approach.py --input user.json    # your own user and candidate
    python model_inference_reranker_avgemb_approach.py --artifacts /content/drive/.../gru_xattn_reranker

Input JSON (the random example prints in this form, so it can be saved and edited):

    {"user_id": "demo",
     "history": [{"name": ..., "ingredients": [...], "price": 12.5, "nutrients": {"calories (g)": ..., ...},
                  "rating": 5, "date": "2010-03-02"}, ...],
     "candidate": {"name": ..., "ingredients": [...], "price": ..., "nutrients": {...}, "rating": 4}}

Prices and nutrients use the units of input_stage2/recipe_user_ratings_prices.csv. Every history recipe needs its
rating (0-5): the GRU sees it centered on the mean of the history's ratings and divided by the scale saved in
config.json (spec 2.3). The candidate's "rating" is optional: when given, it is the ground truth that the
prediction is compared with.

Requires: torch, numpy, nltk (wordnet, stopwords) and gliner (pip install gliner; the first run downloads the
GLiNER model).
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
# The model: the classes of model_training_reranker_avgemb_approach.ipynb.
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


# Checkpoints saved before the encoders became separate classes, with nutrients only in the item embedding and no
# history ratings.
LEGACY_KEY_PREFIXES = {"item_encoder.nutrient_projection.": "item_encoder.numeric_projection.",
                       "profile_encoder.": "user_encoder.mlp.", "gru.": "history_encoder.gru."}


class Reranker:
    """The trained model with the vocabulary and normalization statistics saved next to it."""

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

        state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        for old, new in LEGACY_KEY_PREFIXES.items():
            state = {new + key[len(old):] if key.startswith(old) else key: value for key, value in state.items()}
        # 8 inputs: [price, nutrients]. Legacy checkpoints take the 7 nutrients only.
        self.n_numeric = state["item_encoder.numeric_projection.weight"].shape[1]
        self.use_rating = "history_encoder.rating_projection.weight" in state
        self.rating_scale = saved.get("history_rating_scale")
        if self.use_rating and self.rating_scale is None:
            raise ValueError("model.pt uses history ratings, but config.json has no history_rating_scale.")
        self.model = GRUCrossAttentionReranker(
            len(self.token_lookup), self.n_numeric, state["user_encoder.mlp.0.weight"].shape[1],
            len(self.text_fields), self.config, self.use_rating)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.tagger = tagger

    @lru_cache(maxsize=None)
    def _normalize_word(self, word, pos):
        """Lemmatize with the column's part of speech, then singularize (spec 1.1)."""
        return _lemmatizer.lemmatize(_lemmatizer.lemmatize(word, pos), "n")

    def recipe_entities(self, recipe):
        """product / adj / verb: one list of phrases per ingredient, from the NER."""
        if self.tagger is None:
            self.tagger = IngredientTagger()
        tagged = self.tagger(list(recipe["ingredients"]))
        return {field: [ingredient[field] for ingredient in tagged] for field in self.text_fields}

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
        """log1p [price, nutrients] of one recipe."""
        values = [recipe["price"], *[recipe["nutrients"][column] for column in self.nutrient_columns]]
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

    @torch.no_grad()
    def predict(self, history, candidate):
        """P(positive) of candidate for a user with these rated recipes, plus what went into it.

        Every recipe needs "ingredients", "price" and "nutrients"; history recipes also need "date"."""
        if not history:
            raise ValueError("The reranker needs at least one recipe in the history.")
        history = sorted(history, key=lambda recipe: recipe["date"])
        recipes = [*history, candidate]
        for recipe in recipes:
            if "entities" not in recipe:
                recipe["entities"] = self.recipe_entities(recipe)
        item_emb, tokens = self.encode_items(recipes)
        # History: the last t_max recipes in date order, each rating centered on the mean of all the history's
        # ratings and divided by the training scale. Profile: the mean over all of them.
        recent = item_emb[:len(history)][-self.config["t_max"]:].unsqueeze(0)
        history_rating = self.normalized_ratings(history)[-self.config["t_max"]:]
        profile = (np.mean([self.log_numeric(recipe) for recipe in history], axis=0)
                   - self.profile_mean) / self.profile_std
        keys, key_padding_mask = self.model.user_keys(
            recent, torch.as_tensor(history_rating, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.tensor([recent.size(1)]), torch.as_tensor(profile, dtype=torch.float32,
                                                            device=self.device).unsqueeze(0))
        logit = self.model.score_candidates(item_emb[-1:].unsqueeze(0), keys, key_padding_mask)[0, 0]
        p_positive = float(torch.sigmoid(logit))
        return {"p_positive": p_positive, "predicted_label": int(p_positive >= THRESHOLD),
                "history": history, "candidate": candidate, "tokens": tokens,
                "history_rating": self.normalized_ratings(history)}

    def normalized_ratings(self, history):
        """(rating - mean rating of the history) / the training scale, per history recipe in order. Zeros for a
        checkpoint without history ratings (the model then ignores them)."""
        if not self.use_rating:
            return np.zeros(len(history))
        ratings = np.array([recipe["rating"] for recipe in history], dtype=np.float64)
        return (ratings - ratings.mean()) / self.rating_scale


# ---------------------------------------------------------------------------------------------------------------
# A randomly generated user: two rated recipes and one candidate.
# ---------------------------------------------------------------------------------------------------------------
RECIPE_POOL = [
    {"name": "garlic ginger chicken stir fry",
     "ingredients": ["boneless skinless chicken breasts", "soy sauce", "fresh ginger", "garlic cloves",
                     "red bell pepper", "broccoli florets", "vegetable oil", "cornstarch"]},
    {"name": "creamy mushroom risotto",
     "ingredients": ["arborio rice", "sliced mushrooms", "chicken broth", "dry white wine", "unsalted butter",
                     "grated parmesan cheese", "yellow onion", "salt and pepper"]},
    {"name": "black bean and corn salad",
     "ingredients": ["canned black beans", "frozen corn", "red onion", "fresh cilantro", "lime juice",
                     "extra virgin olive oil", "ground cumin"]},
    {"name": "classic banana bread",
     "ingredients": ["ripe bananas", "all-purpose flour", "white sugar", "baking soda", "eggs", "melted butter",
                     "vanilla extract", "chopped walnuts"]},
    {"name": "spicy beef tacos",
     "ingredients": ["lean ground beef", "taco seasoning", "flour tortillas", "shredded cheddar cheese",
                     "diced tomatoes", "sour cream", "shredded lettuce", "hot sauce"]},
    {"name": "lemon herb baked salmon",
     "ingredients": ["salmon fillets", "lemon", "fresh dill", "minced garlic", "olive oil", "sea salt",
                     "black pepper"]},
    {"name": "chocolate chip cookies",
     "ingredients": ["butter", "brown sugar", "white sugar", "eggs", "all-purpose flour", "baking powder",
                     "semi-sweet chocolate chips"]},
]


def random_numbers(reranker, rng):
    """Price and nutrients drawn around the training recipes' log1p mean, in the dataset's units."""
    values = np.expm1(reranker.item_mean + reranker.item_std * np.clip(rng.normal(0, 0.5, len(reranker.item_mean)),
                                                                       -1.5, 1.5))
    return {"price": round(float(values[0]), 2),
            "nutrients": {column: round(float(value), 1)
                          for column, value in zip(reranker.nutrient_columns, values[1:])}}


def random_example(reranker, seed):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(RECIPE_POOL), size=3, replace=False)
    days = np.sort(rng.choice(np.arange(np.datetime64("2010-01-01"), np.datetime64("2012-01-01")), size=2,
                              replace=False))
    history = [{**RECIPE_POOL[pick], **random_numbers(reranker, rng), "rating": int(rng.integers(0, 6)),
                "date": str(day)} for pick, day in zip(picks[:2], days)]
    candidate = {**RECIPE_POOL[picks[2]], **random_numbers(reranker, rng)}
    return {"user_id": f"random_user_{seed}", "history": history, "candidate": candidate}


def describe(recipe, tokens, token_lookup):
    known = sum(token in token_lookup for token, _ in tokens)
    lines = [f"    ingredients: {', '.join(recipe['ingredients'])}"]
    lines += [f"    {field:<7}  {[phrase for ingredient in recipe['entities'][field] for phrase in ingredient]}"
              for field in recipe["entities"]]
    lines.append(f"    tokens:  {len(tokens)}, {known} in the model's vocabulary (the rest map to <unk>)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="folder holding model.pt and config.json")
    parser.add_argument("--input", type=Path, help="JSON file with a user's history and a candidate")
    parser.add_argument("--seed", type=int, default=7, help="seed of the random example (no --input)")
    args = parser.parse_args()

    reranker = Reranker(args.artifacts)
    if args.input:
        example = json.loads(args.input.read_text())
    else:
        example = random_example(reranker, args.seed)
        print("Randomly generated input (save it as JSON and pass --input to edit it):")
        print(json.dumps(example, indent=2))
        print()
    result = reranker.predict(example["history"], example["candidate"])

    print(f"User {example.get('user_id', '?')}: {len(result['history'])} rated recipes in the history")
    for recipe, tokens, normalized in zip(result["history"], result["tokens"], result["history_rating"]):
        rating_input = f"normalized {normalized:+.3f}" if reranker.use_rating else "not used by this checkpoint"
        print(f"  {recipe['date']}  rating {recipe.get('rating', '?')} ({rating_input})  {recipe['name']}")
        print(describe(recipe, tokens, reranker.token_lookup))
    candidate = result["candidate"]
    print(f"Candidate: {candidate['name']}")
    print(describe(candidate, result["tokens"][-1], reranker.token_lookup))
    print()
    predicted = "positive" if result["predicted_label"] else "negative"
    print(f"Prediction:   P(positive) = {result['p_positive']:.4f} → {predicted} (label "
          f"{result['predicted_label']}, threshold {THRESHOLD})")
    if candidate.get("rating") is not None:
        truth = int(candidate["rating"] >= reranker.config["positive_min_rating"])
        print(f"Ground truth: rating {candidate['rating']} → {'positive' if truth else 'negative'} (label {truth}); "
              f"prediction {'correct' if truth == result['predicted_label'] else 'wrong'}")


if __name__ == "__main__":
    main()
