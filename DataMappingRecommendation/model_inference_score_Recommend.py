"""Recommend catalog recipes to a user: HNSW retrieval over item embeddings, then the GRU + cross-attention reranker.

1. Index (once per model and catalog): every recipe of price_mapped_nutrients.csv is embedded by the reranker's
   ItemEncoder, i.e. its product / adj / verb tokens concatenated with its standardized [price, nutrients], and the
   embeddings go into an HNSW index with cosine distance.
2. Query: the user's representative embedding is the average of their history recipes' embeddings.
3. Retrieval: HNSW returns the recipes closest to it by cosine similarity, the history's own recipes excluded.
4. Reranking: the reranker scores each candidate against the user's history, as model_inference_reranker_avgemb_
   approach.py does, and the candidates are sorted by P(like).

Usage:
    python model_inference_score_Recommend.py build --catalog path/to/price_mapped_nutrients.csv
    python model_inference_score_Recommend.py recommend --catalog path/to/price_mapped_nutrients.csv \\
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
from torch.nn import functional as F

from model_inference_reranker_avgemb_approach import DEFAULT_ARTIFACTS, THRESHOLD, Reranker as BaseReranker

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


class Reranker(BaseReranker):
    """The reranker of model_inference_reranker_avgemb_approach.py, scoring candidates from item embeddings."""

    def normalized_ratings(self, history):
        """History ratings centered on the mean of all of them, divided by the training scale (spec 2.3).

        A recipe without a rating counts as the user's mean, i.e. 0."""
        ratings = np.array([np.nan if recipe.get("rating") is None else recipe["rating"] for recipe in history],
                           dtype=np.float64)
        if not self.use_rating or np.isnan(ratings).all():
            return np.zeros(len(history), dtype=np.float32)
        return np.nan_to_num((ratings - np.nanmean(ratings)) / self.rating_scale).astype(np.float32)

    @torch.no_grad()
    def score(self, history, history_emb, candidate_emb):
        """P(like) (C,) of candidates for the user who rated history, from item embeddings.

        history is in date order and history_emb (len(history), d) holds its item embeddings; candidate_emb is
        (C, d). Both come from encode_items or from the index's cache of it."""
        t_max = self.config["t_max"]
        # The last t_max history recipes in date order go to the GRU; the profile is the mean over all.
        recent = history_emb[-t_max:].to(self.device).unsqueeze(0)
        history_rating = torch.as_tensor(self.normalized_ratings(history)[-t_max:], device=self.device).unsqueeze(0)
        profile = np.mean([self.log_numeric(recipe) for recipe in history], axis=0)[-len(self.profile_mean):]
        profile = (profile - self.profile_mean) / self.profile_std
        keys, key_padding_mask = self.model.user_keys(
            recent, history_rating, torch.tensor([recent.size(1)]),
            torch.as_tensor(profile, dtype=torch.float32, device=self.device).unsqueeze(0))
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

    @staticmethod
    def user_embedding(history_emb):
        """Representative embedding: the average of the history's embeddings, each L2-normalized first so every
        recipe weighs the same under cosine similarity."""
        return F.normalize(history_emb, dim=1).mean(0)

    def recommend(self, user, n_candidates=100, top_k=10):
        """The top_k of n_candidates retrieved recipes by P(like), most likely first."""
        if not user.get("history"):
            raise ValueError("The pipeline needs at least one recipe in the user's history.")
        history, history_emb = self.resolve_history(user["history"])
        seen = {self.row_of[recipe["fdc_id"]] for recipe in history if "fdc_id" in recipe}
        candidates = self.recipe_index.search(self.user_embedding(history_emb).numpy(), n_candidates, exclude=seen)
        if not candidates:
            return []
        rows = [row for row, _ in candidates]
        p_likes = self.reranker.score(history, history_emb, torch.from_numpy(self.recipe_index.embeddings[rows]))
        results = [{"fdc_id": self.recipes[row]["fdc_id"], "name": self.recipes[row]["name"],
                    "price": self.recipes[row]["price"], "p_like": p_like, "like": p_like >= THRESHOLD,
                    "similarity": similarity, "retrieval_rank": rank}
                   for rank, ((row, similarity), p_like) in enumerate(zip(candidates, p_likes.tolist()), start=1)]
        return sorted(results, key=lambda result: -result["p_like"])[:top_k]


def print_recommendations(user, results):
    print(f"User {user.get('user_id', '?')}: {len(user['history'])} history recipes")
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
    args = parser.parse_args()

    pipeline = RetrievalPipeline.from_paths(args.catalog, args.artifacts, args.index_dir,
                                            rebuild=args.rebuild or args.command == "build")
    if args.command == "build":
        print(f"Indexed {len(pipeline.recipes):,} recipes × {pipeline.recipe_index.embeddings.shape[1]} dims "
              f"in {args.index_dir}")
        return
    user = json.loads(args.user.read_text())
    results = pipeline.recommend(user, args.candidates, args.top_k)
    print_recommendations(user, results)
    if args.output:
        args.output.write_text(json.dumps({"user_id": user.get("user_id"), "recommendations": results}, indent=2))


if __name__ == "__main__":
    main()
