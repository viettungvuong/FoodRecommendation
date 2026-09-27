"""Retrieval pipeline checks on a tiny catalog and a randomly initialized reranker, without the trained weights.

Run with: python3 -m unittest DataMappingRecommendation/tests/test_retrieval_pipeline.py -v
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_inference_reranker_avgemb_approach as inference  # noqa: E402
import retrieval_pipeline as retrieval  # noqa: E402

PRODUCTS = ["chicken breast", "salmon", "quinoa", "broccoli", "waffle", "chocolate", "lentil", "rice", "apple",
            "cheese"]
ADJECTIVES = ["fresh", "sweet", "ready", "plain", "red"]
VERBS = ["frozen", "cooked", "smoked", "toasted", "boiled"]


def write_catalog(path, count=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        product = rng.choice(PRODUCTS, size=rng.integers(1, 3), replace=False).tolist()
        unpriced = [product[-1]] if i % 7 == 0 and len(product) > 1 else []
        item_prices = {item: round(float(rng.uniform(1, 20)), 2) for item in product if item not in unpriced}
        rows.append({
            "fdc_id": 100000 + i, "description": f"Item {i}, {', '.join(product)}", "product": str(product),
            "item_prices": str(item_prices), "unpriced_items": str(unpriced),
            "adj": str(rng.choice(ADJECTIVES, size=rng.integers(0, 3), replace=False).tolist()),
            "verb": str(rng.choice(VERBS, size=rng.integers(0, 3), replace=False).tolist()),
            "total_price": 0.0 if unpriced else sum(item_prices.values()),
            "protein_g": rng.uniform(0, 40), "fat_g": rng.uniform(0, 30), "carb_g": rng.uniform(0, 60),
            "energy_kcal": rng.uniform(20, 600), "fiber_g": rng.uniform(0, 5), "sodium_mg": rng.uniform(0, 900),
            "cholesterol_mg": rng.uniform(0, 100), "satfat_g": rng.uniform(0, 10),
            "sugars_g": np.nan if i % 5 == 0 else rng.uniform(0, 30)})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_artifacts(directory, seed=0):
    """config.json and model.pt of an untrained reranker whose vocabulary covers the catalog's words."""
    config = {"d": 16, "n_heads": 2, "text_layers": 1, "text_dropout": 0.0, "dropout": 0.0, "max_tokens": 40,
              "t_max": 4}
    text_fields = {"product": ["n", 1], "adj": ["a", 2], "verb": ["v", 3]}
    nutrient_columns = list(inference.NUTRIENT_INPUTS)
    words = sorted({inference._lemmatizer.lemmatize(inference._lemmatizer.lemmatize(word, pos), "n")
                    for phrases, pos in [(PRODUCTS, "n"), (ADJECTIVES, "a"), (VERBS, "v")]
                    for phrase in phrases for word in phrase.split()})
    vocabulary = ["<pad>", "<unk>", *words]
    rng = np.random.default_rng(seed)
    (directory / "config.json").write_text(json.dumps({
        "config": config, "text_fields": text_fields, "nutrient_columns": nutrient_columns,
        "vocabulary": vocabulary,
        "item_log1p_mean": rng.uniform(1, 5, 8).tolist(), "item_log1p_std": rng.uniform(0.5, 2, 8).tolist(),
        "profile_mean": rng.uniform(1, 5, 8).tolist(), "profile_std": rng.uniform(0.5, 2, 8).tolist(),
        "history_rating_scale": 1.3}))
    torch.manual_seed(seed)
    model = inference.GRUCrossAttentionReranker(len(vocabulary), 8, 8, len(text_fields), config)
    torch.save(model.state_dict(), directory / "model.pt")


class RetrievalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.catalog_path = cls.dir / "price_mapped_nutrients.csv"
        cls.artifacts = cls.dir / "artifacts"
        cls.artifacts.mkdir()
        write_catalog(cls.catalog_path)
        write_artifacts(cls.artifacts)
        cls.index_dir = cls.dir / "index"
        cls.pipeline = retrieval.RetrievalPipeline.from_paths(cls.catalog_path, cls.artifacts, cls.index_dir,
                                                              device=torch.device("cpu"))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        cls.tmp.cleanup()

    def history(self, rows=(3, 11, 25, 30), ratings=(5, 2, 4, 5)):
        dates = ["2024-01-05", "2024-03-01", "2024-02-10", "2024-04-20"]
        return [{"fdc_id": 100000 + row, "rating": rating, "date": date}
                for row, rating, date in zip(rows, ratings, dates)]

    # Catalog -------------------------------------------------------------------------------------------

    def test_catalog_prices_sum_item_prices_and_missing_nutrients_take_the_median(self):
        frame = pd.read_csv(self.catalog_path)
        recipes = self.pipeline.recipes
        self.assertEqual([recipe["fdc_id"] for recipe in recipes], frame["fdc_id"].tolist())
        unpriced = frame.index[frame["total_price"] == 0][0]
        self.assertGreater(recipes[unpriced]["price"], 0)  # total_price is 0 there, item_prices is not empty.
        self.assertAlmostEqual(recipes[1]["price"], frame.loc[1, "total_price"], places=6)
        self.assertAlmostEqual(recipes[0]["nutrients"]["sugar_g"], frame["sugars_g"].median())
        self.assertEqual(recipes[0]["entities"]["product"], [[phrase] for phrase in eval(frame.loc[0, "product"])])

    # Index ---------------------------------------------------------------------------------------------

    def test_index_holds_the_reranker_item_embeddings(self):
        expected, _ = self.pipeline.reranker.encode_items(self.pipeline.recipes)
        np.testing.assert_allclose(self.pipeline.recipe_index.embeddings, expected.numpy(), atol=1e-5)

    def test_search_returns_the_highest_cosine_similarities(self):
        embeddings = self.pipeline.recipe_index.embeddings
        query = np.random.default_rng(1).normal(size=embeddings.shape[1]).astype(np.float32)
        cosine = embeddings @ query / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query))
        found = self.pipeline.recipe_index.search(query, 5)
        self.assertEqual([row for row, _ in found], np.argsort(-cosine)[:5].tolist())
        np.testing.assert_allclose([similarity for _, similarity in found], np.sort(cosine)[::-1][:5], atol=1e-5)

    def test_search_leaves_out_excluded_rows_and_still_returns_k(self):
        query = self.pipeline.recipe_index.embeddings[0]
        nearest = [row for row, _ in self.pipeline.recipe_index.search(query, 6)]
        found = [row for row, _ in self.pipeline.recipe_index.search(query, 6, exclude=nearest[:3])]
        self.assertEqual(len(found), 6)
        self.assertFalse(set(found) & set(nearest[:3]))
        self.assertEqual(found[:3], nearest[3:])
        self.assertEqual(len(self.pipeline.recipe_index.search(query, 1000)), len(self.pipeline.recipes))

    def test_saved_index_is_reused_until_the_model_or_catalog_changes(self):
        saved = json.loads((self.index_dir / "meta.json").read_text())["fingerprint"]
        loaded = retrieval.RecipeIndex.load_or_build(None, self.pipeline.recipes, self.index_dir, saved)  # No build.
        query = self.pipeline.recipe_index.embeddings[4]
        self.assertEqual(loaded.search(query, 5), self.pipeline.recipe_index.search(query, 5))
        with tempfile.TemporaryDirectory() as other:
            retrieval.RecipeIndex.load_or_build(self.pipeline.reranker, self.pipeline.recipes, other, saved)
            rebuilt = retrieval.RecipeIndex.load_or_build(self.pipeline.reranker, self.pipeline.recipes, other,
                                                          "changed")
            self.assertEqual(json.loads((Path(other) / "meta.json").read_text())["fingerprint"], "changed")
            np.testing.assert_allclose(rebuilt.embeddings, loaded.embeddings, atol=1e-6)

    def test_pipeline_rejects_an_index_of_another_catalog(self):
        with self.assertRaises(ValueError):
            retrieval.RetrievalPipeline(self.pipeline.reranker, self.pipeline.recipes[1:], self.pipeline.recipe_index)

    # User embedding and recommendations ----------------------------------------------------------------

    def test_history_is_sorted_by_date_and_reads_embeddings_from_the_index(self):
        history, history_emb = self.pipeline.resolve_history(self.history())
        self.assertEqual([recipe["fdc_id"] for recipe in history], [100003, 100025, 100011, 100030])
        self.assertEqual(history[1]["rating"], 4)
        np.testing.assert_array_equal(history_emb[1].numpy(), self.pipeline.recipe_index.embeddings[25])

    def test_user_embedding_is_the_mean_of_normalized_history_embeddings(self):
        _, history_emb = self.pipeline.resolve_history(self.history())
        expected = np.mean([row / np.linalg.norm(row) for row in history_emb.numpy()], axis=0)
        np.testing.assert_allclose(self.pipeline.user_embedding(history_emb).numpy(), expected, atol=1e-6)

    def test_recommend_reranks_the_retrieved_candidates_like_the_inference_script(self):
        user = {"user_id": "u1", "history": self.history()}
        results = self.pipeline.recommend(user, n_candidates=12, top_k=5)
        history, history_emb = self.pipeline.resolve_history(user["history"])
        retrieved = self.pipeline.recipe_index.search(self.pipeline.user_embedding(history_emb).numpy(), 12,
                                                      exclude=[3, 11, 25, 30])
        self.assertEqual(len(results), 5)
        self.assertFalse({result["fdc_id"] for result in results} & {100003, 100011, 100025, 100030})
        self.assertTrue({result["fdc_id"] - 100000 for result in results} <= {row for row, _ in retrieved})
        self.assertEqual([result["p_like"] for result in results],
                         sorted((result["p_like"] for result in results), reverse=True))
        # The inference script's own path (encode everything, then score) gives the same probabilities.
        rows = [result["fdc_id"] - 100000 for result in results]
        direct = self.pipeline.reranker.predict(history, [self.pipeline.recipes[row] for row in rows])
        np.testing.assert_allclose([result["p_like"] for result in results],
                                   [result["p_like"] for result in direct], atol=1e-5)
        top = max(results, key=lambda result: result["p_like"])
        self.assertEqual(top["similarity"], dict(retrieved)[top["fdc_id"] - 100000])

    def test_off_catalog_history_recipes_are_encoded(self):
        recipe = {"name": "home made salmon rice", "date": "2024-01-01", "rating": 5, "price": 12.0,
                  "nutrients": dict(self.pipeline.recipes[0]["nutrients"]),
                  "entities": {"product": [["salmon"], ["rice"]], "adj": [["fresh"]], "verb": [["smoked"]]}}
        history, history_emb = self.pipeline.resolve_history([*self.history(), recipe])
        self.assertIs(history[0]["name"], recipe["name"])
        expected, _ = self.pipeline.reranker.encode_items([recipe])
        np.testing.assert_allclose(history_emb[0].numpy(), expected[0].numpy(), atol=1e-5)
        self.assertEqual(len(self.pipeline.recommend({"history": [recipe]}, n_candidates=8, top_k=3)), 3)

    def test_unknown_fdc_id_and_empty_history_are_errors(self):
        with self.assertRaises(KeyError):
            self.pipeline.recommend({"history": [{"fdc_id": 1, "rating": 5, "date": "2024-01-01"}]})
        with self.assertRaises(ValueError):
            self.pipeline.recommend({"history": []})

    def test_history_ratings_are_centered_on_their_mean_and_scaled(self):
        ratings = self.pipeline.reranker.normalized_ratings([{"rating": 5}, {"rating": 2}, {}, {"rating": 5}])
        np.testing.assert_allclose(ratings, [1 / 1.3, -2 / 1.3, 0, 1 / 1.3], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
