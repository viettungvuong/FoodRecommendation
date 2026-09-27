"""Retrieval pipeline checks on a tiny catalog and a randomly initialized reranker, without the trained weights.

Run with: python3 -m unittest DataMappingRecommendation/tests/test_score_recommend.py -v
"""
import ast
from types import SimpleNamespace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_inference_reranker_avgemb as inference  # noqa: E402
import model_inference_score_recommend as retrieval  # noqa: E402

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


def write_artifacts(directory, seed=0, windowed=True):
    """config.json and model.pt of an untrained reranker whose vocabulary covers the catalog's words.

    windowed=False writes the older model (one history sequence, ratings scaled by history_rating_scale)."""
    config = {"d": 16, "n_heads": 2, "text_layers": 1, "text_dropout": 0.0, "dropout": 0.0, "max_tokens": 40,
              "t_max": 4}
    if windowed:
        config.update(max_windows=3, window_gap_days=183)
    text_fields = {"product": ["n", 1], "adj": ["a", 2], "verb": ["v", 3]}
    nutrient_columns = list(inference.NUTRIENT_INPUTS)
    words = sorted({inference._lemmatizer.lemmatize(inference._lemmatizer.lemmatize(word, pos), "n")
                    for phrases, pos in [(PRODUCTS, "n"), (ADJECTIVES, "a"), (VERBS, "v")]
                    for phrase in phrases for word in phrase.split()})
    vocabulary = ["<pad>", "<unk>", *words]
    rng = np.random.default_rng(seed)
    saved = {"config": config, "text_fields": text_fields, "nutrient_columns": nutrient_columns,
             "vocabulary": vocabulary,
             "item_log1p_mean": rng.uniform(1, 5, 8).tolist(), "item_log1p_std": rng.uniform(0.5, 2, 8).tolist(),
             "profile_mean": rng.uniform(1, 5, 8).tolist(), "profile_std": rng.uniform(0.5, 2, 8).tolist()}
    saved.update({"history_rating": "centered"} if windowed else {"history_rating_scale": 1.3})
    (directory / "config.json").write_text(json.dumps(saved))
    torch.manual_seed(seed)
    model_class = retrieval.WindowGRUCrossAttentionReranker if windowed else inference.GRUCrossAttentionReranker
    model = model_class(len(vocabulary), 8, 8, len(text_fields), config)
    torch.save(model.state_dict(), directory / "model.pt")


def notebook_definitions(*names):
    """The named top-level functions and classes of the reranker notebook, run in a namespace of their own."""
    notebook = json.loads((ROOT / "model_training_reranker_avgemb.ipynb").read_text())
    namespace = {"np": np, "torch": torch, "nn": torch.nn, "pack_padded_sequence": pack_padded_sequence,
                 "pad_packed_sequence": pad_packed_sequence, "TEXT_FIELDS": {"product": 1, "adj": 2, "verb": 3}}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]).replace("%pip", "#"))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        exec(compile(ast.Module(nodes, type_ignores=[]), "notebook", "exec"), namespace)
    return namespace


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
        results = self.pipeline.recommend(user, n_candidates=12, top_k=5, mmr_lambda=1.0,
                                          max_similarity=None)  # P(like) order.
        history, history_emb = self.pipeline.resolve_history(user["history"])
        retrieved = self.pipeline.recipe_index.search(self.pipeline.user_embedding(history_emb).numpy(), 12,
                                                      exclude=[3, 11, 25, 30])
        self.assertEqual(len(results), 5)
        self.assertFalse({result["fdc_id"] for result in results} & {100003, 100011, 100025, 100030})
        self.assertTrue({result["fdc_id"] - 100000 for result in results} <= {row for row, _ in retrieved})
        self.assertEqual([result["p_like"] for result in results],
                         sorted((result["p_like"] for result in results), reverse=True))
        # Encoding everything afresh, as the inference script does, gives the same probabilities as the cache.
        rows = [result["fdc_id"] - 100000 for result in results]
        fresh, _ = self.pipeline.reranker.encode_items([*history, *(self.pipeline.recipes[row] for row in rows)])
        direct = self.pipeline.reranker.score(history, fresh[:len(history)], fresh[len(history):])
        np.testing.assert_allclose([result["p_like"] for result in results], direct.numpy(), atol=1e-5)
        top = max(results, key=lambda result: result["p_like"])
        self.assertEqual(top["similarity"], dict(retrieved)[top["fdc_id"] - 100000])
        self.assertEqual([result["rerank_rank"] for result in results], [1, 2, 3, 4, 5])

    def test_recommend_picks_from_the_reranked_candidates_with_mmr(self):
        user = {"user_id": "u1", "history": self.history()}
        reranked = self.pipeline.recommend(user, n_candidates=12, top_k=12, mmr_lambda=1.0, max_similarity=None)
        results = self.pipeline.recommend(user, n_candidates=12, top_k=5, mmr_lambda=0.3)  # Cutoff 0.97.
        emb = self.pipeline.recipe_index.embeddings[[result["fdc_id"] - 100000 for result in reranked]]
        unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        cosine = unit @ unit.T
        picked = retrieval.mmr([result["p_like"] for result in reranked], cosine, 5, 0.3, 0.97)
        self.assertEqual([result["fdc_id"] for result in results], [reranked[i]["fdc_id"] for i in picked])
        self.assertEqual(results[0]["fdc_id"], reranked[0]["fdc_id"])  # The first pick is the most likely.
        self.assertEqual([result["rerank_rank"] for result in results], [i + 1 for i in picked])
        self.assertLess(cosine[np.ix_(picked, picked)][np.triu_indices(len(picked), 1)].max(), 0.97)
        # A cutoff below the lowest cosine between candidates leaves every one of them too close to the first pick.
        lowest = cosine.min() - 1e-4
        self.assertEqual(len(self.pipeline.recommend(user, n_candidates=12, top_k=5, max_similarity=lowest)), 1)

    # Maximal marginal relevance ------------------------------------------------------------------------

    def test_mmr_with_lambda_one_and_no_cutoff_is_the_order_of_relevance(self):
        relevance, same = [0.2, 0.9, 0.5, 0.9, 0.1], np.ones((5, 5))
        self.assertEqual(retrieval.mmr(relevance, same, 5, 1.0, None), [1, 3, 2, 0, 4])  # Ties in input order.
        self.assertEqual(retrieval.mmr(relevance, same, 9, 1.0, None), [1, 3, 2, 0, 4])
        self.assertEqual(retrieval.mmr(relevance, same, 0, 1.0, None), [])
        with self.assertRaises(ValueError):
            retrieval.mmr(relevance, same, 3, 7)

    def test_mmr_passes_over_a_near_duplicate_of_a_pick(self):
        # 1 is nearly a copy of 0; 2 is less likely but different.
        relevance = [0.9, 0.89, 0.7]
        similarity = np.array([[1.0, 0.99, 0.2], [0.99, 1.0, 0.2], [0.2, 0.2, 1.0]])
        self.assertEqual(retrieval.mmr(relevance, similarity, 2, 0.5, None), [0, 2])
        self.assertEqual(retrieval.mmr(relevance, similarity, 2, 1.0, None), [0, 1])

    def test_mmr_never_picks_an_item_at_the_cutoff_or_above_even_if_fewer_than_k_remain(self):
        relevance = [0.9, 0.89, 0.7, 0.6]
        similarity = np.array([[1.0, 0.99, 0.2, 0.97],
                               [0.99, 1.0, 0.2, 0.5],
                               [0.2, 0.2, 1.0, 0.1],
                               [0.97, 0.5, 0.1, 1.0]])
        self.assertEqual(retrieval.mmr(relevance, similarity, 4, 1.0, 0.97), [0, 2])  # 1 and 3 too close to 0.
        self.assertEqual(retrieval.mmr(relevance, similarity, 4, 1.0, 0.98), [0, 2, 3])  # 0.97 is below 0.98.
        self.assertEqual(retrieval.mmr(relevance, similarity, 4, 1.0, None), [0, 1, 2, 3])

    def test_each_mmr_pick_maximizes_the_marginal_relevance_among_items_below_the_cutoff(self):
        rng = np.random.default_rng(2)
        emb = rng.normal(size=(30, 8))
        unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        similarity, relevance = unit @ unit.T, rng.uniform(size=30)
        for cutoff, count in [(None, 10), (0.6, None)]:
            picked = retrieval.mmr(relevance, similarity, 10, 0.6, cutoff)
            self.assertEqual(len(set(picked)), len(picked))
            if count:
                self.assertEqual(len(picked), count)
            for step in range(len(picked) + 1):
                # Similarities can be negative: the penalty is the highest one, not clipped at 0.
                penalty = similarity[:, picked[:step]].max(axis=1) if step else np.zeros(30)
                allowed = [i for i in range(30) if i not in picked[:step]
                           and (cutoff is None or not step or penalty[i] < cutoff)]
                if step == len(picked):
                    self.assertTrue(len(picked) == 10 or not allowed)  # Stopped early only when none was allowed.
                else:
                    score = 0.6 * relevance - 0.4 * penalty
                    self.assertEqual(picked[step], max(allowed, key=lambda i: score[i]))

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

    def test_history_ratings_are_centered_on_their_mean_and_not_scaled(self):
        # Mean of 5, 2, 5 is 4: centered 1, -2, (missing) 0, 1.
        ratings = self.pipeline.reranker.normalized_ratings([{"rating": 5}, {"rating": 2}, {}, {"rating": 5}])
        np.testing.assert_allclose(ratings, [1, -2, 0, 1], atol=1e-6)

    def test_config_without_windows_is_the_older_model(self):
        with tempfile.TemporaryDirectory() as other:
            other = Path(other)
            write_artifacts(other, windowed=False)
            reranker = retrieval.Reranker(other, device=torch.device("cpu"))
            self.assertFalse(reranker.windowed)
            np.testing.assert_allclose(reranker.normalized_ratings([{"rating": 5}, {"rating": 2}]),
                                       [1.5 / 1.3, -1.5 / 1.3], atol=1e-6)
            pipeline = retrieval.RetrievalPipeline(reranker, self.pipeline.recipes,
                                                   retrieval.RecipeIndex.build(reranker, self.pipeline.recipes))
            self.assertEqual(len(pipeline.recommend({"history": self.history()}, n_candidates=6, top_k=3)), 3)

    # Windows -------------------------------------------------------------------------------------------

    def windowed_history(self):
        """Seven ratings in three windows: gaps of 200 and 183 days start windows, 182 days does not."""
        dates = ["2020-01-01", "2020-01-10", "2020-07-29",  # +201 days: new window.
                 "2021-01-27",  # +182 days: same window.
                 "2021-07-29", "2021-08-01", "2021-08-02"]  # +183 days: new window.
        return [{"fdc_id": 100000 + row, "rating": rating, "date": date}
                for row, rating, date in zip([1, 2, 3, 4, 5, 6, 7], [5, 4, 2, 3, 5, 5, 1], dates)]

    def test_history_is_cut_into_windows_at_gaps_of_six_months(self):
        history, _ = self.pipeline.resolve_history(self.windowed_history())
        positions, ratings, lengths = self.pipeline.reranker.windows(history)
        # Most recent first, in date order inside a window, trimmed to the longest window.
        self.assertEqual(lengths.tolist(), [3, 2, 2])
        self.assertEqual(positions.tolist(), [[4, 5, 6], [2, 3, -1], [0, 1, -1]])
        mean = np.mean([5, 4, 2, 3, 5, 5, 1])
        np.testing.assert_allclose(ratings[0], np.array([5, 5, 1]) - mean, atol=1e-6)
        self.assertEqual(ratings[1, 2], 0.0)  # Padding.

    def test_only_the_last_max_windows_windows_and_t_max_recipes_are_kept(self):
        dates = [f"{2000 + year}-01-0{day}" for year in range(5) for day in range(1, 7)]  # 5 windows of 6.
        history = [{"fdc_id": 100000 + i, "rating": 4, "date": date} for i, date in enumerate(dates)]
        history, _ = self.pipeline.resolve_history(history)
        positions, _, lengths = self.pipeline.reranker.windows(history)
        self.assertEqual(lengths.tolist(), [4, 4, 4])  # max_windows = 3, t_max = 4.
        self.assertEqual(positions[0].tolist(), [26, 27, 28, 29])

    def test_windows_are_scored_from_their_own_gru_states(self):
        history, history_emb = self.pipeline.resolve_history(self.windowed_history())
        states = self.pipeline.reranker.window_states(history, history_emb)
        encoder = self.pipeline.reranker.model.history_encoder
        ratings = torch.from_numpy(self.pipeline.reranker.normalized_ratings(history))
        with torch.no_grad():
            for state, members in zip(states, [[4, 5, 6], [2, 3], [0, 1]]):
                steps_in = history_emb[members] + encoder.rating_projection(ratings[members].unsqueeze(-1))
                _, last = encoder.gru(steps_in.unsqueeze(0))  # One window alone, from a zero state.
                np.testing.assert_allclose(state.numpy(), last[0, 0].numpy(), atol=1e-5)

    def test_recommend_stores_each_window_state(self):
        self.pipeline.recommend({"user_id": "windows", "history": self.windowed_history()}, n_candidates=6, top_k=2)
        stored = self.pipeline.user_states["windows"]
        self.assertEqual([window["recipes"] for window in stored["windows"]], [3, 2, 2])
        self.assertEqual(stored["windows"][0]["from"], "2021-07-29")
        self.assertEqual(stored["windows"][2]["fdc_ids"], [100001, 100002])
        self.assertEqual(tuple(stored["states"].shape), (3, 16))

    # Training notebook alignment -----------------------------------------------------------------------

    def test_window_functions_match_the_training_notebook(self):
        notebook = notebook_definitions("window_starts", "history_windows")
        rng = np.random.default_rng(5)
        users = np.sort(rng.integers(0, 6, 200))
        days = np.concatenate([np.sort(rng.integers(0, 3000, (users == user).sum())) for user in range(6)])
        ratings = rng.integers(0, 6, 200).astype(np.float64)
        recipe_idx = rng.integers(0, 50, 200)
        np.testing.assert_array_equal(retrieval.window_starts(users, days, 183), notebook["window_starts"](users, days, 183))
        first = retrieval.window_starts(users, days, 183)
        starts = np.searchsorted(users, users)
        ends = np.arange(1, 201)
        mu = rng.normal(size=200)
        for ours, theirs in zip(retrieval.history_windows(ends, starts, mu, first, recipe_idx, ratings, 4, 5),
                                notebook["history_windows"](ends, starts, mu, first, recipe_idx, ratings, 4, 5)):
            np.testing.assert_array_equal(ours, theirs)

    def test_model_matches_the_training_notebook(self):
        notebook = notebook_definitions("ItemEncoder", "UserEncoder", "HistoryEncoder", "GRUCrossAttentionReranker")
        config = {"d": 16, "n_heads": 2, "text_layers": 1, "text_dropout": 0.0, "dropout": 0.0, "max_windows": 3}
        torch.manual_seed(3)
        tokens = torch.randint(1, 20, (10, 6))
        trained = notebook["GRUCrossAttentionReranker"](tokens, torch.randint(1, 4, (10, 6)), torch.randn(10, 8), 20,
                                                        8, SimpleNamespace(**config)).eval()
        served = retrieval.WindowGRUCrossAttentionReranker(20, 8, 8, 3, config).eval()
        served.load_state_dict(trained.state_dict())  # Same parameter names and shapes.
        history_idx = torch.tensor([[[1, 2, 3], [4, -1, -1]], [[5, 6, -1], [-1, -1, -1]]])
        rating = torch.randn(2, 2, 3) * (history_idx >= 0)
        window_len = torch.tensor([[3, 1], [2, 0]])
        profile = torch.randn(2, 8)
        with torch.no_grad():
            expected = trained(torch.tensor([7, 8]), history_idx, rating, window_len, profile)
            emb = trained.encode_items(torch.arange(10))
            keys, mask = served.user_keys(emb[history_idx.clamp(min=0)] * (history_idx >= 0).unsqueeze(-1), rating,
                                          window_len, profile)
            got = served.score_candidates(emb[[7, 8]].unsqueeze(1), keys, mask).squeeze(1)
        np.testing.assert_allclose(got.numpy(), expected.numpy(), atol=1e-5)

if __name__ == "__main__":
    unittest.main()
