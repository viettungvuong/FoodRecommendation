"""Reranker notebook checks on tiny fixtures, without BERT weights or the real data.

Run with: python3 -m unittest discover -s DataMappingRecommendation/tests -v
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
TWO_TOWER_NOTEBOOK = ROOT / "model_training_twotowers.ipynb"
RERANKER_NOTEBOOK = ROOT / "model_training_reranker.ipynb"


def tagged_source(notebook, tag):
    cells = [cell for cell in notebook["cells"] if tag in cell.get("metadata", {}).get("tags", [])]
    if len(cells) != 1:
        raise AssertionError(f"Expected exactly one cell tagged {tag!r}.")
    return "".join(cells[0]["source"])


def defined_names(source):
    """Top-level function and class names of a cell."""
    return {node.name for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def bound_names(source):
    """Every name a cell binds at any depth: assignments, loop targets, imports, defs, with-as."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return names


class RerankerNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.two_tower = json.loads(TWO_TOWER_NOTEBOOK.read_text())
        cls.notebook = json.loads(RERANKER_NOTEBOOK.read_text())
        cls.namespace = {"__name__": "reranker_notebook_under_test"}
        for tag in ("two-tower-core", "two-tower-features"):
            exec(compile(tagged_source(cls.two_tower, tag), f"{TWO_TOWER_NOTEBOOK}:{tag}", "exec"), cls.namespace)
        exec(compile(tagged_source(cls.notebook, "reranker-core"), f"{RERANKER_NOTEBOOK}:reranker-core", "exec"),
             cls.namespace)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def n(self, name):
        return self.namespace[name]

    # Fixtures -------------------------------------------------------------------------------

    def catalog(self, count=8, seed=0):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"recipe_id": [str(100 + i) for i in range(count)],
                              "name": [f"Recipe {i}" for i in range(count)],
                              "nutrition_cluster": np.arange(count) % 3})
        for column in self.n("NUTRITION_COLUMNS"):
            frame[column] = rng.uniform(0, 50, count)
        frame["price_total_known"] = rng.uniform(1, 20, count)
        frame["price_missing_count"] = rng.integers(0, 3, count)
        frame["price_error_dollars"] = 2.0 * frame["price_missing_count"]
        frame["price_lower"] = (frame["price_total_known"] - frame["price_error_dollars"]).clip(lower=0)
        frame["price_upper"] = frame["price_total_known"] + frame["price_error_dollars"]
        return frame

    @staticmethod
    def ratings(rows):
        frame = pd.DataFrame(rows, columns=["user_id", "item_index", "rating", "date"])
        frame["rating"] = frame["rating"].astype(float)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, format="mixed")
        return frame

    def random_ratings(self, users=15, items=60, per_user=50, seed=3):
        rng = np.random.default_rng(seed)
        rows = []
        start = pd.Timestamp("2023-01-01", tz="UTC")
        for user in range(users):
            for item in rng.choice(items, size=per_user, replace=False):
                rows.append((str(1000 + user), int(item), float(rng.choice(5, p=[.1, .1, .2, .3, .3]) + 1),
                             start + pd.Timedelta(int(rng.integers(0, 365)), unit="D")))
        return self.ratings(rows)

    def synthetic_features(self, count=600, seed=0, drop_rating=None):
        """Random features whose ratings follow bias_baseline, for fitting tiny boosters."""
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({column: rng.normal(size=count).astype(np.float32)
                              for column in self.n("FEATURE_COLUMNS")})
        frame["nutrition_cluster"] = rng.integers(0, 3, count)
        ratings = np.clip(np.rint(3 + 1.5 * frame["bias_baseline"] + rng.normal(scale=0.5, size=count)), 1, 5)
        if drop_rating is not None:
            ratings[ratings == drop_rating] = drop_rating + 1
        return frame, pd.Series(ratings, name="rating")

    def tiny_booster(self, **kwargs):
        features, ratings = self.synthetic_features(**kwargs)
        booster, history, _ = self.n("train_reranker")(
            features[:400], ratings[:400], features[400:], ratings[400:],
            params={"min_data_in_leaf": 5, "num_leaves": 4}, num_boost_round=40,
            early_stopping_rounds=10, log_period=0)
        return booster, features, ratings, history

    def tiny_two_tower(self, feature_dim=6, num_users=2, seed=0):
        torch.manual_seed(seed)
        return self.n("TwoTowerModel")(feature_dim, num_users, embedding_dim=4, hidden_dim=8, user_id_dim=4).eval()

    # As-of statistics -----------------------------------------------------------------------

    def test_as_of_statistics_count_only_strictly_earlier_days(self):
        stats = self.n("AsOfRatingStats")([0, 0, 0, 1], [10, 10, 12, 5], [5, 3, 1, 4])
        result = stats.query([0, 0, 0, 1, 1, -1], [10, 11, 13, 5, 6, 20])
        np.testing.assert_array_equal(result["count"], [0, 2, 3, 0, 1, 0])
        self.assertEqual(result["mean"][1], 4.0)
        self.assertAlmostEqual(result["std"][1], np.sqrt(2.0))  # Sample std of 5 and 3.
        np.testing.assert_allclose(result["shares"][1], [0, 0, 0.5, 0, 0.5])
        self.assertEqual((result["first_day"][2], result["last_day"][2]), (10, 12))
        self.assertTrue(np.isnan(result["mean"][[0, 3, 5]]).all())
        self.assertTrue(np.isnan(result["std"][4]))  # One rating has no spread.
        self.assertTrue(np.isnan(result["first_day"][5]))

    def test_features_ignore_the_target_day_and_later_ratings(self):
        catalog = self.catalog(4)
        catalog["nutrition_cluster"] = [0, 1, 0, 1]
        history = self.ratings([("u1", 0, 5, "2024-01-01"), ("u1", 1, 3, "2024-01-02"),
                                ("u1", 2, 1, "2024-01-05"), ("u1", 3, 2, "2024-01-09"),
                                ("u2", 0, 4, "2024-01-03"), ("u2", 2, 5, "2024-01-04")])
        cutoff = self.n("utc_day_numbers")(["2024-01-05"])[0]
        builder = self.n("RerankFeatureBuilder")(history, catalog, prior_mean=3.0, user_lookup={"u1": 1},
                                                 bayes_strength=2.0)
        row = builder.transform(["u1"], [2], cutoff, 0.25).iloc[0]
        self.assertEqual(list(builder.transform(["u1"], [2], cutoff, 0.25).columns), self.n("FEATURE_COLUMNS"))
        self.assertEqual((row["user_count"], row["user_mean"]), (2, 4))
        self.assertAlmostEqual(row["user_bayes_mean"], (8 + 2 * 3.0) / (2 + 2), places=5)  # float32 features.
        self.assertEqual((row["user_days_since_first"], row["user_days_since_last"]), (4, 3))
        self.assertEqual((row["item_count"], row["item_mean"]), (1, 5))  # Only u2's earlier rating.
        self.assertAlmostEqual(row["item_bayes_mean"], (5 + 2 * 3.0) / (1 + 2), places=5)
        self.assertEqual((row["user_cluster_count"], row["user_cluster_mean"], row["user_cluster_share"]), (1, 5, 0.5))
        self.assertAlmostEqual(row["bias_baseline"], row["user_bayes_mean"] + row["item_bayes_mean"] - 3.0, places=5)
        self.assertEqual((row["two_tower_similarity"], row["user_known"], row["nutrition_cluster"]), (0.25, 1, 0))
        self.assertAlmostEqual(row["calories_g"], catalog.loc[2, "calories (g)"], places=4)
        # Changing or adding ratings on or after the cutoff day must not change any feature.
        altered = pd.concat([history.assign(rating=np.where(history["date"] >= pd.Timestamp("2024-01-05", tz="UTC"),
                                                     5.0, history["rating"])),
                             self.ratings([("u1", 1, 1, "2024-01-05 23:00"), ("u2", 2, 1, "2024-01-06")])],
                            ignore_index=True)
        again = self.n("RerankFeatureBuilder")(altered, catalog, prior_mean=3.0, user_lookup={"u1": 1},
                                               bayes_strength=2.0).transform(["u1"], [2], cutoff, 0.25)
        pd.testing.assert_frame_equal(again, builder.transform(["u1"], [2], cutoff, 0.25))
        unknown = builder.transform(["stranger"], [0], cutoff, 0.0).iloc[0]
        self.assertEqual((unknown["user_count"], unknown["user_known"], unknown["user_bayes_mean"]), (0, 0, 3.0))
        self.assertTrue(np.isnan(unknown["user_mean"]) and np.isnan(unknown["user_cluster_share"]))

    # LightGBM softmax ----------------------------------------------------------------------

    def test_feature_names_are_lightgbm_safe_and_survive_training(self):
        names = self.n("FEATURE_COLUMNS")
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(re.fullmatch(r"[a-z0-9_]+", name) for name in names), names)
        booster, *_ = self.tiny_booster()
        self.assertEqual(booster.feature_name(), names)

    def test_multiclass_softmax_gives_five_probabilities_even_without_a_rating(self):
        features, ratings = self.synthetic_features(drop_rating=2)
        booster, _, _ = self.n("train_reranker")(
            features[:400], ratings[:400], features[400:], ratings[400:],
            params={"objective": "multiclassova", "num_class": 3, "min_data_in_leaf": 5, "num_leaves": 4},
            num_boost_round=40, early_stopping_rounds=10, log_period=0)
        self.assertEqual((booster.params["objective"], booster.params["num_class"]), ("multiclass", 5))
        probabilities = self.n("predict_rating_probabilities")(booster, features)
        self.assertEqual(probabilities.shape, (len(features), 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        self.assertEqual(probabilities.mean(axis=0).argmin(), 1)  # Rating 2 never appears in training.
        raw = booster.predict(features[self.n("FEATURE_COLUMNS")], raw_score=True,
                              num_iteration=booster.best_iteration, num_threads=1)
        softmax = np.exp(raw - raw.max(axis=1, keepdims=True))
        np.testing.assert_allclose(softmax / softmax.sum(axis=1, keepdims=True), probabilities, atol=1e-9)
        scores = self.n("rerank_scores")(probabilities)
        np.testing.assert_allclose(scores["expected_rating"], probabilities @ np.arange(1, 6))
        np.testing.assert_allclose(scores["p_at_least_4"], probabilities[:, 3] + probabilities[:, 4])
        with self.assertRaises(ValueError):
            self.n("predict_rating_probabilities")(booster, features.drop(columns="bias_baseline"))

    def test_balanced_class_weights_follow_scikit_learn(self):
        from sklearn.utils.class_weight import compute_class_weight

        ratings = np.array([1.0, 2, 3, 4, 4, 5, 5, 5, 5, 5])
        np.testing.assert_allclose(self.n("balanced_class_weights")(ratings),
                                   compute_class_weight("balanced", classes=np.arange(1, 6), y=ratings))
        np.testing.assert_allclose(self.n("balanced_class_weights")(ratings), 10 / (5 * np.array([1, 1, 1, 2, 5])))
        missing_two = self.n("balanced_class_weights")([1.0, 4, 5, 5])  # n / (5 x count); absent 2, 3 get the max.
        np.testing.assert_allclose(missing_two, [0.8, 0.8, 0.8, 0.8, 0.4])
        np.testing.assert_array_equal(self.n("balanced_class_weights")(ratings, class_weight=None), np.ones(5))
        with self.assertRaises(ValueError):
            self.n("balanced_class_weights")(ratings, class_weight="sqrt")
        features, labels = self.synthetic_features()
        _, _, weights = self.n("train_reranker")(
            features[:400], labels[:400], features[400:], labels[400:],
            params={"min_data_in_leaf": 5, "num_leaves": 4}, num_boost_round=5, early_stopping_rounds=5, log_period=0)
        np.testing.assert_allclose(weights, self.n("balanced_class_weights")(labels[:400]))  # Balanced by default.

    def test_undoing_class_weights_recovers_the_true_rating_mix(self):
        truth = np.array([[0.05, 0.05, 0.10, 0.30, 0.50], [0.01, 0.02, 0.07, 0.20, 0.70]])
        weights = np.array([8.0, 6.0, 3.0, 1.0, 0.3])
        weighted = truth * weights
        weighted /= weighted.sum(axis=1, keepdims=True)  # What weighted cross-entropy converges to.
        np.testing.assert_allclose(self.n("undo_class_weighting")(weighted, weights), truth)
        with self.assertRaises(ValueError):
            self.n("undo_class_weighting")(weighted, np.array([1.0, 1, 1, 1, 0]))
        # A skewed mix like the real one (5% / 5% / 10% / 20% / 60%), driven by bias_baseline.
        features, _ = self.synthetic_features(count=2000, seed=5)
        signal = features["bias_baseline"] + np.random.default_rng(6).normal(scale=0.5, size=len(features))
        ratings = pd.Series(1.0 + np.digitize(signal, np.quantile(signal, [0.05, 0.10, 0.20, 0.40])))
        booster, _, weights = self.n("train_reranker")(
            features[:1500], ratings[:1500], features[1500:], ratings[1500:],
            params={"min_data_in_leaf": 20, "num_leaves": 4, "learning_rate": 0.2}, num_boost_round=300,
            early_stopping_rounds=20, log_period=0)  # Undoing the weights is exact only near the optimum.
        raw = self.n("predict_rating_probabilities")(booster, features[1500:])
        calibrated = self.n("predict_rating_probabilities")(booster, features[1500:], weights)
        shares = np.bincount(self.n("rating_classes")(ratings[1500:]), minlength=5) / 500
        self.assertLess(np.abs(calibrated.mean(axis=0) - shares).max(), 0.04)
        self.assertGreater(np.abs(raw.mean(axis=0) - shares).max(), 0.1)  # Balanced softmax overstates rare ratings.

    def test_balanced_decisions_divide_probabilities_by_the_prior(self):
        probabilities = np.array([[0.05, 0.02, 0.03, 0.20, 0.70], [0.0, 0.0, 0.0, 0.1, 0.9]])
        prior = np.array([0.01, 0.02, 0.06, 0.20, 0.71])
        np.testing.assert_array_equal(self.n("balanced_decisions")(probabilities, prior), [0, 4])
        with self.assertRaises(ValueError):
            self.n("balanced_decisions")(probabilities, np.array([0.5, 0.5, 0, 0, 0]))
        np.testing.assert_allclose(self.n("class_prior")([5, 5, 4]), np.array([1, 1, 1, 2, 3]) / 8)
        report = self.n("probability_report")([1, 5], probabilities, prior)
        self.assertEqual(report.loc[1, "balanced_recall"], 1.0)
        self.assertEqual(report.loc[1, "argmax_recall"], 0.0)
        self.assertAlmostEqual(report.loc[5, "mean_probability"], 0.8)

    # Splits and ranking ---------------------------------------------------------------------

    def test_split_is_disjoint_stratified_and_keeps_features_aligned(self):
        ratings = np.repeat([1.0, 2, 3, 4, 5], [20, 20, 40, 60, 60])
        rows = pd.DataFrame({"marker": np.arange(len(ratings)), "rating": ratings})
        features = pd.DataFrame({"marker_copy": rows["marker"] * 10})
        splits = self.n("split_reranker_rows")(rows, features, 0.15, 0.15, seed=7)
        markers = [set(part_rows["marker"]) for part_rows, _ in splits.values()]
        self.assertEqual(sum(map(len, markers)), len(rows))
        self.assertEqual(set.union(*markers), set(rows["marker"]))
        for part_rows, part_features in splits.values():
            np.testing.assert_array_equal(part_features["marker_copy"], part_rows["marker"] * 10)
            np.testing.assert_allclose(part_rows["rating"].value_counts(normalize=True).sort_index(),
                                       [0.1, 0.1, 0.2, 0.3, 0.3], atol=0.04)
        self.assertEqual([len(part_rows) for part_rows, _ in splits.values()], [140, 30, 30])

    def test_mean_ndcg_matches_a_hand_computation(self):
        users, ratings, scores = ["a", "a", "a", "b", "b"], [5, 3, 1, 4, 4], [0.1, 0.9, 0.5, 0.3, 0.2]
        dcg = 7 + 1 / np.log2(3) + 31 / 2
        ideal = 31 + 7 / np.log2(3) + 1 / 2
        ndcg, evaluated = self.n("mean_ndcg")(users, ratings, scores, k=10)
        self.assertAlmostEqual(ndcg, dcg / ideal)
        self.assertEqual(evaluated, 1)  # User b's ratings are all equal, so every order is ideal.
        self.assertAlmostEqual(self.n("mean_ndcg")(users, ratings, scores, k=1)[0], 7 / 31)
        many_users, many_ratings = np.repeat(["a", "b", "c"], 5), np.tile([1.0, 2, 3, 4, 5], 3)
        report = self.n("ranking_report")(many_users, many_ratings,
                                          {"perfect": many_ratings, "reversed": -many_ratings}, k=10)
        self.assertEqual(list(report.index), ["random_order", "perfect", "reversed"])
        self.assertAlmostEqual(report.loc["perfect", "ndcg@10"], 1.0)
        self.assertAlmostEqual(report.loc["perfect", "gap_closed"], 1.0)
        self.assertLess(report.loc["reversed", "gap_closed"], 0.0)
        self.assertEqual(report.loc["perfect", "users"], 3)

    def test_shared_cutoff_rows_use_each_users_earliest_day(self):
        rows = self.ratings([("u1", 0, 5, "2024-01-03 10:00"), ("u1", 1, 3, "2024-01-05"),
                             ("u2", 2, 4, "2024-02-01 23:59")])
        shared = self.n("shared_cutoff_rows")(rows)
        self.assertEqual(shared["date"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                         ["2024-01-03 00:00", "2024-01-03 00:00", "2024-02-01 00:00"])
        self.assertEqual(rows["date"].iloc[0], pd.Timestamp("2024-01-03 10:00", tz="UTC"))  # Input untouched.

    # Two-tower integration -----------------------------------------------------------------

    def test_training_similarity_matches_recommend_recipes_similarity(self):
        model = self.tiny_two_tower()
        item_features = np.random.default_rng(1).normal(size=(6, 6)).astype(np.float32)
        history = self.ratings([("u1", 0, 5, "2024-01-01"), ("u1", 1, 2, "2024-01-03"), ("u1", 2, 4, "2024-01-06"),
                                ("u1", 5, 4, "2024-01-10"), ("u1", 3, 1, "2024-01-12"), ("u2", 4, 3, "2024-01-02")])
        lookup = {"u1": 1, "u2": 2}
        # Row 1 (a 2 on Jan 3) is an opposite-level recipe of the target (a 4 on Jan 10). Serving never
        # hides earlier ratings, so the reranker's history must keep it, whatever pairs the two-tower draws.
        targets = history.iloc[[1, 3]]
        similarity = self.n("two_tower_similarity")(model, targets, history, item_features, lookup, device="cpu")
        served = self.n("recommend_recipes")(model, self.catalog(6), item_features, history, "u1", user_lookup=lookup,
                                             top_k=6, as_of="2024-01-10")
        expected = served.loc[served["recipe_id"] == "105", "similarity"].to_numpy()  # Item 5 is rated on Jan 10.
        np.testing.assert_allclose(similarity[1:], expected, atol=1e-5)
        target = history.iloc[[3]]
        self.assertEqual(len(self.n("two_tower_similarity")(model, target.iloc[:0], history, item_features, lookup)), 0)

    def test_rerank_recipes_orders_candidates_by_probability_and_skips_rated_recipes(self):
        catalog = self.catalog(8)
        model = self.tiny_two_tower()
        item_features = np.random.default_rng(2).normal(size=(8, 6)).astype(np.float32)
        history = self.ratings([("u1", 0, 5, "2024-01-01"), ("u1", 1, 2, "2024-01-03"), ("u1", 2, 4, "2024-01-06"),
                                ("u2", 3, 3, "2024-01-02"), ("u2", 4, 5, "2024-01-04")])
        lookup = {"u1": 1, "u2": 2}
        builder = self.n("RerankFeatureBuilder")(history, catalog, prior_mean=4.0, user_lookup=lookup)
        booster, *_ = self.tiny_booster()
        candidates = self.n("recommend_recipes")(model, catalog, item_features, history, "u1", user_lookup=lookup,
                                                 top_k=5, as_of="2024-02-01")
        weights = np.array([4.0, 3.0, 2.0, 1.0, 0.5])
        prior = np.array([0.05, 0.05, 0.1, 0.3, 0.5])
        result = self.n("rerank_recipes")(booster, builder, model, catalog, item_features, history, "u1",
                                          user_lookup=lookup, retrieve_k=5, top_k=3, as_of="2024-02-01",
                                          class_weights=weights, prior=prior)
        self.assertEqual(len(result), 3)
        self.assertFalse(set(result["recipe_id"]) & {"100", "101", "102"})
        self.assertTrue(result["expected_rating"].is_monotonic_decreasing)
        probabilities = result[self.n("PROBABILITY_COLUMNS")].to_numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(result["expected_rating"], probabilities @ np.arange(1, 6))
        ranks = dict(zip(candidates["recipe_id"], range(1, len(candidates) + 1)))
        self.assertEqual(result["two_tower_rank"].tolist(), [ranks[recipe] for recipe in result["recipe_id"]])
        similarity = dict(zip(candidates["recipe_id"], candidates["similarity"]))
        np.testing.assert_allclose(result["two_tower_similarity"], [similarity[r] for r in result["recipe_id"]])
        np.testing.assert_array_equal(result["predicted_rating"],
                                      self.n("balanced_decisions")(probabilities, prior) + 1)
        positions = result["recipe_id"].map(lambda recipe: int(recipe) - 100).to_numpy()
        features = builder.transform(["u1"] * 3, positions, self.n("utc_day_numbers")(["2024-02-01"])[0],
                                     result["two_tower_similarity"].to_numpy(np.float32))
        np.testing.assert_allclose(probabilities, self.n("predict_rating_probabilities")(booster, features, weights))
        with_vectors = self.n("rerank_recipes")(
            booster, builder, model, catalog, item_features, history, "u1", user_lookup=lookup, retrieve_k=5,
            top_k=3, as_of="2024-02-01", item_embeddings=self.n("encode_catalog")(model, item_features).numpy(),
            class_weights=weights, prior=prior)
        pd.testing.assert_frame_equal(with_vectors, result, atol=1e-5)
        with self.assertRaises(ValueError):
            self.n("rerank_recipes")(booster, builder, model, catalog, item_features, history, "u1",
                                     retrieve_k=2, top_k=3)

    def test_load_two_tower_rejects_a_checkpoint_from_another_architecture(self):
        model = self.tiny_two_tower(feature_dim=6)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "config.json").write_text(json.dumps({"format_version": "test_v1"}))
            torch.save({"state_dict": model.state_dict(), "model_config": model.config}, directory / "two_tower.pt")
            restored, config = self.n("load_two_tower")(directory)
            self.assertFalse(restored.training)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
            other = self.tiny_two_tower(feature_dim=7)
            torch.save({"state_dict": other.state_dict(), "model_config": model.config}, directory / "two_tower.pt")
            with self.assertRaisesRegex(RuntimeError, "Retrain and save the two-tower"):
                self.n("load_two_tower")(directory)

    def test_saved_reranker_restores_probabilities_and_rejects_another_two_tower(self):
        booster, features, _, _ = self.tiny_booster()
        builder = self.n("RerankFeatureBuilder")(self.ratings([("u1", 0, 5, "2024-01-01")]), self.catalog(2),
                                                 prior_mean=4.2, bayes_strength=7.0)
        with tempfile.TemporaryDirectory() as directory:
            two_tower_dir, reranker_dir = Path(directory) / "two_tower", Path(directory) / "reranker"
            two_tower_dir.mkdir()
            (two_tower_dir / "two_tower.pt").write_bytes(b"checkpoint one")
            (two_tower_dir / "config.json").write_text(json.dumps({"format_version": "test_v1"}))
            config = self.n("save_reranker")(reranker_dir, booster, builder, two_tower_dir, {"note": np.float64(0.5)})
            self.assertEqual((config["prior_mean"], config["bayes_strength"], config["note"]), (4.2, 7.0, 0.5))
            self.assertEqual(config["two_tower_artifact"]["format_version"], "test_v1")
            restored, loaded_config = self.n("load_reranker")(reranker_dir, two_tower_dir)
            self.assertEqual(loaded_config, json.loads(json.dumps(config)))
            np.testing.assert_allclose(self.n("predict_rating_probabilities")(restored, features),
                                       self.n("predict_rating_probabilities")(booster, features), rtol=0, atol=1e-12)
            (two_tower_dir / "two_tower.pt").write_bytes(b"checkpoint two")
            with self.assertRaisesRegex(RuntimeError, "retrain the reranker"):
                self.n("load_reranker")(reranker_dir, two_tower_dir)

    # Notebook structure ---------------------------------------------------------------------

    def test_later_cells_do_not_rebind_core_or_two_tower_definitions(self):
        two_tower_defs = set().union(*(defined_names(tagged_source(self.two_tower, tag))
                                       for tag in ("two-tower-core", "two-tower-features")))
        core = tagged_source(self.notebook, "reranker-core")
        self.assertFalse(defined_names(core) & two_tower_defs, "reranker-core shadows a two-tower definition")
        self.assertFalse(bound_names(core) & two_tower_defs, "reranker-core rebinds a two-tower definition")
        protected = two_tower_defs | defined_names(core)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code" or "reranker-core" in cell.get("metadata", {}).get("tags", []):
                continue
            clashes = bound_names("".join(cell["source"])) & protected
            self.assertFalse(clashes, f"Cell {cell['id']} rebinds {sorted(clashes)}")

    def test_notebook_runs_end_to_end_on_tiny_two_tower_artifacts(self):
        observed = self.random_ratings()
        catalog = self.catalog(60)
        item_features = np.random.default_rng(4).normal(size=(60, 6)).astype(np.float32)
        train, heldout = self.n("stratified_split")(observed, validation_fraction=0.2, seed=42)
        lookup = self.n("build_user_lookup")(train)
        model = self.tiny_two_tower(num_users=len(lookup))
        item_vectors = self.n("encode_catalog")(model, item_features).numpy()
        code_cells = [cell for cell in self.notebook["cells"] if cell["cell_type"] == "code"]
        with tempfile.TemporaryDirectory() as directory:
            two_tower_dir = Path(directory) / "two_tower"
            two_tower_dir.mkdir()
            torch.save({"state_dict": model.state_dict(), "model_config": model.config}, two_tower_dir / "two_tower.pt")
            (two_tower_dir / "config.json").write_text(json.dumps({
                "format_version": "test_two_tower", "max_history": 64, "validation_fraction": 0.2, "random_state": 42}))
            catalog.to_csv(two_tower_dir / "catalog.csv", index=False)
            np.save(two_tower_dir / "item_features.npy", item_features)

            observed.to_csv(two_tower_dir / "observed_ratings.csv", index=False)
            pd.DataFrame({"user_id": list(lookup), "user_index": list(lookup.values())}).to_csv(
                two_tower_dir / "user_vocabulary.csv", index=False)

            def run(saved_vectors):
                np.save(two_tower_dir / "item_vectors.npy", saved_vectors)
                namespace = {"__name__": "__main__"}
                with contextlib.chdir(ROOT), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for cell in code_cells:
                        exec(compile("".join(cell["source"]), f"{RERANKER_NOTEBOOK.name}:{cell['id']}", "exec"),
                             namespace)
                        if cell["id"] == "reranker-imports":
                            namespace["display"] = lambda *args, **kwargs: None
                        if cell["id"] == "reranker-configuration":
                            namespace.update(
                                TWO_TOWER_DIR=two_tower_dir, RERANKER_DIR=Path(directory) / "reranker",
                                device=torch.device("cpu"), NUM_BOOST_ROUND=60, EARLY_STOPPING_ROUNDS=10,
                                LIGHTGBM_OVERRIDES={"min_data_in_leaf": 5, "num_leaves": 4})
                plt.close("all")
                return namespace

            with self.assertRaisesRegex(RuntimeError, "does not reproduce item_vectors.npy"):
                run(item_vectors + 0.01)
            namespace = run(item_vectors)
            reranker_dir = Path(directory) / "reranker"
            for name in ("lightgbm_reranker.txt", "config.json", "training_history.csv", "test_metrics.csv",
                         "test_ranking.csv"):
                self.assertTrue((reranker_dir / name).is_file(), name)
            config = json.loads((reranker_dir / "config.json").read_text())
            self.assertEqual(config["two_tower_artifact"]["sha256"],
                             self.n("file_sha256")(two_tower_dir / "two_tower.pt"))
            self.assertEqual((config["objective"], config["activation"], config["num_class"], config["class_weight"]),
                             ("multiclass", "softmax", 5, "balanced"))
            self.assertIn("lightgbm_without_two_tower", namespace["evaluation"].index)
            self.assertIn("lightgbm_without_two_tower_expected_rating", namespace["ranking"].index)
            self.assertEqual(config["split"]["rows"]["test"], len(namespace["test_rows"]))
            self.assertEqual(sum(config["split"]["rows"].values()), len(heldout))
            self.assertGreater(namespace["ranking"].loc["lightgbm_expected_rating", "users"], 0)
            np.testing.assert_allclose(namespace["test_probabilities"].sum(axis=1), 1.0, atol=1e-9)
            reranked = namespace["reranked"]
            self.assertGreater(len(reranked), 0)
            rated = set(observed.loc[observed["user_id"] == namespace["example_user_id"], "item_index"])
            self.assertFalse(set(catalog.loc[sorted(rated), "recipe_id"]) & set(reranked["recipe_id"]))
            self.assertTrue(reranked[namespace["RERANK_SCORE"]].is_monotonic_decreasing)
            self.assertTrue(reranked["predicted_rating"].between(1, 5).all())


if __name__ == "__main__":
    unittest.main()
