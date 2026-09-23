"""Behavioral regressions for the notebook, without downloading BERT weights.

Run with: python3 -m unittest discover -s DataMappingRecommendation/tests -v
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import torch


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "model_training_twotowers.ipynb"


class TwoTowerNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text())
        cls.namespace = {"__name__": "two_tower_notebook_under_test"}
        for tag in ("two-tower-core", "two-tower-features"):
            cells = [cell for cell in cls.notebook["cells"]
                     if tag in cell.get("metadata", {}).get("tags", [])]
            if len(cells) != 1:
                raise AssertionError(f"Expected exactly one definition cell tagged {tag!r}.")
            exec(compile("".join(cells[0]["source"]), f"{NOTEBOOK_PATH}:{tag}", "exec"),
                 cls.namespace)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def call(self, name, *args, **kwargs):
        return self.namespace[name](*args, **kwargs)

    def recipe_rows(self):
        rows = []
        for index, base in enumerate((2.0, 6.0, 12.0), start=1):
            row = {
                "recipe_id": index, "name": f"Recipe {index}",
                "ingredients": '["Tomato", "Rice"]',
                "product": '[["Tomato"], ["Rice"]]',
                "adj": '[[], ["soft"]]', "verb": '[["chopped"], []]',
                "price_breakdown": '{"tomato": 3.5, "rice": null}',
                "user_id": f"user{index}", "rating": 5, "date": "2024-01-01",
            }
            row.update({column: base + offset
                        for offset, column in enumerate(self.namespace["NUTRITION_COLUMNS"])})
            rows.append(row)
        return pd.DataFrame(rows)

    def interaction_fixture(self):
        catalog = pd.DataFrame({"recipe_id": [str(i) for i in range(1, 6)],
                                "name": [f"Recipe {i}" for i in range(1, 6)]})
        frame = pd.DataFrame([
            ["u", "1", 5, "2024-01-01"], ["u", "2", 1, "2024-01-02"],
            ["u", "3", 4, "2024-01-04"], ["v", "2", 4, "2024-01-01"],
            ["v", "3", 1, "2024-01-02"], ["v", "4", 5, "2024-01-04"],
        ], columns=["user_id", "recipe_id", "rating", "date"])
        return catalog, self.call("build_interactions", frame, catalog), np.eye(5, dtype=np.float32)

    def test_nested_prices_sum_known_leaves_and_accumulate_missing_error(self):
        result = self.call("summarize_price",
                           '{"a": {"x": 3.5, "y": null}, "b": [2, null]}', 4)
        self.assertEqual(result["price_total_known"], 5.5)
        self.assertEqual(result["price_missing_count"], 2)
        self.assertEqual(result["price_error_dollars"], 4.0)
        self.assertEqual((result["price_lower"], result["price_upper"]), (1.5, 9.5))
        self.assertEqual(result["price_missing_fraction"], 0.5)

    def test_csv_missing_prices_keep_unknown_flag_and_two_dollar_error(self):
        for value in (None, np.nan, pd.NA, "null", "{}", "[]"):
            with self.subTest(value=value):
                result = self.call("summarize_price", value, 3)
                self.assertEqual(result["price_total_known"], 0.0)
                self.assertEqual(result["price_missing_count"], 3)
                self.assertEqual(result["price_error_dollars"], 6.0)
                self.assertEqual(result["price_all_missing"], 1)
        with self.assertRaises(ValueError):
            self.call("summarize_price", '{"rice": -1}', 1)

    def test_nested_text_fields_and_nullable_text(self):
        self.assertEqual(self.call("flatten_terms", '[["Green Onion"], [], ["RICE"]]'),
                         ["green onion", "rice"])
        self.assertEqual(self.call("flatten_terms", pd.NA), [])
        frame = self.recipe_rows()
        frame.loc[0, "product"] = "[]"
        catalog = self.call("build_recipe_catalog", frame)
        self.assertEqual(catalog.loc[0, "ingredient_terms"], ["tomato", "rice"])
        self.assertEqual(catalog.loc[0, "adj_terms"], ["soft"])
        self.assertEqual(catalog.loc[0, "verb_terms"], ["chopped"])

    def test_recipe_nutrition_and_clusters_ignore_repeated_rating_rows(self):
        frame = self.recipe_rows()
        alternate = frame.iloc[[0]].copy()
        columns = self.namespace["NUTRITION_COLUMNS"]
        alternate[columns] += 4
        frame = pd.concat([frame, alternate], ignore_index=True)
        duplicated = pd.concat([frame, pd.concat([frame.iloc[[0]]] * 20)], ignore_index=True)
        first = self.call("build_recipe_catalog", frame)
        second = self.call("build_recipe_catalog", duplicated)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_allclose(first.loc[0, columns].to_numpy(dtype=float), np.arange(4, 11))
        first_numeric, first_means, _ = self.call("build_numeric_features", first, n_clusters=1)
        second_numeric, second_means, _ = self.call("build_numeric_features", second, n_clusters=1)
        np.testing.assert_allclose(first_numeric, second_numeric)
        pd.testing.assert_frame_equal(first_means, second_means)
        np.testing.assert_allclose(first_means.iloc[0], first[columns].mean())

    def test_numeric_missing_values_are_finite_and_preserve_missingness(self):
        frame = self.recipe_rows()
        columns = self.namespace["NUTRITION_COLUMNS"]
        frame[columns[0]] = np.nan
        frame.loc[0, columns[1]] = np.nan
        catalog = self.call("build_recipe_catalog", frame)
        features, _, preprocessing = self.call("build_numeric_features", catalog, n_clusters=2)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(preprocessing["nutrition_medians"][0], 0)
        np.testing.assert_array_equal(features[:, len(columns)], np.ones(3))
        np.testing.assert_array_equal(features[:, len(columns) + 1], [1, 0, 0])

    def test_unlabeled_rows_are_excluded_before_date_validation(self):
        catalog = pd.DataFrame({"recipe_id": ["1", "2", "3"]})
        frame = pd.DataFrame({"recipe_id": [1, 2, 3], "user_id": [10, 10, 10],
                              "rating": [5, 0, None], "date": ["2024-01-01", None, None]})
        result = self.call("build_interactions", frame, catalog)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].user_id, "10")
        self.assertEqual(result.iloc[0].rating, 5)
        for invalid in (6, -1, "bad"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bad = frame.copy()
                bad["rating"] = bad["rating"].astype(object)
                bad.loc[0, "rating"] = invalid
                self.call("build_interactions", bad, catalog)

    def test_labeled_rows_require_original_valid_dates(self):
        catalog = pd.DataFrame({"recipe_id": ["1"]})
        frame = pd.DataFrame({"recipe_id": [1], "user_id": [10], "rating": [5], "date": [None]})
        with self.assertRaisesRegex(ValueError, "missing/invalid dates"):
            self.call("build_interactions", frame, catalog)
        with self.assertRaisesRegex(ValueError, "date"):
            self.call("build_interactions", frame.drop(columns="date"), catalog)

    def test_latest_rating_wins_after_identifier_normalization(self):
        catalog = pd.DataFrame({"recipe_id": ["1"]})
        frame = pd.DataFrame({"recipe_id": ["1.0", 1], "user_id": ["10.0", 10],
                              "rating": [2, 5], "date": ["2024-01-01", "2024-02-01"]})
        result = self.call("build_interactions", frame, catalog)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].rating, 5)

    def test_temporal_split_keeps_same_day_blocks_and_an_earliest_training_day(self):
        frame = pd.DataFrame([
            ["u", 0, 5, "2024-01-01T08:00Z"], ["u", 1, 4, "2024-01-02T01:00Z"],
            ["u", 2, 3, "2024-01-02T20:00Z"], ["v", 0, 4, "2024-01-03T01:00Z"],
            ["v", 1, 5, "2024-01-03T20:00Z"],
        ], columns=["user_id", "item_index", "rating", "date"])
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        train, validation = self.call("temporal_split", frame, validation_fraction=0.2)
        self.assertEqual(set(validation.item_index), {1, 2})
        self.assertEqual(set(validation.user_id), {"u"})
        self.assertEqual(len(train.loc[train.user_id.eq("v")]), 2)
        self.assertLess(train.loc[train.user_id.eq("u"), "date"].max().floor("D"),
                        validation.date.min().floor("D"))

    def test_profile_excludes_target_same_day_and_future_ratings(self):
        _, history, features = self.interaction_fixture()
        profile = self.call("user_profile", history, features, "u", "2024-01-02T23:00Z")
        np.testing.assert_allclose(profile[:5], features[0])
        np.testing.assert_array_equal(profile[5:10], np.zeros(5))
        earlier = history.loc[history.date < pd.Timestamp("2024-01-02", tz="UTC")]
        np.testing.assert_allclose(profile, self.call("user_profile", earlier, features, "u", "2024-01-02"))
        target_excluded = self.call("user_profile", history, features, "u", "2024-01-05", exclude_item=2)
        self.assertEqual(target_excluded[2], 0)
        self.assertEqual(target_excluded[7], 0)

    def test_recent_ratings_have_stronger_profile_and_training_weights(self):
        history = pd.DataFrame({"user_id": ["u", "u"], "item_index": [0, 1],
                                "rating": [5.0, 5.0],
                                "date": pd.to_datetime(["2024-01-01", "2024-01-11"], utc=True)})
        features = np.eye(2, dtype=np.float32)
        profile = self.call("user_profile", history, features, "u", "2024-01-12", half_life_days=10)
        self.assertAlmostEqual(float(profile[1] / profile[0]), 2.0, places=5)
        dataset = self.call("HistoryDataset", history, history, features, half_life_days=10)
        np.testing.assert_allclose(dataset.weights, [0.5, 1.0])
        validation = self.call("HistoryDataset", history, history, features, training=False)
        np.testing.assert_array_equal(validation.weights, [1, 1])

    def test_both_towers_receive_gradients(self):
        torch.manual_seed(9)
        model = self.call("TwoTowerModel", 5, 13, embedding_dim=4, hidden_dim=8)
        predictions = model(torch.randn(3, 13), torch.eye(5)[:3])
        ((predictions - torch.tensor([1.0, 3.0, 5.0])) ** 2).mean().backward()
        for tower in (model.user_tower, model.item_tower):
            gradients = [parameter.grad for parameter in tower.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_tiny_training_recommends_unseen_recipes_and_ignores_future(self):
        torch.manual_seed(11)
        catalog, interactions, features = self.interaction_fixture()
        train, validation = self.call("temporal_split", interactions)
        train_dataset = self.call("HistoryDataset", train, train, features)
        validation_dataset = self.call("HistoryDataset", validation, train, features, training=False)
        model = self.call("TwoTowerModel", 5, 13, embedding_dim=4, hidden_dim=8)
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        with contextlib.redirect_stdout(io.StringIO()):
            model, history, metrics = self.call("train_two_tower", model, train_dataset,
                validation_dataset, features, epochs=3, batch_size=2, device="cpu")
        self.assertTrue(np.isfinite(history.train_weighted_rmse).all())
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        for prefix in ("item_tower", "user_tower"):
            self.assertTrue(any(not torch.equal(initial[name], parameter)
                                for name, parameter in model.named_parameters() if name.startswith(prefix)))
        result = self.call("recommend_recipes", model, catalog, features, interactions, "u",
                           top_k=10, as_of="2024-01-05")
        self.assertEqual(set(result.recipe_id), {"4", "5"})
        self.assertTrue(result.predicted_rating.between(1, 5).all())
        self.assertTrue(result.predicted_rating.is_monotonic_decreasing)
        before_future = self.call("recommend_recipes", model, catalog, features, interactions, "u",
                                  top_k=10, as_of="2024-01-03")
        reduced = interactions.loc[interactions.date < pd.Timestamp("2024-01-03", tz="UTC")]
        without_future = self.call("recommend_recipes", model, catalog, features, reduced, "u",
                                   top_k=10, as_of="2024-01-03")
        pd.testing.assert_frame_equal(before_future, without_future)
        self.assertIn("3", set(before_future.recipe_id))

    def test_notebook_artifacts_restore_identical_recommendations(self):
        torch.manual_seed(13)
        catalog = self.call("build_recipe_catalog", self.recipe_rows())
        features, _, numeric = self.call("build_numeric_features", catalog, n_clusters=2)
        observations = self.call("build_interactions", self.recipe_rows(), catalog)
        dim = features.shape[1]
        model = self.call("TwoTowerModel", dim, 2 * dim + 3, embedding_dim=4, hidden_dim=8)
        expected = self.call("recommend_recipes", model, catalog, features, observations,
                             "user1", as_of="2024-01-02")
        save_cells = [cell for cell in self.notebook["cells"]
                      if "save-artifacts" in cell.get("metadata", {}).get("tags", [])]
        self.assertEqual(len(save_cells), 1)
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            context = dict(self.namespace, ARTIFACT_DIR=artifact_dir, item_features=features,
                EMBEDDING_DIM=4, HIDDEN_DIM=8, recommendation_model=model, device=torch.device("cpu"),
                item_catalog=catalog, observed_ratings=observations, semantic_preprocessing={},
                numeric_preprocessing=numeric, joblib=joblib, BERT_MODEL="test-no-download",
                BERT_REVISION="test", RECENCY_HALF_LIFE_DAYS=180, MAX_HISTORY=64,
                MISSING_PRICE_ERROR_DOLLARS=2, RANDOM_STATE=13, VALIDATION_FRACTION=0.2,
                DATA_PATH=Path("fixture.csv"), training_history=pd.DataFrame({"epoch": [1]}))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile("".join(save_cells[0]["source"]), "save-artifacts", "exec"), context)
            checkpoint = torch.load(artifact_dir / "two_tower.pt", map_location="cpu", weights_only=True)
            restored = self.call("TwoTowerModel", **checkpoint["model_config"])
            restored.load_state_dict(checkpoint["state_dict"])
            restored_catalog = pd.read_csv(artifact_dir / "catalog.csv", dtype={"recipe_id": "string"})
            restored_history = pd.read_csv(artifact_dir / "observed_ratings.csv", dtype={"user_id": "string"})
            restored_history["date"] = pd.to_datetime(restored_history["date"], utc=True)
            restored_features = np.load(artifact_dir / "item_features.npy", allow_pickle=False)
            item_vectors = np.load(artifact_dir / "item_vectors.npy", allow_pickle=False)
            result = self.call("recommend_recipes", restored, restored_catalog, restored_features,
                               restored_history, "user1", as_of="2024-01-02", item_embeddings=item_vectors)
            self.assertEqual(result.recipe_id.tolist(), expected.recipe_id.tolist())
            np.testing.assert_allclose(result.predicted_rating, expected.predicted_rating, atol=1e-6)
            self.assertIn("nutrition_scaler", joblib.load(artifact_dir / "content_preprocessing.joblib")["numeric"])


if __name__ == "__main__":
    unittest.main()
