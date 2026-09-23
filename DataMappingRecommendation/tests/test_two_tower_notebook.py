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
        # Strings inside a parsed list are terms, never re-parsed as numbers, booleans or null.
        self.assertEqual(self.call("flatten_terms", '["Salt", "2", "(optional)", "true", "null", "[half"]'),
                         ["salt", "2", "(optional)", "true", "null", "[half"])
        self.assertEqual(self.call("flatten_terms", ['["Green Onion"]', " Rice "]), ["green onion", "rice"])
        with self.assertRaises(ValueError):
            self.call("flatten_terms", "[unclosed")
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

    def test_stratified_split_keeps_each_rating_share_in_both_splits(self):
        frame = pd.DataFrame({"user_id": [f"u{i % 7}" for i in range(100)], "item_index": range(100),
                              "rating": [5.0] * 70 + [4.0] * 20 + [1.0] * 10,
                              "date": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")})
        train, validation = self.call("stratified_split", frame, validation_fraction=0.2, seed=0)
        self.assertEqual(validation.rating.value_counts().to_dict(), {5.0: 14, 4.0: 4, 1.0: 2})
        self.assertEqual(train.rating.value_counts().to_dict(), {5.0: 56, 4.0: 16, 1.0: 8})
        self.assertFalse(set(train.item_index) & set(validation.item_index))
        pd.testing.assert_frame_equal(train, self.call("stratified_split", frame, 0.2, seed=0)[0])
        with self.assertRaises(ValueError):
            self.call("stratified_split", frame, validation_fraction=1.0)

    def test_history_is_oldest_first_and_excludes_target_same_day_and_future_ratings(self):
        _, history, _ = self.interaction_fixture()
        index = self.call("HistoryIndex", history)
        items, classes, ages = self.call("user_history", index, "u", "2024-01-02T23:00Z")
        # Only recipe "1" (index 0, rating 5) is from an earlier day; same-day recipe "2" stays hidden.
        np.testing.assert_array_equal(items, [0])
        np.testing.assert_array_equal(classes, [4])
        np.testing.assert_allclose(ages, [1.0])
        items, _, ages = self.call("user_history", index, "u", "2024-01-05", exclude_item=1)
        np.testing.assert_array_equal(items, [0, 2])
        np.testing.assert_allclose(ages, [4.0, 1.0])
        np.testing.assert_array_equal(self.call("user_history", index, "u", "2024-01-05", max_history=1)[0], [2])
        self.assertEqual(len(self.call("user_history", index, "nobody", "2024-01-05")[0]), 0)
        padded = self.call("pad_history", items, np.array([4, 3]), ages, max_history=4)
        np.testing.assert_array_equal(padded["history_mask"], [True, True, False, False])
        np.testing.assert_array_equal(padded["history_items"], [0, 2, 0, 0])

    def test_recency_prior_halves_attention_per_half_life_and_weights_recent_targets(self):
        history = pd.DataFrame({"user_id": ["u", "u"], "item_index": [0, 1],
                                "rating": [5.0, 5.0],
                                "date": pd.to_datetime(["2024-01-01", "2024-01-11"], utc=True)})
        dataset = self.call("HistoryDataset", history, history, half_life_days=10)
        np.testing.assert_allclose(dataset.weights, [0.5, 1.0])
        validation = self.call("HistoryDataset", history, history, training=False)
        np.testing.assert_array_equal(validation.weights, [1, 1])
        model = self.call("TwoTowerModel", 2, 1, embedding_dim=4, hidden_dim=8, half_life_days=10)
        with torch.no_grad():  # Without learned scores, only the recency prior sets the weights.
            model.history_attention.weight.zero_()
            model.history_attention.bias.zero_()
        weights = model.history_weights(torch.randn(1, 3, 8), torch.tensor([[11.0, 1.0, 0.0]]),
                                        torch.tensor([[True, True, False]]))
        torch.testing.assert_close(weights, torch.tensor([[1 / 3, 2 / 3, 0.0]]))

    def test_user_id_gru_item_towers_and_cut_points_receive_weighted_cross_entropy_gradients(self):
        torch.manual_seed(9)
        model = self.call("TwoTowerModel", 5, 3, embedding_dim=4, hidden_dim=8, user_id_dropout=0.0)
        mask = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
        logits = model(torch.tensor([1, 2, 0]), torch.randn(3, 3, 5), torch.tensor([[4, 0, 0], [2, 0, 0], [0, 0, 0]]),
                       torch.tensor([[30.0, 2.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mask, torch.eye(5)[:3])
        torch.nn.functional.nll_loss(model.rating_log_probabilities(logits), torch.tensor([0, 2, 4]),
                                     weight=torch.tensor([3.0, 2.0, 1.5, 1.0, 0.5])).backward()
        for module in (model.item_tower, model.user_embedding, model.history_projection,
                       model.history_rnn, model.user_head):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        for parameter in (model.first_threshold, model.threshold_gaps):
            self.assertTrue(parameter.grad is not None and torch.isfinite(parameter.grad).all())

    def test_cosine_score_ignores_vector_length_and_unknown_users_share_row_zero(self):
        torch.manual_seed(4)
        model = self.call("TwoTowerModel", 5, 2, embedding_dim=4, hidden_dim=8)
        users, items = torch.randn(3, 4), torch.randn(3, 4)
        torch.testing.assert_close(model.similarity(users, items),
                                   torch.nn.functional.cosine_similarity(users, items, dim=-1))
        torch.testing.assert_close(model.score_embeddings(users, items), model.score_embeddings(7 * users, 0.1 * items))
        lookup = self.call("build_user_lookup", pd.DataFrame({"user_id": ["b", "a", "b"]}))
        self.assertEqual(lookup, {"a": 1, "b": 2})
        targets = pd.DataFrame({"user_id": ["a", "new"], "item_index": [0, 1], "rating": [5.0, 3.0],
                                "date": pd.to_datetime(["2024-01-02"] * 2, utc=True)})
        np.testing.assert_array_equal(
            self.call("HistoryDataset", targets, targets, lookup, training=False).user_indices, [1, 0])
        model.eval()  # ID dropout only acts in training, so evaluation is deterministic.
        inputs = (torch.tensor([1, 2]), torch.randn(2, 1, 5), torch.zeros(2, 1, dtype=torch.long),
                  torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool))
        torch.testing.assert_close(model.encode_users(*inputs), model.encode_users(*inputs))

    def test_in_batch_loss_skips_false_negatives_and_corrects_for_popularity(self):
        model = self.call("TwoTowerModel", 5, 3, embedding_dim=4, hidden_dim=8)
        users, items = torch.randn(3, 4), torch.randn(3, 4)
        # Row 0's other columns are the same user (1) and the same recipe (0): no negatives, loss 0.
        # Row 2 is rated 2, so it is not a query.
        batch = {"rating": torch.tensor([5.0, 4.0, 2.0]), "user_index": torch.tensor([1, 1, 2]),
                 "item_index": torch.tensor([0, 1, 0])}
        losses, rows = self.call("in_batch_losses", model, users, items, batch, torch.zeros(3))
        np.testing.assert_array_equal(rows, [0, 1])
        self.assertAlmostEqual(float(losses[0]), 0.0, places=6)
        self.assertGreater(float(losses[1]), 0.0)  # Column 2 (user 2, recipe 0) is its negative.
        # A negative that batches sample often is penalized less once log q is subtracted.
        pair = {"rating": torch.tensor([5.0, 5.0]), "user_index": torch.tensor([1, 2]),
                "item_index": torch.tensor([0, 1])}
        uniform, _ = self.call("in_batch_losses", model, users[:2], items[:2], pair, torch.log(torch.full((2,), 0.5)))
        popular, _ = self.call("in_batch_losses", model, users[:2], items[:2], pair, torch.log(torch.tensor([0.1, 0.9])))
        self.assertLess(float(popular[0]), float(uniform[0]))

    def test_validation_in_batch_loss_mixes_users_instead_of_reading_sorted_runs(self):
        torch.manual_seed(5)
        targets = pd.DataFrame({"user_id": ["a"] * 16 + ["b"] * 16, "item_index": np.arange(32),
                                "rating": [5.0] * 32, "date": pd.to_datetime(["2024-01-02"] * 32, utc=True)})
        dataset = self.call("HistoryDataset", targets, targets.iloc[:0], {"a": 1, "b": 2}, training=False)
        model = self.call("TwoTowerModel", 4, 2, embedding_dim=4, hidden_dim=8)
        features = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float32)
        in_batch = {"log_q": torch.zeros(32), "temperature": 0.05, "min_rating": 4.0}
        # In row order, each batch of 16 is one user, every negative is masked, and CE is exactly 0.
        metrics = self.call("evaluate_two_tower", model, dataset, features, batch_size=16, device="cpu",
                            in_batch=in_batch)
        self.assertGreater(metrics["in_batch_cross_entropy"], 0.1)

    def test_serving_now_hides_recipes_rated_earlier_today(self):
        catalog, _, features = self.interaction_fixture()
        today = pd.DataFrame({"user_id": ["u"], "item_index": [2], "rating": [5.0],
                              "date": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=1)]})
        model = self.call("TwoTowerModel", 5, 1, embedding_dim=4, hidden_dim=8)
        result = self.call("recommend_recipes", model, catalog, features, today, "u", top_k=10)
        self.assertEqual(sorted(result.recipe_id), ["1", "2", "4", "5"])

    def test_ordinal_head_gives_valid_probabilities_and_monotone_expected_rating(self):
        torch.manual_seed(3)
        model = self.call("TwoTowerModel", 5, 3, embedding_dim=4, hidden_dim=8)
        with torch.no_grad():
            model.threshold_gaps.copy_(torch.tensor([-12.0, 0.0, 3.0]))  # Includes a near-zero step.
        logits = model.exceedance_logits(torch.linspace(-1, 1, 21))
        log_probabilities = model.rating_log_probabilities(logits)
        self.assertTrue(torch.isfinite(log_probabilities).all())
        probabilities = log_probabilities.exp()
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(21))
        expected = model.expected_rating(logits)
        torch.testing.assert_close(expected, (probabilities * torch.arange(1.0, 6.0)).sum(-1))
        self.assertTrue(torch.all(expected.diff() > 0))
        self.assertTrue(1 < expected.min() and expected.max() < 5)

    def test_balanced_class_weights_match_scikit_learn_formula(self):
        ratings = pd.Series([5.0] * 6 + [4.0] * 3 + [1.0])
        # n_samples / (n_present_classes * count); absent ratings 2 and 3 get the rarest weight.
        np.testing.assert_allclose(self.call("rating_class_weights", ratings),
                                   [10 / 3, 10 / 3, 10 / 3, 10 / 9, 10 / 18], rtol=1e-6)
        np.testing.assert_array_equal(self.call("rating_class_weights", ratings, class_weight=None),
                                      np.ones(5))
        for bad in ([4.5], [0.0], [6.0], [np.nan]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.call("rating_class_weights", bad)

    def test_five_star_thinning_samples_only_users_with_many_fives(self):
        rows = ([["heavy", 5]] * 400 + [["heavy", 3]] * 2          # limit max(20, 4) = 20 of 400
                + [["generous", 5]] * 60 + [["generous", 4]] * 40  # limit max(20, 80) >= 60
                + [["light", 5]] * 15 + [["light", 1]])
        targets = pd.DataFrame(rows, columns=["user_id", "rating"])
        targets["item_index"] = np.arange(len(targets))
        thinned = self.call("downsample_five_star_targets", targets, max_five_star=20,
                            max_five_to_other_ratio=2.0, seed=0)
        counts = thinned.groupby(["user_id", "rating"]).size()
        self.assertTrue(5 <= counts[("heavy", 5)] <= 40)  # Binomial(400, 0.05), mean 20.
        self.assertEqual(counts[("heavy", 3)], 2)
        self.assertEqual((counts[("generous", 5)], counts[("generous", 4)]), (60, 40))
        self.assertEqual((counts[("light", 5)], counts[("light", 1)]), (15, 1))
        pd.testing.assert_frame_equal(thinned, self.call("downsample_five_star_targets", targets,
                                                          max_five_star=20, max_five_to_other_ratio=2.0, seed=0))
        stricter = self.call("downsample_five_star_targets", targets, max_five_star=5,
                             max_five_to_other_ratio=1.0, seed=0).groupby(["user_id", "rating"]).size()
        self.assertTrue(20 <= stricter[("generous", 5)] <= 60)  # limit max(5, 40): keep p = 2/3 of 60.
        self.assertEqual((stricter[("generous", 4)], stricter[("light", 1)]), (40, 1))
        self.assertLess(stricter[("light", 5)], 15)  # limit max(5, 1): keep p = 1/3 of 15.

    def test_tiny_training_recommends_unseen_recipes_and_ignores_future(self):
        torch.manual_seed(11)
        catalog, interactions, features = self.interaction_fixture()
        train, validation = self.call("stratified_split", interactions, validation_fraction=0.5, seed=0)
        lookup = self.call("build_user_lookup", train)
        train_dataset = self.call("HistoryDataset", train, train, lookup)
        validation_dataset = self.call("HistoryDataset", validation, train, lookup, training=False)
        model = self.call("TwoTowerModel", 5, len(lookup), embedding_dim=4, hidden_dim=8)
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        with contextlib.redirect_stdout(io.StringIO()):
            model, history, metrics = self.call("train_two_tower", model, train_dataset,
                validation_dataset, features, epochs=3, batch_size=2, device="cpu")
        self.assertTrue(np.isfinite(history[["train_cross_entropy", "train_in_batch_cross_entropy",
                                             "validation_total_loss"]]).all().all())
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertEqual(set(metrics), {"rmse", "mae", "macro_mae", "balanced_accuracy", "cross_entropy",
                                        "in_batch_cross_entropy"})
        report = self.call("rating_class_report",
                           *self.call("predict_ratings", model, validation_dataset, features))
        self.assertEqual(report.index.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(report["count"].sum(), len(validation))
        self.assertEqual(report["predicted_count"].sum(), len(validation))
        for prefix in ("item_tower", "user_head"):
            self.assertTrue(any(not torch.equal(initial[name], parameter)
                                for name, parameter in model.named_parameters() if name.startswith(prefix)))
        result = self.call("recommend_recipes", model, catalog, features, interactions, "u",
                           user_lookup=lookup, top_k=10, as_of="2024-01-05")
        self.assertEqual(set(result.recipe_id), {"4", "5"})
        self.assertTrue(result.predicted_rating.between(1, 5).all())
        self.assertTrue(result.predicted_rating.is_monotonic_decreasing)
        before_future = self.call("recommend_recipes", model, catalog, features, interactions, "u",
                                  user_lookup=lookup, top_k=10, as_of="2024-01-03")
        reduced = interactions.loc[interactions.date < pd.Timestamp("2024-01-03", tz="UTC")]
        without_future = self.call("recommend_recipes", model, catalog, features, reduced, "u",
                                   user_lookup=lookup, top_k=10, as_of="2024-01-03")
        pd.testing.assert_frame_equal(before_future, without_future)
        self.assertIn("3", set(before_future.recipe_id))

    def test_notebook_artifacts_restore_identical_recommendations(self):
        torch.manual_seed(13)
        catalog = self.call("build_recipe_catalog", self.recipe_rows())
        features, _, numeric = self.call("build_numeric_features", catalog, n_clusters=2)
        observations = self.call("build_interactions", self.recipe_rows(), catalog)
        dim = features.shape[1]
        lookup = self.call("build_user_lookup", observations)
        model = self.call("TwoTowerModel", dim, len(lookup), embedding_dim=4, hidden_dim=8)
        expected = self.call("recommend_recipes", model, catalog, features, observations,
                             "user1", user_lookup=lookup, as_of="2024-01-02")
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
                CLASS_WEIGHT="balanced", training_class_weights=np.ones(5, np.float32),
                IN_BATCH_WEIGHT=0.5, IN_BATCH_TEMPERATURE=0.05, IN_BATCH_MIN_RATING=4,
                MAX_FIVE_STAR_TARGETS_PER_USER=20, MAX_FIVE_TO_OTHER_RATIO=2.0,
                DATA_PATH=Path("fixture.csv"), training_history=pd.DataFrame({"epoch": [1]}),
                user_lookup=lookup)
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
            vocabulary = pd.read_csv(artifact_dir / "user_vocabulary.csv", dtype={"user_id": "string"})
            restored_lookup = dict(zip(vocabulary.user_id, vocabulary.user_index))
            self.assertEqual(restored_lookup, lookup)
            result = self.call("recommend_recipes", restored, restored_catalog, restored_features,
                               restored_history, "user1", user_lookup=restored_lookup,
                               as_of="2024-01-02", item_embeddings=item_vectors)
            self.assertEqual(result.recipe_id.tolist(), expected.recipe_id.tolist())
            np.testing.assert_allclose(result.predicted_rating, expected.predicted_rating, atol=1e-6)
            self.assertIn("nutrition_scaler", joblib.load(artifact_dir / "content_preprocessing.joblib")["numeric"])
            config = json.loads((artifact_dir / "config.json").read_text())
            self.assertEqual((config["loss"], config["validation_split"], config["similarity"]),
                             ("class_weighted_cross_entropy", "stratified_random", "cosine"))
            self.assertEqual((config["retrieval_loss"], config["in_batch_weight"]),
                             ("in_batch_sampled_softmax_logq", 0.5))


if __name__ == "__main__":
    unittest.main()
