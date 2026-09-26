"""Behavioral regressions for the notebook, without downloading BERT or sentence-transformer weights.

Run with: python3 -m unittest discover -s DataMappingRecommendation/tests -v
"""
import ast
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

    # Fixtures -------------------------------------------------------------------------------

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

    def tiny_model(self, feature_dim, num_items, **kwargs):
        options = dict(embedding_dim=4, part_dim=4, hidden_dim=8)
        options.update(kwargs)
        return self.call("TwoTowerModel", [["dense", 0, feature_dim]], num_items, **options)

    @staticmethod
    def ratings(rows, date="2024-01-02"):
        frame = pd.DataFrame(rows, columns=["user_id", "item_index", "rating"])
        frame["date"] = pd.Timestamp(date, tz="UTC")
        return frame

    # Notebook structure ---------------------------------------------------------------------

    def test_later_cells_do_not_rebind_core_or_feature_definitions(self):
        # Every cell shares one namespace, so e.g. a table variable named like a helper breaks it.
        definition_tags = {"two-tower-core", "two-tower-features"}
        definitions = set()
        for cell in self.notebook["cells"]:
            if definition_tags & set(cell.get("metadata", {}).get("tags", [])):
                definitions |= {node.name for node in ast.parse("".join(cell["source"])).body
                                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code" or definition_tags & set(cell.get("metadata", {}).get("tags", [])):
                continue
            assigned = {node.id for node in ast.walk(ast.parse("".join(cell["source"])))
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
            self.assertFalse(assigned & definitions, f"cell {cell['id']} rebinds {sorted(assigned & definitions)}")

    # Catalog, prices, and text ----------------------------------------------------------------

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
        self.assertEqual(catalog.loc[0, "ingredient_terms"], ["tomato", "rice"])  # Falls back to the ingredients.
        self.assertEqual(catalog.loc[0, "adj_terms"], ["soft"])
        self.assertEqual(catalog.loc[0, "verb_terms"], ["chop"])  # Verbs take their base form.

    def test_terms_are_lowercased_lemmatized_singularized_and_deduplicated(self):
        self.assertEqual(self.call("normalize_term", "Green Onions", "n"), "green onion")
        self.assertEqual(self.call("normalize_term", "Cherry Tomatoes,", "n"), "cherry tomato")
        self.assertEqual(self.call("normalize_term", "Leaves", "n"), "leaf")
        self.assertEqual(self.call("normalize_term", "chopped", "v"), "chop")
        self.assertEqual(self.call("normalize_term", "Softer", "a"), "soft")
        self.assertEqual(self.call("normalize_terms", ["Tomatoes", "tomato", "", "2", "Onions"], "n"),
                         ["tomato", "onion"])
        frame = self.recipe_rows()
        frame["product"] = '[["Green Onions"], ["green onion"], ["Tomatoes"]]'
        frame["adj"] = '[["Softer", "fresh"], ["soft"]]'
        frame["verb"] = '[["Chopped", "minced"], ["chop"]]'
        catalog = self.call("build_recipe_catalog", frame)
        self.assertEqual(catalog.loc[0, "ingredient_terms"], ["green onion", "tomato"])
        self.assertEqual(catalog.loc[0, "adj_terms"], ["soft", "fresh"])
        self.assertEqual(catalog.loc[0, "verb_terms"], ["chop", "mince"])

    # Nutrition, price, and time ---------------------------------------------------------------

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
        first_groups, first_means, _ = self.call("build_numeric_features", first, n_clusters=1)
        second_groups, second_means, _ = self.call("build_numeric_features", second, n_clusters=1)
        for name in ("nutrition", "price"):
            np.testing.assert_allclose(first_groups[name], second_groups[name])
        pd.testing.assert_frame_equal(first_means, second_means)
        np.testing.assert_allclose(first_means.iloc[0], first[columns].mean())

    def test_numeric_missing_values_are_finite_and_preserve_missingness(self):
        frame = self.recipe_rows()
        columns = self.namespace["NUTRITION_COLUMNS"]
        frame[columns[0]] = np.nan
        frame.loc[0, columns[1]] = np.nan
        catalog = self.call("build_recipe_catalog", frame)
        groups, _, preprocessing = self.call("build_numeric_features", catalog, n_clusters=2)
        nutrition = groups["nutrition"]
        self.assertTrue(np.isfinite(nutrition).all() and np.isfinite(groups["price"]).all())
        self.assertEqual(preprocessing["nutrition_medians"][0], 0)
        np.testing.assert_array_equal(nutrition[:, len(columns)], np.ones(3))
        np.testing.assert_array_equal(nutrition[:, len(columns) + 1], [1, 0, 0])

    def test_nutrition_and_price_are_log_clipped_and_standardized_on_training_recipes(self):
        rng = np.random.default_rng(5)
        columns = self.namespace["NUTRITION_COLUMNS"]
        catalog = pd.DataFrame(rng.lognormal(5, 1, size=(50, len(columns))), columns=columns)
        catalog.iloc[0] *= 1e6  # An outlier recipe.
        catalog["price_total_known"] = rng.lognormal(3, 1, 50)
        catalog["price_error_dollars"] = 2.0 * rng.integers(0, 6, 50)
        catalog["price_missing_fraction"] = rng.uniform(0, 1, 50)
        catalog["price_all_missing"] = (rng.uniform(size=50) < 0.1).astype(int)
        train = np.arange(40)
        groups, _, preprocessing = self.call("build_numeric_features", catalog, train, n_clusters=3,
                                             clip_quantile=0.9)
        nutrition = groups["nutrition"]
        self.assertEqual(nutrition.shape, (50, 2 * len(columns) + 3))  # scaled, missing flags, cluster one-hot
        scaled = nutrition[:, :len(columns)]
        np.testing.assert_allclose(scaled[train].mean(axis=0), 0, atol=1e-5)  # Standardized on training recipes.
        limits = np.quantile(np.log1p(catalog[columns].to_numpy()), 0.9, axis=0)  # Over every recipe.
        np.testing.assert_allclose(preprocessing["nutrition_clip_limits"], limits)
        scaler = preprocessing["nutrition_scaler"]
        np.testing.assert_allclose(scaled[0], (limits - scaler.mean_) / scaler.scale_, rtol=1e-5)  # Clipped.
        np.testing.assert_array_equal(nutrition[:, 2 * len(columns):].argmax(axis=1), catalog["nutrition_cluster"])
        self.assertEqual(groups["price"].shape, (50, 4))
        np.testing.assert_allclose(groups["price"][train].mean(axis=0), 0, atol=1e-5)

    def test_saved_nutrition_clustering_reassigns_every_recipe_to_its_cluster(self):
        rng = np.random.default_rng(3)
        columns = self.namespace["NUTRITION_COLUMNS"]
        catalog = pd.DataFrame(rng.lognormal(3, 1.5, size=(300, len(columns))), columns=columns)
        catalog.iloc[:20, 0] = np.nan  # Missing values take the training medians, as in build_numeric_features.
        for column in self.namespace["PRICE_FEATURE_COLUMNS"]:
            catalog[column] = rng.uniform(0, 1, len(catalog))
        _, _, preprocessing = self.call("build_numeric_features", catalog, np.arange(200), n_clusters=6)
        weights = self.call("nutrition_clustering_weights", preprocessing)
        self.assertEqual(set(weights), {"columns", "medians", "clip_limits", "mean", "scale", "centers"})
        self.assertEqual(tuple(weights["centers"].shape), (6, len(columns)))
        assigned = self.call("assign_nutrition_clusters", catalog[columns].to_numpy(), weights)
        np.testing.assert_array_equal(assigned, catalog["nutrition_cluster"])
        filled = catalog[columns].fillna(pd.Series(preprocessing["nutrition_medians"], index=columns)).to_numpy()
        scaled = preprocessing["nutrition_scaler"].transform(
            np.minimum(np.log1p(filled), preprocessing["nutrition_clip_limits"])).astype(np.float32)
        np.testing.assert_array_equal(assigned, preprocessing["nutrition_clusterer"].predict(scaled))
        # Plain tensors and lists: a weights_only checkpoint holds and restores them.
        with tempfile.TemporaryDirectory() as directory:
            torch.save({"nutrition_clustering": weights}, Path(directory) / "weights.pt")
            restored = torch.load(Path(directory) / "weights.pt", weights_only=True)["nutrition_clustering"]
        np.testing.assert_array_equal(self.call("assign_nutrition_clusters", catalog[columns].to_numpy(), restored),
                                      assigned)

    def test_recipe_metadata_and_texts_follow_the_catalog_order(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "RAW_recipes.csv"
            pd.DataFrame({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "minutes": [10, 20, 30, 40],
                          "n_steps": [3, 4, 5, 6], "description": ["one", None, "three", "four"],
                          "ingredients": ["['salt']"] * 4,
                          "tags": ["['course', '60-minutes-or-less', 'Desserts']", "[]", "['easy', 'preparation', 'easy']",
                                   None]}).to_csv(raw_path, index=False)
            catalog = pd.DataFrame({"recipe_id": ["3", "1", "5"], "name": ["C", "A", "E"]})
            metadata = self.call("load_recipe_metadata", catalog, raw_path)
        self.assertEqual(metadata["recipe_id"].tolist(), ["3", "1", "5"])  # Catalog order, missing recipes too.
        self.assertEqual(metadata["minutes"].iloc[:2].tolist(), [30, 10])
        self.assertTrue(np.isnan(metadata["minutes"].iloc[2]))
        self.assertEqual(self.call("recipe_texts", catalog, metadata), ["C. three", "A. one", "E."])
        # Tags: hyphens become spaces, lowercase, deduplicated, category headers ("course", "preparation") dropped.
        self.assertEqual(self.call("recipe_tags", metadata), [["easy"], ["60 minutes or less", "desserts"], []])

    def test_recipe_extras_are_log_scaled_clipped_standardized_and_flag_missing_values(self):
        metadata = pd.DataFrame({"minutes": [10, 20, 30, 1e7, None, -5], "n_steps": [1, 2, 3, 4, 5, None]})
        extras, preprocessing = self.call("build_recipe_extras", metadata, clip_quantile=0.75)
        self.assertEqual(extras.shape, (6, 4))
        self.assertTrue(np.isfinite(extras).all())
        np.testing.assert_allclose(extras[:, :2].mean(axis=0), 0, atol=1e-6)
        np.testing.assert_array_equal(extras[:, 2], [0, 0, 0, 0, 1, 1])  # Missing or negative minutes.
        np.testing.assert_array_equal(extras[:, 3], [0, 0, 0, 0, 0, 1])
        self.assertLess(np.expm1(preprocessing["clip_limits"][0]), 1e7)  # A ten-million-minute outlier is clipped.
        self.assertEqual(preprocessing["medians"][0], 25)  # Median of the valid minutes 10, 20, 30, 1e7.
        trained, _ = self.call("build_recipe_extras", metadata, train_items=[0, 1, 2], clip_quantile=0.75)
        np.testing.assert_allclose(trained[:3, :2].mean(axis=0), 0, atol=1e-6)  # Standardized on training recipes.

    # Ratings, filters, and the split ----------------------------------------------------------

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

    def test_rating_zero_and_three_star_rows_are_cut_and_levels_are_negative_or_positive(self):
        catalog = pd.DataFrame({"recipe_id": [str(i) for i in range(1, 6)]})
        frame = pd.DataFrame({"recipe_id": [1, 2, 3, 4, 5], "user_id": 7, "rating": [5, 0, 3, 2, 1],
                              "date": "2024-01-01"})
        result = self.call("build_interactions", frame, catalog).sort_values("rating")
        self.assertEqual(result.rating.tolist(), [1, 2, 5])  # The review without stars and the 3-star rating are cut.
        self.assertEqual(result.level.tolist(), ["negative", "negative", "positive"])
        np.testing.assert_array_equal(self.call("rating_to_level", [1, 2, 4, 5]), [0, 0, 1, 1])

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

    def test_core_filter_drops_users_and_recipes_below_their_minimums(self):
        rows = [("u1", 0), ("u1", 1), ("u2", 0), ("u2", 1), ("u3", 1), ("u3", 2), ("u4", 3)]
        frame = pd.DataFrame(rows, columns=["user_id", "item_index"]).assign(rating=5.0)
        kept = self.call("core_filter", frame, 2, 2)
        # Recipe 2 has one rating; dropping it leaves u3 with one, and dropping u3 must be followed up.
        self.assertEqual(sorted(zip(kept.user_id, kept.item_index)), [("u1", 0), ("u1", 1), ("u2", 0), ("u2", 1)])
        self.assertEqual(len(self.call("core_filter", frame, 1, 1)), len(frame))
        # Users need 3 ratings; recipes any number.
        many = pd.DataFrame([("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 3)], columns=["user_id", "item_index"])
        self.assertEqual(self.call("core_filter", many, 3, 1)["user_id"].unique().tolist(), ["a"])
        self.assertTrue(self.call("core_filter", frame, 3, 3).empty)

    def test_per_user_split_holds_out_each_users_latest_ratings(self):
        frame = pd.DataFrame({"user_id": ["a"] * 5 + ["b"] * 2, "item_index": [4, 0, 3, 1, 2, 5, 6],
                              "rating": [5.0, 1.0, 4.0, 1.0, 3.0, 5.0, 4.0],
                              "date": pd.to_datetime(["2024-01-05", "2024-01-01", "2024-01-04", "2024-01-02",
                                                      "2024-01-03", "2024-01-01", "2024-01-02"], utc=True)})
        train, validation = self.call("per_user_split", frame, holdout_per_user=2)
        # a's last two by date are held out; b has no more than two, so it keeps both for its history.
        self.assertEqual(list(zip(validation.user_id, validation.item_index)), [("a", 3), ("a", 4)])
        self.assertEqual(sorted(zip(train.user_id, train.item_index)),
                         [("a", 0), ("a", 1), ("a", 2), ("b", 5), ("b", 6)])
        self.assertTrue((train.groupby("user_id")["date"].max().loc["a"] < validation["date"].min()))
        pd.testing.assert_frame_equal(train, self.call("per_user_split", frame.iloc[::-1], 2)[0])

    def test_positive_rows_keep_four_and_five_star_ratings(self):
        frame = pd.DataFrame({"rating": [1.0, 2.0, 4.0, 5.0], "item_index": range(4)})
        self.assertEqual(self.call("positive_rows", frame)["rating"].tolist(), [4.0, 5.0])

    def test_history_is_oldest_first_and_excludes_target_same_day_and_future_ratings(self):
        _, history, _ = self.interaction_fixture()
        index = self.call("HistoryIndex", history)
        items, classes, ages = self.call("user_history", index, "u", "2024-01-02T23:00Z")
        # Only recipe "1" (index 0, rating 5) is from an earlier day; same-day recipe "2" stays hidden.
        np.testing.assert_array_equal(items, [0])
        np.testing.assert_array_equal(classes, [4])  # Rating classes 0..4 are 1..5 stars.
        np.testing.assert_allclose(ages, [1.0])
        items, _, ages = self.call("user_history", index, "u", "2024-01-05", exclude_item=1)
        np.testing.assert_array_equal(items, [0, 2])
        np.testing.assert_allclose(ages, [4.0, 1.0])
        np.testing.assert_array_equal(self.call("user_history", index, "u", "2024-01-05", max_history=1)[0], [2])
        self.assertEqual(len(self.call("user_history", index, "nobody", "2024-01-05")[0]), 0)
        # One spare row per excluded recipe keeps max_history real ratings.
        np.testing.assert_array_equal(
            self.call("user_history", index, "u", "2024-01-05", max_history=1, exclude_item=[2, 1])[0], [0])
        padded = self.call("pad_history", items, np.array([4, 3]), ages, max_history=4)
        np.testing.assert_array_equal(padded["history_mask"], [True, True, False, False])
        np.testing.assert_array_equal(padded["history_items"], [0, 2, 0, 0])

    def test_history_rows_hide_their_target_recipe(self):
        ratings = pd.DataFrame({"user_id": ["u"] * 3, "item_index": [0, 1, 2], "rating": [1.0, 4.0, 5.0],
                                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"], utc=True)})
        dataset = self.call("HistoryDataset", self.call("positive_rows", ratings), ratings)
        sample = dataset[1]  # Target: recipe 2 (5 stars). Its history holds the 1-star and 4-star recipes.
        self.assertEqual(sample["item_index"], 2)
        self.assertEqual(sample["history_items"][sample["history_mask"]].tolist(), [0, 1])
        self.assertEqual(sample["history_ratings"][sample["history_mask"]].tolist(), [0, 3])

    def test_pooled_user_vector_weights_liked_recipes_by_rating_and_recency(self):
        # Classes 4, 4, 3, 0 are 5, 5, 4, 1 stars (weights 2, 2, 1, 0); the second 5 is one half-life older.
        weights = self.call("history_pool_weights", torch.tensor([[4, 4, 3, 0, 0]]),
                            torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0]]),
                            torch.tensor([[True, True, True, True, False]]), 10.0)
        torch.testing.assert_close(weights, torch.tensor([[2.0, 1.0, 1.0, 0.0, 0.0]]) / 4.0)
        # Without a liked recipe (a 2-star rating only) or without any history, nothing is pooled.
        for ratings, mask in (([[1, 0]], [[True, False]]), ([[0, 0]], [[False, False]])):
            torch.testing.assert_close(self.call("history_pool_weights", torch.tensor(ratings), torch.zeros(1, 2),
                                                 torch.tensor(mask), 10.0), torch.zeros(1, 2))
        # Training rows are weighted by recency too; validation rows count equally.
        history = pd.DataFrame({"user_id": ["u", "u"], "item_index": [0, 1], "rating": [5.0, 5.0],
                                "date": pd.to_datetime(["2024-01-01", "2024-01-11"], utc=True)})
        np.testing.assert_allclose(self.call("HistoryDataset", history, history, half_life_days=10).weights, [0.5, 1.0])
        np.testing.assert_array_equal(self.call("HistoryDataset", history, history, training=False).weights, [1, 1])

    def test_users_are_described_by_their_history_alone(self):
        torch.manual_seed(4)
        model = self.tiny_model(5, 6).eval()
        self.assertFalse(any("user" in name and "embedding" in name for name, _ in model.named_parameters()))
        features = torch.randn(6, 5)
        history_items = torch.tensor([[1, 2], [1, 2], [3, 0]])
        inputs = (history_items, features[history_items], torch.tensor([[4, 2], [4, 2], [0, 0]]),
                  torch.tensor([[3.0, 1.0], [3.0, 1.0], [2.0, 0.0]]),
                  torch.tensor([[True, True], [True, True], [True, False]]))
        users = model.encode_users(*inputs)
        torch.testing.assert_close(users[0], users[1])  # Same history, same vector: there is no user ID.
        self.assertFalse(torch.allclose(users[0], users[2]))
        torch.testing.assert_close(users.norm(dim=-1), torch.ones(3))
        items = torch.randn(3, 4)
        torch.testing.assert_close(model.similarity(users, items), (users * items).sum(-1))
        self.assertEqual(self.call("build_user_lookup", pd.DataFrame({"user_id": ["b", "a", "b"]})), {"a": 1, "b": 2})

    # Item tower -------------------------------------------------------------------------------

    def test_each_feature_group_gets_its_own_projection(self):
        matrix, groups = self.call("stack_feature_groups", {"a": np.ones((3, 2)), "b": np.zeros((3, 5))})
        self.assertEqual((matrix.shape, matrix.dtype, groups), ((3, 7), np.float32, [["a", 0, 2], ["b", 2, 7]]))
        torch.manual_seed(6)
        model = self.call("TwoTowerModel", groups, 3, embedding_dim=4, part_dim=3, hidden_dim=8, normalize=False).eval()
        self.assertEqual({name: tuple(layer.weight.shape) for name, layer in model.group_projections.items()},
                         {"a": (3, 2), "b": (3, 5)})
        self.assertEqual(model.item_mlp[0].in_features, 2 * 3)  # Each group contributes part_dim columns.
        features, index = torch.randn(3, 7), torch.arange(3)
        base = model.encode_items(features, index)
        for start, end in ((0, 2), (2, 7)):
            changed = features.clone()
            changed[:, start:end] += 1.0
            self.assertFalse(torch.allclose(model.encode_items(changed, index), base))

    def test_term_bags_average_frozen_pretrained_vectors_then_project(self):
        catalog = pd.DataFrame({"ingredient_terms": [["egg", "salt"], [], ["salt"]],
                                "adj_terms": [["fresh"], [], []], "verb_terms": [[], ["chop"], []]})
        vocabulary = self.call("build_term_vocabulary", catalog)
        self.assertEqual(vocabulary, ["chop", "egg", "fresh", "salt"])  # Term t has id t + 1; 0 pads.
        matrix = self.call("term_id_matrix", catalog["ingredient_terms"], vocabulary)
        np.testing.assert_array_equal(matrix, [[2, 4], [0, 0], [4, 0]])
        np.testing.assert_array_equal(self.call("term_id_matrix", [["egg", "unknown"]], vocabulary), [[2]])
        torch.manual_seed(7)
        model = self.call("TwoTowerModel", [["dense", 0, 2]], 3, embedding_dim=4, part_dim=3, hidden_dim=8,
                          num_terms=4, term_dim=5, term_fields=[("ingredients", 2)], normalize=False)
        vectors = torch.randn(4, 5)
        model.set_term_vectors(vectors).set_item_terms("ingredients", matrix)
        self.assertFalse(model.term_bag.weight.requires_grad)  # Frozen, like EmbeddingBag.from_pretrained(freeze=True).
        torch.testing.assert_close(model.term_bag.weight[0], torch.zeros(5))
        seen = []
        model.item_mlp.register_forward_hook(lambda module, inputs, output: seen.append(inputs[0]))
        model.encode_items(torch.zeros(3, 2), torch.arange(3))
        bags = seen[0][:, 3:6]  # [dense group, ingredients]; each part is part_dim wide.
        project = model.term_projections["ingredients"]
        with torch.no_grad():
            torch.testing.assert_close(bags[0], project((vectors[1] + vectors[3]) / 2))  # egg and salt
            torch.testing.assert_close(bags[1], project(torch.zeros(5)))  # No terms: an empty bag.
            torch.testing.assert_close(bags[2], project(vectors[3]))
        with self.assertRaises(ValueError):
            model.set_item_terms("ingredients", matrix + 9)  # Ids beyond num_terms.
        with self.assertRaises(ValueError):
            model.set_term_vectors(torch.randn(3, 5))

    def test_recipe_id_embeddings_id_dropout_and_normalized_towers(self):
        rows, count = self.call("build_item_id_rows", [0, 0, 2, 2, 2, 3], 5, min_ratings=2)
        np.testing.assert_array_equal(rows, [1, 0, 2, 0, 0])
        self.assertEqual(count, 2)
        torch.manual_seed(2)
        model = self.tiny_model(3, 5, part_dim=2, num_item_ids=count).set_item_id_rows(rows).eval()
        same_content = torch.ones(5, 3)
        vectors = model.encode_items(same_content, torch.arange(5))
        torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(5))  # L2-normalized
        torch.testing.assert_close(vectors[1], vectors[3])  # rare recipes: zero PAD row, content only
        self.assertFalse(torch.allclose(vectors[0], vectors[2]))  # their own ID rows differ
        torch.testing.assert_close(model.item_embedding.weight[0], torch.zeros(2))
        # In training, a hidden ID falls back to the PAD row: the recipe is scored like a rare one.
        model.config["item_id_dropout"] = 0.999999
        with torch.no_grad():
            hidden = model.train().encode_items(same_content, torch.arange(5))
        torch.testing.assert_close(hidden, vectors[1].expand(5, -1))
        raw = self.tiny_model(3, 4, normalize=False)
        self.assertFalse(torch.allclose(raw.encode_items(torch.randn(4, 3), torch.arange(4)).norm(dim=-1),
                                        torch.ones(4)))

    # Training ---------------------------------------------------------------------------------

    def test_both_towers_learn_from_the_in_batch_softmax(self):
        torch.manual_seed(9)
        model = self.call("TwoTowerModel", [["a", 0, 2], ["b", 2, 5]], 6, embedding_dim=4, part_dim=4,
                          hidden_dim=8, num_item_ids=3, item_id_dropout=0.0, num_terms=4, term_dim=3,
                          term_fields=[("ingredients", 2)])
        model.set_item_id_rows([1, 2, 3, 0, 0, 0]).set_term_vectors(torch.randn(4, 3))
        model.set_item_terms("ingredients", [[1, 2], [3, 0], [4, 1], [0, 0], [2, 0], [1, 0]])
        features = torch.randn(6, 5)
        history_items = torch.tensor([[3, 4, 0], [5, 0, 0], [0, 0, 0]])
        history = (history_items, features[history_items], torch.tensor([[4, 0, 0], [2, 0, 0], [0, 0, 0]]),
                   torch.tensor([[30.0, 2.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                   torch.tensor([[True, True, False], [True, False, False], [False, False, False]]))
        item_index = torch.tensor([0, 1, 2])
        # The model's only output is the dot product of the tower vectors: no rating head.
        torch.testing.assert_close(model(*history, features[item_index], item_index),
                                   (model.encode_users(*history) * model.encode_items(features[item_index],
                                                                                      item_index)).sum(-1))
        losses = self.call("batch_softmax_losses", model.encode_users(*history),
                           model.encode_items(features[item_index], item_index), {"item_index": item_index},
                           torch.zeros(6))
        losses.mean().backward()
        for module in (model.group_projections, model.term_projections, model.item_embedding, model.item_mlp,
                       model.history_rnn, model.rating_embedding, model.user_mlp):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        self.assertIsNone(model.term_bag.weight.grad)  # The pretrained term vectors never train.

    def test_in_batch_softmax_masks_other_liked_recipes_and_corrects_for_popularity(self):
        torch.manual_seed(1)
        users, items = torch.randn(3, 4), torch.randn(3, 4)
        logits = users @ items.T / 0.5
        batch = {"item_index": torch.tensor([10, 15, 20]),
                 "mask": torch.tensor([[False, True, False], [False, False, False], [False, False, False]])}
        losses = self.call("batch_softmax_losses", users, items, batch, torch.zeros(50), 0.5)
        torch.testing.assert_close(losses[0], -torch.log_softmax(logits[0, [0, 2]], dim=0)[0])  # Column 1 masked.
        torch.testing.assert_close(losses[1], -torch.log_softmax(logits[1], dim=0)[1])
        # Without a collate mask, only a repeat of the row's own recipe is masked.
        repeated = self.call("batch_softmax_losses", users, items, {"item_index": torch.tensor([10, 10, 20])},
                             torch.zeros(50), 0.5)
        torch.testing.assert_close(repeated[0], -torch.log_softmax(logits[0, [0, 2]], dim=0)[0])
        # log q: a popular recipe is penalized less as a negative (row 0) and helped less as a positive (row 1).
        plain = self.call("batch_softmax_losses", users, items, {"item_index": batch["item_index"]},
                          torch.zeros(50), 0.5)
        popular_q = torch.zeros(50)
        popular_q[15] = 3.0
        popular = self.call("batch_softmax_losses", users, items, {"item_index": batch["item_index"]}, popular_q, 0.5)
        self.assertLess(float(popular[0]), float(plain[0]))
        self.assertGreater(float(popular[1]), float(plain[1]))
        # The same log q for every recipe shifts each row equally, which changes nothing.
        torch.testing.assert_close(self.call("batch_softmax_losses", users, items, {"item_index": batch["item_index"]},
                                             torch.full((50,), -12.0), 0.5), plain)

    def test_collate_masks_liked_recipes_but_keeps_disliked_ones_as_negatives(self):
        # u1 liked 10 (5) and 12 (4), disliked 22 (2); u2 liked 15 and 10, disliked 12 (1); u3 liked 44 and 22.
        ratings = self.ratings([["u1", 10, 5.0], ["u1", 12, 4.0], ["u1", 22, 2.0], ["u2", 15, 5.0], ["u2", 10, 4.0],
                                ["u2", 12, 1.0], ["u3", 44, 5.0], ["u3", 22, 4.0]])
        targets = self.call("positive_rows", ratings)
        dataset = self.call("HistoryDataset", targets, ratings)
        rows = [int(np.flatnonzero((targets.user_id == user) & (targets.item_index == item))[0])
                for user, item in [("u1", 10), ("u2", 15), ("u1", 12), ("u3", 22)]]
        roles = dataset.batch_roles(rows, [10, 15, 12, 22])
        names = np.array(self.namespace["BATCH_ROLES"], dtype=object)[roles].tolist()
        # Columns I:10, I:15, I:12, I:22. A 4-star recipe is liked (masked); a 1-2 star recipe stays a negative.
        self.assertEqual(names, [["P", "N", "mask: liked", "N"],             # u1 liked 12, disliked 22
                                 ["mask: liked", "P", "N", "N"],             # u2 liked 10, disliked 12
                                 ["mask: liked", "N", "P", "N"],             # u1 again
                                 ["N", "N", "N", "P"]])                      # u3 rated none of the others
        batch = dataset.collate([dataset[row] for row in rows])
        np.testing.assert_array_equal(batch["mask"].numpy(), roles == self.namespace["ROLE_LIKED"])
        self.assertEqual(tuple(batch["mask"].shape), (4, 4))

    def test_resample_caps_rows_per_user_and_covers_every_row_over_epochs(self):
        ratings = pd.DataFrame({"user_id": ["a"] * 6 + ["b"] * 2, "item_index": range(8), "rating": 5.0,
                                "date": pd.Timestamp("2024-01-02", tz="UTC")})
        dataset = self.call("HistoryDataset", ratings, ratings)
        self.assertEqual(len(dataset), 8)
        used = set()
        for epoch in range(20):
            dataset.resample(max_per_user=2, seed=epoch)
            users = dataset.targets["user_id"].to_numpy()[dataset.active]
            self.assertEqual(((users == "a").sum(), (users == "b").sum(), len(dataset)), (2, 2, 4))
            self.assertEqual({dataset[i]["row"] for i in range(len(dataset))}, set(dataset.active.tolist()))
            used |= set(dataset.active.tolist())
        self.assertEqual(used, set(range(8)))  # every row is used across epochs
        self.assertEqual(len(dataset.resample(None)), 8)

    def test_validation_in_batch_loss_mixes_users_instead_of_reading_sorted_runs(self):
        torch.manual_seed(5)
        targets = pd.DataFrame({"user_id": ["a"] * 16 + ["b"] * 16, "item_index": np.arange(32),
                                "rating": [5.0] * 32, "date": pd.to_datetime(["2024-01-02"] * 32, utc=True)})
        dataset = self.call("HistoryDataset", targets, targets.iloc[:0], training=False)
        model = self.tiny_model(4, 32)
        features = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float32)
        # In row order, each batch of 16 is one user, every negative is masked, and CE is exactly 0.
        metrics = self.call("evaluate_two_tower", model, dataset, features, batch_size=16, device="cpu")
        self.assertGreater(metrics["in_batch_cross_entropy"], 0.1)
        self.assertEqual(set(metrics), {"in_batch_cross_entropy", "in_batch_rows"})

    def test_tiny_training_recommends_unseen_recipes_and_ignores_future(self):
        torch.manual_seed(11)
        catalog, interactions, features = self.interaction_fixture()
        train, validation = self.call("per_user_split", interactions, holdout_per_user=1)
        train_targets = self.call("positive_rows", train)
        validation_targets = self.call("positive_rows", validation)
        train_dataset = self.call("HistoryDataset", train_targets, train)
        validation_dataset = self.call("HistoryDataset", validation_targets, train, training=False,
                                       extra_interactions=validation)
        item_rows, item_ids = self.call("build_item_id_rows", train_targets["item_index"], 5, min_ratings=1)
        model = self.tiny_model(5, 5, num_item_ids=item_ids, num_terms=3, term_dim=4,
                                term_fields=[("ingredients", 2)]).set_item_id_rows(item_rows)
        model.set_term_vectors(torch.randn(3, 4)).set_item_terms("ingredients", [[1, 2], [2, 0], [3, 0], [1, 3], [0, 0]])
        recall = self.call("RecallAtK", train, validation_targets, 5, ks=(1, 2))
        self.assertTrue(recall.users)
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            model, history, metrics = self.call("train_two_tower", model, train_dataset,
                validation_dataset, features, epochs=3, batch_size=2, device="cpu", max_rows_per_user=1,
                recall_evaluator=recall)
        self.assertIn("in-batch softmax CE", output.getvalue())
        self.assertIn("of ln 2", output.getvalue())
        self.assertTrue(np.isfinite(history[["train_in_batch_cross_entropy", "validation_in_batch_cross_entropy",
                                             "validation_recall@1", "validation_recall@2"]]).all().all())
        self.assertTrue((history["train_rows"] <= train_dataset.targets["user_id"].nunique()).all())
        self.assertEqual(set(metrics), {"in_batch_cross_entropy", "in_batch_rows"})
        for prefix in ("item_mlp", "user_mlp", "group_projections", "term_projections"):
            self.assertTrue(any(not torch.equal(initial[name], parameter)
                                for name, parameter in model.named_parameters() if name.startswith(prefix)))
        torch.testing.assert_close(model.term_bag.weight, initial["term_bag.weight"])  # Frozen.
        result = self.call("recommend_recipes", model, catalog, features, interactions, "u", top_k=10,
                           as_of="2024-01-05")
        self.assertEqual(set(result.recipe_id), {"4", "5"})
        self.assertTrue(result.similarity.is_monotonic_decreasing)
        before_future = self.call("recommend_recipes", model, catalog, features, interactions, "u",
                                  top_k=10, as_of="2024-01-03")
        reduced = interactions.loc[interactions.date < pd.Timestamp("2024-01-03", tz="UTC")]
        without_future = self.call("recommend_recipes", model, catalog, features, reduced, "u",
                                   top_k=10, as_of="2024-01-03")
        pd.testing.assert_frame_equal(before_future, without_future)
        self.assertIn("3", set(before_future.recipe_id))

    # Evaluation and serving -------------------------------------------------------------------

    def test_recall_at_k_counts_heldout_positives_and_never_recommends_seen_recipes(self):
        history = pd.DataFrame({"user_id": ["u", "u", "v"], "item_index": [0, 1, 0], "rating": [5.0, 2.0, 4.0],
                                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"], utc=True)})
        heldout = pd.DataFrame({"user_id": ["u", "u", "w"], "item_index": [2, 4, 1], "rating": [5.0, 4.0, 5.0],
                                "date": pd.Timestamp("2024-01-03", tz="UTC")})
        recall = self.call("RecallAtK", history, heldout, 5, ks=(1, 2, 3))
        self.assertEqual(recall.users, ["u"])  # w has no training history; v has nothing held out
        # Most-liked order is 0 > 1 > 2 > 3 > 4; u already rated 0 and 1, so it gets 2, 3, 4.
        result = recall.popularity([10, 9, 8, 7, 6])
        self.assertEqual((result["recall@1"], result["recall@2"], result["recall@3"]), (0.5, 0.5, 1.0))
        self.assertEqual(result["heldout_positives"], 2)
        scored = recall.evaluate(self.tiny_model(3, 5), np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32),
                                 "cpu")
        self.assertEqual(scored["recall@3"], 1.0)  # only 3 unseen recipes remain, holding both targets
        # Recipes outside candidate_items (e.g. dropped by the filter) are never recommended: 3 drops out.
        in_core = self.call("RecallAtK", history, heldout, 5, ks=(1, 2), candidate_items=[True, True, True, False, True])
        result = in_core.popularity([10, 9, 8, 7, 6])
        self.assertEqual((result["recall@1"], result["recall@2"]), (0.5, 1.0))

    def test_item_index_serves_the_same_recommendations_as_scoring_every_recipe(self):
        vectors = np.random.default_rng(1).normal(size=(30, 4)).astype(np.float32)
        index = self.call("ItemIndex", vectors)  # FAISS on Linux when installed; exact torch scan on macOS
        queries = np.random.default_rng(2).normal(size=(3, 4)).astype(np.float32)
        scores, ids = index.search(queries, 5)
        exact = torch.topk(torch.from_numpy(queries) @ torch.from_numpy(vectors).T, 5, dim=1)
        np.testing.assert_array_equal(ids, exact.indices.numpy())
        np.testing.assert_allclose(scores, exact.values.numpy(), rtol=1e-5, atol=1e-5)
        torch.manual_seed(3)
        catalog, interactions, features = self.interaction_fixture()
        model = self.tiny_model(5, 5)
        catalog_index = self.call("ItemIndex", self.call("encode_catalog", model, features).numpy())
        scanned = self.call("recommend_recipes", model, catalog, features, interactions, "u", top_k=3,
                            as_of="2024-01-05")
        indexed = self.call("recommend_recipes", model, catalog, features, interactions, "u", top_k=3,
                            as_of="2024-01-05", index=catalog_index)
        self.assertEqual(indexed.recipe_id.tolist(), scanned.recipe_id.tolist())
        np.testing.assert_allclose(indexed.similarity, scanned.similarity, atol=1e-5)

    def test_users_without_history_get_the_most_liked_recipes(self):
        catalog, interactions, features = self.interaction_fixture()
        extra = pd.DataFrame({"user_id": ["w", "x"], "item_index": [3, 3], "rating": [5.0, 3.0], "level": "positive",
                              "date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)})
        interactions = pd.concat([interactions, extra], ignore_index=True)
        model = self.tiny_model(5, 5)
        result = self.call("recommend_recipes", model, catalog, features, interactions, "nobody", top_k=2,
                           as_of="2024-01-05")
        # Recipe "4" (index 3) is liked three times; ties keep catalog order.
        self.assertEqual(result.recipe_id.tolist(), ["4", "1"])
        self.assertEqual(result.source.unique().tolist(), ["popularity"])
        self.assertTrue(result.similarity.isna().all())
        known = self.call("recommend_recipes", model, catalog, features, interactions, "u", top_k=2, as_of="2024-01-05")
        self.assertEqual(known.source.unique().tolist(), ["two_tower"])

    def test_serving_now_hides_recipes_rated_earlier_today(self):
        catalog, _, features = self.interaction_fixture()
        today = pd.DataFrame({"user_id": ["u"], "item_index": [2], "rating": [5.0],
                              "date": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(1, unit="s")]})
        model = self.tiny_model(5, 5)
        result = self.call("recommend_recipes", model, catalog, features, today, "u", top_k=10)
        self.assertEqual(sorted(result.recipe_id), ["1", "2", "4", "5"])

    def test_sentiment_rows_follow_the_mf_als_rules_and_sample_one_unobserved_negative_per_positive(self):
        cell = [cell for cell in self.notebook["cells"]
                if "sentiment-definitions" in cell.get("metadata", {}).get("tags", [])]
        self.assertEqual(len(cell), 1)
        context = dict(self.namespace, RANDOM_STATE=0, SENTIMENT_RATINGS=(0, 1, 2, 4, 5), SENTIMENT_POSITIVE_MIN=4,
                       MIN_USER_RATINGS=3, SENTIMENT_NEGATIVES_PER_POSITIVE=1)
        exec(compile("".join(cell[0]["source"]), "sentiment-definitions", "exec"), context)
        catalog = pd.DataFrame({"recipe_id": [str(i) for i in range(12)]})
        # u1: 0, 5, 4, 3, 5 (by date); u2: 2, 5, 3 (two kept ratings -> dropped); u3: 1, 4, 5, 5.
        rows = [("u1", 0, 0), ("u1", 1, 5), ("u1", 2, 4), ("u1", 3, 3), ("u1", 4, 5), ("u2", 5, 2), ("u2", 6, 5),
                ("u2", 7, 3), ("u3", 8, 1), ("u3", 9, 4), ("u3", 10, 5), ("u3", 11, 5)]
        frame = pd.DataFrame({"user_id": [u for u, _, _ in rows], "recipe_id": [str(r) for _, r, _ in rows],
                              "rating": [x for _, _, x in rows],
                              "date": [f"2024-01-{day:02d}" for day in range(1, len(rows) + 1)]})
        every = context["ratings_with_zero"](frame, catalog)
        self.assertNotIn(3, every["rating"].tolist())  # 3-star ratings are never loaded
        labelled, history = context["prepare_sentiment_rows"](every)
        sampled = labelled[labelled["source"] == context["SAMPLED_SOURCE"]]
        observed = labelled[labelled["source"] != context["SAMPLED_SOURCE"]]
        self.assertEqual(sorted(labelled["user_id"].unique()), ["u1", "u3"])  # u2 has fewer than 3 kept ratings
        self.assertEqual(observed["rating"].tolist().count(5), 4)  # every 5-star rating stays
        # Sentiment one-hot: 4-5 positive, 0-2 negative, and every sampled row is a rating-0 negative.
        np.testing.assert_array_equal(labelled["sentiment_positive"] + labelled["sentiment_negative"], 1)
        np.testing.assert_array_equal(observed["sentiment_positive"], (observed["rating"] >= 4).astype(int))
        self.assertTrue((sampled["rating"] == 0).all() and (sampled["sentiment"] == "negative").all())
        # One unobserved recipe per positive, in the positive's split, never one the user rated.
        positives = observed[observed["sentiment_positive"] == 1]
        self.assertEqual(len(sampled), len(positives))
        pd.testing.assert_series_equal(sampled.groupby(["user_id", "split"]).size(),
                                       positives.groupby(["user_id", "split"]).size())
        rated = set(zip(every["user_id"], every["item_index"]))
        self.assertFalse(any(pair in rated for pair in zip(sampled["user_id"], sampled["item_index"])))
        # Per user by date: the latest rating is test and the one before valid; the history is the observed train rows.
        latest = observed.sort_values("date").groupby("user_id").tail(2)
        self.assertEqual(sorted(latest["split"].tolist()), ["test", "test", "valid", "valid"])
        self.assertTrue((history["split"] == "train").all() and (history["source"] != context["SAMPLED_SOURCE"]).all())
        self.assertIn(0.0, history["rating"].tolist())  # a 0-star review is history too

