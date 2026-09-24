"""Behavioral regressions for the notebook, without downloading BERT weights.

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
                                        torch.tensor([[4, 4, 0]]), torch.tensor([[True, True, False]]))
        torch.testing.assert_close(weights, torch.tensor([[1 / 3, 2 / 3, 0.0]]))
        # Same age: a neutral rating (3) starts at half a positive one's weight; negative counts fully.
        same_age, both = torch.zeros(1, 2), torch.tensor([[True, True]])
        torch.testing.assert_close(model.history_weights(torch.randn(1, 2, 8), same_age, torch.tensor([[2, 4]]), both),
                                   torch.tensor([[1 / 3, 2 / 3]]))
        torch.testing.assert_close(model.history_weights(torch.randn(1, 2, 8), same_age, torch.tensor([[0, 4]]), both),
                                   torch.tensor([[0.5, 0.5]]))

    def test_both_towers_learn_from_the_batch_softmax(self):
        torch.manual_seed(9)
        model = self.call("TwoTowerModel", 5, 3, embedding_dim=4, hidden_dim=8, user_id_dropout=0.0)
        mask = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
        user_args = (torch.tensor([1, 2, 0]), torch.randn(3, 3, 5), torch.tensor([[4, 0, 0], [2, 0, 0], [0, 0, 0]]),
                     torch.tensor([[30.0, 2.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mask)
        features = torch.eye(5)
        # The model's only output is the dot product of the tower vectors: no rating head.
        similarity = model(*user_args, features[:3])
        torch.testing.assert_close(similarity, (model.encode_users(*user_args) * model.encode_items(features[:3])).sum(-1))
        self.assertFalse(any(name.startswith(("logit_scale", "first_threshold", "threshold_gaps"))
                             for name, _ in model.named_parameters()))
        users, items = model.encode_users(*user_args), model.encode_items(features[:3])
        batch = {"user_index": torch.tensor([1, 2, 0]), "item_index": torch.tensor([0, 1, 2]),
                 "contrast_sign": torch.tensor([1.0, 0.0, 1.0]), "contrast_item_index": torch.tensor([3, 0, 4])}
        losses, _, _ = self.call("batch_softmax_losses", model, users, items, batch, features, torch.zeros(5))
        losses.mean().backward()
        for module in (model.item_tower, model.user_embedding, model.history_projection,
                       model.history_rnn, model.user_head):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_dot_product_score_and_unknown_users_share_row_zero(self):
        torch.manual_seed(4)
        model = self.call("TwoTowerModel", 5, 2, embedding_dim=4, hidden_dim=8)
        users, items = torch.randn(3, 4), torch.randn(3, 4)
        torch.testing.assert_close(model.similarity(users, items), (users * items).sum(-1))
        torch.testing.assert_close(model.similarity(2 * users, items), 2 * model.similarity(users, items))
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

    def test_batch_softmax_masks_false_negatives_and_pad_like_option_b(self):
        # The worked example: u1 likes r10, dislikes r22; u2 likes r15, dislikes r30; u3 likes r10, no dislike.
        model = self.call("TwoTowerModel", 50, 3, embedding_dim=4, hidden_dim=8)
        model.encode_items = lambda features, item_index=None: features
        features = torch.randn(50, 4)
        batch = {"user_index": torch.tensor([1, 2, 3]), "item_index": torch.tensor([10, 15, 10]),
                 "contrast_sign": torch.tensor([1.0, 1.0, 0.0]), "contrast_item_index": torch.tensor([22, 30, 0])}
        users = torch.randn(3, 4)
        losses, has_hard, wins = self.call("batch_softmax_losses", model, users, features[batch["item_index"]],
                                           batch, features, torch.zeros(50), 1.0)
        np.testing.assert_array_equal(has_hard, [True, True, False])
        # Recompute row u1 by hand: columns [r10, r15, r10(u3: mask), r22, r30, PAD(mask)].
        columns = features[[10, 15, 22, 30]]
        expected = -torch.log_softmax(columns @ users[0], dim=0)[0]
        torch.testing.assert_close(losses[0], expected)
        # u3's positive r10 masks u1's r10 column too; its PAD column never counts.
        expected = -torch.log_softmax(features[[15, 10, 22, 30]] @ users[2], dim=0)[1]
        torch.testing.assert_close(losses[2], expected)
        self.assertEqual(bool(wins[0]), bool(users[0] @ features[10] > users[0] @ features[22]))
        # Another positive of the same user is masked as a false negative.
        same = dict(batch, user_index=torch.tensor([1, 1, 3]))
        same_losses, _, _ = self.call("batch_softmax_losses", model, users, features[batch["item_index"]],
                                      same, features, torch.zeros(50), 1.0)
        expected = -torch.log_softmax(features[[10, 22, 30]] @ users[0], dim=0)[0]
        torch.testing.assert_close(same_losses[0], expected)
        # A popular in-batch column (large q) is penalized less as a negative.
        popular_q = torch.zeros(50); popular_q[15] = 3.0
        popular, _, _ = self.call("batch_softmax_losses", model, users, features[batch["item_index"]],
                                  batch, features, popular_q, 1.0)
        self.assertLess(float(popular[0]), float(losses[0]))

    def test_log_q_is_centered_so_hard_negatives_stay_in_play_and_w_weights_them(self):
        model = self.call("TwoTowerModel", 50, 3, embedding_dim=4, hidden_dim=8)
        model.encode_items = lambda features, item_index=None: features
        features = torch.randn(50, 4)
        batch = {"user_index": torch.tensor([1, 2, 3]), "item_index": torch.tensor([10, 15, 20]),
                 "contrast_sign": torch.tensor([1.0, 1.0, 0.0]), "contrast_item_index": torch.tensor([22, 30, 0])}
        users, items = torch.randn(3, 4), features[batch["item_index"]]
        plain, _, _ = self.call("batch_softmax_losses", model, users, items, batch, features, torch.zeros(50), 1.0)
        # A log q shared by every recipe (real values are about -12) changes nothing once centered,
        # so it can't lift the in-batch block above the hard negatives.
        shared, _, _ = self.call("batch_softmax_losses", model, users, items, batch, features,
                                 torch.full((50,), -12.0), 1.0)
        torch.testing.assert_close(shared, plain)
        # log(w) on the hard block: each hard negative counts w times in the denominator.
        heavier, _, _ = self.call("batch_softmax_losses", model, users, items, batch, features,
                                  torch.zeros(50), 1.0, 3.0)
        logits = users[0] @ features[[10, 15, 20, 22, 30]].T + torch.log(torch.tensor([1.0, 1, 1, 3, 3]))
        torch.testing.assert_close(heavier[0], torch.logsumexp(logits, 0) - logits[0])
        self.assertTrue(torch.all(heavier > plain))

    def test_collate_masks_other_positives_neutral_recipes_and_pad_like_the_guide(self):
        # u1 likes 10 and 12, dislikes 22, is neutral on 44; u2 likes 15 and 10, dislikes 30; u3 likes 44 and 10.
        ratings = pd.DataFrame([["u1", 10, 5.0], ["u1", 12, 5.0], ["u1", 22, 2.0], ["u1", 44, 3.0],
                                ["u2", 15, 5.0], ["u2", 10, 4.0], ["u2", 30, 1.0],
                                ["u3", 44, 5.0], ["u3", 10, 4.0]], columns=["user_id", "item_index", "rating"])
        ratings["date"] = pd.Timestamp("2024-01-02", tz="UTC")
        targets = self.call("positive_rows", ratings)
        dataset = self.call("HistoryDataset", targets, ratings, contrast_pool=ratings)
        rows = [int(np.flatnonzero((targets.user_id == user) & (targets.item_index == item))[0])
                for user, item in [("u1", 10), ("u2", 15), ("u3", 44)]]
        roles = dataset.batch_roles(rows, [10, 15, 44], [22, 30, -1])
        names = np.array(self.namespace["BATCH_ROLES"], dtype=object)[roles].tolist()
        # Columns: I:10, I:15, I:44, H:22, H:30, H:PAD. The diagonal and own dislikes are never masked.
        self.assertEqual(names, [["P", "N", "mask: neutral", "HN", "HN", "mask: PAD"],   # 44 is u1's neutral
                                 ["mask: liked", "P", "N", "HN", "HN", "mask: PAD"],     # u2 also liked 10
                                 ["mask: liked", "N", "P", "HN", "HN", "mask: PAD"]])    # u3 also liked 10
        batch = dataset.collate([dataset[row] for row in rows])
        self.assertEqual(batch["hard_item_index"].tolist(), [22, 30, -1])  # u3 has no dislike: PAD
        np.testing.assert_array_equal(batch["mask"].numpy(), roles >= self.namespace["ROLE_LIKED"])

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

    def test_recipe_id_embeddings_and_normalized_towers(self):
        rows, count = self.call("build_item_id_rows", [0, 0, 2, 2, 2, 3], 5, min_ratings=2)
        np.testing.assert_array_equal(rows, [1, 0, 2, 0, 0])
        self.assertEqual(count, 2)
        torch.manual_seed(2)
        model = self.call("TwoTowerModel", 3, 1, embedding_dim=4, hidden_dim=8, num_items=5,
                          num_item_ids=count, item_id_dim=2).set_item_id_rows(rows)
        model.eval()
        same_content = torch.ones(5, 3)
        vectors = model.encode_items(same_content, torch.arange(5))
        torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(5))  # L2-normalized
        torch.testing.assert_close(vectors[1], vectors[3])  # rare recipes: zero PAD row, content only
        self.assertFalse(torch.allclose(vectors[0], vectors[2]))  # their own ID rows differ
        torch.testing.assert_close(model.item_embedding.weight[0], torch.zeros(2))
        with self.assertRaises(ValueError):
            model.encode_items(same_content)  # recipe indices are required with ID embeddings
        raw = self.call("TwoTowerModel", 3, 1, embedding_dim=4, hidden_dim=8, normalize=False)
        self.assertFalse(torch.allclose(raw.encode_items(torch.randn(4, 3)).norm(dim=-1), torch.ones(4)))

    def test_recall_at_k_counts_heldout_positives_and_never_recommends_seen_recipes(self):
        history = pd.DataFrame({"user_id": ["u", "u", "v"], "item_index": [0, 1, 0], "rating": [5.0, 2.0, 4.0],
                                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"], utc=True)})
        heldout = pd.DataFrame({"user_id": ["u", "u", "w"], "item_index": [2, 4, 1], "rating": [5.0, 4.0, 5.0],
                                "date": pd.Timestamp("2024-01-03", tz="UTC")})
        recall = self.call("RecallAtK", history, heldout, {"u": 1, "v": 2}, 5, ks=(1, 2, 3))
        self.assertEqual(recall.users, ["u"])  # w has no training history; v has nothing held out
        # Most-rated order is 0 > 1 > 2 > 3 > 4; u already rated 0 and 1, so it gets 2, 3, 4.
        result = recall.popularity([10, 9, 8, 7, 6])
        self.assertEqual((result["recall@1"], result["recall@2"], result["recall@3"]), (0.5, 0.5, 1.0))
        self.assertEqual(result["heldout_positives"], 2)
        model = self.call("TwoTowerModel", 3, 2, embedding_dim=4, hidden_dim=8)
        scored = recall.evaluate(model, np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32), "cpu")
        self.assertEqual(scored["recall@3"], 1.0)  # only 3 unseen recipes remain, holding both targets

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
        model = self.call("TwoTowerModel", 5, 2, embedding_dim=4, hidden_dim=8)
        catalog_index = self.call("ItemIndex", self.call("encode_catalog", model, features).numpy())
        scanned = self.call("recommend_recipes", model, catalog, features, interactions, "u", top_k=3,
                            as_of="2024-01-05")
        indexed = self.call("recommend_recipes", model, catalog, features, interactions, "u", top_k=3,
                            as_of="2024-01-05", index=catalog_index)
        self.assertEqual(indexed.recipe_id.tolist(), scanned.recipe_id.tolist())
        np.testing.assert_allclose(indexed.similarity, scanned.similarity, atol=1e-5)

    def test_positive_rows_keep_only_four_and_five_star_ratings(self):
        frame = pd.DataFrame({"rating": [1.0, 2.0, 3.0, 4.0, 5.0], "item_index": range(5)})
        self.assertEqual(self.call("positive_rows", frame)["rating"].tolist(), [4.0, 5.0])

    def test_validation_in_batch_loss_mixes_users_instead_of_reading_sorted_runs(self):
        torch.manual_seed(5)
        targets = pd.DataFrame({"user_id": ["a"] * 16 + ["b"] * 16, "item_index": np.arange(32),
                                "rating": [5.0] * 32, "date": pd.to_datetime(["2024-01-02"] * 32, utc=True)})
        dataset = self.call("HistoryDataset", targets, targets.iloc[:0], {"a": 1, "b": 2}, training=False)
        model = self.call("TwoTowerModel", 4, 2, embedding_dim=4, hidden_dim=8)
        features = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float32)
        # In row order, each batch of 16 is one user, every negative is masked, and CE is exactly 0.
        metrics = self.call("evaluate_two_tower", model, dataset, features, batch_size=16, device="cpu")
        self.assertGreater(metrics["in_batch_cross_entropy"], 0.1)

    def test_serving_now_hides_recipes_rated_earlier_today(self):
        catalog, _, features = self.interaction_fixture()
        today = pd.DataFrame({"user_id": ["u"], "item_index": [2], "rating": [5.0],
                              "date": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(1, unit="s")]})
        model = self.call("TwoTowerModel", 5, 1, embedding_dim=4, hidden_dim=8)
        result = self.call("recommend_recipes", model, catalog, features, today, "u", top_k=10)
        self.assertEqual(sorted(result.recipe_id), ["1", "2", "4", "5"])

    def test_level_column_marks_negative_neutral_and_positive_ratings(self):
        catalog = pd.DataFrame({"recipe_id": [str(i) for i in range(1, 6)]})
        frame = pd.DataFrame({"recipe_id": range(1, 6), "user_id": 7, "rating": [1, 2, 3, 4, 5],
                              "date": "2024-01-01"})
        result = self.call("build_interactions", frame, catalog).sort_values("rating")
        self.assertEqual(result.level.tolist(), ["negative", "negative", "neutral", "positive", "positive"])
        np.testing.assert_array_equal(self.call("rating_to_level", [1, 2, 3, 4, 5]), [0, 0, 1, 2, 2])

    def test_contrast_pairs_positive_and_negative_recipes_of_the_same_user(self):
        ratings = pd.DataFrame({"user_id": ["u", "u", "u", "u", "v"], "item_index": [0, 1, 2, 3, 4],
                                "rating": [5.0, 4.0, 1.0, 3.0, 5.0],
                                "date": pd.to_datetime(["2024-01-02"] * 5, utc=True)})
        dataset = self.call("HistoryDataset", ratings, ratings, training=False, seed=0)
        samples = [dataset[index] for index in range(len(dataset))]
        # u's positives pair with its only negative (+1); the negative pairs with a positive (-1).
        self.assertEqual([(s["contrast_item_index"], float(s["contrast_sign"])) for s in samples[:2]],
                         [(2, 1.0), (2, 1.0)])
        self.assertIn(samples[2]["contrast_item_index"], {0, 1})
        self.assertEqual(float(samples[2]["contrast_sign"]), -1.0)
        # A neutral target, and a user with only positive ratings, have no pair.
        self.assertEqual([float(s["contrast_sign"]) for s in samples[3:]], [0.0, 0.0])
        np.testing.assert_array_equal(dataset.has_contrast, [True, True, True, False, False])
        self.assertEqual(dataset[2]["contrast_item_index"], samples[2]["contrast_item_index"])  # Fixed in validation.
        # An empty pool draws no pairs, so only the target is hidden (the reranker relies on this).
        unpaired = self.call("HistoryDataset", ratings, ratings, training=False, contrast_pool=ratings.iloc[:0])
        self.assertFalse(unpaired.has_contrast.any())
        self.assertEqual(float(unpaired[0]["contrast_sign"]), 0.0)

    def test_training_pairs_hide_both_recipes_from_the_history(self):
        ratings = pd.DataFrame({"user_id": ["u"] * 3, "item_index": [0, 1, 2], "rating": [1.0, 3.0, 5.0],
                                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"], utc=True)})
        dataset = self.call("HistoryDataset", ratings, ratings, contrast_pool=ratings)
        sample = dataset[2]  # Positive recipe 2; its only negative, recipe 0, was rated earlier.
        self.assertEqual((sample["contrast_item_index"], float(sample["contrast_sign"])), (0, 1.0))
        self.assertEqual(sample["history_items"][sample["history_mask"]].tolist(), [1])
        # One spare row per excluded recipe keeps max_history real ratings.
        index = self.call("HistoryIndex", ratings)
        np.testing.assert_array_equal(
            self.call("user_history", index, "u", "2024-01-06", max_history=1, exclude_item=[2, 1])[0], [0])

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
        train_dataset = self.call("HistoryDataset", self.call("positive_rows", train), train, lookup,
                                  contrast_pool=interactions)
        validation_dataset = self.call("HistoryDataset", self.call("positive_rows", validation), train, lookup,
                                       training=False, contrast_pool=interactions)
        item_rows, item_ids = self.call("build_item_id_rows", self.call("positive_rows", train)["item_index"], 5,
                                        min_ratings=1)
        model = self.call("TwoTowerModel", 5, len(lookup), embedding_dim=4, hidden_dim=8, num_items=5,
                          num_item_ids=item_ids, item_id_dim=2).set_item_id_rows(item_rows)
        recall = self.call("RecallAtK", train, self.call("positive_rows", validation), lookup, 5, ks=(1, 2))
        self.assertTrue(recall.users)
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        with contextlib.redirect_stdout(io.StringIO()):
            model, history, metrics = self.call("train_two_tower", model, train_dataset,
                validation_dataset, features, epochs=3, batch_size=2, device="cpu", max_rows_per_user=1,
                recall_evaluator=recall)
        self.assertTrue(np.isfinite(history[["train_in_batch_cross_entropy", "validation_in_batch_cross_entropy",
                                             "validation_recall@1", "validation_recall@2"]]).all().all())
        self.assertTrue((history["train_rows"] <= train_dataset.targets["user_id"].nunique()).all())
        self.assertEqual(set(metrics), {"in_batch_cross_entropy", "in_batch_rows", "hard_negative_rows",
                                        "hard_negative_accuracy"})
        for prefix in ("item_tower", "user_head"):
            self.assertTrue(any(not torch.equal(initial[name], parameter)
                                for name, parameter in model.named_parameters() if name.startswith(prefix)))
        result = self.call("recommend_recipes", model, catalog, features, interactions, "u",
                           user_lookup=lookup, top_k=10, as_of="2024-01-05")
        self.assertEqual(set(result.recipe_id), {"4", "5"})
        self.assertTrue(result.similarity.is_monotonic_decreasing)
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
        item_rows, item_ids = self.call("build_item_id_rows", observations["item_index"], len(catalog), min_ratings=1)
        model = self.call("TwoTowerModel", dim, len(lookup), embedding_dim=4, hidden_dim=8, num_items=len(catalog),
                          num_item_ids=item_ids, item_id_dim=2).set_item_id_rows(item_rows)
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
                IN_BATCH_TEMPERATURE=0.05, HARD_NEGATIVE_WEIGHT=1.0, MAX_ROWS_PER_USER=50,
                MIN_ITEM_RATINGS_FOR_ID=5, NORMALIZE_EMBEDDINGS=True, NEUTRAL_HISTORY_WEIGHT=0.5,
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
            np.testing.assert_allclose(result.similarity, expected.similarity, atol=1e-6)
            self.assertIn("nutrition_scaler", joblib.load(artifact_dir / "content_preprocessing.joblib")["numeric"])
            config = json.loads((artifact_dir / "config.json").read_text())
            self.assertEqual((config["loss"], config["output"], config["validation_split"], config["similarity"]),
                             ("in_batch_softmax_hard_negatives_masked_logq", "dot_product_only", "stratified_random",
                              "dot_product"))
            self.assertEqual((config["format_version"], config["normalized_embeddings"],
                              config["model_config"]["neutral_weight"], config["model_config"]["num_item_ids"]),
                             ("recipe_two_tower_v8", True, 0.5, item_ids))
            self.assertNotIn("class_weight", config)
            # The reranker relies on these two staying in the saved artifacts.
            np.testing.assert_allclose(item_vectors, self.call("encode_catalog", restored, restored_features), atol=1e-6)
            self.assertTrue({"max_history", "validation_fraction", "random_state"} <= set(config))


if __name__ == "__main__":
    unittest.main()
