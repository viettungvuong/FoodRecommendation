"""CPU correctness checks: python -m unittest discover -s tests -v."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

# Exercise the notebook's actual implementation, without importing local modules.
_NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "model_training.ipynb"
_notebook = json.loads(_NOTEBOOK_PATH.read_text())
_model_cells = [cell for cell in _notebook["cells"] if cell.get("id") == "recipe-model-definitions"]
if len(_model_cells) != 1:
    raise ValueError("Expected one combined model definitions cell in the notebook.")
exec(compile("".join(_model_cells[0]["source"]), str(_NOTEBOOK_PATH) + "#models", "exec"), globals())


def make_table(count=12):
    return prepare_term_features({"recipes": pd.DataFrame({
        "product": [[f"ingredient {i}", "olive oil"] for i in range(count)],
        "adj": [["fresh", f"adjective {i}"] for i in range(count)],
        "verb": [["stirred", f"verb {i}"] for i in range(count)],
    })})


def vectors(table, dim=8):
    array = np.random.default_rng(42).normal(size=(len(table.terms), dim)).astype(np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def tiny_encoder():
    torch.manual_seed(42)
    return RecipeContextEncoder(embedding_dim=8, num_heads=2, feedforward_dim=16, dropout=0).eval()


class RecipeEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_nested_terms_preserve_phrases_and_repeats(self):
        self.assertEqual(feature_to_terms('[[" Olive  Oil "], ["salt", "salt"], []]'),
                         ("olive oil", "salt", "salt"))
        self.assertEqual(feature_to_terms("(['Fresh basil'], ['salt'])"), ("fresh basil", "salt"))
        self.assertEqual(feature_to_terms(np.array(["olive oil", "salt"])), ("olive oil", "salt"))
        for value in (None, np.nan, pd.NA, "nan", "null", "[[], []]", []):
            self.assertEqual(feature_to_terms(value), ())
        with self.assertRaises(ValueError):
            feature_to_terms('["broken"')
        with self.assertRaises(TypeError):
            feature_to_terms(123)

    def test_complete_combinations_deduplicate_across_sources(self):
        recipe = pd.DataFrame({"product": ['[["salt"], ["olive oil", "salt"]]',
                                            '["olive oil", "salt", "salt"]', '["salt"]'],
                               "adj": ['["fresh"]', '["fresh"]', '[]'], "verb": ['[]'] * 3,
                               "rating": [1, 5, 0]}, index=[10, 10, 99])
        food = pd.DataFrame({"products": [["salt", "salt", "olive oil"]],
                             "adj": [["fresh"]], "verb": [[]]})
        original = recipe.copy(deep=True)
        table = prepare_term_features({"recipe": recipe, "food": food})
        np.testing.assert_array_equal(table.row_embedding_ids, [0, 0, 1, 0])
        np.testing.assert_array_equal(table.row_counts, [3, 1])
        self.assertEqual(len(table.term_ids["products"][0]), 3)
        self.assertEqual(table.terms.count("salt"), 1)
        pd.testing.assert_frame_equal(recipe, original)
        recipe["rating"] = [5, 0, 3]
        changed = prepare_term_features({"recipe": recipe, "food": food})
        pd.testing.assert_frame_equal(changed.unique_features, table.unique_features)
        with self.assertRaisesRegex(ValueError, "Missing embedding columns"):
            prepare_term_features({"recipe": recipe.drop(columns="verb")})
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_term_features({"recipe": recipe.iloc[:0]})

    def test_bert_cache_is_frozen_and_excludes_padding_and_special_tokens(self):
        class FakeBert(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=4)
                self.embedding = nn.Embedding(10, 4)
                self.calls = 0

            def forward(self, input_ids, attention_mask):
                self.calls += 1
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        def tokenizer(texts, **kwargs):
            ids = torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])[:len(texts)]
            return {"input_ids": ids, "attention_mask": ids != 0,
                    "special_tokens_mask": (ids == 1) | (ids == 2) | (ids == 0)}

        bert = FakeBert()
        old = bert.embedding.weight.detach().clone()
        result = embed_terms(("olive oil", "salt"), tokenizer, bert, "cpu")
        expected = torch.stack([old[3:5].mean(0), old[5]])
        expected = nn.functional.normalize(expected, dim=-1).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-7)
        self.assertEqual(bert.calls, 1)
        self.assertFalse(bert.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in bert.parameters()))
        torch.testing.assert_close(bert.embedding.weight, old)

    def test_recipe_context_changes_ingredients_but_not_other_recipes(self):
        table = make_table(2)
        inputs = TermBatchCollator(vectors(table), table.term_ids).collate([0, 1])["inputs"]
        model = tiny_encoder()
        baseline = model(**inputs)
        for field in FEATURE_COLUMNS:
            changed = {key: value.clone() for key, value in inputs.items()}
            changed[field][0, 1] = torch.arange(8, dtype=torch.float32) * 3
            result = model(**changed)
            self.assertFalse(torch.allclose(result["term_embeddings"]["products"][0, 0],
                                            baseline["term_embeddings"]["products"][0, 0]))
            torch.testing.assert_close(result["recipe_embedding"][1], baseline["recipe_embedding"][1])
            if field == "products":
                for other in ("adj", "verb"):
                    torch.testing.assert_close(result["term_embeddings"][other],
                                               baseline["term_embeddings"][other])
            else:
                other = "verb" if field == "adj" else "adj"
                torch.testing.assert_close(result["term_embeddings"][other], baseline["term_embeddings"][other])
                self.assertFalse(torch.allclose(result["term_embeddings"][field][0, 0],
                                                baseline["term_embeddings"][field][0, 0]))

    def test_pooling_is_permutation_invariant(self):
        table = make_table(2)
        inputs = TermBatchCollator(vectors(table), table.term_ids).collate([0, 1])["inputs"]
        model = tiny_encoder()
        expected = model(**inputs)
        shuffled = {key: value.flip(1) for key, value in inputs.items()}
        actual = model(**shuffled)
        torch.testing.assert_close(actual["recipe_embedding"], expected["recipe_embedding"], atol=1e-6, rtol=1e-6)
        for field in FEATURE_COLUMNS:
            torch.testing.assert_close(actual["term_embeddings"][field].flip(1),
                                       expected["term_embeddings"][field], atol=1e-6, rtol=1e-6)

    def test_padding_empty_groups_singletons_and_weighted_pooling(self):
        table = prepare_term_features({"recipes": pd.DataFrame({
            "product": [["salt"], ["salt", "pepper"], [], [], []],
            "adj": [[], ["fresh"], [], ["green"], []],
            "verb": [[], ["stirred"], [], [], ["boiled"]],
        })})
        collator = TermBatchCollator(vectors(table), table.term_ids)
        inputs = collator.collate(range(5))["inputs"]
        model = tiny_encoder()
        output = model(**inputs)
        self.assertTrue(torch.isfinite(output["recipe_embedding"]).all())
        for field in FEATURE_COLUMNS:
            self.assertTrue((output["term_embeddings"][field][~inputs[field + "_mask"]] == 0).all())
        torch.testing.assert_close(output["recipe_embedding"][2], torch.zeros(8))
        torch.testing.assert_close(output["recipe_embedding"][0].norm(), torch.tensor(0.6))
        torch.testing.assert_close(output["recipe_embedding"][3].norm(), torch.tensor(0.2))
        torch.testing.assert_close(output["recipe_embedding"][4].norm(), torch.tensor(0.2))
        expected = sum(weight * output["field_embeddings"][field]
                       for weight, field in zip((0.6, 0.2, 0.2), FEATURE_COLUMNS))
        torch.testing.assert_close(output["recipe_embedding"], expected)
        # A singleton with zero-width adj/verb fields matches the padded batch.
        alone = model(**collator.collate([0])["inputs"])
        torch.testing.assert_close(alone["recipe_embedding"][0], output["recipe_embedding"][0])
        empty = model(**collator.collate([2])["inputs"])
        torch.testing.assert_close(empty["recipe_embedding"], torch.zeros(1, 8))
        # The absent cross-attention context must bypass the entire cross block.
        x = inputs["products"][:1] + model.field_types[0]
        mask = inputs["products_mask"][:1]
        self_only = model.self_attention["products"](x.masked_fill(~mask.unsqueeze(-1), 0), x, mask, mask)
        torch.testing.assert_close(output["term_embeddings"]["products"][:1], self_only)
        changed = {key: value.clone() for key, value in inputs.items()}
        for field in FEATURE_COLUMNS:
            changed[field][~changed[field + "_mask"]] = float("nan")
        torch.testing.assert_close(model(**changed)["recipe_embedding"], output["recipe_embedding"])

    def test_mask_eligibility_and_fixed_validation(self):
        table = prepare_term_features({"recipes": pd.DataFrame({
            "product": [["salt"], ["salt"], [], ["a", "b"]],
            "adj": [[], ["fresh"], ["fresh", "green"], ["red"]],
            "verb": [[], ["stirred"], ["boiled"], ["cut", "mixed"]],
        })})
        self.assertEqual(eligible_mask_targets(table.term_ids, 0), [])
        self.assertEqual(eligible_mask_targets(table.term_ids, 1), [(0, 0)])
        self.assertEqual(eligible_mask_targets(table.term_ids, 2), [(1, 0), (1, 1)])
        self.assertEqual(eligible_mask_targets(table.term_ids, 3), [(0, 0), (0, 1), (2, 0), (2, 1)])
        training, validation = split_term_features(table.term_ids)
        self.assertFalse(set(training) & set(validation))
        self.assertNotIn(0, training)
        self.assertNotIn(0, validation)
        fixed = MaskedTermDataset(table.term_ids, validation, seed=43, fixed=True)
        self.assertEqual(fixed[0], fixed[0])
        same = MaskedTermDataset(table.term_ids, validation, seed=43, fixed=True)
        self.assertEqual(same[0], fixed[0])

    def test_hidden_original_vectors_cannot_leak_into_attention(self):
        table = make_table(3)
        batch = TermBatchCollator(vectors(table), table.term_ids)([(0, 0, 0), (1, 1, 0), (2, 2, 0)])
        model = tiny_encoder()
        original = model(**batch["inputs"])
        for owner, field in enumerate(FEATURE_COLUMNS):
            batch["inputs"][field][owner, 0] = 999
        actual = model(**batch["inputs"])
        torch.testing.assert_close(original["recipe_embedding"], actual["recipe_embedding"])
        loss = masked_term_loss(model, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for field in FEATURE_COLUMNS:
            grad = model.self_attention[field].attention.in_proj_weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        grad = model.cross_attention.attention.in_proj_weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.mask_tokens.grad).all())

    def test_pretraining_learns_restores_best_validation_and_freezes(self):
        table = make_table(16)
        embedding = np.zeros((len(table.terms), 8), dtype=np.float32)
        embedding[:, 0] = 1  # An intentionally learnable fixed target for every field.
        original = embedding.copy()
        model = tiny_encoder()
        with contextlib.redirect_stdout(io.StringIO()):
            history = train_recipe_encoder(model, embedding, table.term_ids, "cpu",
                                           epochs=12, batch_size=8, learning_rate=0.01)
        self.assertLess(history.iloc[-1].train_cosine_loss, history.iloc[0].train_cosine_loss * 0.5)
        _, validation = split_term_features(table.term_ids)
        fixed = MaskedTermDataset(table.term_ids, validation, seed=43, fixed=True)
        batch = TermBatchCollator(embedding, table.term_ids)([fixed[i] for i in range(len(fixed))])
        with torch.inference_mode():
            restored_loss = masked_term_loss(model, batch).item()
        self.assertAlmostEqual(restored_loss, history.validation_cosine_loss.min(), places=6)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        np.testing.assert_array_equal(embedding, original)

    def test_no_validation_and_no_eligible_targets(self):
        table = make_table(1)
        model = tiny_encoder()
        with contextlib.redirect_stdout(io.StringIO()):
            history = train_recipe_encoder(model, vectors(table), table.term_ids, "cpu", epochs=1)
        self.assertEqual(history.iloc[0].validation_features, 0)
        self.assertTrue(np.isnan(history.iloc[0].validation_cosine_loss))
        empty = {field: [np.array([], dtype=np.int64)] for field in FEATURE_COLUMNS}
        with self.assertRaisesRegex(ValueError, "No feature combination"):
            train_recipe_encoder(model, np.empty((0, 8), dtype=np.float32), empty, "cpu")

    def test_checkpoint_roundtrip_identity_and_compatibility(self):
        model = tiny_encoder()
        table = make_table(3)
        embedding = vectors(table)
        fields, expected = encode_feature_table(model, embedding, table.term_ids, "cpu", batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe_context_encoder.h5"
            identity = save_recipe_encoder(model, path, training_config={"seed": 42})
            loaded, metadata = load_recipe_encoder(path)
            self.assertEqual(identity, metadata["encoder_identity"])
            self.assertEqual(metadata["feature_format_version"], FEATURE_FORMAT_VERSION)
            self.assertEqual(metadata["training_config"], {"seed": 42})
            actual_fields, actual = encode_feature_table(loaded, embedding, table.term_ids, "cpu", batch_size=1)
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            for field in FEATURE_COLUMNS:
                np.testing.assert_allclose(fields[field], actual_fields[field], atol=1e-6)
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            self.assertNotEqual(identity, encoder_identity(model, "different-bert", 512))
            with h5py.File(path, "r+") as handle:
                handle["state_dict"]["field_types"][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "identity"):
                load_recipe_encoder(path)
            with h5py.File(path, "r+") as handle:
                handle.attrs["feature_format_version"] = "old"
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                load_recipe_encoder(path)

    def test_default_encoder_preserves_768_dimensions(self):
        model = RecipeContextEncoder().eval()
        table = make_table(1)
        inputs = TermBatchCollator(vectors(table, dim=768), table.term_ids).collate([0])["inputs"]
        with torch.inference_mode():
            output = model(**inputs)
        self.assertEqual(output["recipe_embedding"].shape, (1, 768))
        self.assertTrue(torch.isfinite(output["recipe_embedding"]).all())

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is unavailable")
    def test_mps_training_and_inference_with_empty_fields(self):
        table = prepare_term_features({"recipes": pd.DataFrame({
            "product": [["salt", "pepper"], ["butter"], [], ["tomato", "olive oil"]],
            "adj": [["fresh", "green"], [], [], ["ripe"]],
            "verb": [["cut", "mixed"], [], [], []],
        })})
        embedding = vectors(table, dim=768)
        model = RecipeContextEncoder()
        with contextlib.redirect_stdout(io.StringIO()):
            history = train_recipe_encoder(model, embedding, table.term_ids, "mps", epochs=1, batch_size=2)
        self.assertTrue(np.isfinite(history.validation_cosine_loss).all())
        _, accelerated = encode_feature_table(model, embedding, table.term_ids, "mps", batch_size=4)
        _, cpu = encode_feature_table(model, embedding, table.term_ids, "cpu", batch_size=4)
        np.testing.assert_allclose(accelerated, cpu, atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(accelerated[2], np.zeros(768))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_balanced_rating_class_weights_match_scikit_learn(self):
        # Ratings 5 x6, 4 x3 and 1 x1, scaled to 0..1 as RatingGroup stores them.
        groups = [SimpleNamespace(ratings=(np.array([5.0] * 6 + [4.0] * 3) - 1) / 4),
                  SimpleNamespace(ratings=np.array([0.0]))]
        counts, weights = build_rating_class_weights(groups)
        np.testing.assert_array_equal(counts, [1, 0, 0, 3, 6])
        # n_samples / (n_present_classes * count); absent ratings 2 and 3 get the rarest weight.
        np.testing.assert_allclose(weights, [10 / 3, 10 / 3, 10 / 3, 10 / 9, 10 / 18], rtol=1e-6)
        np.testing.assert_array_equal(build_rating_class_weights(groups, class_weight=None)[1], np.ones(5))
        with self.assertRaises(ValueError):
            build_rating_class_weights(groups, class_weight="sqrt")

    def test_notebook_pipeline_through_clustering_training_prediction_and_save(self):
        from scipy import sparse
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import normalize
        from umap import UMAP

        recipe = pd.DataFrame({
            "recipe_id": range(6), "name": [f"Recipe {i}" for i in range(6)],
            "user_id": ["user"] * 6, "rating": [1, 2, 3, 4, 5, 0],
            "product": [[f"ingredient {i}", "salt"] for i in range(6)],
            "adj": [["fresh"]] * 6, "verb": [["mixed"]] * 6,
        })
        food = pd.DataFrame({"fdc_id": [1, 2], "description": ["Food one", "Food two"],
                             "product": [["olive oil"], ["butter"]],
                             "adj": [["extra virgin"], []], "verb": [[], []]})
        datasets = {"recipes": recipe, "foods": food}
        table = prepare_term_features(datasets)
        embedding = vectors(table)
        model = tiny_encoder()
        with contextlib.redirect_stdout(io.StringIO()):
            train_recipe_encoder(model, embedding, table.term_ids, "cpu", epochs=2, batch_size=4)
        state_before = {key: value.clone() for key, value in model.state_dict().items()}
        notebook = json.loads((Path(__file__).resolve().parents[1] / "model_training.ipynb").read_text())
        sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        scope = dict(
            np=np, pd=pd, torch=torch, sparse=sparse, normalize=normalize, UMAP=UMAP, HDBSCAN=HDBSCAN,
            encode_feature_table=encode_feature_table, recipe_encoder=model, term_embeddings=embedding,
            term_feature_table=table, datasets=datasets, row_embedding_ids=table.row_embedding_ids,
            row_counts=table.row_counts, device="cpu", FEATURE_COLUMNS=FEATURE_COLUMNS,
            ENCODER_BATCH_SIZE=4, UMAP_COMPONENTS=2, UMAP_NEIGHBORS=3, UMAP_MIN_DIST=0,
            RANDOM_STATE=42, MIN_CLUSTER_SIZE=2, MIN_SAMPLES=1, CLUSTER_SELECTION_EPSILON=0.0,
            CLUSTER_SELECTION_METHOD="eom", HDBSCAN_N_JOBS=1,
            display=lambda *args: None,
        )
        for prefix in ("contextual_field_embeddings,", "valid_embedding_ids =",
                       "def cluster_contextualized_embeddings(", "def average_cluster_embeddings("):
            source = next(source for source in sources if source.startswith(prefix))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "notebook-integration", "exec"), scope)
        features = scope["contextualized_embeddings"]
        self.assertIs(features, scope["weighted_embeddings"])
        self.assertTrue(np.isfinite(scope["reduced_embeddings"]).all())
        # Each row inherits its feature combination's HDBSCAN label, and cluster means count every row.
        row_labels = np.concatenate([df["cluster"].to_numpy() for df in datasets.values()])
        np.testing.assert_array_equal(row_labels, scope["unique_cluster_labels"][table.row_embedding_ids])
        self.assertGreater((row_labels >= 0).sum(), 0)
        expected_sizes = pd.Series(row_labels[row_labels >= 0]).value_counts().sort_index()
        np.testing.assert_array_equal(scope["cluster_embeddings_df"]["row_count"].to_numpy(),
                                      expected_sizes.to_numpy())
        catalog, ratings = build_catalog_and_ratings(recipe, food)
        self.assertEqual(len(ratings), 5)  # Unrated recipe/foods never become zero targets.
        training, validation = split_rating_groups(ratings)
        held_out = set(validation[0][1].items)
        self.assertFalse(set(training[0].items) & held_out)
        masked = MaskedRatingDataset(training)
        for _ in range(10):
            _, context, _, targets, _ = masked[0]
            self.assertFalse(set(context) & set(targets))
            self.assertFalse((set(context) | set(targets)) & held_out)
        collator = RatingCollator(features, catalog, scope["cluster_embeddings_df"])
        autoencoder = ClusterDenoisingAutoencoder(embedding_dim=8, feature_dim=64, latent_dim=4)
        with contextlib.redirect_stdout(io.StringIO()):
            history = train_autoencoder(autoencoder, training, validation, collator, "cpu", epochs=2)
        self.assertTrue(np.isfinite(history.validation_rmse).all())
        predictions = predict_unrated_items(autoencoder, "user", catalog, ratings, features,
                                             scope["cluster_embeddings_df"], "cpu")
        self.assertEqual(set(predictions.item_key), {"recipe:5", "food:1", "food:2"})
        self.assertTrue(predictions.predicted_rating.between(1, 5).all())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state_before[key])
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model_artifacts"
            encoder_path = output / "recipe_context_encoder.h5"
            identity = save_recipe_encoder(model, encoder_path)
            scope.update(recipe_encoder=model, encoder_identity=encoder_identity,
                         MODEL_NAME="bert-base-uncased", MAX_LENGTH=512, ENCODER_IDENTITY=identity,
                         ENCODER_WEIGHTS_PATH=encoder_path, FEATURE_FORMAT_VERSION=FEATURE_FORMAT_VERSION,
                         recommendation_model=autoencoder, LATENT_DIM=4, RATING_MIN=1, RATING_MAX=5,
                         NUM_RATING_CLASSES=5, FOCAL_GAMMA=2.0, CLASS_WEIGHT="balanced",
                         rating_class_weights=build_rating_class_weights(training)[1],
                         DOMINANT_RATING=5.0, HEAVY_USER_MIN_RATINGS=50,
                         MAX_DOMINANT_TO_OTHER_RATIO=2.0, MIN_DOMINANT_KEEP=10,
                         Path=lambda value: Path(directory) / value)
            save_cell = next(source for source in sources if source.startswith("# Save recommender weights"))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(save_cell, "notebook-save", "exec"), scope)
            with h5py.File(output / "recipe_recommender_autoencoder.h5") as handle:
                self.assertEqual(handle.attrs["encoder_identity"], identity)
                self.assertEqual(handle.attrs["clustering_algorithm"], "HDBSCAN")
                self.assertEqual(handle.attrs["num_rating_classes"], autoencoder.num_rating_classes)
                self.assertEqual(set(handle["state_dict"]), set(autoencoder.state_dict()))
                self.assertEqual(handle.attrs["feature_format_version"], FEATURE_FORMAT_VERSION)
                self.assertEqual(handle.attrs["encoder_artifact"], encoder_path.name)
                self.assertEqual(handle.attrs["class_weight"], "balanced")


if __name__ == "__main__":
    unittest.main()
