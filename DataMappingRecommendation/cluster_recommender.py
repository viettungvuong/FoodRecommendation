"""Cluster-conditioned denoising autoencoder for sparse explicit ratings.

Inputs are item embeddings and observed ratings, not a dense user-item matrix.
The model encodes a set of ratings from one user/cluster and reconstructs ratings
for masked items in that cluster. Content features also permit unrated food items
to be scored without learned item-ID parameters.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class RatingGroup:
    user_id: str
    cluster: int
    items: np.ndarray
    ratings: np.ndarray


def build_catalog_and_ratings(recipe_df, food_df, rating_min=1.0, rating_max=5.0,
                              zero_is_missing=True):
    """Namespace item IDs, retain each item once, and keep observed ratings only."""
    if rating_max <= rating_min:
        raise ValueError("rating_max must be greater than rating_min.")
    catalogs = []
    for frame, source, id_column, title_column in (
        (recipe_df, "recipe", "recipe_id", "name"),
        (food_df, "food", "fdc_id", "description"),
    ):
        if frame[id_column].isna().any():
            raise ValueError(f"Missing item IDs in {source} data.")
        items = pd.DataFrame({
            "item_key": source + ":" + frame[id_column].astype(str),
            "source": source,
            "title": frame[title_column].fillna("").astype(str),
            "embedding_row_id": frame["embedding_row_id"],
            "cluster": frame["cluster"],
        })
        inconsistent = items.groupby("item_key")[["embedding_row_id", "cluster"]].nunique().gt(1).any(axis=1)
        if inconsistent.any():
            raise ValueError(f"An item has inconsistent features/clusters: {inconsistent[inconsistent].index[0]}")
        catalogs.append(items.drop_duplicates("item_key"))
    catalog = pd.concat(catalogs, ignore_index=True)
    catalog.index.name = "item_index"

    raw = recipe_df[["user_id", "recipe_id", "rating"]].copy()
    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce")
    raw = raw.loc[raw["user_id"].notna() & raw["recipe_id"].notna() & raw["rating"].notna()]
    if zero_is_missing:
        raw = raw.loc[raw["rating"] != 0]
    if not raw["rating"].between(rating_min, rating_max).all():
        raise ValueError("Observed ratings lie outside the configured rating scale.")
    raw["user_id"] = raw["user_id"].astype(str)
    raw["item_key"] = "recipe:" + raw["recipe_id"].astype(str)
    ratings = raw.groupby(["user_id", "item_key"], as_index=False, sort=False)["rating"].mean()
    ratings = ratings.merge(catalog.reset_index()[["item_key", "item_index", "cluster"]],
                            on="item_key", how="left", validate="many_to_one")
    if ratings["item_index"].isna().any():
        raise ValueError("Some rated items are missing from the catalog.")
    ratings["scaled_rating"] = (ratings["rating"] - rating_min) / (rating_max - rating_min)
    return catalog, ratings


def split_rating_groups(ratings, validation_fraction=0.2, seed=42):
    """Hold out fixed targets first. At least two training observations remain.

    Training uses groups with >=2 observations. Singleton groups remain available
    for recommendation context but cannot support masked reconstruction with a
    nonempty context. DBSCAN noise is never treated as a shared cluster.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    training, validation = [], []
    eligible = ratings.loc[ratings["cluster"] >= 0]
    for (user, cluster), group in eligible.groupby(["user_id", "cluster"], sort=False):
        items = group["item_index"].to_numpy(dtype=np.int64)
        values = group["scaled_rating"].to_numpy(dtype=np.float32)
        if len(items) < 2:
            continue
        order = rng.permutation(len(items))
        n_val = min(max(1, round(len(items) * validation_fraction)), len(items) - 2) if len(items) >= 3 else 0
        val_ids, train_ids = order[:n_val], order[n_val:]
        train_group = RatingGroup(str(user), int(cluster), items[train_ids], values[train_ids])
        training.append(train_group)
        if n_val:
            targets = RatingGroup(str(user), int(cluster), items[val_ids], values[val_ids])
            validation.append((train_group, targets))
    if not training:
        raise ValueError("No user has two observed ratings in the same non-noise cluster. Inspect clustering and rating coverage.")
    return training, validation


class MaskedRatingDataset(Dataset):
    """Resample masked observed targets every access; targets never enter context."""

    def __init__(self, groups, mask_fraction=0.3, max_context=64, max_targets=32, seed=42):
        if not 0 < mask_fraction < 1 or max_context < 1 or max_targets < 1:
            raise ValueError("Use 0 < mask_fraction < 1 and positive context/target limits.")
        self.groups = groups
        self.mask_fraction = mask_fraction
        self.max_context, self.max_targets = max_context, max_targets
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        order = self.rng.permutation(len(group.items))
        n_mask = min(len(order) - 1, max(1, round(len(order) * self.mask_fraction)))
        targets = order[:n_mask][:self.max_targets]
        context = order[n_mask:][:self.max_context]
        return (group.cluster, group.items[context], group.ratings[context],
                group.items[targets], group.ratings[targets])


class ValidationRatingDataset(Dataset):
    """Fixed holdouts; context contains only the training part of each group."""

    def __init__(self, pairs, max_context=64, seed=42):
        rng = np.random.default_rng(seed)
        self.samples = []
        for context, target in pairs:
            chosen = rng.permutation(len(context.items))[:max_context]
            self.samples.append((context.cluster, context.items[chosen], context.ratings[chosen],
                                 target.items, target.ratings))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class RatingCollator:
    """Gather only the embeddings needed by this minibatch from the shared table."""

    def __init__(self, feature_matrix, catalog, cluster_embeddings):
        self.features = feature_matrix
        self.embedding_ids = catalog["embedding_row_id"].to_numpy(dtype=np.int64)
        self.centroids = cluster_embeddings["average_embedding"].to_dict()

    def __call__(self, samples):
        width = max(len(sample[1]) for sample in samples)
        context_ids = np.zeros((len(samples), width), dtype=np.int64)
        context_ratings = np.zeros((len(samples), width), dtype=np.float32)
        context_mask = np.zeros((len(samples), width), dtype=bool)
        target_ids, target_ratings, owners, cluster_features = [], [], [], []
        for owner, (cluster, items, ratings, targets, values) in enumerate(samples):
            length = len(items)
            context_ids[owner, :length] = self.embedding_ids[items]
            context_ratings[owner, :length] = ratings
            context_mask[owner, :length] = True
            target_ids.extend(self.embedding_ids[targets])
            target_ratings.extend(values)
            owners.extend([owner] * len(targets))
            cluster_features.append(self.centroids[cluster])
        return {
            "context_features": torch.from_numpy(self.features[context_ids]),
            "context_ratings": torch.from_numpy(context_ratings),
            "context_mask": torch.from_numpy(context_mask),
            "cluster_features": torch.from_numpy(np.stack(cluster_features).astype(np.float32)),
            "target_features": torch.from_numpy(self.features[np.asarray(target_ids, dtype=np.int64)]),
            "target_owner": torch.tensor(owners, dtype=torch.long),
            "target_ratings": torch.tensor(target_ratings, dtype=torch.float32),
        }


class ClusterDenoisingAutoencoder(nn.Module):
    """Set encoder -> user/cluster bottleneck -> item-conditioned rating decoder."""

    def __init__(self, embedding_dim=768, feature_dim=64, latent_dim=32):
        super().__init__()
        self.item_encoder = nn.Sequential(nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Linear(128, feature_dim))
        self.rating_encoder = nn.Sequential(nn.Linear(feature_dim + 1, 128), nn.ReLU(), nn.Linear(128, feature_dim))
        self.cluster_encoder = nn.Sequential(nn.Linear(embedding_dim, feature_dim), nn.ReLU())
        self.encoder = nn.Sequential(nn.Linear(2 * feature_dim + 1, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim + 2 * feature_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 1), nn.Sigmoid())

    def encode(self, context_features, context_ratings, context_mask, cluster_features):
        item_codes = self.item_encoder(context_features)
        observations = self.rating_encoder(torch.cat([item_codes, context_ratings.unsqueeze(-1)], dim=-1))
        mask = context_mask.unsqueeze(-1).to(observations.dtype)
        counts = mask.sum(dim=1)
        pooled = (observations * mask).sum(dim=1) / counts.clamp(min=1)
        cluster_codes = self.cluster_encoder(cluster_features)
        latent = self.encoder(torch.cat([pooled, cluster_codes, counts.log1p()], dim=-1))
        return latent, cluster_codes

    def decode(self, latent, cluster_codes, target_features, target_owner):
        target_codes = self.item_encoder(target_features)
        inputs = torch.cat([latent[target_owner], target_codes, cluster_codes[target_owner]], dim=-1)
        return self.decoder(inputs).squeeze(-1)

    def forward(self, batch):
        latent, cluster_codes = self.encode(batch["context_features"], batch["context_ratings"],
                                            batch["context_mask"], batch["cluster_features"])
        return self.decode(latent, cluster_codes, batch["target_features"], batch["target_owner"])


def evaluate_autoencoder(model, loader, device, rating_span):
    model.eval()
    squared_error = absolute_error = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            errors = (model(batch) - batch["target_ratings"]) * rating_span
            squared_error += errors.square().sum().item()
            absolute_error += errors.abs().sum().item()
            count += errors.numel()
    return {"rmse": (squared_error / count) ** 0.5 if count else np.nan,
            "mae": absolute_error / count if count else np.nan, "count": count}


def train_autoencoder(model, training_groups, validation_pairs, collator, device,
                      epochs=10, batch_size=64, learning_rate=1e-3, mask_fraction=0.3,
                      max_context=64, max_targets=32, rating_span=4.0, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    model.to(device)
    train_data = MaskedRatingDataset(training_groups, mask_fraction, max_context, max_targets, seed)
    val_data = ValidationRatingDataset(validation_pairs, max_context, seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator,
                        num_workers=0, generator=generator)
    validation_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collator,
                                   num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_state, best_rmse = [], None, np.inf
    for epoch in range(1, epochs + 1):
        model.train()
        squared_error = 0.0
        count = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            errors = model(batch) - batch["target_ratings"]
            loss = errors.square().mean()  # Only masked, truly observed targets.
            loss.backward()
            optimizer.step()
            squared_error += errors.detach().square().sum().item() * rating_span ** 2
            count += errors.numel()
        metrics = evaluate_autoencoder(model, validation_loader, device, rating_span)
        history.append({"epoch": epoch, "train_masked_rmse": (squared_error / count) ** 0.5,
                        "validation_rmse": metrics["rmse"], "validation_mae": metrics["mae"],
                        "validation_ratings": metrics["count"]})
        print(history[-1], flush=True)
        if metrics["count"] and metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return pd.DataFrame(history)


def validation_baselines(training_groups, validation_pairs, rating_span=4.0):
    """Global and user/cluster means fitted exclusively on training observations."""
    total = sum(float(group.ratings.sum()) for group in training_groups)
    count = sum(len(group.ratings) for group in training_groups)
    global_mean = total / count
    errors = {"global_mean": [], "user_cluster_mean": []}
    for context, targets in validation_pairs:
        errors["global_mean"].extend((global_mean - targets.ratings) * rating_span)
        errors["user_cluster_mean"].extend((context.ratings.mean() - targets.ratings) * rating_span)
    rows = []
    for name, values in errors.items():
        values = np.asarray(values)
        rows.append({"baseline": name, "rmse": np.sqrt(np.mean(values ** 2)) if len(values) else np.nan,
                     "mae": np.mean(np.abs(values)) if len(values) else np.nan, "count": len(values)})
    return pd.DataFrame(rows).set_index("baseline")


def predict_unrated_items(model, user_id, catalog, observed_ratings, feature_matrix,
                          cluster_embeddings, device, rating_min=1.0, rating_max=5.0,
                          candidate_sources=("recipe", "food"), batch_size=1024,
                          max_context=64, top_k=None, seed=42):
    """Score unrated items only in clusters where this user has an observed rating.

    Use training-only observations for evaluation. At serving time it is valid to
    use all known observations, including the ratings held out during validation.
    A user without a non-noise rated cluster receives an empty result.
    """
    if batch_size < 1 or max_context < 1 or (top_k is not None and top_k < 1):
        raise ValueError("Batch size, context limit, and optional top_k must be positive.")
    user_rows = observed_ratings.loc[observed_ratings["user_id"].astype(str) == str(user_id)]
    rated = set(user_rows["item_index"])
    eligible = user_rows.loc[user_rows["cluster"] >= 0]
    output_columns = ["user_id", "item_key", "source", "title", "cluster", "predicted_rating", "context_ratings"]
    results = []
    model.eval()
    embedding_ids = catalog["embedding_row_id"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    with torch.inference_mode():
        for cluster, history in eligible.groupby("cluster", sort=True):
            candidates = catalog.loc[(catalog["cluster"] == cluster) & catalog["source"].isin(candidate_sources)
                                     & ~catalog.index.isin(rated)]
            if candidates.empty:
                continue
            chosen = rng.permutation(len(history))[:max_context]
            context = history.iloc[chosen]
            context_ids = context["item_index"].to_numpy(dtype=np.int64)
            context_features = torch.from_numpy(feature_matrix[embedding_ids[context_ids]][None]).to(device)
            values = torch.tensor(context["scaled_rating"].to_numpy()[None], dtype=torch.float32, device=device)
            mask = torch.ones(values.shape, dtype=torch.bool, device=device)
            centroid = torch.tensor(cluster_embeddings.loc[cluster, "average_embedding"][None],
                                    dtype=torch.float32, device=device)
            latent, cluster_codes = model.encode(context_features, values, mask, centroid)
            for start in range(0, len(candidates), batch_size):
                selected = candidates.iloc[start:start + batch_size]
                target_features = torch.from_numpy(feature_matrix[embedding_ids[selected.index.to_numpy()]])
                owners = torch.zeros(len(selected), dtype=torch.long, device=device)
                predictions = model.decode(latent, cluster_codes, target_features.to(device), owners)
                output = selected[["item_key", "source", "title", "cluster"]].copy()
                output["user_id"] = str(user_id)
                output["predicted_rating"] = rating_min + predictions.cpu().numpy() * (rating_max - rating_min)
                output["context_ratings"] = len(context)
                results.append(output[output_columns])
    if not results:
        return pd.DataFrame(columns=output_columns)
    result = pd.concat(results, ignore_index=True).sort_values("predicted_rating", ascending=False, kind="stable")
    return result.head(top_k).reset_index(drop=True) if top_k is not None else result.reset_index(drop=True)
