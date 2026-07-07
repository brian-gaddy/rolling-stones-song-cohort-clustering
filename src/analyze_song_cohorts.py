from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

RAW_PATH = Path("data/raw/rolling_stones_spotify.csv")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")
MODELS_DIR = Path("models")

AUDIO_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness", "liveness",
    "loudness", "speechiness", "tempo", "valence", "popularity", "duration_ms"
]

INTERPRETABLE_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness", "liveness",
    "loudness", "speechiness", "tempo", "valence", "popularity"
]


def load_and_clean(path=RAW_PATH):
    """Load the Rolling Stones Spotify dataset and apply repeatable cleaning rules."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
    return df


def build_cluster_features(df):
    """Return standardized audio features for clustering."""
    features = [col for col in AUDIO_FEATURES if col in df.columns]
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features, scaler


def choose_k_by_silhouette(X_scaled, k_values=range(2, 9)):
    """Evaluate KMeans cluster counts with silhouette score."""
    scores = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def compare_clustering_models(X_scaled, best_k):
    """Compare KMeans, Gaussian Mixture, and DBSCAN as baseline clustering options."""
    results = []

    kmeans_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)
    results.append({
        "model": "KMeans",
        "cluster_count": int(len(set(kmeans_labels))),
        "silhouette_score": float(silhouette_score(X_scaled, kmeans_labels)),
        "notes": "Selected baseline model because it is simple, interpretable, and stable for cohort creation."
    })

    gmm_labels = GaussianMixture(n_components=best_k, random_state=42).fit_predict(X_scaled)
    results.append({
        "model": "GaussianMixture",
        "cluster_count": int(len(set(gmm_labels))),
        "silhouette_score": float(silhouette_score(X_scaled, gmm_labels)),
        "notes": "Useful comparison when clusters may overlap or have different covariance structures."
    })

    dbscan_labels = DBSCAN(eps=1.8, min_samples=8).fit_predict(X_scaled)
    non_noise = dbscan_labels != -1
    dbscan_cluster_count = len(set(dbscan_labels[non_noise]))
    if dbscan_cluster_count >= 2 and non_noise.sum() > 1:
        dbscan_score = float(silhouette_score(X_scaled[non_noise], dbscan_labels[non_noise]))
    else:
        dbscan_score = None
    results.append({
        "model": "DBSCAN",
        "cluster_count": int(dbscan_cluster_count),
        "silhouette_score": dbscan_score,
        "notes": "Density-based comparison; noise handling can be helpful but may underperform when cohorts are broad."
    })

    return pd.DataFrame(results)


def create_cluster_names(cluster_profile):
    """Create business-friendly names from each cluster's strongest audio traits."""
    available = [c for c in INTERPRETABLE_FEATURES if c in cluster_profile.columns]
    global_means = cluster_profile[available].mean()
    names = {}

    for cluster_id, row in cluster_profile.iterrows():
        strengths = (row[available] - global_means).sort_values(ascending=False)
        top_traits = strengths.head(3).index.tolist()
        label_parts = []

        if "energy" in top_traits or "loudness" in top_traits:
            label_parts.append("High-Energy")
        if "danceability" in top_traits or "tempo" in top_traits:
            label_parts.append("Groove")
        if "acousticness" in top_traits:
            label_parts.append("Acoustic")
        if "instrumentalness" in top_traits:
            label_parts.append("Instrumental")
        if "valence" in top_traits:
            label_parts.append("Upbeat")
        if "popularity" in top_traits:
            label_parts.append("Popular")
        if "liveness" in top_traits:
            label_parts.append("Live-Feel")
        if not label_parts:
            label_parts = [trait.replace("_", " ").title() for trait in top_traits[:2]]

        names[cluster_id] = " / ".join(dict.fromkeys(label_parts)) + " Cohort"

    return names


def fit_clusters(df):
    X_scaled, features, scaler = build_cluster_features(df)
    best_k, scores = choose_k_by_silhouette(X_scaled)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    clustered = df.copy()
    clustered["cluster"] = labels

    preliminary_profile = clustered.groupby("cluster")[features].mean().round(3)
    preliminary_profile["song_count"] = clustered.groupby("cluster").size()
    cluster_names = create_cluster_names(preliminary_profile)
    clustered["cluster_name"] = clustered["cluster"].map(cluster_names)

    pca = PCA(n_components=2, random_state=42)
    pca_embedding = pca.fit_transform(X_scaled)
    clustered["pca_dim1"] = pca_embedding[:, 0]
    clustered["pca_dim2"] = pca_embedding[:, 1]

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    embedding = tsne.fit_transform(X_scaled)
    clustered["tsne_dim1"] = embedding[:, 0]
    clustered["tsne_dim2"] = embedding[:, 1]

    model_comparison = compare_clustering_models(X_scaled, best_k)

    artifacts = {
        "scaler": scaler,
        "kmeans_model": model,
        "pca_model": pca,
        "features": features,
        "cluster_names": cluster_names,
    }

    return clustered, features, best_k, scores, model_comparison, artifacts


def save_outputs(clustered, features, best_k, scores, model_comparison, artifacts):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    clustered.to_csv(PROCESSED_DIR / "rolling_stones_song_clusters.csv", index=False)

    cluster_profile = clustered.groupby(["cluster", "cluster_name"])[features].mean().round(3)
    cluster_profile["song_count"] = clustered.groupby(["cluster", "cluster_name"]).size()
    cluster_profile.to_csv(PROCESSED_DIR / "cluster_profile.csv")

    model_comparison.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    with open(PROCESSED_DIR / "model_selection_scores.json", "w", encoding="utf-8") as f:
        json.dump({"best_k": int(best_k), "silhouette_scores": scores}, f, indent=2)

    joblib.dump(artifacts["scaler"], MODELS_DIR / "scaler.joblib")
    joblib.dump(artifacts["kmeans_model"], MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(artifacts["pca_model"], MODELS_DIR / "pca_model.joblib")
    joblib.dump({"features": artifacts["features"], "cluster_names": artifacts["cluster_names"]}, MODELS_DIR / "cluster_metadata.joblib")

    plt.figure(figsize=(9, 6))
    plt.scatter(clustered["tsne_dim1"], clustered["tsne_dim2"], c=clustered["cluster"], alpha=0.75)
    plt.title("Song Cohorts Visualized with t-SNE")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clusters_tsne_generated.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(clustered["pca_dim1"], clustered["pca_dim2"], c=clustered["cluster"], alpha=0.75)
    plt.title("Song Cohorts Visualized with PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clusters_pca_generated.png", dpi=180)
    plt.close()


def main():
    df = load_and_clean()
    clustered, features, best_k, scores, model_comparison, artifacts = fit_clusters(df)
    save_outputs(clustered, features, best_k, scores, model_comparison, artifacts)
    print(f"Completed clustering with k={best_k}. Outputs saved to {PROCESSED_DIR}, {FIGURES_DIR}, and {MODELS_DIR}.")


if __name__ == "__main__":
    main()
