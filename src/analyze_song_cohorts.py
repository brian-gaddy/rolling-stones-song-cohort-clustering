from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

RAW_PATH = Path("data/raw/rolling_stones_spotify.csv")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")

AUDIO_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness", "liveness",
    "loudness", "speechiness", "tempo", "valence", "popularity", "duration_ms"
]


def load_and_clean(path=RAW_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
    return df


def build_cluster_features(df):
    features = [col for col in AUDIO_FEATURES if col in df.columns]
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, features


def choose_k_by_silhouette(X_scaled, k_values=range(2, 9)):
    scores = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_clusters(df):
    X_scaled, features = build_cluster_features(df)
    best_k, scores = choose_k_by_silhouette(X_scaled)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    clustered = df.copy()
    clustered["cluster"] = labels

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    embedding = tsne.fit_transform(X_scaled)
    clustered["tsne_dim1"] = embedding[:, 0]
    clustered["tsne_dim2"] = embedding[:, 1]
    return clustered, features, best_k, scores


def save_outputs(clustered, features, best_k, scores):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(PROCESSED_DIR / "rolling_stones_song_clusters.csv", index=False)

    cluster_profile = clustered.groupby("cluster")[features].mean().round(3)
    cluster_profile["song_count"] = clustered.groupby("cluster").size()
    cluster_profile.to_csv(PROCESSED_DIR / "cluster_profile.csv")

    with open(PROCESSED_DIR / "model_selection_scores.json", "w", encoding="utf-8") as f:
        json.dump({"best_k": int(best_k), "silhouette_scores": scores}, f, indent=2)

    plt.figure(figsize=(9, 6))
    plt.scatter(clustered["tsne_dim1"], clustered["tsne_dim2"], c=clustered["cluster"], alpha=0.75)
    plt.title("Song Cohorts Visualized with t-SNE")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clusters_tsne_generated.png", dpi=180)
    plt.close()


def main():
    df = load_and_clean()
    clustered, features, best_k, scores = fit_clusters(df)
    save_outputs(clustered, features, best_k, scores)
    print(f"Completed clustering with k={best_k}.")


if __name__ == "__main__":
    main()
