import pandas as pd
from src.analyze_song_cohorts import load_and_clean, build_cluster_features


def test_load_and_clean_removes_duplicates(tmp_path):
    df = pd.DataFrame({
        "name": ["A", "A"],
        "album": ["X", "X"],
        "release_date": ["2020-01-01", "2020-01-01"],
        "danceability": [0.5, 0.5],
        "energy": [0.7, 0.7],
        "popularity": [50, 50],
    })
    path = tmp_path / "songs.csv"
    df.to_csv(path, index=False)
    clean = load_and_clean(path)
    assert len(clean) == 1
    assert "release_year" in clean.columns


def test_build_cluster_features_returns_numeric_matrix():
    df = pd.DataFrame({
        "danceability": [0.5, 0.7, 0.9],
        "energy": [0.7, 0.6, 0.8],
        "popularity": [50, 60, 70],
    })
    X_scaled, features = build_cluster_features(df)
    assert X_scaled.shape == (3, 3)
    assert features == ["danceability", "energy", "popularity"]
