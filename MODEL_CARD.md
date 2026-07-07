# Model Card: Rolling Stones Song Cohort Clustering

## Model Purpose

This project creates interpretable song cohorts from Spotify audio-feature data for Rolling Stones tracks. The cohorts are intended for exploratory recommendation strategy, playlist design, and music catalog analysis.

## Model Type

Primary model: **KMeans clustering**  
Comparison models: **Gaussian Mixture Model** and **DBSCAN**  
Visualization methods: **t-SNE** and **PCA**

## Intended Use

The model is useful for grouping songs with similar audio characteristics. Example uses include:

- Playlist seed creation
- Similar-song recommendation analysis
- Album and catalog exploration
- Music cohort labeling based on audio traits
- Portfolio demonstration of unsupervised machine learning

## Not Intended For

This model should not be treated as a production recommendation engine by itself. It does not include user behavior signals such as skips, saves, replays, playlist additions, likes, or collaborative filtering data.

## Input Features

The clustering pipeline uses available Spotify-style audio and track features, including:

- acousticness
- danceability
- energy
- instrumentalness
- liveness
- loudness
- speechiness
- tempo
- valence
- popularity
- duration_ms

## Preprocessing

The pipeline removes exact duplicate rows, derives release year from release date, fills missing numerical feature values with feature medians, and standardizes numerical features with `StandardScaler` before clustering.

## Model Selection

The pipeline evaluates KMeans cluster counts using silhouette score. It also compares KMeans against Gaussian Mixture and DBSCAN baselines. KMeans remains the default model because it is simple, explainable, stable, and easy to translate into business-friendly song cohorts.

## Outputs

The pipeline generates:

- clustered song-level dataset
- cluster profile table
- model comparison table
- silhouette score JSON
- fitted scaler artifact
- fitted KMeans model artifact
- fitted PCA artifact
- cluster metadata artifact
- t-SNE and PCA visualizations

## Cluster Naming

Cluster names are generated from each cluster's strongest audio traits relative to the global cluster profile. Example labels may include high-energy, groove, acoustic, instrumental, upbeat, popular, or live-feel cohorts.

## Limitations

The model is based on audio features and popularity only. It does not account for listener context, user taste, listening history, geography, seasonality, playlist behavior, or long-term engagement outcomes. t-SNE is used only for visualization and should not be interpreted as a formal clustering model.

## Future Improvements

- Add collaborative filtering or user behavior data.
- Compare additional algorithms such as HDBSCAN or hierarchical clustering.
- Add cluster stability checks across random seeds.
- Deploy the Streamlit dashboard and add a live app link.
- Add automated regeneration of model artifacts in CI.
