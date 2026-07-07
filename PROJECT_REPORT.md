# Project Report: Rolling Stones Song Cohort Clustering

## Executive Summary

This project creates song cohorts from Spotify audio features for Rolling Stones tracks. The goal is to group similar songs so that recommendation, playlist, and content teams can better understand how songs relate to one another by musical attributes such as danceability, energy, valence, acousticness, tempo, loudness, and popularity.

## Business Context

Recommendation systems rely on similarity. If a streaming platform can identify groups of songs with similar audio characteristics, it can recommend songs more effectively, build more coherent playlists, and improve listener engagement.

## Analytical Workflow

1. Inspect the dataset for duplicates, missing values, irrelevant fields, and outliers.
2. Clean and refine the dataset for modeling.
3. Explore song-level audio features and popularity.
4. Evaluate album-level popularity patterns.
5. Standardize numerical features.
6. Use clustering to create song cohorts.
7. Use dimensionality reduction to visualize high-dimensional patterns.
8. Profile clusters so the cohorts are interpretable.

## Methodology

The clustering pipeline uses standardized Spotify audio features and KMeans clustering. Silhouette scores are used to compare candidate cluster counts. t-SNE is used as a visualization technique to project the high-dimensional feature space into two dimensions.

## Interpretation

The t-SNE visualization shows two clear broad song cohorts. This suggests that the selected audio features contain enough structure to separate Rolling Stones songs into distinct similarity groups. Cluster profiling should be used to label those groups based on their dominant feature patterns, such as higher energy, greater acousticness, stronger danceability, or higher popularity.

## Recommendations

1. Use clusters as playlist seeds for recommendation experiments.
2. Label each cluster based on dominant feature patterns to make cohorts business-friendly.
3. Compare cluster membership by album to understand how Rolling Stones albums differ musically.
4. Use popularity within each cluster to identify representative songs for recommendations.
5. Treat t-SNE as a visualization tool, not the clustering model itself.

## Limitations

This project clusters songs by audio and popularity attributes only. It does not include user listening behavior, skip rates, playlist saves, or collaborative filtering signals. A production recommendation system should combine audio similarity with listener behavior.

## Future Improvements

- Add PCA comparison alongside t-SNE.
- Add DBSCAN or Gaussian Mixture Models for comparison.
- Add a cluster naming function based on feature profiles.
- Deploy the Streamlit dashboard.
- Add GitHub Actions for testing.
- Add model artifact export for reproducible cluster assignment.
