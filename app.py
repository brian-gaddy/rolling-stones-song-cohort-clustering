import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Rolling Stones Song Cohorts", layout="wide")
st.title("Rolling Stones Spotify Song Cohort Dashboard")

@st.cache_data
def load_clustered_data():
    return pd.read_csv("data/processed/rolling_stones_song_clusters.csv")

@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv("data/processed/model_comparison.csv")
    except FileNotFoundError:
        return pd.DataFrame()


df = load_clustered_data()
model_comparison = load_model_comparison()

if "cluster_name" not in df.columns:
    df["cluster_name"] = "Cluster " + df["cluster"].astype(str)

cluster_options = sorted(df["cluster_name"].unique())
selected_clusters = st.sidebar.multiselect("Song cohort", cluster_options, default=cluster_options)

album_options = sorted(df["album"].dropna().unique()) if "album" in df.columns else []
selected_albums = st.sidebar.multiselect("Album", album_options, default=[])

filtered = df[df["cluster_name"].isin(selected_clusters)]
if selected_albums and "album" in filtered.columns:
    filtered = filtered[filtered["album"].isin(selected_albums)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Songs", f"{len(filtered):,}")
if "album" in filtered.columns:
    c2.metric("Albums", f"{filtered['album'].nunique():,}")
if "popularity" in filtered.columns:
    c3.metric("Avg Popularity", f"{filtered['popularity'].mean():.1f}")
c4.metric("Cohorts", f"{filtered['cluster_name'].nunique():,}")

st.subheader("t-SNE Cohort Map")
st.plotly_chart(
    px.scatter(
        filtered,
        x="tsne_dim1",
        y="tsne_dim2",
        color="cluster_name",
        hover_data=[c for c in ["name", "album", "popularity", "danceability", "energy", "valence"] if c in filtered.columns],
        title="Song Cohorts Visualized with t-SNE",
    ),
    use_container_width=True,
)

if {"pca_dim1", "pca_dim2"}.issubset(filtered.columns):
    st.subheader("PCA Cohort Map")
    st.plotly_chart(
        px.scatter(
            filtered,
            x="pca_dim1",
            y="pca_dim2",
            color="cluster_name",
            hover_data=[c for c in ["name", "album", "popularity", "danceability", "energy", "valence"] if c in filtered.columns],
            title="Song Cohorts Visualized with PCA",
        ),
        use_container_width=True,
    )

if "popularity" in filtered.columns and "album" in filtered.columns:
    album_popularity = filtered.groupby("album", as_index=False)["popularity"].mean().sort_values("popularity", ascending=False).head(15)
    st.plotly_chart(px.bar(album_popularity, x="popularity", y="album", orientation="h", title="Top Albums by Average Popularity"), use_container_width=True)

st.subheader("Cluster Profile")
numeric_cols = [c for c in ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence", "popularity"] if c in filtered.columns]
st.dataframe(filtered.groupby("cluster_name")[numeric_cols].mean().round(3))

if not model_comparison.empty:
    st.subheader("Model Comparison")
    st.dataframe(model_comparison)

st.subheader("Executive Interpretation")
st.write(
    "Use the cohort map and cluster profile to identify musically similar Rolling Stones songs. "
    "The cluster names are automatically generated from each cohort's strongest audio traits, "
    "making the outputs easier to use for playlist design and recommendation strategy."
)
