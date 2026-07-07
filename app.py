import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Rolling Stones Song Cohorts", layout="wide")
st.title("Rolling Stones Spotify Song Cohort Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/rolling_stones_song_clusters.csv")

df = load_data()

cluster_options = sorted(df["cluster"].unique())
selected_clusters = st.sidebar.multiselect("Cluster", cluster_options, default=cluster_options)

album_options = sorted(df["album"].dropna().unique()) if "album" in df.columns else []
selected_albums = st.sidebar.multiselect("Album", album_options, default=album_options[:10] if album_options else [])

filtered = df[df["cluster"].isin(selected_clusters)]
if selected_albums and "album" in filtered.columns:
    filtered = filtered[filtered["album"].isin(selected_albums)]

c1, c2, c3 = st.columns(3)
c1.metric("Songs", f"{len(filtered):,}")
if "album" in filtered.columns:
    c2.metric("Albums", f"{filtered['album'].nunique():,}")
if "popularity" in filtered.columns:
    c3.metric("Avg Popularity", f"{filtered['popularity'].mean():.1f}")

st.plotly_chart(
    px.scatter(
        filtered,
        x="tsne_dim1",
        y="tsne_dim2",
        color="cluster",
        hover_data=[c for c in ["name", "album", "popularity", "danceability", "energy", "valence"] if c in filtered.columns],
        title="Song Cohorts Visualized with t-SNE",
    ),
    use_container_width=True,
)

if "popularity" in filtered.columns and "album" in filtered.columns:
    album_popularity = filtered.groupby("album", as_index=False)["popularity"].mean().sort_values("popularity", ascending=False).head(15)
    st.plotly_chart(px.bar(album_popularity, x="popularity", y="album", orientation="h", title="Top Albums by Average Popularity"), use_container_width=True)

st.subheader("Cluster Profile")
numeric_cols = [c for c in ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence", "popularity"] if c in filtered.columns]
st.dataframe(filtered.groupby("cluster")[numeric_cols].mean().round(3))
