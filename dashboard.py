# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Clustering Dashboard", layout="wide")

st.title("Clustering Dashboard (Interactive)")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Sidebar controls
    st.sidebar.header("Settings")
    num_clusters = st.sidebar.slider("Number of clusters (K)", 2, 10, 3)
    features = st.sidebar.multiselect("Select features for clustering",
                                      data.select_dtypes("number").columns.tolist(),
                                      default=list(data.select_dtypes("number").columns)[:3])

    # Run clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data["Cluster"] = kmeans.fit_predict(data[features])

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Scatter plot (2D)
    if len(features) >= 2:
        st.subheader("Scatter Plot")
        fig = px.scatter(data, x=features[0], y=features[1], color="Cluster", title="Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)

    # 3D scatter
    if len(features) >= 3:
        st.subheader("3D Scatter Plot")
        fig3d = px.scatter_3d(data, x=features[0], y=features[1], z=features[2],
                              color="Cluster", title="3D Scatter")
        st.plotly_chart(fig3d, use_container_width=True)

    # Cluster size bar chart
    st.subheader("Cluster Sizes")
    cluster_counts = data["Cluster"].value_counts().reset_index()
    fig_bar = px.bar(cluster_counts, x="index", y="Cluster",
                     labels={"index": "Cluster", "Cluster": "Count"},
                     title="Cluster Sizes")
    st.plotly_chart(fig_bar, use_container_width=True)

    # PCA projection
    if len(features) >= 2:
        st.subheader("PCA Projection (2D)")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data[features])
        data["PCA1"], data["PCA2"] = pca_result[:, 0], pca_result[:, 1]
        fig_pca = px.scatter(data, x="PCA1", y="PCA2", color="Cluster", title="PCA Projection")
        st.plotly_chart(fig_pca, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = data[features].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Box plot
    col_choice = st.selectbox("Select column for Box Plot", features)
    if col_choice:
        st.subheader(f"Box Plot of {col_choice}")
        fig_box = px.box(data, y=col_choice, color="Cluster", title=f"Box Plot of {col_choice}")
        st.plotly_chart(fig_box, use_container_width=True)

else:
    st.info("👆 Upload a CSV file to start.")
