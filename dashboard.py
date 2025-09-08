import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================
# Judul Aplikasi
# ==========================
st.set_page_config(page_title="Clustering Dashboard", layout="wide")
st.title("Clustering Dashboard")

# ==========================
# Upload Data
# ==========================
uploaded_file = st.file_uploader("Masukkan sembarang Dataset (CSV/XLS/XLSX)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Coba baca data
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(data.head())

    # ==========================
    # Preprocessing
    # ==========================
    numeric_data = data.select_dtypes(include=["number"]).copy()

    if numeric_data.empty:
        st.error("❌ Dataset tidak punya kolom numerik untuk clustering.")
        st.stop()

    # Tangani missing values
    numeric_data = numeric_data.dropna()

    if numeric_data.shape[0] < 2:
        st.error("Dataset terlalu sedikit setelah menghapus missing values.")
        st.stop()

    # ==========================
    # Sidebar Controls
    # ==========================
    st.sidebar.header("⚙Pengaturan Clustering")
    num_clusters = st.sidebar.number_input(
        "Jumlah Cluster",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

    # ==========================
    # KMeans Clustering
    # ==========================
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clustered_data = data.loc[numeric_data.index].copy()
    clustered_data["Cluster"] = kmeans.fit_predict(numeric_data)

    st.subheader("🔎 Data dengan Label Cluster")
    st.dataframe(clustered_data.head())

    # ==========================
    # Visualisasi
    # ==========================
    st.subheader("Visualisasi Cluster")

    # Cluster Size Bar Chart
    cluster_counts = clustered_data["Cluster"].value_counts().reset_index(name="Count")
    cluster_counts.rename(columns={"index": "Cluster"}, inplace=True)
    fig_bar = px.bar(cluster_counts, x="Cluster", y="Count", title="Cluster Sizes")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter Plot 2D
    if numeric_data.shape[1] >= 2:
        x_axis = st.sidebar.selectbox("Pilih X-axis", numeric_data.columns, index=0)
        y_axis = st.sidebar.selectbox("Pilih Y-axis", numeric_data.columns, index=1)

        fig_scatter = px.scatter(
            clustered_data,
            x=x_axis,
            y=y_axis,
            color="Cluster",
            title=f"Scatter Plot: {y_axis} vs {x_axis}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Scatter Plot membutuhkan minimal 2 kolom numerik.")

    # Scatter Plot 3D
    if numeric_data.shape[1] >= 3:
        cols_3d = st.sidebar.multiselect(
            "Pilih 3 kolom untuk Scatter 3D",
            numeric_data.columns,
            default=numeric_data.columns[:3]
        )
        if len(cols_3d) == 3:
            fig_scatter3d = px.scatter_3d(
                clustered_data,
                x=cols_3d[0],
                y=cols_3d[1],
                z=cols_3d[2],
                color="Cluster",
                title=f"3D Scatter Plot: {cols_3d[0]}, {cols_3d[1]}, {cols_3d[2]}"
            )
            st.plotly_chart(fig_scatter3d, use_container_width=True)

    # PCA Projection (2D)
    if numeric_data.shape[1] >= 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_data)
        clustered_data["PCA1"] = pca_result[:, 0]
        clustered_data["PCA2"] = pca_result[:, 1]

        fig_pca = px.scatter(
            clustered_data,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            title="PCA Projection (2D)"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # Box Plot
    col_box = st.sidebar.selectbox("Pilih kolom untuk Box Plot", numeric_data.columns)
    fig_box = px.box(clustered_data, y=col_box, color="Cluster", title=f"Box Plot: {col_box}")
    st.plotly_chart(fig_box, use_container_width=True)

    # Histogram
    col_hist = st.sidebar.selectbox("Pilih kolom untuk Histogram", numeric_data.columns)
    fig_hist = px.histogram(
        clustered_data,
        x=col_hist,
        color="Cluster",
        barmode="overlay",
        title=f"Histogram: {col_hist}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr = numeric_data.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("👆 Silakan upload file CSV/XLS/XLSX untuk memulai.")
