import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------
# Konfigurasi halaman
# ------------------------------
st.set_page_config(page_title="Clustering Dashboard", layout="wide")
st.title("Clustering Dashboard (Interactive)")

# ------------------------------
# Upload file
# ------------------------------
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Baca file sesuai ekstensi
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.success(f"File berhasil dibaca: {uploaded_file.name}")

    # ------------------------------
    # Sidebar controls
    # ------------------------------
    st.sidebar.header("Settings")
    num_clusters = st.sidebar.slider("Number of clusters (K)", 2, 10, 3)
    numeric_cols = data.select_dtypes("number").columns.tolist()
    features = st.sidebar.multiselect(
        "Select features for clustering",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )

    # ------------------------------
    # Jalankan clustering
    # ------------------------------
    if len(features) >= 2:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data["Cluster"] = kmeans.fit_predict(data[features])

        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # ------------------------------
        # Scatter Plot (2D)
        # ------------------------------
        st.subheader("Scatter Plot (2D)")
        fig = px.scatter(data, x=features[0], y=features[1], color="Cluster",
                         title="Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------
        # 3D Scatter Plot
        # ------------------------------
        if len(features) >= 3:
            st.subheader("3D Scatter Plot")
            fig3d = px.scatter_3d(data, x=features[0], y=features[1], z=features[2],
                                  color="Cluster", title="3D Scatter Plot")
            st.plotly_chart(fig3d, use_container_width=True)

        # ------------------------------
        # Cluster Size Bar Chart
        # ------------------------------
        st.subheader("Cluster Sizes")
        cluster_counts = data["Cluster"].value_counts().reset_index()
        fig_bar = px.bar(cluster_counts, x="index", y="Cluster",
                         labels={"index": "Cluster", "Cluster": "Count"},
                         title="Cluster Sizes")
        st.plotly_chart(fig_bar, use_container_width=True)

        # ------------------------------
        # PCA Projection
        # ------------------------------
        if len(features) >= 2:
            st.subheader("PCA Projection (2D)")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data[features])
            data["PCA1"], data["PCA2"] = pca_result[:, 0], pca_result[:, 1]
            fig_pca = px.scatter(data, x="PCA1", y="PCA2", color="Cluster",
                                 title="PCA Projection (2D)")
            st.plotly_chart(fig_pca, use_container_width=True)

        # ------------------------------
        # Correlation Heatmap
        # ------------------------------
        st.subheader("Correlation Heatmap")
        corr = data[features].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                             title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

        # ------------------------------
        # Box Plot (pilihan kolom)
        # ------------------------------
        col_choice = st.selectbox("Select column for Box Plot", features)
        if col_choice:
            st.subheader(f"Box Plot of {col_choice}")
            fig_box = px.box(data, y=col_choice, color="Cluster",
                             title=f"Box Plot of {col_choice}")
            st.plotly_chart(fig_box, use_container_width=True)

        # ------------------------------
        # Download hasil clustering
        # ------------------------------
        csv_download = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download clustered data (CSV)",
            data=csv_download,
            file_name="clustered_data.csv",
            mime="text/csv",
        )
    else:
        st.error("Pilih minimal 2 fitur numerik untuk clustering.")
else:
    st.info("👆 Upload file CSV/XLS/XLSX untuk mulai analisis.")
