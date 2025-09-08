# dashboard_maker.py
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class Dashboard_Maker:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.data = pd.read_csv(self.data_path)
        self.cluster_labels = None
        self.figures = []
        self.html_path = os.path.join(self.output_dir, "dashboard.html")
        self.png_paths, self.pdf_paths, self.svg_paths, self.html_paths = [], [], [], []
        self.cluster_column = 'Cluster'
        self.num_clusters = 3
        self.cluster_colors = px.colors.qualitative.Plotly
        self.cluster_color_map = {}
        self._prepare_data()

    def _prepare_data(self):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.data[self.cluster_column] = kmeans.fit_predict(
            self.data.select_dtypes(include=['number'])
        )
        self.cluster_labels = self.data[self.cluster_column].unique()
        self.cluster_color_map = {
            label: self.cluster_colors[i % len(self.cluster_colors)]
            for i, label in enumerate(self.cluster_labels)
        }

    # --- Grafik dasar ---
    def create_scatter_plot(self, x_col, y_col):
        fig = px.scatter(self.data, x=x_col, y=y_col, color=self.cluster_column,
                         color_discrete_map=self.cluster_color_map,
                         title=f'Scatter Plot of {y_col} vs {x_col}')
        self.figures.append(fig)
        return fig

    def create_histogram(self, col):
        fig = px.histogram(self.data, x=col, color=self.cluster_column,
                           color_discrete_map=self.cluster_color_map,
                           title=f'Histogram of {col}', barmode='overlay')
        self.figures.append(fig)
        return fig

    def create_box_plot(self, col):
        fig = px.box(self.data, y=col, color=self.cluster_column,
                     color_discrete_map=self.cluster_color_map,
                     title=f'Box Plot of {col}')
        self.figures.append(fig)
        return fig

    # --- Grafik tambahan ---
    def create_cluster_size_bar(self):
        counts = self.data[self.cluster_column].value_counts().reset_index()
        fig = px.bar(counts, x="index", y=self.cluster_column,
                     title="Cluster Sizes", labels={"index": "Cluster", self.cluster_column: "Count"})
        self.figures.append(fig)
        return fig

    def create_centroid_heatmap(self):
        features = self.data.select_dtypes(include=['number']).drop(columns=[self.cluster_column])
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(features)
        centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
        fig = px.imshow(centroid_df, text_auto=True, aspect="auto", title="Cluster Centroids Heatmap")
        self.figures.append(fig)
        return fig

    def create_correlation_heatmap(self):
        corr = self.data.select_dtypes(include=['number']).corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        self.figures.append(fig)
        return fig

    def create_scatter_matrix(self):
        num_cols = self.data.select_dtypes(include=['number']).columns
        fig = px.scatter_matrix(self.data, dimensions=num_cols, color=self.cluster_column,
                                color_discrete_map=self.cluster_color_map,
                                title="Scatter Matrix of Features")
        self.figures.append(fig)
        return fig

    def create_3d_scatter(self, x_col, y_col, z_col):
        fig = px.scatter_3d(self.data, x=x_col, y=y_col, z=z_col, color=self.cluster_column,
                            color_discrete_map=self.cluster_color_map,
                            title=f'3D Scatter Plot ({x_col}, {y_col}, {z_col})')
        self.figures.append(fig)
        return fig

    def create_pca_projection(self):
        features = self.data.select_dtypes(include=['number']).drop(columns=[self.cluster_column])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        self.data["PCA1"], self.data["PCA2"] = pca_result[:, 0], pca_result[:, 1]
        fig = px.scatter(self.data, x="PCA1", y="PCA2", color=self.cluster_column,
                         color_discrete_map=self.cluster_color_map,
                         title="PCA Projection (2D)")
        self.figures.append(fig)
        return fig

    # --- Dashboard export ---
    def create_dashboard(self):
        with open(self.html_path, 'w') as f:
            f.write('<html><head><title>Clustering Dashboard</title></head><body>')
            f.write('<h1>Clustering Dashboard</h1>')
            for i, fig in enumerate(self.figures):
                f.write(f'<h2>Figure {i+1}</h2>')
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('</body></html>')
        print(f'Dashboard saved to {self.html_path}')
